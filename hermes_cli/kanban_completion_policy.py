"""Operator-declared required checks for GitHub completion contracts.

GitHub's own policy is the preferred source of truth, but it is not always
readable: a private repository on a free plan exposes no
``branchProtectionRule`` and answers 403 for the Repository Rules API, so a
card whose contract could only be judged from that API would be impossible to
complete. For those repositories the required checks are *declared* in
``config.yaml``; they are never inferred from whichever checks happen to be
green on the head commit, which is why "no declared policy" fails closed
instead of accepting an all-green commit.

Config shape (``required_checks`` is the only key today; the per-repository
dict is what lets the policy grow without a migration)::

    kanban:
      completion_checks:
        "acme/repo":
          required_checks: ["build", "unit-tests"]

A bare list is accepted as shorthand for ``{"required_checks": [...]}``.
"""
from __future__ import annotations

CONFIG_DOTPATH = "kanban.completion_checks"


def configured_required_checks(repo: str) -> tuple[list[str], str | None]:
    """Declared check names for ``repo`` (``OWNER/REPO``) and a config problem.

    The config is read at call time through ``load_config_readonly()`` so each
    served profile answers with its own ``config.yaml``; a module-level
    snapshot would pin the launch profile's for every profile in the process.

    A malformed entry is *reported*, never raised: the acceptance receipt is
    the operator-visible surface and a bad config line must read as "fix this
    line", not as a generic GitHub API failure. Either way the caller is left
    with no declared checks, so the gate still fails closed.
    """
    section, problem = _completion_checks_section()
    if problem is not None:
        return [], problem
    entry = _entry_for(section, repo)
    if entry is None:
        return [], None
    if isinstance(entry, list):
        raw = entry
    elif isinstance(entry, dict):
        if "required_checks" not in entry:
            return [], (
                f"{CONFIG_DOTPATH}.{repo!r} declares no required_checks key; "
                f"add required_checks: [...] or remove the entry"
            )
        raw = entry.get("required_checks")
    else:
        return [], (
            f"{CONFIG_DOTPATH}.{repo!r} must be a mapping with a required_checks "
            f"list (or a bare list of check names)"
        )
    if not isinstance(raw, list):
        return [], f"{CONFIG_DOTPATH}.{repo!r}.required_checks must be a list of check names"
    names: list[str] = []
    for value in raw:
        if not isinstance(value, str) or not value.strip():
            return [], (
                f"{CONFIG_DOTPATH}.{repo!r}.required_checks must contain non-empty "
                f"check-name strings"
            )
        name = value.strip()
        if name not in names:
            names.append(name)
    return names, None


def _completion_checks_section() -> tuple[dict, str | None]:
    """``kanban.completion_checks`` as a mapping, plus a problem description."""
    from hermes_yaml import YAMLError

    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()
    except (OSError, YAMLError, ValueError):
        # The two ways a config read fails: the file cannot be read (OSError) or
        # cannot be parsed (YAMLError/ValueError). Both are an operator problem,
        # reported as one — never raised, since a bad config line must read as
        # "fix this line" on the receipt, not as a GitHub API failure.
        return {}, "config.yaml could not be loaded, so no required checks are declared"
    kanban_cfg = config.get("kanban") if isinstance(config, dict) else None
    section = kanban_cfg.get("completion_checks") if isinstance(kanban_cfg, dict) else None
    if section is None:
        return {}, None
    if not isinstance(section, dict):
        return {}, f"{CONFIG_DOTPATH} must be a mapping of OWNER/REPO to required checks"
    return section, None


def _entry_for(section: dict, repo: str):
    """Entry for ``repo``, matched case-insensitively the way GitHub treats owner/repo."""
    if repo in section:
        return section[repo]
    folded = repo.casefold()
    for key, value in section.items():
        if isinstance(key, str) and key.strip().casefold() == folded:
            return value
    return None
