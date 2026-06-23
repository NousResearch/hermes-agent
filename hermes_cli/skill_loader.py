"""Skill frontmatter validator.

Adds a strict ``validate_skill_frontmatter(path)`` function used by the skill
loader to reject malformed SKILL.md files before they reach the gateway
startup path.  Lives in ``hermes_cli`` (not ``agent``) so the validator can
be reused by the ``hermes skills`` CLI subcommand, the curator, and the
prompt-builder snapshot loader without creating a circular import.

The contract:

* ``validate_skill_frontmatter(path) -> (ok: bool, errors: list[str])``
* Returns ``(True, [])`` for a fully valid SKILL.md.
* Returns ``(False, [...])`` with one actionable error per problem found
  (``missing required field 'name'``, ``status 'production' is not one of
  {draft, vetted, deprecated}``, ``version '1.0' is not valid semver
  (expected MAJOR.MINOR.PATCH)``, ``name 'foo' does not match parent
  directory 'bar'``).
* Never raises — file-system and YAML parse errors are caught and reported
  in the returned ``errors`` list so callers can aggregate.

Required fields: ``name``, ``version``, ``status``, ``description``,
``author``.  ``status`` must be one of ``draft`` / ``vetted`` /
``deprecated``.  ``version`` must match MAJOR.MINOR.PATCH semver (per the
official semver.org regex).  ``name`` must match the parent directory name
so a skill at ``.../skills/<name>/SKILL.md`` cannot silently rename itself
in the metadata.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ── Constants ──────────────────────────────────────────────────────────────

REQUIRED_FIELDS: Tuple[str, ...] = (
    "name",
    "version",
    "status",
    "description",
    "author",
)

ALLOWED_STATUSES: Tuple[str, ...] = ("draft", "vetted", "deprecated")

# Official semver.org regular expression (slightly trimmed — we don't accept
# build metadata with leading ``+`` without separators, matching the strict
# reading on https://semver.org/#is-there-a-suggested-regular-expression-regex-to-check-a-semver-string).
_SEMVER_RE = re.compile(
    r"^(?P<major>0|[1-9]\d*)"
    r"\.(?P<minor>0|[1-9]\d*)"
    r"\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*)?"
    r"(?:\+[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*)?$"
)


# ── Local fallback frontmatter parser ──────────────────────────────────────


def _local_parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """Minimal YAML-frontmatter splitter used when agent.skill_utils is unavailable.

    Splits on ``---\\n`` markers at the top of the file.  Returns
    ``({}, content)`` if no markers are present (matching the agent helper's
    fallback semantics).  Body parsing is intentionally dumb — we use a
    plain ``key: value`` split because the validator only needs scalar
    fields; structured YAML can wait for the agent helper.
    """
    frontmatter: Dict[str, Any] = {}
    if not content.startswith("---"):
        return frontmatter, content

    # Find the closing "\n---" (allow trailing whitespace) on a fresh line.
    end_match = re.search(r"\n---\s*(?:\n|$)", content[3:])
    if not end_match:
        return frontmatter, content

    yaml_block = content[3 : end_match.start() + 3]
    body = content[end_match.end() + 3 :]

    for raw_line in yaml_block.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        # Strip matching quotes so ``name: "foo-bar"`` reads as ``foo-bar``.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        frontmatter.setdefault(key, value)

    return frontmatter, body


# ── Validation helpers ─────────────────────────────────────────────────────


def _is_semver(value: str) -> bool:
    return bool(_SEMVER_RE.match(value))


def _validate_field_present(errors: List[str], frontmatter: Dict[str, Any], field: str) -> Any:
    """Append a missing-field error and return ``None``; return value otherwise.

    Tracks whether the field is actually absent vs. present-but-empty so the
    error message is specific (``missing`` vs. ``empty``).
    """
    if field not in frontmatter:
        errors.append(f"missing required field '{field}'")
        return None
    value = frontmatter[field]
    # Allow ``None`` explicitly (YAML literal ``null``); treat everything
    # else that is "empty" as missing.
    if value is None or (isinstance(value, str) and not value.strip()):
        errors.append(f"missing required field '{field}' (empty value)")
        return None
    return value


# ── Public API ─────────────────────────────────────────────────────────────


def parse_skill_frontmatter(path: Path) -> Tuple[Dict[str, Any], str]:
    """Parse the YAML frontmatter block at the top of a SKILL.md file.

    Prefers ``agent.skill_utils.parse_frontmatter`` (full YAML support
    including nested ``metadata:`` blocks); falls back to the local
    splitter when the agent package isn't importable.  The agent helper
    is lazy-imported here so this module can be imported without
    triggering a hermes_constants cold-start chain.
    """
    try:  # pragma: no cover - exercised in integration, not unit tests
        from agent.skill_utils import parse_frontmatter as _agent_parse_frontmatter
    except Exception:  # pragma: no cover
        _agent_parse_frontmatter = None

    content = Path(path).read_text(encoding="utf-8")
    if _agent_parse_frontmatter is not None:
        return _agent_parse_frontmatter(content)
    return _local_parse_frontmatter(content)


def validate_skill_frontmatter(
    path: Path | str,
) -> Tuple[bool, List[str]]:
    """Validate a SKILL.md file's frontmatter.

    Parameters
    ----------
    path:
        Filesystem path to the SKILL.md file.  Strings are accepted for
        convenience but coerced to ``pathlib.Path``.

    Returns
    -------
    (ok, errors)
        ``ok`` is ``True`` iff ``errors`` is empty.  ``errors`` is a list
        of human-readable strings — each one names exactly one problem so
        callers can show the user a checklist of fixes.
    """
    errors: List[str] = []
    skill_path = Path(path)

    # 1. File exists and is readable.
    if not skill_path.is_file():
        return False, [f"SKILL.md not found at {skill_path}"]

    # 2. Parse frontmatter.  YAML errors are reported, not raised.
    try:
        content = skill_path.read_text(encoding="utf-8")
    except OSError as exc:
        return False, [f"could not read {skill_path}: {exc}"]

    try:
        frontmatter, _body = parse_skill_frontmatter(skill_path)
    except Exception as exc:  # pragma: no cover - defensive
        return False, [f"frontmatter parse error in {skill_path}: {exc}"]

    if not frontmatter:
        errors.append(
            "no YAML frontmatter block found "
            "(expected '---' on line 1 and a closing '---' line)"
        )
        # Cannot validate field-level rules without parsed metadata.
        return False, errors

    # 3. Required-field presence checks.
    name = _validate_field_present(errors, frontmatter, "name")
    version = _validate_field_present(errors, frontmatter, "version")
    status = _validate_field_present(errors, frontmatter, "status")
    _validate_field_present(errors, frontmatter, "description")
    _validate_field_present(errors, frontmatter, "author")

    # 4. Field-level rules — only run when the field is present so we
    #    don't double-report "missing X" AND "X is invalid".
    if status is not None:
        status_str = str(status).strip().lower()
        if status_str not in ALLOWED_STATUSES:
            allowed = ", ".join(ALLOWED_STATUSES)
            errors.append(
                f"status {status_str!r} is not one of {{{allowed}}}"
            )

    if version is not None:
        version_str = str(version).strip()
        if not _is_semver(version_str):
            errors.append(
                f"version {version_str!r} is not valid semver "
                f"(expected MAJOR.MINOR.PATCH, e.g. '1.2.3' or '0.1.0-rc.1')"
            )

    # 5. name ↔ parent directory consistency.
    if name is not None:
        name_str = str(name).strip()
        parent_dir_name = skill_path.parent.name
        if name_str != parent_dir_name:
            errors.append(
                f"name {name_str!r} does not match parent directory "
                f"{parent_dir_name!r} (skills must live in skills/<name>/)"
            )

    return (len(errors) == 0), errors


def validate_all_skills(
    skills_root: Path | str,
) -> Dict[str, Tuple[bool, List[str]]]:
    """Run :func:`validate_skill_frontmatter` against every SKILL.md under ``skills_root``.

    Walks one level deep (``skills_root/<name>/SKILL.md``) — matches the
    current Hermes skills layout.  Nested-category skills
    (``skills_root/<category>/<name>/SKILL.md``) are also walked because
    `iter_skill_index_files` in ``agent.skill_utils`` does the same.

    Returns a dict keyed by the absolute path of the SKILL.md file.
    """
    root = Path(skills_root)
    results: Dict[str, Tuple[bool, List[str]]] = {}
    if not root.is_dir():
        return results

    for skill_md in sorted(root.rglob("SKILL.md")):
        results[str(skill_md)] = validate_skill_frontmatter(skill_md)
    return results


def validate_or_warn(
    skill_path: Path | str,
    *,
    logger: Any = None,
) -> bool:
    """Validate a SKILL.md and log a warning on failure.

    Convenience wrapper for the agent loader: returns ``True`` iff the
    frontmatter is valid.  On failure, logs each error message at WARNING
    level (or via ``print`` if no logger is provided) and returns ``False``
    so the caller can skip auto-loading the skill.

    This function never raises — it's the wire-in point that keeps the
    gateway startup crash-free even when skills are malformed.

    Parameters
    ----------
    skill_path:
        Filesystem path to the SKILL.md file.
    logger:
        Optional logger instance.  If ``None``, falls back to the module
        logger (``hermes_cli.skill_loader``) and finally to ``print`` if
        the module logger has no handlers.  Pass ``logging.getLogger()``
        from your caller to keep log records attributed correctly.
    """
    ok, errors = validate_skill_frontmatter(skill_path)
    if ok:
        return True

    # Lazy logger resolution — avoids forcing every caller to construct one.
    if logger is None:
        import logging

        logger = logging.getLogger("hermes_cli.skill_loader")

    for err in errors:
        logger.warning("skill frontmatter validation failed for %s: %s", skill_path, err)
    return False


__all__ = [
    "REQUIRED_FIELDS",
    "ALLOWED_STATUSES",
    "parse_skill_frontmatter",
    "validate_skill_frontmatter",
    "validate_all_skills",
    "validate_or_warn",
]