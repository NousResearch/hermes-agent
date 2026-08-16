"""
``hermes validate-config`` — run all config validations and exit non-zero on
warnings.

Scans the user's config.yaml for common misconfigurations:

- auxiliary.<task> entries where base_url is empty but provider is "auto"
  (the task silently falls through the full auto-detection chain).
- pipeline.<setting> invalid types or values.
"""

import sys


def _get_user_config_raw() -> dict:
    """Load the user's config.yaml without merging DEFAULT_CONFIG defaults."""
    from hermes_cli.config import read_user_config_raw

    try:
        return read_user_config_raw()
    except Exception:
        return {}


def run_validate_config(args) -> int:
    """Run all config validators, print findings, return exit code."""
    user_config = _get_user_config_raw()
    warnings: list[str] = []

    # Validate auxiliary task configs — only check tasks the user explicitly
    # configured in their config.yaml (not built-in DEFAULT_CONFIG defaults).
    user_auxiliary = user_config.get("auxiliary", {})
    if isinstance(user_auxiliary, dict):
        for task_name, task_cfg in user_auxiliary.items():
            if not isinstance(task_cfg, dict):
                continue
            provider = str(task_cfg.get("provider", "")).strip()
            base_url = str(task_cfg.get("base_url", "")).strip()
            if not base_url and provider == "auto":
                warnings.append(
                    f"  WARNING: auxiliary.{task_name}: base_url is empty with "
                    f"provider 'auto'. May fall back unexpectedly."
                )

    # Validate pipeline config — check for common misconfigurations.
    user_pipeline = user_config.get("pipeline", {})
    if isinstance(user_pipeline, dict):
        # Validate max_revise_loops is positive int
        max_revise = user_pipeline.get("max_revise_loops")
        if max_revise is not None:
            if not isinstance(max_revise, int) or max_revise < 1:
                warnings.append(
                    f"  WARNING: pipeline.max_revise_loops must be a positive "
                    f"integer, got {max_revise!r}."
                )
        # Validate token_cap is None or positive int
        token_cap = user_pipeline.get("token_cap")
        if token_cap is not None:
            if not isinstance(token_cap, int) or token_cap < 1:
                warnings.append(
                    f"  WARNING: pipeline.token_cap must be null or a positive "
                    f"integer, got {token_cap!r}."
                )
        # Validate stage_owners is a dict
        stage_owners = user_pipeline.get("stage_owners")
        if stage_owners is not None and not isinstance(stage_owners, dict):
            warnings.append(
                f"  WARNING: pipeline.stage_owners must be a dict, "
                f"got {type(stage_owners).__name__}."
            )
        # Validate stage_owners values are spawnable profiles. A nonspawnable
        # owner (e.g. 'kensei', 'orchestrator') silently starves every task at
        # that pipeline stage — the dispatcher forces reassignment to the
        # configured owner each tick, then skips it as nonspawnable with no
        # alert. This gate makes that failure class impossible to ship.
        if isinstance(stage_owners, dict):
            nonspawnable = user_config.get("kanban", {}).get(
                "nonspawnable_profiles", []
            )
            import os
            from pathlib import Path

            _hm = os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))
            for _stage, _owner in stage_owners.items():
                if _owner in nonspawnable:
                    warnings.append(
                        f"  ERROR: pipeline.stage_owners['{_stage}'] = "
                        f"'{_owner}' is in kanban.nonspawnable_profiles and "
                        f"will block all tasks at that stage. Use a spawnable "
                        f"sub-profile (e.g. kensei-review)."
                    )
                    continue
                _pdir = Path(_hm) / "profiles" / _owner
                if not _pdir.is_dir():
                    warnings.append(
                        f"  ERROR: pipeline.stage_owners['{_stage}'] = "
                        f"'{_owner}' has no profile directory at {_pdir}; "
                        f"tasks at that stage cannot spawn."
                    )

    # Validate council config — panel diversity (D4) + types.
    user_council = user_config.get("council", {})
    if isinstance(user_council, dict):
        panel = user_council.get("panel")
        if panel is not None:
            if not isinstance(panel, list) or not panel:
                warnings.append(
                    "  WARNING: council.panel must be a non-empty list of "
                    "members, each with provider + model."
                )
            else:
                providers: list[str] = []
                for i, member in enumerate(panel):
                    if not isinstance(member, dict):
                        warnings.append(
                            f"  WARNING: council.panel[{i}] must be a dict with "
                            f"provider + model."
                        )
                        continue
                    if not str(member.get("provider", "")).strip():
                        warnings.append(
                            f"  WARNING: council.panel[{i}] missing provider."
                        )
                    if not str(member.get("model", "")).strip():
                        warnings.append(
                            f"  WARNING: council.panel[{i}] missing model."
                        )
                    prov = str(member.get("provider", "")).strip()
                    if prov:
                        providers.append(prov)
                    if not member.get("fallback"):
                        warnings.append(
                            f"  WARNING: council.panel[{i}] ({prov}) has no "
                            f"fallback chain — a provider outage drops this "
                            f"panellist."
                        )
                # D4: warn on same-provider panellists (single point of failure).
                dupes = {p for p in providers if providers.count(p) > 1}
                for p in sorted(dupes):
                    warnings.append(
                        f"  WARNING: council.panel has {providers.count(p)} "
                        f"members on provider '{p}'. Diversity across providers "
                        f"is recommended; an outage drops multiple panellists."
                    )
        token_cap = user_council.get("token_cap")
        if token_cap is not None and (not isinstance(token_cap, int) or token_cap < 1):
            warnings.append(
                f"  WARNING: council.token_cap must be null or a positive "
                f"integer, got {token_cap!r}."
            )
        max_revise = user_council.get("max_revise_loops")
        if max_revise is not None and (not isinstance(max_revise, int) or max_revise < 1):
            warnings.append(
                f"  WARNING: council.max_revise_loops must be a positive "
                f"integer, got {max_revise!r}."
            )

    if warnings:
        for w in warnings:
            print(w)
        return 1

    print("Config validation passed — no warnings found.")
    return 0
