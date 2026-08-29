"""Fleet config-drift checker — extends the enabled_skills allow-list check.

Exit contract (unchanged):
    0 — silent: healthy
    1 — drift: concise, deterministic `[DRIFT] ...` lines (no timestamps)
    2 — execution/parse error: `[DRIFT] Check failed: ...`

Approved surfaces only. The checker NEVER inspects or reports provider,
model, base_url, fallback_providers, fallback_reasoning, auxiliary,
custom_providers, credential pools, MCP servers, or secret values —
arbitrary changes under those surfaces must keep the exit at 0 (proven by
tests/test_config_drift_check_p13.py::test_provider_surface_mutations_*).

Schema-version expectations are DERIVED from the KenseiAgent code
(hermes_cli.config_defaults.DEFAULT_CONFIG). Failure to import that authority
is an execution error, never permission to use a stale copied literal. Root
config.yaml and every HERMES_HOME/profiles/*/config.yaml must carry the current
version — deferred
profile migrations are flagged intentionally (tracked as a separate migration
backlog; no exemptions here).
"""
import os
import sys
from pathlib import Path

import yaml

# hermes-drift-weekly references a dry-run affordance in this script
# (tests/test_defer_agent_jobs_p13.py::test_hermes_drift_weekly_stub).
# CONFIG_DRIFT_DRY_RUN=1 keeps the checker read-only — it already never
# mutates anything, so behaviour is unchanged; the variable is honored for
# contract compatibility and future read-path gating.
DRY_RUN = os.environ.get("CONFIG_DRIFT_DRY_RUN", "") == "1"

MIN_SKILLS = 45
expected = {
    "hermes-update", "kanban-ops", "kensei-triage-processor",
    "kensei-triage-investigator", "kanban-router", "feature-pipeline",
    "kensei-coordinator", "skill-broker-core", "skill-reroute",
    "denji-triage", "denji-skill-audit", "denji-write-skill", "sdlc-review",
    "kanban-blocked-task-resolution", "arxiv", "audit-engine",
    "avoid-ai-writing", "brand-voices", "coachos-voice", "content-pipeline",
    "content-pipeline-standards", "content-review", "cron-output-contract",
    "design-cron-output-format", "design-system-review", "dezzy-design-systems",
    "github-pr-workflow", "hermes-cron-operations", "lesson-delivery",
    "llm-wiki", "mailbox-agent", "mailbox-cleaner", "market-research",
    "matchdaymaestro-voice", "mnemosyne-health-check", "plenishd-voice",
    "remii-triage", "research-digest", "research-paper-synthesis",
    "sahil-linkedin-voice", "sahil-twitter-voice", "social-content",
    "system-script-patterns", "ui-pattern-library-research",
    "upstream-contribution-gate", "weekly-ideas-scan"
}

# Approved step-6 governance policy. Only these surfaces are read.
CURATOR_REQUIRED = {
    "enabled": True,
    "stale_after_days": 45,
    "archive_after_days": 120,
    "consolidate": False,
    "prune_builtins": False,
}
CURATOR_BACKUP_ENABLED = True

CURATOR_ENABLED_PROFILES = [
    "misa-misa", "remii", "wesker", "gojo", "octacon", "ceecee",
    "denji", "light", "quan", "dezzy", "kensei-review", "sirvir",
]
# mrhermagi must keep the curator OFF.
CURATOR_DISABLED_PROFILES = ["mrhermagi"]

MAX_TURNS_NONE_PROFILES = {
    "root", "octacon", "remii", "wesker", "quan", "dezzy", "denji", "light",
    "kensei-review", "orchestrator",
}
DELEGATION_CAP_PROFILES = {
    "root", "dezzy", "gojo", "kensei-review", "light", "octacon", "remii", "wesker",
}
MAX_ITERATIONS = 250
MAX_CONCURRENT_CHILDREN = 10

PERSONALITY_PROFILES = {
    "remii": "minimal", "wesker": "minimal", "gojo": "minimal",
    "octacon": "minimal", "mrhermagi": "minimal", "light": "minimal",
    "kensei-review": "minimal", "sirvir": "kawaii",
}

ROOT_CLI_TOOLSETS_REQUIRED = [
    "guard", "terminal", "file", "skills", "delegation",
    "memory", "session_search", "cronjob", "todo",
]

REQUIRED_PROFILES = (
    set(CURATOR_ENABLED_PROFILES)
    | set(CURATOR_DISABLED_PROFILES)
    | MAX_TURNS_NONE_PROFILES
    | DELEGATION_CAP_PROFILES
    | set(PERSONALITY_PROFILES)
) - {"root"}


def _derive_expected_schema_version() -> int | None:
    """Derive the expected config schema version from the live code authority.

    Return None when hermes_cli cannot be imported. run_checks() converts that
    into the script's rc=2 execution-error contract rather than trusting a
    stale duplicated schema literal.
    """
    try:
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from hermes_cli.config_defaults import DEFAULT_CONFIG
        ver = DEFAULT_CONFIG.get("_config_version")
        return int(ver) if isinstance(ver, int) else None
    except Exception:
        return None


# Derived at import time; the test suite asserts equality with DEFAULT_CONFIG.
EXPECTED_SCHEMA_VERSION = _derive_expected_schema_version()


class DriftCheckFatal(Exception):
    """Execution/parse failure → rc 2."""


class Drift:
    """Accumulates drift lines; carries the final exit code."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.code = 0

    def add(self, line: str) -> None:
        self.lines.append(f"[DRIFT] {line}")
        self.code = 1


def _load_yaml(path: Path) -> dict:
    """Load a YAML config; parse/IO errors raise DriftCheckFatal (→ rc 2)."""
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except (yaml.YAMLError, OSError, UnicodeDecodeError) as e:
        raise DriftCheckFatal(f"failed to parse {path}: {e}")
    if not isinstance(data, dict):
        raise DriftCheckFatal(f"config is not a YAML mapping: {path}")
    return data


def _check_root_skills(drift: Drift, cfg: dict) -> None:
    """Existing allow-list semantics. Absent/None enabled_skills means the
    current schema uses implicit discovery — NOTHING to validate, and this
    check must NOT exit early or suppress the other checks (fail-safe fix)."""
    skills_block = cfg.get("skills")
    if not isinstance(skills_block, dict):
        return
    configured = skills_block.get("enabled_skills")
    if configured is None:
        return
    if not isinstance(configured, (list, set)):
        drift.add(f"skills.enabled_skills has unexpected type "
                  f"{type(configured).__name__}")
        return
    skills = set(configured)
    if len(skills) < MIN_SKILLS:
        drift.add(f"enabled_skills dropped to {len(skills)} "
                  f"(expected >= {MIN_SKILLS})")
        return
    missing = expected - skills
    extra = skills - expected
    if missing:
        drift.add(f"Missing skills: {', '.join(sorted(missing))}")
    if extra:
        drift.add(f"Unexpected skills: {', '.join(sorted(extra))}")


def _record_schema_mismatch(mismatches: list[object], cfg: dict) -> None:
    actual = cfg.get("_config_version")
    if actual != EXPECTED_SCHEMA_VERSION:
        mismatches.append(actual)


def _flush_schema_mismatches(drift: Drift, mismatches: list[object]) -> None:
    if not mismatches:
        return
    buckets: dict[str, int] = {}
    for value in mismatches:
        key = "missing" if value is None else f"v{value}"
        buckets[key] = buckets.get(key, 0) + 1
    detail = ", ".join(f"{key}={buckets[key]}" for key in sorted(buckets))
    drift.add(
        f"schema: {len(mismatches)} config(s) have _config_version != "
        f"v{EXPECTED_SCHEMA_VERSION} ({detail})"
    )


def _check_curator_safety(drift: Drift, label: str, curator: object) -> None:
    if not isinstance(curator, dict):
        drift.add(f"{label}: curator block missing (safety values required)")
        return
    for key, want in CURATOR_REQUIRED.items():
        got = curator.get(key)
        if got != want:
            drift.add(f"{label}: curator.{key} = {got!r} != {want!r}")
    backup = curator.get("backup")
    got = backup.get("enabled") if isinstance(backup, dict) else None
    if got != CURATOR_BACKUP_ENABLED:
        drift.add(f"{label}: curator.backup.enabled = {got!r} != True")


def _check_budgets(drift: Drift, label: str, cfg: dict, profile: str) -> None:
    if profile in MAX_TURNS_NONE_PROFILES:
        agent = cfg.get("agent") or {}
        got = agent.get("max_turns") if isinstance(agent, dict) else None
        if got != "none":
            drift.add(f"{label}: agent.max_turns = {got!r} != 'none'")
    if profile in DELEGATION_CAP_PROFILES:
        delegation = cfg.get("delegation")
        delegation = delegation if isinstance(delegation, dict) else {}
        for key, want in (("max_iterations", MAX_ITERATIONS),
                          ("max_concurrent_children",
                           MAX_CONCURRENT_CHILDREN)):
            got = delegation.get(key)
            if got != want:
                drift.add(f"{label}: delegation.{key} = {got!r} != {want!r}")


def _check_personality_and_soul(drift: Drift, label: str, cfg: dict,
                                profile: str, home: Path) -> None:
    want = PERSONALITY_PROFILES.get(profile)
    if want is None:
        return
    display = cfg.get("display")
    got = display.get("personality") if isinstance(display, dict) else None
    if got != want:
        drift.add(f"{label}: display.personality = {got!r} != {want!r}")
    soul = home / "profiles" / profile / "SOUL.md"
    try:
        has_soul = soul.is_file() and soul.read_text().strip() != ""
    except OSError:
        has_soul = False
    if not has_soul:
        drift.add(f"SOUL.md missing or empty: {soul}")


def _check_root_cli_toolsets(drift: Drift, cfg: dict) -> None:
    pt = cfg.get("platform_toolsets")
    cli = pt.get("cli") if isinstance(pt, dict) else None
    if not isinstance(cli, list):
        drift.add("platform_toolsets.cli missing (required: "
                  + ", ".join(ROOT_CLI_TOOLSETS_REQUIRED) + ")")
        return
    have = set(cli)
    for required in ROOT_CLI_TOOLSETS_REQUIRED:
        if required not in have:
            drift.add(f"platform_toolsets.cli missing required toolset "
                      f"'{required}'")


def run_checks(home: Path, drift: Drift) -> None:
    """Populate `drift`. Raises DriftCheckFatal on parse errors (→ rc 2)."""
    if EXPECTED_SCHEMA_VERSION is None:
        raise DriftCheckFatal(
            "cannot derive current _config_version from hermes_cli.config_defaults"
        )

    root_path = home / "config.yaml"
    if not root_path.is_file():
        raise DriftCheckFatal(f"root config.yaml not found at {root_path}")
    root_cfg = _load_yaml(root_path)
    schema_mismatches: list[object] = []

    # Order matters for the fail-safe property: the skills check runs FIRST
    # but can never short-circuit the remaining checks.
    _check_root_skills(drift, root_cfg)
    _record_schema_mismatch(schema_mismatches, root_cfg)
    _check_curator_safety(drift, "config.yaml", root_cfg.get("curator"))
    _check_budgets(drift, "config.yaml", root_cfg, profile="root")
    _check_root_cli_toolsets(drift, root_cfg)

    profiles_dir = home / "profiles"
    seen_profiles: set[str] = set()
    profile_dirs = sorted(profiles_dir.iterdir()) if profiles_dir.is_dir() else []
    for profile_dir in profile_dirs:
        if not profile_dir.is_dir() or profile_dir.name.startswith((".", "_")):
            continue
        name = profile_dir.name
        cfg_path = profile_dir / "config.yaml"
        if not cfg_path.is_file():
            continue
        seen_profiles.add(name)
        cfg = _load_yaml(cfg_path)
        label = f"profiles/{name}/config.yaml"

        # Schema version: every profile config on disk must match the code.
        _record_schema_mismatch(schema_mismatches, cfg)

        if name in CURATOR_ENABLED_PROFILES:
            _check_curator_safety(drift, label, cfg.get("curator"))
        elif name in CURATOR_DISABLED_PROFILES:
            curator = cfg.get("curator")
            got = curator.get("enabled") if isinstance(curator, dict) else None
            if got is not False:
                drift.add(f"{label}: curator.enabled = {got!r} != False")

        # Budgets are independent of the curator scope (e.g. orchestrator is
        # budget-only). _check_budgets gates on profile membership itself.
        _check_budgets(drift, label, cfg, name)

        if name in PERSONALITY_PROFILES:
            _check_personality_and_soul(drift, label, cfg, name, home)

    if profiles_dir.is_dir():
        for name in sorted(REQUIRED_PROFILES - seen_profiles):
            drift.add(f"required profile config missing: {name}")
    _flush_schema_mismatches(drift, schema_mismatches)


def main() -> int:
    home = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
    drift = Drift()
    try:
        run_checks(home, drift)
    except DriftCheckFatal as e:
        print(f"[DRIFT] Check failed: {e}")
        return 2
    except Exception as e:  # defensive: unexpected execution failure
        print(f"[DRIFT] Check failed: {e}")
        return 2
    if drift.lines:
        print("\n".join(drift.lines))
    return drift.code


if __name__ == "__main__":
    sys.exit(main())