"""Skills configuration defaults, assembled into DEFAULT_CONFIG by config_defaults."""

SKILLS_DEFAULTS = {
    "external_dirs": [],   # e.g. ["~/.agents/skills", "/shared/team-skills"]
    # Where skill_manage-created skills go (empty = profile-local dir). When set, new skills
    # land here AND agent-facing instructions name this path; expanded (~, ${VAR}), relative to
    # HERMES_HOME, scanned alongside the local dir.
    "create_dir": "",
    # In a git checkout, <root>/.hermes/skills/ and <root>/.agents/skills/ load as the
    # highest-precedence tier — ONLY if the root is in trusted_project_dirs. false = no scan, no
    # untrusted-skills notice.
    "project_discovery": True,
    # Trusted project roots; managed by `hermes skills trust` / `untrust`.
    "trusted_project_dirs": [],
    # Skill names pinned as fully loaded in every new session (CLI, TUI, gateway, cron, API).
    # Resolved once when the agent's prompt is first built; missing/disabled names warn and
    # skip; HERMES_IGNORE_RULES suppresses the list like the other auto-injected context.
    "auto_load": [],
    # Substitute ${HERMES_SKILL_DIR} / ${HERMES_SESSION_ID} in SKILL.md content.
    "template_vars": True,
    # Pre-execute !`cmd` snippets in SKILL.md, inlining stdout (dates, git state...). Off:
    # host-unapproved skill-author code; community hub installs never auto-execute (#63307).
    "inline_shell": False,
    "inline_shell_timeout": 10,  # seconds per !`cmd` snippet
    # Security-scan skills the agent writes via skill_manage. Off: the agent can run the same
    # code via terminal() ungated, so it mostly blocks prose with risky keywords. On: a
    # dangerous verdict is a tool error the agent can retry. Hub installs are always scanned.
    "guard_agent_created": False,
    # Advisory NVIDIA SkillEvaluator Tier 1 scan on `hermes skills install` (alongside the
    # enforcing built-in guard), only if `skillevaluator` is on PATH (uv tool install
    # "skillevaluator @ git+https://github.com/NVIDIA/SkillEvaluator.git"). Informational, never
    # blocking; secrets-class findings shown red. No-op without it.
    "tier1_advisory": True,
    # Approval gate for skill_manage mutations on BOTH foreground turns and the background
    # review fork. true = ALWAYS stage (SKILL.md too large for an inline prompt): /skills
    # pending, /skills diff <id>, /skills approve|reject <id>.
    "write_approval": False,
    # Optional lower budgets for main SKILL.md writes; non-positive/unset = off.
    # Growth past the cap is refused, but shrinking an already-large skill stays possible.
    "max_skill_md_chars": None,
    "size_warn_chars": None,
    # Audit ledger: every skill mutation appends to ~/.hermes/skills/.curator_ledger.jsonl with
    # before/after hashes (blobs under ~/.hermes/.curator_backups/blobs/); powers `hermes
    # curator ledger` / `rollback <entry-id>`. Never a gate — failures can't block.
    # See #79686.
    "ledger": True,
    # Size cap for that ledger: once the file grows past this, the next append rewrites it
    # through the unchanged-file dedup and, if still over, drops the oldest entries (0 = keep
    # the ledger append-only forever, the previous behaviour).
    "ledger_max_bytes": 5 * 1024 * 1024,
}
