# Operator review: host-side Hermes skills

Host-side skills live under `~/.hermes/skills/`, optional `skills.external_dirs`
paths, or team mirrors. They run in the agent process with the same privileges
as the operator account. **Review before trust** — see [SECURITY.md](../../SECURITY.md)
§2.4 and §2.5.

This document describes the **operator security audit layer**
(`tools/audit_hermes_skill_security.py`), which complements:

| Layer | Role | When |
|-------|------|------|
| **Skills Guard** (`tools/skills_guard.py`) | Install-time regex gate for hub/community skills | `hermes skills install`, quarantine |
| **AST deep scan** (`tools/skills_ast_audit.py`) | Optional dynamic-import hints | `hermes skills audit --deep` |
| **Operator security audit** (this layer) | Categorized review checklist | Before enabling `external_dirs`, after manual copies, periodic re-audit |

None of these are security boundaries. They reduce the chance of missing obvious
risk during human review.

## What the audit checks

| Category | Examples flagged |
|----------|------------------|
| `unsafe_instruction` | Prompt injection, “ignore previous instructions”, hidden HTML comments, deception |
| `shell_api_mutation` | `rm -rf /`, `sudo`, `skill_manage(delete)`, writes to `config.yaml` / `SOUL.md`, `cronjob(` |
| `credential_use` | `~/.hermes/.env`, `os.getenv` for secrets, curl with `$API_KEY`, hardcoded tokens |
| `hidden_background` | `terminal(background=True)`, `notify_on_complete=True`, `cronjob(`, `nohup`, “without informing the user” |
| `source_mirror_drift` | Content hash differs from hub lock, golden copy, or team mirror |

Verdicts:

- **pass** — no findings and no drift
- **review** — medium/high findings only; operator should read scripts and SKILL.md
- **fail** — critical findings and/or hash drift; do not enable until resolved

## Review workflow

### 1. New skill from an untrusted source

1. Install into quarantine or a staging directory (do not add to `external_dirs` yet).
2. Run Skills Guard (automatic on `hermes skills install`) and read the report.
3. Run the operator audit:

   ```bash
   python -m tools.audit_hermes_skill_security ~/.hermes/skills/.hub/quarantine/my-skill
   ```

4. Read **Python under `scripts/`**, not only `SKILL.md` (scripts run on the host).
5. Optional: `hermes skills audit my-skill --deep` for AST hints.
6. If verdict is **pass** or you accept **review** findings, move/install and record the content hash.

### 2. Team mirror or `external_dirs`

When a skill is copied from a git mirror or NFS share:

```bash
python -m tools.audit_hermes_skill_security \
  /path/to/installed/my-skill \
  --mirror /path/to/team-mirror/my-skill
```

Drift between installed and mirror copies produces `source_mirror_drift` findings.
Re-sync or investigate before the agent loads the skill.

### 3. Hub-installed skills (recorded hash)

After install, the hub lock stores `content_hash`. Re-audit after manual edits:

```bash
python -m tools.audit_hermes_skill_security \
  ~/.hermes/skills/my-skill \
  --expected-hash sha256:abc123...
```

Or use the combined CLI (Skills Guard + operator audit):

```bash
hermes skills audit my-skill
hermes skills audit my-skill --deep   # adds AST scan
```

### 4. Periodic re-audit

Re-run when:

- Upstream publishes an update (`hermes skills check` / `update`)
- A skill under `external_dirs` changes on disk
- You rotate credentials that the skill might reference

Install/remove history: `~/.hermes/skills/.hub/audit.log`.

## Interpreting results

- **fail** on `unsafe_instruction` or `credential_use` — treat as blocking until SKILL.md and scripts are cleaned or removed.
- **review** on `shell_api_mutation` — may be legitimate (e.g. `subprocess` in a maintained script); confirm scope and approvals.
- **hidden_background** — ensure the operator explicitly asked for background/cron work; hidden side effects violate the trust model.
- **source_mirror_drift** — never ignore; either refresh from trusted upstream or diff manually.

## Agent-created skills

`skill_manage` writes to `~/.hermes/skills/` by default. Optional gate:
`skills.guard_agent_created` runs Skills Guard on agent writes. The operator
audit is still recommended before promoting agent-drafted skills to
`external_dirs` or shared profiles.

## Related docs

- [Skills feature guide](../../website/docs/user-guide/features/skills.md) — install, trust levels, `hermes skills audit`
- [Creating skills](../../website/docs/developer-guide/creating-skills.md) — authoring standards
- [SECURITY.md](../../SECURITY.md) — trust model and reporting scope
