# Blast Radius + Integration Analysis

## Scripts Under Review

| Script | Path | Type |
|--------|------|------|
| `curator-governance-hook.py` | `~/.hermes/scripts/curator-governance-hook.py` | Python |
| `pin-assigned-skills.sh` | `~/.hermes/scripts/pin-assigned-skills.sh` | Bash |

---

## 1. Caller Map

### `curator-governance-hook.py`

| Caller | Source | Details |
|--------|--------|---------|
| **Cron job** `curator-governance-hook` (id `ce5d9e9e7d29`) | `~/.hermes/cron/jobs.json` | Schedule: `0 10 * * 1` (Mon 10:00). `no_agent: true`. Deliver: `discord:#governance`. Created 2026-06-09. Never run yet (last_status: null, completed: 0). |
| Ad-hoc CLI | `python3 curator-governance-hook.py` | Scoped in skill docs. |

### `pin-assigned-skills.sh`

| Caller | Source | Details |
|--------|--------|---------|
| **Cron job** `curator-proactive-pin` (id `8c721e494e55`) | `~/.hermes/cron/jobs.json` | Schedule: `0 6 * * *` (daily 06:00). `no_agent: true`. Deliver: `discord:#governance`. Created 2026-06-09. Never run yet (last_status: null, completed: 0). |
| Ad-hoc CLI | `bash pin-assigned-skills.sh` | |

**No other scripts, skills, or agents reference either script by name anywhere in the codebase or config.** Both are exclusively cron-driven.

---

## 2. Cron Contract Analysis

### `no_agent` contract (from `cron/scheduler.py`)

Both jobs use `no_agent: true`, which means the scheduler short-circuits to `_run_job_script()`:

1. Script path is resolved against `~/.hermes/scripts/` (with a fallback to the global `HERMES_HOME` if the profile dir doesn't have it)
2. Script is executed as a subprocess — `.sh` → `/bin/bash`, `.py` → `sys.executable`
3. **Default timeout: 120 seconds** (`_DEFAULT_SCRIPT_TIMEOUT`)
4. On success: stdout is delivered to Discord channel (#governance for both)
5. On failure (non-zero exit / timeout): error alert delivered to Discord (#governance)
6. On empty stdout: silent run (no delivery)
7. The wakeAgent=false gate also suppresses delivery

### RISK — 120s timeout for `pin-assigned-skills.sh`

With ~118 profile-referenced skills, each calling `hermes curator pin <skill>` (which is a subprocess call back into the Hermes CLI), plus the `find` commands and embedded `python3 -c` calls, the 120s default script timeout could be tight. A single `hermes curator pin` call involves:
- CLI startup overhead (Python init, tool loading)
- Reading `~/.hermes/skills/.usage.json`
- Checking bundled/hub-installed manifests
- Atomic write to `.usage.json`

At 0.3-0.5s per pin, 118 skills = ~35-60s. Comfortable but leaves no margin. If `hermes` CLI is slow to start (e.g., cold caches), it could bust the timeout mid-run — partial pin state, no clear indicator of how many succeeded.

**Recommendation:** Consider raising `script_timeout_seconds` in cron config, or adding a `timeout` field to the job itself.

---

## 3. Downstream Effects — Data Writes

### `curator-governance-hook.py`

| Operation | Target File | Risk |
|-----------|-------------|------|
| **Reads** `~/.hermes/skills/.curator_state` | JSON | Read-only. File may not exist on first run → returns `None` → graceful exit. |
| **Reads** last curator run.json | `~/.hermes/logs/curator/{timestamp}/run.json` | Read-only. Path from `.curator_state["last_report_path"]`. If state file exists but report dir was deleted, returns `None` → graceful exit. |
| **Reads** root `config.yaml` | `~/.hermes/config.yaml` | Read-only for `skills.always_skills` and `skills.enabled_skills`. |
| **Reads** all profile configs | `~/.hermes/profiles/*/config.yaml` | Read-only for `skills.always_skills` and `skills.enabled_skills`. Skips dirs starting with `_`. |
| **WRITES** root config.yaml | `~/.hermes/config.yaml` | **MUTATION.** `add_skill_to_enabled()` appends to `skills.enabled_skills`. Partial failure: writes via `yaml.dump` overwrite the file. If yaml library reorders keys, could break human-edited config. |
| **WRITES** profile config.yaml | `~/.hermes/profiles/{name}/config.yaml` | **MUTATION.** Same as root. |
| **WRITES** skill SKILL.md | `~/.hermes/skills/{category}/{name}/SKILL.md` | **MUTATION.** `set_adoption_status()` uses regex `re.sub` on `adoption_status:` or inserts after frontmatter `---`. Regex is fragile — if `adoption_status:` appears in non-frontmatter content, it could corrupt the file. |
| **WRITES** governance log | `~/.hermes/governance/logboard/curator-governance-{date}.mdl` | Append-only JSONL. Safe on partial failure. |
| **Calls** `hermes curator pin` (subprocess) | `~/.hermes/skills/.usage.json` | Side effect of blocking archival overrides. May race with concurrent curator runs. |

### `pin-assigned-skills.sh`

| Operation | Target File | Risk |
|-----------|-------------|------|
| **Reads** root `config.yaml` | `~/.hermes/config.yaml` | Via embedded `python3 -c`. Shell word-splitting of output is fragile if skill names contain spaces. |
| **Reads** all profile configs | `~/.hermes/profiles/*/config.yaml` | Same as root. |
| **Calls** `hermes curator pin <skill>` (subprocess) | `~/.hermes/skills/.usage.json` | **MUTATION.** Each call sets `pinned: true` on the skill's .usage.json record. Atomic writes per skill (via `_mutate()`), so partial failure in a batch is safe. |
| Writes stdout only | Cron delivery to Discord | Output includes `⚠️` for stale refs, `✅` for pinned, `❌` for failures. |

---

## 4. Configuration Coupling

### Dependencies shared by BOTH scripts:

- **`~/.hermes/config.yaml`** root skills block (`always_skills`, `enabled_skills`) must be parseable YAML. Both handle JSON-string encoding (via `json.loads` fallback). If config has unreachable YAML syntax, the `python3 -c` calls in `pin-assigned-skills.sh` and `yaml.safe_load` in the .py will silently skip.
- **`~/.hermes/profiles/*/config.yaml`** — same contract. Profile dirs starting with `_` are skipped.
- **`hermes` CLI on PATH** — both scripts call `hermes curator pin` as a subprocess. If `hermes` is broken, misconfigured, or path changes, both scripts fail silently (`.py` uses `stderr=subprocess.DEVNULL`, `.sh` uses `|| true`).

### `curator-governance-hook.py`-specific contracts:

- **`~/.hermes/skills/.curator_state`** — must exist with `last_report_path` pointing to a valid directory containing `run.json`. The curator has never completed a live run yet (last run was a dry-run, `run_count: 0`).
- **`CATEGORY_TO_LEAD` dict** — hardcoded mapping of 23 category paths to profile names. Adding a new category without updating this dict means confidence=0 and manual review gate.
- **`_SKILL_NAME_RE` regex** — `^[A-Za-z0-9._-]+$` — skills with special characters or spaces will get `None, 0` classification.
- **Classification via `find` subprocess** — depends on SKILL.md existing at `skills/{category}/{name}/SKILL.md`. This is the Hermes convention but not enforced by a schema.

### `pin-assigned-skills.sh`-specific contracts:

- **Skill existence check** — runs `find ... -path */{SKILL}/SKILL.md -not -path '*/_archived/*'`. If the skill dir was renamed but the profile config wasn't updated, the stale reference gets `⚠️ skipped`. This is by design — it's not a failure scenario.
- **Shell word-splitting** — `for s in $SKILLS` iterates over raw output from `python3 -c`. Skill names with spaces would cause split issues. `_SKILL_NAME_RE` in the .py would reject these, but the .sh doesn't validate. Risk: a skill named `"my skill"` would be treated as two separate skills `my` and `skill`.

---

## 5. Database / Migration Safety

### Persistence layer: `~/.hermes/skills/.usage.json`

Both scripts ultimately touch this file through `hermes curator pin`:
- `skill_usage.set_pinned()` → `_mutate()` → atomic write via tempfile + rename
- Safe on partial failure: each call writes atomically
- No database — file-based JSON

### YAML config writes (governance hook)

`add_skill_to_enabled()`:
1. Reads full YAML into memory
2. Modifies `skills.enabled_skills` list
3. Dumps full YAML back to file via `yaml.dump(data, sort_keys=False, ...)`

**Safety concern**: `yaml.dump` does not preserve comments, formatting, or ordering beyond `sort_keys=False`. If `config.yaml` or profile configs have YAML comments or manual formatting, they will be stripped by the first auto-assignment. This is a one-time data loss event per edited file.

**Mitigation**: After the first auto-assignment changes a config, subsequent edits only add new entries to existing lists — no reordering of other fields. But the one-time formatting loss is irreversible without a backup.

### Governance logs

`log_event()` appends JSONL to `~/.hermes/governance/logboard/curator-governance-{YYYYMMDD}.mdl`. Append-only, atomic per line. Safe.

---

## 6. API Contract Stability

### `run.json` contract (curator output → governance hook input)

The governance hook expects these keys in the curator's `run.json`:
- `started_at` — exists ✓
- `counts.checked` — **NOT present** in dry-run report (absent when `after=0` skills). Governance hook prints `None`.
- `counts.archived_this_run` — exists ✓
- `counts.added_this_run` — exists ✓
- `archived` (list) — exists ✓
- `pruned_names` (list) — exists ✓
- `added` (list) — exists ✓

The governance hook uses `report.get("archived", []) or report.get("pruned_names", [])` as a fallback chain — safe.

**Edge case**: When `counts.checked` is missing (`report.get('counts', {}).get('checked', 0)` returns `0`/`None`), the governance hook prints `Checked: None`. Cosmetic only — doesn't affect logic.

### `hermes curator pin` contract

The `.sh` script expects stdout to contain "already pinned" or "pinned". Verified from `_cmd_pin()`:
- Success: `f"curator: pinned '{args.skill}' ..."` → `.sh` matches `"pinned"`
- Already pinned: not explicitly printed; `set_pinned(True)` is idempotent, re-setting `pinned: true` in .usage.json, then prints `"pinned"...` anyway. ✓
- Failure: returns exit code 1 if skill is bundled/hub-installed → `.sh` catches via `|| true` and reports `❌`

---

## 7. Observability

### When `curator-governance-hook.py` fails:

| Failure Mode | Observability |
|-------------|---------------|
| `yaml.safe_load` fails | Caught by `except Exception` → silently skipped. Profile config may be loaded with partial data. |
| `find` subprocess fails | `subprocess.run` with no timeout. Could hang indefinitely. |
| `yaml.dump` → disk full | Exception propagates → script exits non-zero → cron delivers error alert to Discord. |
| Regex corruption in SKILL.md | Script exits non-zero → cron delivers error. Content damage is already done. |
| Governance log write fails | Logged event is lost. Script continues. |

**Gap**: There is no logging to `~/.hermes/logs/` (Hermes standard). The script only writes to stdout (captured by cron) and the governance logboard. Operators monitoring Hermes logs (`agent.log`) will not see governance hook errors.

### When `pin-assigned-skills.sh` fails:

| Failure Mode | Observability |
|-------------|---------------|
| Timeout (120s) | Cron scheduler catches `TimeoutExpired` → delivers error to Discord. |
| Embedded `python3 -c` fails | `2>&1` captures stderr into `$SKILLS` variable. If `python3` isn't available, command fails silently — `$SKILLS` will be empty, no skills pinned, script reports "0 unique skills". |
| `hermes` not on PATH | `hermes curator pin` subprocess fails → `|| true` swallows error → `FAILED` count increments. Output reports skill as `❌`. |
| Partial pin (halfway through loop) | With `set -euo pipefail`, a single failed `hermes curator pin` would normally abort the script. But `|| true` protects the pin loop. Loop continues. |

**Gap**: The `declare -A ALL_SKILLS` pattern is Bash 4+. Works on modern Linux but would fail on macOS (Bash 3). The shebang is `#!/bin/bash`.

---

## 8. Performance Anti-Patterns

### `curator-governance-hook.py`

1. **N+1 `find` subprocess calls**: `classify_skill()` runs `find .../SKILL.md` for EACH new skill. With 118+ existing skills, a large batch of new discoveries could mean dozens of `find` calls. Mitigation: uses `classifications` dict cache but re-fetches for each skill independently.
2. **Same pattern in `set_adoption_status()`**: Another independent `find` per skill — double the subprocess cost.
3. **Full YAML read/write per profile edit**: `load_profile_skills()` reads every profile config, then `add_skill_to_enabled()` reads + writes the same file again. Could consolidate.
4. **Governance log append per event**: Each event writes one line. Fine for normal operation.

### `pin-assigned-skills.sh`

1. **Per-skill `find` + `hermes curator pin` subprocess**: 2 subprocesses per unique skill (one `find` for existence check, one `hermes curator pin`). 118 unique skills = 236 subprocess calls. At 0.3-0.5s each = ~70-120s total.
2. **N+1 `python3 -c` calls**: 1 per profile config (root + ~47 profiles = 48 calls). Each starts Python, imports yaml+json, reads a file, prints output. Optimizable: could batch with one Python invocation that reads all configs.
3. **Shell word-splitting loop**: `for s in $SKILLS` iterates over unquoted output. If any skill name output contains glob characters (`*`, `?`), they'd be expanded. Unlikely given the regex validation in the .py counterpart.

---

## 9. Revert Safety Assessment

### Scenario: Governance hook adds wrong skill to wrong profile

1. `add_skill_to_enabled("new-skill", "wrong-profile")` adds to `config.yaml`
2. **Revert**: Manually edit `~/.hermes/config.yaml` or `~/.hermes/profiles/wrong-profile/config.yaml` to remove the entry.
3. **Impact**: The skill would be loaded by the wrong profile until reverted. No data loss.
4. **Rollback with curator backup**: `hermes curator backup` only backs up skills, not config files. Config mutation is NOT reversible via curator rollback.

### Scenario: Governance hook corrupts SKILL.md via regex

1. `re.sub(r"adoption_status:.*", ...)` matches the first occurrence in content. If `adoption_status:` appears in a code block or YAML value before the frontmatter, it could corrupt.
2. **Revert**: Restore from curator backup (`hermes curator rollback`) — the backup includes the full skills tree. ✓
3. **Impact**: Skill content corrupted between curator runs. Max 7 days (curator interval) or until detected.

### Scenario: `pin-assigned-skills.sh` times out mid-run

1. Skills 1-60 are pinned, skills 61-118 are not.
2. **Revert**: Re-run the script. Idempotent — pinning an already-pinned skill is a no-op. ✓
3. **Impact**: Unpinned skills could be archived by the next curator run. Mitigated by the governance hook (layer 2), which re-pins them reactively.

### Scenario: `yaml.dump` strips config.yaml formatting

1. First auto-assignment writes `~/.hermes/config.yaml` with `yaml.dump(sort_keys=False, default_flow_style=False)`.
2. **Revert**: Restore from manual backup or git. `~/.hermes/config.yaml` is NOT backed up by curator backups.
3. **Impact**: Comments and hand-tuned formatting are permanently lost. Mitigation: use the git-tracked version in `repos/KenseiAgent/` (if config is symlinked).

---

## 10. Summary: Risk Register

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| 120s timeout on pin script | Medium | Low | Add `script_timeout_seconds` to cron config or increase default |
| yaml.dump strips config formatting | High | **Certain** (first auto-assignment will do this) | None built-in. Pre-backup config.yaml manually. |
| Regex corruption in SKILL.md adoption_status | Medium | Low | Regex targets first match only. Risk if `adoption_status:` appears in non-frontmatter. |
| Shell word-splitting on skill names with spaces | Low | Very Low | `_SKILL_NAME_RE` prevents creation of such skills. |
| `hermes` CLI missing/off PATH | High | Low | Both scripts fail silently. Governance hook uses `stderr=subprocess.DEVNULL` — errors invisible. |
| No logging to Hermes standard logs | Low | Always | Operator checks Discord, not `agent.log`. |
| No test coverage | Medium | Always | Both scripts are untested in CI. |
| Config files not in curator backup scope | High | Always | Config mutation is NOT revertable via `hermes curator rollback`. |

---

## Key Recommendations

1. **Pre-backup config.yaml** before the first governance hook auto-assignment fires (next Monday 10:00).
2. **Add `script_timeout_seconds: 300`** to cron config for the pin script (118 skills × ~0.5s = safety margin).
3. **Redirect subprocess stderr** in the governance hook — `stderr=subprocess.DEVNULL` hides `hermes` failures. Use `subprocess.PIPE` and log to stdout on error.
4. **Add a validation log** to `~/.hermes/logs/` (not just governance logboard) so operators monitoring Hermes logs see governance hook activity.
5. **Test the scripts end-to-end** before the first live cron run — ideally with a dry-run curator report to verify the contract.