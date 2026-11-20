# FRS-012: `hermes skills doctor` — Skills Registry↔Disk Reconciliation

Version: 1.0.0
Status: Implemented (ORCA-06 follow-up, 2026-10-26)

## Objective

Prevent silent registry/disk desync of the skills usage registry. On 2026-10-26
(ORCA-06) a skill directory (`~/.hermes/skills/orca_orchestrator`) was raw-deleted
by a foreground CLI `rm -rf` while `skills/.usage.json` still listed it
`state: active`; no platform check reconciled the two, and the desync went
unnoticed until a dispatched worker hit it. This FRS specifies the first-class
command `hermes skills doctor` that detects that class of desync, plus the
pin-aware delete guards that keep pin a meaningful protection promise.

---

## FRS-1: Doctor command

### Functional Requirements

**FR-1.1** — `hermes skills doctor` SHALL reconcile `skills/.usage.json` against
the on-disk skill dirs of the active profile (`get_hermes_home() / "skills"`) using
NAME-based reconciliation, not registry-key↔path comparison.

**FR-1.2** — The reconciliation SHALL treat registry keys in all observed forms as
equivalent to their on-disk skill: flat frontmatter names (`orca_orchestrator`),
category paths (`devops/kanban-worker`), and legacy forms (`./spec-driven-development`,
`agent-skills:using-agent-skills`).

**FR-1.3** — The reconciliation SHALL resolve a registry entry to an on-disk skill
when ANY of: the normalized key matches a disk relpath; the key (or its basename, or
its normalized form) matches a SKILL.md frontmatter `name:` or a dir basename; or the
name exists under `skills/.archive/` (top-level entries or nested legacy archives).

**FR-1.4** — Hub-viewed-only registry entries (skills seen in a hub browse but never
installed locally) SHALL be excluded from findings. `created_at` is not evidence of
local activity: only `last_used_at`/`last_viewed_at` count.

**FR-1.5** — Pinned entries SHALL always be checked regardless of activity age: pin
is a protection promise, and a pinned entry resolving to no on-disk dir and no
`.archive` copy is a desync by definition.

**FR-1.6** — A registry entry with `archived_at` set, or state other than
`active`/`stale`, SHALL be excluded from CRITICAL findings (archived skills
legitimately live only in `.archive`).

**FR-1.7** — CRITICAL SHALL be reported iff: `state` ∈ {active, stale} AND
(pinned OR used/viewed within the last 14 days) AND the skill resolves to no
on-disk dir AND no `.archive` copy.

**FR-1.8** — On-disk SKILL.md-bearing dirs with no registry entry at all SHALL be
reported as INFO only (fresh skills register on first use/view — normal).

**FR-1.9** — The command SHALL exit 0 when reconciliation completes with no CRITICAL
findings, and exit 1 when CRITICAL findings exist OR the registry is missing/unreadable
(nothing can be reconciled). The exit code follows the hermes CLI convention (handler
int return → process exit code) so cron/no-agent runs surface as errored jobs.

**FR-1.10** — `--json` SHALL print the full machine-readable report
(`ts_utc`, `skills_root`, `registry_present`, `registry_error`, entry/dir counts,
`critical`, `info_unregistered_dirs`) while preserving the exit-code contract.

### Scenarios

#### Scenario: ORCA-06 signature detected
- Given a skills root with `.usage.json` entry `orca_orchestrator`
  (`state: active`, `pinned: true`, `last_used_at` 400 days ago) and no
  `skills/orca_orchestrator/` dir and no `.archive/orca_orchestrator/` copy
- When `hermes skills doctor` runs
- Then the entry is reported as CRITICAL and the process exits 1

#### Scenario: hub-viewed-only ledger rows are noise
- Given 160 registry entries with `state: active`, `use_count: 1-2`,
  `last_used_at` >14 days ago, and no local dirs
- When `hermes skills doctor` runs
- Then zero CRITICAL findings are reported and the process exits 0

#### Scenario: archived skill is not a desync
- Given a registry entry with `archived_at` set (or state `archived`) whose live
  dir is gone but whose `.archive/<name>/` copy exists
- When `hermes skills doctor` runs
- Then no CRITICAL finding is reported for it

### Acceptance Criteria

- [x] `hermes skills doctor` exits non-zero on a simulated active+pinned entry with
      missing dir (unit: `tests/tools/test_skills_health.py`,
      `tests/hermes_cli/test_skills_doctor_cli.py`)
- [x] Legacy key forms, frontmatter-name resolution, `.archive` validity, and the
      14-day activity window are each covered by unit tests on a temp skills root
- [x] Implementation is stdlib-only (plus `hermes_constants` home resolution) so it
      runs under cron/no-agent contexts

### Verification

```bash
hermes skills doctor            # human report; exit 1 on CRITICAL
hermes skills doctor --json     # machine-readable report
.venv/bin/python -m pytest tests/tools/test_skills_health.py tests/hermes_cli/test_skills_doctor_cli.py
```

---

## FRS-2: Pin-aware delete guards (all platform delete paths)

### Functional Requirements

**FR-2.1** — Skill deletion SHALL be refused when the target's `.usage.json` record
has `pinned: true`, on every platform delete path. Patches/edits to pinned skills
remain allowed (pin protects against deletion only — the irrecoverable operation).

**FR-2.2** — The agent-facing `skill_manage(action="delete")` path SHALL honor pin
(`tools/skill_manager_guards.py::_pinned_guard`, wired in
`tools/skill_manager_tool.py::_delete_skill`).

**FR-2.3** — The CLI hub uninstall path (`hermes skills uninstall`,
`tools/skills_hub_install.py::uninstall_skill`) SHALL honor pin, refusing with
guidance to unpin first (`hermes curator unpin <name>`). Without this guard the CLI
would be the pin bypass.

**FR-2.4** — The refusal message SHALL name the skill, state that it is pinned, and
point at `hermes curator unpin` as the deliberate escape hatch (mirroring the curator
invariant "pinned skills bypass").

### Scenarios

#### Scenario: pinned hub skill uninstall refused
- Given a hub-installed skill whose `.usage.json` record has `pinned: true`
- When `hermes skills uninstall <name>` runs
- Then the uninstall is refused, the skill dir remains on disk, and the message
      names `hermes curator unpin`

#### Scenario: unpinned hub skill uninstalls normally
- Given a hub-installed skill whose record has `pinned: false` (or no record)
- When `hermes skills uninstall <name>` runs
- Then the uninstall proceeds (guard is not over-broad)

### Acceptance Criteria

- [x] `skill_manage` delete of a pinned skill refused (pre-existing; regression-tested
      in `tests/tools/test_skill_manager_tool.py::TestPinnedGuard`)
- [x] Hub uninstall of a pinned skill refused without `--force` and proceeds when
      unpinned (`tests/tools/test_skills_hub.py::TestInstallPathSafety`)

---

## FRS-3: Kanban worker guidance — delivered-files vanish lesson

### Functional Requirements

**FR-3.1** — `KANBAN_GUIDANCE` (agent/prompt_builder.py), the canonical repo channel
for kanban-worker knowledge since #50473 folded the worker/orchestrator skills into
injected guidance, SHALL teach the ORCA-06 lessons: delivered skill files can vanish
mid-run via raw deletion from ANY session source (worker, cron, or the user's
foreground CLI); forensic rm-scans MUST parse `state.db` `messages.tool_calls` JSON
across ALL session sources, not just worker message content; the registry desyncs
silently and `hermes skills doctor` reconciles it; recovery is replay from
transcripts/ledger blobs, not re-implementation; and a concurrent worker's own
restore must be ruled out first.

### Acceptance Criteria

- [x] The compact lesson ships in `KANBAN_GUIDANCE` (every dispatched worker's system
      prompt); the deep-dive pitfall + byte-exact recovery playbook remain in the
      host-local `devops/kanban-worker` skill (upstream removed the bundled skill in
      #50473 by design — see card t_d305c955 for the re-bundle decision deferred to
      the user)
