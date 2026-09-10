# Profile & Governance System: as-built reference (2026-06-14)

Single source of truth for how the lead/sub profile fleet, capability scoping,
capability-request brokers, profile lifecycle, and Denji governance actually
work after the 2026-06-14 hardening programme. Written because the prior design
docs (MultiProfileOperatingModel.md, the sov-setup fleet guide, governance
decision-records) described competing models and predated PROFILE-GATE, so no
single doc matched the code. Where this file and an older doc disagree, this
file wins.

## 1. Lead / sub profiles

- Profiles live under `~/.hermes/profiles/<name>/` (declarative config tracked in
  a git repo). Leads (remii, octacon, quan, gojo, light, dezzy, ceecee, denji,
  misa-misa, mrhermagi, orchestrator, market-scanner, triage-router) are listed
  in `config.yaml: kanban.nonspawnable_profiles` and are NOT spawned as kanban
  workers. Their specialist sub-profiles (e.g. quan-arch, octacon-backend) are
  spawnable workers.
- `_is_profile_spawnable` (hermes_cli/kanban_db.py) gates spawn eligibility and
  now fails CLOSED: if spawnability cannot be determined (profiles import or
  config read error) the profile is treated as non-spawnable and the task waits
  in `ready` (recoverable) rather than dispatching to a possibly-invalid profile.
- Intake `_build_roster` filters assignees through `_is_profile_spawnable`, so a
  fresh task can never be routed to a non-spawnable lead (the historical I-2
  strand).

## 2. Capability scoping (skills + tools)

Two layers, both allowlist-based, both with an `off | shadow | enforce` mode.

### Skills
- `skills.enabled_skills` (union with `skills.always_skills`) is the per-profile
  allowlist. `skills.enforcement_mode` controls loading skills outside it:
  `off` = no gate, `shadow` = load proceeds but logs `skill.access.would_block`,
  `enforce` = load blocked, agent told to use `skill_request`.
- Active temporary grants (see brokers) also satisfy the allowlist.

### Tools (NEW 2026-06-14)
- A runtime toolset-scope fence enforces that a profile only EXECUTES tools in
  its enabled toolsets (`config.yaml: toolsets:`), mirroring the skills model.
  Previously `toolsets:` only filtered which tool schemas the model was shown;
  a model could still call a tool outside its set.
- `tools.enforcement_mode` (`off | shadow | enforce`, default `shadow` in code)
  governs the fence. The fence runs in both the sequential and concurrent tool
  executors.
- Always-allowed escape hatches: `tool_search`, `skill_request`, `tool_request`,
  `skill_view`, `skills_list`, the Tool Search bridge. Kanban worker lifecycle
  tools are always retained under `HERMES_KANBAN_TASK`, even past an explicit
  `disabled_toolsets`, so a dispatched worker can never be fenced off its
  completion/block/heartbeat surface.
- Active tool grants satisfy the fence. Shadow logs `tool.access.would_block`;
  enforce blocks and logs `tool.access.blocked`. Both fail to `shadow` (not
  off) on a config-read error.

## 3. Capability-request brokers

Two parallel brokers over the central profile-activity ledger
(`~/.hermes/governance/profile-activity-ledger.sqlite`).

- `skill_request(skill, task_id, reason)` -> `tools/skill_grants.py`.
- `tool_request(tool, task_id, reason)` -> `tools/tool_grants.py` (NEW).

Both: task-scoped grants, 6 borrows / 30 days per (profile, capability) before
escalating to Denji, 24h TTL sweep, auto-revoke on `kanban.task.completed`,
fail-closed `has_active_grant`. Each maintains a NEVER_GRANT deny set for
capabilities that change auth/providers/service/other-profiles or execute
arbitrary commands; tool NEVER_GRANT hardcodes `terminal`/`process` plus the
profile-lifecycle and skill-management mutators as a floor. Skills additionally
honour a quarantine gate for externally-sourced skills.

There is no auto-grant path for the NEVER_GRANT set: those require KENSEI/Denji
or the profile-edit / lifecycle gates.

## 4. Profile lifecycle (create / delete / spawn-a-sub)

- All profile create/delete is fail-closed behind `profiles.lifecycle_authorised`
  and must go through the PROFILE-GATE: `submit_lifecycle_request` records a
  pending row in `profile_lifecycle_approvals`, blocks the requesting task, and
  the Discord watcher posts approve/reject. Execution happens only on Sahil's
  approval via Discord.
- Lead front door (NEW): `kanban_request_subprofile(profile, reason, clone_from?)`
  lets a lead request creation of a dedicated sub-profile. It records a pending
  create approval and blocks the task; the sub-profile is created only after
  Sahil approves. Previously this capability was backend-only with no agent
  entry point.
- The approve path executes `create_profile`/`delete_profile` under a one-shot
  authorisation token; `_run_op` coerces missing args to `{}` so a create with
  no clone executes cleanly.

## 5. Denji governance cadence

Crons in the base store (`~/.hermes/cron/jobs.json`), delivered to #governance:

| Cron | Schedule | What |
|------|----------|------|
| denji-profile-review-weekly | Mon 09:00 | lightweight activity snapshot |
| denji-profile-review-monthly | 1st 09:00 | usage + config drift + auto-promotions |
| denji-profile-review-quarterly | 1st of Jan/Apr/Jul/Oct 09:00 | full audit: sessions + config + identity + trends |
| denji-skill-audit | Mon 11:00 | skill metadata/reference integrity |
| denji-worker-failure-analysis | Mon 09:00 | blocked/protocol-violation root-cause scan |
| denji-self-eval-reminder | Fri 10:00 | weekly self-eval prompt (created 2026-06-14) |
| governance-crossref | Mon 11:15 | cross-reference reviews vs self-evals; silent when nothing new |
| denji-logboard-monitor | every 4h | agent.log/errors.log anomaly scan |
| system-health | daily 05:00 | fleet health snapshot |

The three review cycles are differentiated in `scripts/denji-review-cycle.py`
by depth and event type (`profile.review.{weekly,monthly,quarterly}`); they are
not duplicates.

## 6. Cron-store resilience (NEW)

`cron/jobs.py` keeps a `jobs.json.lastgood` snapshot refreshed on every
successful load/save. On structural corruption (e.g. a non-atomic external
write) `load_jobs` recovers from the snapshot, preserves the corrupted file as
`jobs.json.corrupt.<ts>` for forensics, and self-heals, instead of raising and
taking down the cron subsystem. It raises only when no usable snapshot exists.

## 7. Known follow-ups (not yet actioned)

- Stale governance ledger artefacts (`skill-broker-ledger.jsonl`,
  `skill-borrow-ledger.jsonl`, the 0-byte root `profile-activity-ledger.sqlite`)
  are superseded by the sqlite ledger and should be removed manually; the live
  source of truth is `governance/profile-activity-ledger.sqlite`.
- `tests/tools/test_kanban_tools.py::test_kanban_guidance_prompt_size_bounded`
  fails pre-existing on main (KANBAN_GUIDANCE is ~4202 chars vs the 4096 bound);
  trim the guidance prompt or raise the bound as a separate decision.
