# Living Subsystem Framework → Hermes primitives

Issue #11604 proposed twelve always-on subsystems under a core `subsystems/` package. Each maps onto
a surface that already ships; this table is what the skill routes to. "Gap" rows are the only
places the skill adds anything (`scripts/fitness.py`).

| Proposed subsystem | Use instead | Notes |
|---|---|---|
| Governance — pre-flight block | `approvals.mode`, `approvals.deny`, `command_allowlist`; `hooks.pre_tool_call` shell hooks (`{"action":"block"}` / exit 2); `hermes approvals test -- <cmd>` | Built-in dangerous-command detection + hardline floor already cover recursive root deletes, filesystem formatting, piping downloads into a shell, and `/etc/` writes |
| Governance — audit trail | Session DB (every tool call is stored); `hermes sessions export --session-id ID --format jsonl`; `hermes logs --component tools`; a `hooks.post_tool_call` script if an external log is required | No cross-session "list every tool call" CLI — export per session |
| Governance — code-level danger | `hermes plugins enable security-guidance` | Warns (or blocks) on `eval`, `shell=True`, `pickle.load`, ... in written code |
| ScienceLoop (hypothesis → experiment → verdict) | Kanban: one task per hypothesis, `kanban_comment` per experiment, `kanban_complete(metadata={"verdict": ...})`; `/goal` with `verification:` + `/goal gate add` for a single measurable hypothesis | Group with `tenant` (kanban has no tags) |
| ReflectiveEvolution (lessons from failure) | `/refine`, `/learn`, `skill_manage` (procedures + pitfalls), `memory` (cross-task facts), `session_search` (recall past failures) | Lessons belong in the skill used for the task, not a parallel `learnings.json` |
| FitnessBuilder (weighted multi-dimension score) | **Gap** → `scripts/fitness.py` + a cron job | Deterministic arithmetic, append-only JSONL history, delta vs last run |
| Knowledge (typed units + graph) | `memory`, skills with `related_skills`, `hermes journey` (learning graph) | |
| Reasoning (typed chains, decisions) | Model reasoning; record decisions as kanban comments or goal contracts | No persisted structure needed |
| Perception / Integration / Adaptation | Tools + context files; no separate layer | |
| Evolution / Reflection / Metacognitive / Self-model / Identity | Curator (`/curator`), `/refine`, `SOUL.md`, `USER.md` | |
| Memory tiering | Built-in memory + external memory providers (`hermes memory setup`) | |
| Orchestrator | `delegate_task`, kanban swarm, `dynamic-workflow` skill | |
| Quota display | `hermes insights`, `/usage`, `/status` | |
| `.run()` / `.status()` per subsystem | `cronjob_manage` / `hermes cron` with `skills=["self-improvement-loops"]`; `hermes status`, `hermes cron status`, `hermes kanban stats`, `hermes hooks doctor` | |
