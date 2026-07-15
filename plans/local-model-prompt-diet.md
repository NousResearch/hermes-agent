# Local-Model Prompt Diet — scoped plan (androux-8dg0)

> Goal: make Hermes automation viable on the local MLX model by cutting the fixed
> per-session prompt from ~26k tokens to ≤8k on machine platforms (cron/api_server),
> and stop paying Claude Max quota for housekeeping jobs.
> Measured baseline (Euripides, 2026-07-14): 55.9KB system prompt + 46.7KB tool
> schemas (26-27 tools) + 31.5KB skills index (311 entries) ≈ 26k tokens; Qwen3.5-9B
> completed ZERO cron turns in 25+ min (sonnet: 18s) — prefill-bound, no KV reuse.
> Verdict: option 1 (local default) is correct for cost but NOT viable without this diet.

*(Restored 2026-07-15: the original commit lived in the Sophocles hermes-agent clone,
which was deleted by the Sophocles-Hermes teardown (androux-mr6hi). Status updates
live on the beads, not here.)*

## Context

- 2026-07-07: Hermes default silently flipped to `claude-sonnet-4-6` (provider
  `claude-max` → claude-proxy :11436). Likely cause: MLX server was wedged (verified
  hung 2026-07-14, respawned clean by watchdog after kill).
- 8 days on sonnet: 573 sessions, 561 sonnet, ~7.8M tokens; cron platform alone
  518 sessions / ~4.9M tokens (~65 automated sessions/day) against Max quota.
- 2026-07-14: default flipped back to `mlx-community/Qwen3.5-9B-OptiQ-4bit`
  (:11437, ctx 131072). Backup: `~/.hermes/config.yaml.bak-sonnet-20260714`.
- Trim attempts that missed, with lessons:
  - `no_mcp` sentinel on all platforms: −1 tool (~1.5KB). mneme MCP was never the bulk.
  - Cron toolset cut (browser/vision/image_gen/tts/computer_use/delegation/
    code_execution removed): schema bytes ~unchanged — most of those tools were
    already gated off by check_fns, AND `delegate_task`/`execute_code`/feishu-plugin
    schemas SURVIVED explicit exclusion (see Workstream B).
  - Real weight: 4 core tools = half the schema budget — `delegate_task` 6,919B,
    `terminal` 5,399B, `session_search` 5,073B, `skill_manage` 4,130B.

## Phase 0 — stabilize now (before any repo work) — DONE 2026-07-15 (androux-5b5p4)

Per-job model overrides in `~/.hermes/cron/jobs.json`:

- `ntfy-awareness` (144/day) + `beads-kanban-sync` (72/day): → `openai/gpt-4o-mini`
  via openrouter.
- Low-frequency quality jobs (praxis-review, self-healing, USER.md sync, security
  digest, open-questions, LLM cheat-sheet): → sonnet via claude-max proxy, pinned
  base_url.
- Everything else: local model.

Acceptance PASSED: 24h insights showed zero sonnet token spend; 4o-mini jobs at
97-98% prompt-cache. Residue: nightly-curator + autonomous-exploration iter-cap
out on Qwen (escalation candidates until Workstream A lands).

## Workstream A — fixed-prompt diet (biggest lever, repo work)

- **A1. Skills index scoping (~31.5KB → ~3KB on machine platforms).** 311 entries
  come from local skills + hermes-agent builtin/optional-skills dirs indexed
  wholesale. Add per-platform skill-category allowlist (mirror of
  `platform_toolsets`, e.g. `platform_skills: {cron: [homelab, mlops, mneme]}`)
  consumed in `agent/prompt_builder.py` where the `<available_skills>` block is
  built. Crons need ~10 skills, not 311.
- **A2. System-prompt tier for machine platforms (~55.9KB → ~10KB).** The kawaii
  personality + full operational guidance is interactive-facing. Add a minimal
  cron/api_server system prompt variant (identity, delivery contract, tool rules).
  `build_system_prompt_parts` already tiers stable/context/volatile — add
  platform-aware stable tier.
- **A3. Fat-schema slimming (~46.7KB → ~20KB).** Tighten descriptions of the top-4
  schemas (above); audit the rest for example-bloat. Schema descriptions are prompt
  engineering for a 200k-window model; rewrite for a 9B (short, imperative,
  no prose examples).

Acceptance: `hermes prompt-size --platform cron` ≤ 8k tokens (~32KB total);
a kanban-sync cron completes on Qwen3.5-9B in <90s end-to-end.

## Workstream B — toolset resolution bug (correctness) — androux-y7ufv

Explicitly excluded toolsets leak into the platform tool list:
`delegation`, `code_execution`, `moa` schemas present on cron despite an explicit
`platform_toolsets.cron` list without them; feishu plugin tools present because
plugin toolsets are default-enabled for platforms never saved via `hermes tools`
(`known_plugin_toolsets` empty for cron). Two fixes in `hermes_cli/tools_config.py`
(`_get_platform_tools`):

- Treat an explicit platform list as authoritative for plugin toolsets too
  (or honor a `no_plugins` sentinel, symmetric with `no_mcp`).
- Repro + unit test: explicit list must produce exactly its resolved tools.
  Existing test files: `tests/hermes_cli/test_*.py`.

Acceptance: per-tool dump for cron contains no feishu/delegate/execute schemas
when excluded; test in CI.

## Workstream C — demote machine crons to scripts (volume killer) — DONE 2026-07-15 (androux-9p8gq, soaking)

`beads-kanban-sync` was already `no_agent: true` (pure script). `ntfy-awareness`
now emits the native wake-gate (`{"wakeAgent": false}`) from `ntfy-poll.py` on
quiet polls — scheduler skips the agent entirely (verified in scheduler log).
~90% of its 144 runs/day were empty polls → now a bare python exec, zero LLM.
Non-empty polls still wake 4o-mini for triage. Acceptance window: 1 week,
agent-escalations <5/day. Future refinement: gate low-priority-only batches.

## Workstream D — provenance + hang hygiene (the meta-lesson) — androux-1cu28

- **D1. Config-change provenance.** The Jul 7 model flip left no record. Hermes
  self-edits `config.yaml` (its system prompt says it may). Add a config-write
  audit line (timestamp, actor/session, diff summary) to a `config-changes.log`,
  or git-track `~/.hermes/config.yaml` on Euripides. Related incident class: the
  nightly-dream-cycle prompt hardcoded "Creon is DEAD" for 2 months (fixed 07-14).
- **D2. MLX watchdog health probe.** The watchdog restarts a dead process but not
  a wedged one (12-day-old process, 60s+ unresponsive). Add a periodic completion
  probe (tiny prompt, 30s timeout) → kill on failure; watchdog respawn is verified
  to work.
- **D3. (Optional) prompt/KV cache re-investigation.** The no-KV-reuse constraint
  is why prefill dominates (`--prompt-cache` flags previously caused hangs).
  Re-test on current mlx_lm (0.31.x) — if fixed upstream, A1-A3 targets relax
  considerably.

## Sequencing

~~Phase 0~~ → ~~C~~ → **B (current)** → A1 → A3 → A2. D1/D2 anytime, small.
Each workstream = its own bead; this doc is the epic's spec.

## Non-goals

- Not switching Hermes off sonnet for interactive/Telegram quality use — per-job
  and per-session overrides stay available.
- Not upgrading the local model (Qwen3.5-9B stays; if a stronger local model
  lands later, the diet still pays).
- ~~Not touching the Sophocles Hermes instance~~ (teardown executed 2026-07-14).
