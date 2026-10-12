## What does this PR do?

Adds `hermes-vcc` to the community plugin catalog: deterministic, no-LLM context compaction — a port of pi-vcc's extraction/formatter core (MIT, © sting8k) onto the built-in `ContextCompressor`. The compaction summary is built by regex extraction instead of a summarizer-model call: zero tokens billed, no auxiliary network round-trip, same transcript in → same bytes out. Everything else stays host-owned: head/tail protection, token thresholds + per-model overrides, orphaned tool-pair repair, archival to the session DB, commit fences, session lifecycle.

I'm the owner of agentbox-tech/hermes-vcc (owner submission, rule 5). The verification commands below were run by our agent against the pinned tree; outputs quoted verbatim.

## Related Issue

None — new catalog submission per `plugin-catalog/README.md`.

## Type of Change

- [x] ✨ New feature (one catalog entry + one contributor-mapping correction)

## Changes Made

- **`plugin-catalog/hermes-vcc.yaml`** — `subdir: engine`, pinned at `72a765b8fdf2af4c830c01ac995fc514e05ae4f2` (= pushed `main` tip, confirmed by `git ls-remote` today), `version: "0.1.0"`, `requires_hermes: ">=0.21.6"`, category `memory` — the shelf context-engine plugins sit on (precedent: `hermes-snapcompact`).
- **`contributors/emails/rain@synth.kitchen`** — first line `matarbot` → `rain-sk`. That email is the commit-author email on all my agentbox-tech commits; the old mapping attributed them to the project bot account from the salvage era. Attribution CI passes either way; this keeps it accurate.
- **Public surfaces only (rule 9):** `register(ctx)` → `ctx.register_context_engine(VccEngine())`; `VccEngine(ContextCompressor)` overrides the two documented LLM choke points (`_generate_summary`, `_augment_summary_lean`) and keeps host framing intact — `SUMMARY_PREFIX` via `_with_summary_prefix`, ghosted-skill marker reinjection. Never auto-activated: only via `context.engine: hermes-vcc`. One deliberate policy choice worth flagging: we **drop** the lean-mode verbatim-user-messages section (verbatim imperatives inside a REFERENCE-ONLY artifact license stale-task resumption); everything else in the framing is untouched.

## Evidence (run at the pinned SHA / this branch)

- `hermes plugins validate ./engine --install-deps` at a **fresh clone of the pinned SHA**: all green — `security scan: safe`, `no core override`, capability probe ran and matches the empty `capabilities:` block (v0 ships zero engine tools, on purpose — every tool is schema re-billed on every API call).
- `scripts/validate_plugin_catalog.py plugin-catalog/hermes-vcc.yaml` → `OK: 1 file(s) valid`.
- Pin resolves and equals `main` tip: `git ls-remote https://github.com/agentbox-tech/hermes-vcc`.
- Rule 17 — nothing unscannable tracked: `git ls-files | grep -c pyc` → `0`; the pinned tree is pure source.
- Rule 13 grep — `grep -rn "urllib\|requests\|socket\|subprocess\|os.system\|eval(\|exec(" engine/` → exactly 1 hit, inside a regex literal (the URL *extractor* pattern); zero executable calls. `vcc_core.py` imports stdlib only.
- Rule 11 — no `requires_env`, no config schema, no `.env`/`auth.json`/keychain reads; the single `os.environ` read is a `HERMES_HOME` fallback for the debug artifact, the primary path uses the host-supplied `hermes_home` kwarg.
- `requires_hermes` floor, seam-verified not guessed: every symbol imported from `agent.context_compressor` (`_generate_summary`, `_augment_summary_lean`, `_with_summary_prefix`, `_collect_ghosted_skill_names`, `_extract_pruned_skill_names`, `_redact_compaction_text`, `_reinject_pruned_skill_markers`, `_SYNTHETIC_USER_ROW_PREFIXES`, `HISTORICAL_TASK_HEADING`) exists in `agent/context_compressor.py` at tags v0.21.6 and v0.21.7.

## How to Test

1. After merge: `hermes plugins install hermes-vcc --enable`
2. `hermes config set context.engine hermes-vcc`
3. New session → `agent.log` shows `Using context engine: hermes-vcc` (activation is per-agent-init; no restart needed).
4. Let a long session compact: summaries render as deterministic briefs whose `(#N)` refs are DB message ids → `session_search(session_id=…, around_message_id=N)` scrolls the full original of any compacted turn (archived `active=0, compacted=1`, never deleted; FTS covers archived rows by default).

Live evidence: running in production on our Pixel 3 / postmarketOS arm64 server since 2026-10-08. Real-session-DB replays: 69–90% char reduction across three 120–284-message sessions, 0 role-alternation violations, byte-identical output on repeated compaction. Repro: `scripts/replay_eval.py` against any session DB.

## Disclosure (rule 13)

No network calls. No shell commands. No background processes (compaction runs in the host's existing path). No telemetry. No credentials required, read, or stored. Reads only the turns the host hands the compressor plus its own `context.vcc` config block. Writes: none by default; `debug: true` (default false) atomically writes one metadata-only diagnostics file, `$HERMES_HOME/vcc-debug.json`. No self-updaters — the pinned SHA is the release. **Origin:** port of [pi-vcc](https://github.com/sting8k/pi-vcc) (MIT, © sting8k), which targets a different harness (pi); this implements the Hermes context-engine seam, and no catalog entry shares this lineage (checked across all 555).

## Checklist

- [x] I've read the Contributing Guide and the catalog admission rules
- [x] My commit messages follow Conventional Commits
- [x] I searched for existing PRs — no `vcc` catalog key, no same-lineage entry
- [x] My PR contains only the catalog entry + the contributor-mapping correction
- [x] I've tested on my platform: Linux (postmarketOS edge, arm64), live in production since 2026-10-08
- `pytest` / `scripts/check`: N/A — no core code touched; catalog CI runs `hermes plugins validate` at the pinned SHA (already green locally at the same SHA)
