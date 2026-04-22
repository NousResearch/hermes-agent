# Hermes Agent v0.10.2 (v2026.4.22)

**Release Date:** April 22, 2026

> Backend coding-agent follow-up release. Adds ratchet snapshots, Spar review/judge flow, structured failure scars, campaign persistence, and a proving-matrix gate.

---

## ✨ Highlights

- **Ratchet snapshots** — Hermes now has session-scoped ratchet refs on top of the checkpoint manager, so a worktree can be pinned, listed, and restored without replacing the older rollback path.

- **Spar review gate** — new opt-in `spar` tool runs a bounded `builder -> reviewer -> fix -> reviewer` loop with cosmetic-only findings filtered out. Default routes are:
  - builder: `xiaomi/mimo-v2-pro`
  - reviewer: `minimax/MiniMax-M2.7-highspeed`
  - judge: `deepseek/deepseek-reasoner`

- **Judge disagreement visibility** — Spar now surfaces reviewer/judge disagreement instead of silently hiding it.

- **ACP chat routing modes** — ACP sessions now persist and expose `standard`, `auto`, `force-spar`, and `force-moa` modes. Scarf and other ACP clients can switch modes explicitly instead of hoping the model picks the right path.

- **Failure scars** — final Spar rejections and failed agent turns now write structured markdown scars under `HERMES_HOME/FAILURES/`.

- **Campaign persistence** — `SessionDB` now supports durable multi-run campaign state with start/log/resume/close/prune operations.

- **Proving matrix scaffold** — rubric-backed `scripts/proving_matrix.py` now emits scorecards for ratchet, spar, and campaigns.

---

## ⚠️ Upgrade Note

- `spar` is **off by default** in toolset config. Enable it explicitly if you want the new adversarial review gate.
- ACP `force-spar` / `force-moa` routing is stronger than normal tool availability: those modes call the routed engines directly once selected.
- Campaign persistence adds a new `campaigns` table to `state.db` (schema version 7).

---

## ✅ Validation

- `scripts/run_tests.sh tests/tools/test_checkpoint_manager.py tests/tools/test_ratchet.py`
- `scripts/run_tests.sh tests/tools/test_spar_tool.py tests/agent/test_failure_registry.py tests/run_agent/test_failure_scars.py tests/test_campaigns.py`
- `python scripts/proving_matrix.py --skill=ratchet`
- `python scripts/proving_matrix.py --skill=spar`
- `python scripts/proving_matrix.py --skill=campaigns`
