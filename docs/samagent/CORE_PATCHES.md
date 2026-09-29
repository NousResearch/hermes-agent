# SamAgent Core Patches Ledger (`CORE_PATCHES.md`)

**Rule:** Target **≤ 4 core patches** to upstream Hermes (`NousResearch/hermes-agent`), each generic and upstream-able. All SamAgent product logic lives in `samagent/` (library), `plugins/samagent/` (hooks, tools, dashboard UI), and `evals/samagent_bench/`. Anything beyond 4 core patches requires an ADR.

Upstream remote:
```bash
git remote add upstream https://github.com/NousResearch/hermes-agent.git
```

## Patch Table (after Phase 0 Spikes S1–S9)

| ID | Candidate Patch | Status | Why / Empirical Spike Finding |
|----|----------------|--------|-------------------------------|
| **P1** | Per-task `model`/`provider`/`profile` in `tools/delegate_tool.py` | **ELIMINATED (0 lines changed)** | **Spike S3 PASS:** (1) `hermes_cli.kanban_swarm.SwarmWorkerSpec` already routes each worker card to a named profile with its own `model`, `provider`, and `toolsets`; (2) `tools.delegate_tool.delegate_task` already accepts `credentials_cfg={"model": ..., "provider": ..., "base_url": ..., "request_overrides": ...}` and `_build_child_agent` accepts `toolsets` and `model` overrides. |
| **P2** | Add `samagent` and `samagent.*` to `[tool.setuptools.packages.find]` and `samagent` template data in `pyproject.toml` | **ACTIVE (1 file)** | **Spike S1 PASS:** Required so `pip install -e .` / wheel builds include the `samagent/` package and its deterministic scaffold templates. |
| **P3** | Product identity strings (`apps/desktop/product-identity.cjs`) | **DEFERRED to Phase 7** | Optional desktop rebrand seam; not needed while Mission Control runs as a bundled dashboard plugin (`plugins/samagent/dashboard/`) and standalone web server. |
| **P4** | Ephemeral user-turn hook for Ledger memory injection | **ELIMINATED (0 lines changed)** | **Spike S2 PASS:** `agent/turn_context.py` (`_collect_pre_llm_call_context`) already injects `pre_llm_call` hook `{"context": "..."}` into the user message via `compose_user_api_content` (never touching the system prompt) and stamps `api_content` onto the message row so intra-turn tool loops and historical turns replay identical bytes. |

**Current active core patch count: 1 (`pyproject.toml`).**
