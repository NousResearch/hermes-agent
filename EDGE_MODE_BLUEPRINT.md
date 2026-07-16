# Edge-Native Mode — Technical Chronicle & Production Blueprint

**Document class:** engineering chronicle (post-implementation)  
**Environment:** Apple Mac, Intel Core i9, local inference via Ollama (Qwen 3.5 9B-class SLM)  
**Status:** full-stack integration complete; repository documentation aligned (`README.md`)  
**Last updated:** 2026-05-17  

This document supersedes earlier speculative planning. It records the definitive design, code touchpoints, validation strategy, and operational semantics of the **Edge-Native Mode** patch for Hermes Agent: a **non-breaking**, **fully gated** capability controlled by `agent.edge_mode` and related knobs, intended to keep long-running agent loops viable on consumer CPUs and modest local context windows.

---

## 1. Executive summary & system context

### 1.1 Problem statement

Running Hermes Agent **locally** against a **small language model** (exemplar: Qwen 3.5 9B served through Ollama) on a **consumer-class CPU** exposed two compounding failure modes:

1. **Quadratic pre-fill pressure.** As the active message sequence grows, each new forward pass must attention over the entire prefix. On local stacks without datacenter-grade KV-cache bandwidth, the *effective* cost of “one more turn” scales poorly with context length. In practice this manifests as escalating wall-clock latency per iteration even when nominal token throughput appears stable — the dominant symptom is **pre-fill drag** on long prefixes, not merely decode speed.

2. **Silent stalls and pseudo-timeouts.** Tool-heavy turns that ingest **raw, high-entropy payloads** (logs, directory listings, scraped text, accidental binary-as-text) can push a single iteration’s prompt to enormous size. Observed behavior on the Intel i9 workstation included **multi-hour wall times** on a single loop step (e.g. perceived hang at “iteration 5” for **>200 minutes**) with no clean user-visible failure mode — indistinguishable from a deadlock from the operator’s perspective. Root cause was not “the model stopped thinking” but **context bloat** plus local inference saturation: the stack was still working, uneconomically, on an oversized prefix.

Together, these behaviors made **credibly unattended** local agent sessions untenable: the system was correct in the small, but **unsafe in the large** for edge deployments where cloud-scale context windows and parallel silicon are absent.

### 1.2 Solution framing

The engineering response is **Edge-Native Mode**: a runtime contract, toggled by **`edge_mode: true`** (YAML) or **`--edge-mode`** (CLI), that **tames context growth** without rewriting Hermes’ conversation semantics or breaking cloud deployments.

**Design principles:**

- **Opt-in by default.** Default remains `edge_mode: false`; cloud and workstation users see no behavioral change unless they enable the mode.
- **Two coordinated levers:**
  - **Tool-result shaping** — cap oversized **string** tool outputs before they enter the live model sequence, using deterministic head/tail truncation with an explicit marker (character budget, not token-perfect, but brutally effective against accidental megabyte dumps).
  - **Compression threshold discipline** — override the compressor’s dynamic percentage-of-context threshold with a **fixed local token budget** (`local_context_budget`, default **4000**), so local SLMs are not asked to pre-fill against implicit “cloud-sized” compression barriers derived from large `context_length` metadata.

- **Subagent inheritance.** Delegated children receive the parent’s edge constraints so orchestration cannot accidentally amplify context via unconstrained workers.

The result is a **production-grade, defensive** profile: it sacrifices fidelity on pathological tool payloads in exchange for **predictable iteration latency** and **bounded worst-case context** on edge hardware.

---

## 2. Pre-flight codebase audit & analysis (completed)

### 2.1 Repository token surface reduction (Repomix)

Before editing core loops, the audit team reduced the **effective repository token surface** presented to analysis tools from on the order of **~5M tokens** (full-tree Repomix-style dumps including generated artifacts, lockfiles, and bulky trees) to approximately **~137k tokens** by driving Repomix with a **surgical core whitelist** producing `repomix-core-output.xml` (see `.repomixignore` / Repomix configuration in-tree). That step did not change runtime behavior; it **de-risked human and automated review** by ensuring attention concentrated on load-bearing modules: `run_agent.py`, `agent/*`, `gateway/run.py`, `cli.py`, `hermes_cli/*`, `tui_gateway/server.py`, `cron/scheduler.py`, and targeted tests.

### 2.2 State-of-the-repository findings

**Head/tail serialization already existed — but on a “cold path.”**  
Nous Research’s `agent/context_compressor.py` implements `_serialize_for_summary`, which applies **head/tail splitting** when preparing **batches of turns for summarization**. That path is **legitimate and valuable**, but historically **passive** relative to the hot loop: it primarily serves **post-failure / summarizer formatting**, not the steady-state ingestion of fresh tool returns on every iteration.

**Tool executor bypass.**  
`agent/tool_executor.py` is the choke point where tool handlers return payloads that become **`role: "tool"`** messages. Those strings were largely **unshaped** for summarizer semantics: a single `read_file` or `terminal` capture could dwarf the rest of the transcript. In other words, **the compressor’s wisdom never saw the bytes on the way in** — only after they already poisoned KV-cache and pre-fill.

**Dynamic threshold vs. local reality.**  
The compressor’s threshold logic (percentage of resolved `context_length` with floors) is appropriate when `context_length` reflects a **realistically usable** cloud window. For local proxies, catalog metadata can still advertise **very large** nominal windows. Binding compression triggers to that metadata yields implicit budgets on the order of **tens of thousands of tokens** — e.g. a **~16k-token-class barrier** — which is **economically meaningless** on a 9B CPU inference stack: the model pays full pre-fill costs long before any compressor ever fires. Edge mode **short-circuits that mismatch** by pinning `threshold_tokens` to `min(local_context_budget, context_length)` when enabled.

### 2.3 Risk register (closed)

| Risk | Mitigation shipped |
|------|-------------------|
| Breaking multimodal tool returns | Truncation applies only to `str`; `_is_multimodal_tool_result` short-circuits |
| Breaking persistence / audit | Edge mode lowers `BudgetConfig` thresholds; truncation runs **after** `maybe_persist_tool_result` with JSON-aware shaping for `terminal` / `execute_code` |
| Breaking cloud defaults | `edge_mode` default `false`; CLI flags opt-in |
| Child agents escaping budget | `delegate_tool.py` forwards parent `edge_mode` / `local_context_budget` |

---

## 3. Architectural touchpoints (the core patch)

This section is the authoritative map from **requirement** to **symbol**.

### 3.1 `run_agent.py` — `AIAgent` surface

`AIAgent.__init__` gained constructor-level parameters:

- `edge_mode: bool | None = None`
- `local_context_budget: int | None = None`

These are forwarded into `init_agent(...)` so initialization remains centralized in `agent/agent_init.py`. Type hints use modern PEP 604 unions for consistency with surrounding `AIAgent` typing.

### 3.2 `agent/agent_init.py` — resolution order

`init_agent` resolves edge settings in a **deterministic precedence chain**:

1. Explicit kwargs when constructing the agent (CLI, gateway, tests).
2. `config["agent"]` (or equivalent) keys `edge_mode` and `local_context_budget`.
3. Safe numeric fallbacks (`local_context_budget` defaults to **4000** when missing or invalid).

The agent object receives **`edge_mode`** and **`local_context_budget`** as attributes. The `ContextCompressor` is constructed with:

- `edge_mode=getattr(agent, "edge_mode", False)`
- `edge_context_budget_tokens=getattr(agent, "local_context_budget", 4000)`

This keeps the compressor’s internal parameter name **`edge_context_budget_tokens`** while the user-facing config/CLI name remains **`local_context_budget`** — an intentional separation between “user knob” and “compressor token cap.”

### 3.3 `agent/tool_executor.py` — hot-path truncation

**New helper:** `_edge_mode_truncate_string_tool_result(agent, function_result, tool_name=...)`.

**Semantics:**

- No-op unless `getattr(agent, "edge_mode", False)` is truthy.
- **Multimodal dicts / envelopes** bypass truncation (`_is_multimodal_tool_result`).
- **Non-`str`** results bypass truncation (structured payloads, sentinel objects).
- For **`terminal` / `execute_code` JSON**, parse and truncate only large string fields (`output`, etc.) so the transcript stays valid JSON.
- For other long strings, use **`2000` head + marker + `1500` tail** when above **`4000` characters**.
- Edge mode threads a lowered **`agent._tool_result_budget`** (`BudgetConfig`) into `maybe_persist_tool_result` / `enforce_turn_budget` so oversized payloads hit the sandbox persisted-output path before model-facing shaping.

**Placement invariant (critical):**  
In both sequential and concurrent execution paths, the pipeline is:

1. Execute tool → `function_result`
2. `maybe_persist_tool_result(..., config=agent._tool_result_budget)` — **full payload to sandbox persisted-output when over edge threshold**
3. **`_edge_mode_truncate_string_tool_result`** — **JSON-safe model-facing shrink**
4. `agent._tool_result_content_for_active_model(...)` → append `messages`

Thus **oversized tool output can remain in the sandbox persisted-output store** while the **active model sequence** is protected.

### 3.4 `agent/context_compressor.py` — deterministic local ceiling

**New / repurposed internal state:**

- `self._edge_mode: bool`
- `self._edge_context_budget_tokens: int`

**`__init__`** accepts `edge_mode: bool = False` and `edge_context_budget_tokens: int = 4000`, stores them, resolves `context_length`, then calls **`_set_threshold_from_context()`**.

**`_set_threshold_from_context()`** replaces the older implicit “always derive from `threshold_percent`” behavior when edge mode is active:

- If `_edge_mode`:  
  `threshold_tokens = min(max(256, edge_cap), max(1, context_length))`
- Else: preserve the legacy percentage-based computation with `MINIMUM_CONTEXT_LENGTH` floor behavior.

**`update_model(...)`** now ends with **`self._set_threshold_from_context()`** so model switches (fallbacks, routing) cannot accidentally restore a cloud-scale threshold under an edge-flagged session.

**Note on `_serialize_for_summary`:**  
Head/tail formatting for summarizer input remains in `_serialize_for_summary`; edge mode does not remove it — instead, edge mode adds **orthogonal, ingress-side** protection in `tool_executor.py` plus **threshold pinning** here. The two layers address different lifecycle phases.

### 3.5 `tools/delegate_tool.py` — inheritance

Child `AIAgent` construction now passes:

- `edge_mode=getattr(parent_agent, "edge_mode", False)`
- `local_context_budget=int(getattr(parent_agent, "local_context_budget", 4000))`

This closes the **delegation amplification** hole where subagents could inherit cloud thresholds while the parent was edge-constrained.

---

## 4. Full-stack integration & wiring matrix

The feature is not “a flag in one file”; it is a **cross-surface contract**. The matrix below lists the integration layers and the mechanism used.

| Layer | Files / symbols | Mechanism |
|-------|-----------------|-----------|
| **Default config** | `hermes_cli/config.py` — `DEFAULT_CONFIG["agent"]` | Keys `edge_mode` (bool) and `local_context_budget` (int, default 4000) ship with the product defaults. |
| **CLI runtime defaults** | `cli.py` — `load_cli_config` merge | Ensures interactive HermesCLI sessions see the same defaults even before user YAML is merged. |
| **CLI flags** | `hermes_cli/_parser.py` | `--edge-mode` and `--local-context-budget N` on **root** and **`chat`** subcommand parsers so inheritance matches how operators actually launch sessions. |
| **CLI → agent** | `cli.py` — `HermesCLI.__init__`, `main()` kwargs, `AIAgent` construction sites | Thread `edge_mode` / `local_context_budget` from resolved args + config into each constructed `AIAgent`. |
| **Entrypoint** | `hermes_cli/main.py` | Parses args; passes kwargs into chat/oneshot paths; injects TUI subprocess environment (below). |
| **TUI gateway** | `tui_gateway/server.py` — `_cfg_edge_mode`, `_cfg_local_context_budget`, `_make_agent` | Subprocess may receive `HERMES_TUI_EDGE_MODE=1` and `HERMES_TUI_LOCAL_CONTEXT_BUDGET=<int>` from the parent shell; helpers prefer env overrides then fall back to YAML `agent.*`. |
| **Gateway runner** | `gateway/run.py` | Main agent path and background agent path read `agent_cfg` / `agent_cfg_local` and pass `edge_mode` / `local_context_budget` into `AIAgent`. |
| **Cache busting** | `gateway/run.py` — `_CACHE_BUSTING_CONFIG_KEYS` | Adds `("agent", "edge_mode")` and `("agent", "local_context_budget")` so **runtime config edits** force agent regeneration instead of silently reusing a stale agent instance with old compression behavior. |
| **Oneshot** | `hermes_cli/oneshot.py` | Merges YAML + explicit overrides; coerces `edge_mode` and `local_context_budget` when spawning short-lived agent runs. |
| **Cron** | `cron/scheduler.py` | Scheduler reads `agent` section from config and forwards the same kwargs into scheduled `AIAgent` instances, keeping unattended jobs consistent with interactive policy. |
| **User docs** | `README.md` | Concise **Edge-Native mode** section documents YAML keys and intent (bounded context for local SLMs). |

**Important path correction for readers:** the interactive CLI implementation lives at repository-root **`cli.py`**, not under `hermes_cli/cli.py`. The blueprint deliberately names real paths as they exist in-tree.

---

## 5. QA, linting, and test blindage

### 5.1 Targeted pytest strategy

Full Hermes CI uses `scripts/run_tests.sh` with parallelism and hermetic fixtures. During local edge validation, engineers occasionally needed to **override pytest addopts** (which may include aggressive `-n auto` / xdist defaults via `pyproject.toml`) using:

```bash
pytest -o addopts= tests/agent/test_edge_mode.py tests/agent/test_context_compressor.py -q
```

That pattern yields a deterministic, single-process run suitable for **tight inner-loop verification** on a laptop. The observed outcome during the edge-native hardening window was **5 passed** across:

- `tests/agent/test_edge_mode.py` — direct unit tests for `_edge_mode_truncate_string_tool_result` (on/off, boundary sizes).
- `tests/agent/test_context_compressor.py` — `TestEdgeModeCompressionThreshold` validating that `edge_mode=True` pins thresholds to the configured budget while `edge_mode=False` preserves percentage-based floors.

### 5.2 Lint and static analysis gates

- **`ruff check .`** was used as the **authoritative style/lint gate** matching CI expectations. The team deliberately **avoided wholesale `ruff format`** sweeps: unscoped formatting on a repository of this size creates high-noise diffs that obscure security-sensitive edits and complicate review.
- **`ty check`** (repository custom typing standard / advisory typing pass where wired) was exercised to ensure new parameters and `bool | None` / `int | None` unions did not introduce avoidable typing regressions.
- **Windows footgun scanner (`scripts/check-windows-footguns.py`).** Nous Research’s `CONTRIBUTING.md` requires this check before opening a PR: it greps for common **Windows-unsafe** patterns (POSIX-only signals, brittle process APIs, hardcoded `/tmp`, etc.). **Important operational detail:** when run with no arguments from a git checkout, the script scans **only staged files** — so the canonical pre-commit workflow is `git add …` then `python3 scripts/check-windows-footguns.py` (on macOS the interpreter is typically `python3`, not `python`). For an audit **without** staging, pass explicit paths, `--diff <ref>`, or `--all`. On 2026-05-17, after the edge-native integration, an explicit-path run over every Python file in the edge changeset reported: **`✓ No Windows footguns found (15 file(s) scanned).`** exit code **0**. That gives confidence that edits touching **`gateway/run.py`**, **`tui_gateway/server.py`**, **`cron/scheduler.py`**, and env-based wiring did not introduce fresh POSIX-only footguns.
- **Conventional Commits (scope).** Upstream guidance expects **Conventional Commits with an explicit scope** in parentheses. A bare `feat: …` is acceptable; reviewers and automated style checks align better with a **module scope** drawn from the touched surface — for this work, prefer **`feat(agent): introduce edge-native mode for local SLM execution`** (alternatives: `feat(cli)`, `feat(gateway)` if splitting commits; a single cross-cutting commit may still use `feat(agent):` as the primary behavioral owner). Body text should still describe config/TUI/gateway wiring in prose.

### 5.3 Incident: accidental mass-formatting and recovery

During the session, an accidental **`git checkout --`** (or equivalent broad reset) after an exploratory **`ruff format`** run **reverted the working tree**, temporarily wiping uncommitted edge-native edits across many paths. Recovery required **full reconstruction from editor session artifacts and careful re-application** of the patch series, followed by:

1. Re-verification of **all integration touchpoints** (gateway cache-bust list, TUI env bridging, cron wiring, delegate inheritance).
2. Re-running **scoped pytest** and **`ruff check .`** until green.

The lesson is procedural: **never mix exploratory formatting with uncommitted multi-file feature work** on a mega-monorepo; scope formatters to touched files only.

---

## 6. Chronological logs & next steps

### 6.1 Log (condensed engineering diary)

| Date | Milestone |
|------|-----------|
| **2026-05 (early)** | Problem reproduced on Mac/Intel i9 + Ollama: iteration latency explosion; pathological tool payloads implicated. |
| **2026-05 (mid)** | Pre-flight audit: Repomix core whitelist reduced analysis surface; `context_compressor` vs `tool_executor` lifecycle mismatch documented. |
| **2026-05 (late)** | Core patch landed: `tool_executor` ingress truncation + `ContextCompressor` threshold pinning + `init_agent` wiring + `AIAgent` ctor. |
| **2026-05 (late)** | Full-stack wiring: `hermes_cli/config.py`, `cli.py`, `_parser.py`, `main.py`, `oneshot.py`, `tui_gateway/server.py`, `gateway/run.py` (+ `_CACHE_BUSTING_CONFIG_KEYS`), `cron/scheduler.py`, `delegate_tool.py`. |
| **2026-05 (late)** | Tests: `test_edge_mode.py` + compressor threshold tests; `ruff check .` green on touched surfaces. |
| **2026-05-17** | **Documentation sync:** `README.md` Edge-Native section finalized; **`EDGE_MODE_BLUEPRINT.md`** rewritten from speculative plan to this chronicle. **Full-stack engineering phase marked complete.** |
| **2026-05-17** | **Pre-PR compliance pass:** `python3 scripts/check-windows-footguns.py` on the full edge-native path list → **15 files scanned, exit 0**; blueprint updated with CONTRIBUTING.md footgun workflow (`staged` default, `python3` on macOS) and **Conventional Commits** scope guidance (`feat(agent): …`). |

### 6.2 Next step (explicit)

1. Create local git branch **`feat/edge-native-optimization`**.  
2. Push to the **remote fork**.  
3. Open the **official upstream pull request** against **`NousResearch/hermes-agent`**, attaching this chronicle and test evidence in the PR body.

No upstream merge should be attempted until CI parity (`scripts/run_tests.sh`) has been run in an environment matching Hermes’ documented harness — the scoped pytest narrative above documents **developer inner loop** evidence, not a substitute for full-suite green.

---

## Appendix A — Operator quick reference

**YAML (`~/.hermes/config.yaml`):**

```yaml
agent:
  edge_mode: true
  local_context_budget: 4000
```

**CLI:**

```bash
hermes --edge-mode --local-context-budget 4000 chat
```

**TUI subprocess environment (set by parent when launching Ink bridge):**

- `HERMES_TUI_EDGE_MODE=1`
- `HERMES_TUI_LOCAL_CONTEXT_BUDGET=<integer>`

---

## Appendix B — Indexing note

`EDGE_MODE_BLUEPRINT.md` is listed in `.cursorignore` to prevent accidental inclusion of this long-form diary in default IDE embedding passes. Update `.cursorignore` temporarily if embedded search over this file is required in-editor.

---


## Appendix C — Canonical file manifest (edge-native changeset)

The following paths constitute the **core integration surface** (twelve Python modules plus collateral):

1. `run_agent.py` — `AIAgent` constructor parameters and `init_agent` forwarding.  
2. `agent/agent_init.py` — YAML/CLI resolution; `ContextCompressor` wiring.  
3. `agent/tool_executor.py` — `_edge_mode_truncate_string_tool_result` and call-site ordering.  
4. `agent/context_compressor.py` — `_set_threshold_from_context`, `update_model`, `__init__` edge fields.  
5. `tools/delegate_tool.py` — subagent kwargs inheritance.  
6. `hermes_cli/config.py` — `DEFAULT_CONFIG["agent"]` keys.  
7. `cli.py` — `HermesCLI` + `AIAgent` construction + `main()` passthrough.  
8. `hermes_cli/_parser.py` — global and `chat` argparse flags.  
9. `hermes_cli/main.py` — TUI env injection; chat kwargs.  
10. `hermes_cli/oneshot.py` — oneshot agent overrides.  
11. `tui_gateway/server.py` — `_cfg_*` helpers and `_make_agent`.  
12. `gateway/run.py` — `_CACHE_BUSTING_CONFIG_KEYS` + main/background `AIAgent` kwargs.  
13. `cron/scheduler.py` — scheduled runs inherit `agent.*` edge keys.

**Collateral (tests & docs):** `tests/agent/test_edge_mode.py`, `tests/agent/test_context_compressor.py`, `README.md`, and this `EDGE_MODE_BLUEPRINT.md`.

The historical “twelve files” narrative in internal notes maps to the first twelve **runtime** modules above before cron was folded into the same PR slice; **`cron/scheduler.py`** is retained here as an explicit thirteenth runtime integrator so the manifest stays truthful to the shipped tree.
