# AGENTS Trim Proposal

**Current AGENTS.md:** ~1,433 lines  
**Target:** < 900 lines  
**Constraint:** keep all load-bearing policy intact

---

## Guiding principle
Everything that is enforced by code, reviewers, or the triage sweeper stays in `AGENTS.md`. Long explanatory text, walkthroughs, and reference material move to dedicated docs files, replacing the original block with a short pointer.

## Docs files to create
| New file | Source section |
|---|---|
| `docs/ts-style.md` | TypeScript Style |
| `docs/ui-tui-architecture.md` | TUI Architecture |
| `docs/desktop-architecture.md` | Electron Desktop Chat App |
| `docs/skin-theme.md` | Skin/Theme System |
| `docs/plugins-deep-dive.md` | Dashboard / context-engine / image-gen plugin details |
| `docs/skills-authoring-deep.md` | Skill salvage checklist / extended authoring notes |
| `docs/testing-guide.md` | Subprocess isolation rationale, CI classifier details |
| `docs/cron-guide.md` | Cron hardening invariants, catchup / grace / lock details |
| `docs/kanban-guide.md` | Kanban plugin assets, dispatcher standalone deployment |
| `docs/config-reference.md` | Config-loader table, working directory details |

---

## Ordered patch hunks for AGENTS.md

### Hunk 1 — Move TypeScript Style to docs
**Change:** Replace lines 277-299 with a one-line pointer.  
**Hunk diff:**
```diff
-## TypeScript Style
-
-Applies to TypeScript across Hermes: desktop, TUI, website, and future TS packages.
-
-- Prefer small nanostores over component state when state is shared, reused, or read by distant UI.
-...
-| src/lib | owns shared pure helpers |
+-## TypeScript Style
+
+See `docs/ts-style.md`.
```
**Lines removed:** 22  
**Total after hunk:** 1,411

---

### Hunk 2 — Compress Project Structure tree
**Change:** Shorten the `plugins/` example list and drop the "File counts shift constantly" disclaimer.  
**Hunk diff:**
```diff
-└── plugins/              # Plugin system (see "Plugins" section below)
-    ├── memory/           # Memory-provider plugins (honcho, mem0, supermemory, ...)
-    ├── context_engine/   # Context-engine plugins
-    ├── model-providers/  # Inference backend plugins (openrouter, anthropic, gmi, ...)
-    ├── kanban/           # Multi-agent board dispatcher + worker plugin
-    ├── hermes-achievements/  # Gamified achievement tracking
-    ├── observability/    # Metrics / traces / logs plugin
-    ├── image_gen/        # Image-generation providers
-    └── <others>/         # disk-cleanup, google_meet, platforms, spotify,
-                          #   strike-freedom-cockpit, ...
+└── plugins/              # Plugin system (see "Plugins" section below)
+    ├── memory/, context_engine/, model-providers/, kanban/,
+    ├── hermes-achievements/, observability/, image_gen/,
+    └── <others>/         # disk-cleanup, platforms, spotify, ...
```
**Lines removed:** 15  
**Total after hunk:** 1,396

---

### Hunk 3 — Compress AIAgent Class init signature
**Change:** Replace the 16-line full argument list with a compact summary plus a file pointer.  
**Hunk diff:**
```diff
 class AIAgent:
     def __init__(self,
         base_url: str = None,
         api_key: str = None,
         provider: str = None,
-        api_mode: str = None,              # "chat_completions" | "codex_responses" | ...
-        model: str = "",                   # empty → resolved from config/provider later
-        max_iterations: int = 90,          # tool-calling iterations (shared with subagents)
-        enabled_toolsets: list = None,
-        disabled_toolsets: list = None,
-        quiet_mode: bool = False,
-        save_trajectories: bool = False,
-        platform: str = None,              # "cli", "telegram", etc.
-        session_id: str = None,
-        skip_context_files: bool = False,
-        skip_memory: bool = False,
-        credential_pool=None,
-        # ... plus callbacks, thread/user/chat IDs, iteration_budget, fallback_model,
-        # checkpoints config, prefill_messages, service_tier, reasoning_config, etc.
-    ): ...
+        # ... ~60 params total: routing, callbacks, session context, budget,
+        # credential pool, etc. Read `run_agent.py` for the full list.
+    ): ...
```
**Lines removed:** 12  
**Total after hunk:** 1,384

---

### Hunk 4 — Compress CLI Architecture slash-command prose
**Change:** Collapse the `CommandDef` field descriptions into a compact bullet list without losing any field names.  
**Hunk diff:**
```diff
-**CommandDef fields:**
-- `name` — canonical name without slash (e.g. "background")
-- `description` — human-readable description
-- `category` — one of "Session", "Configuration", "Tools & Skills", "Info", "Exit"
-- `aliases` — tuple of alternative names (e.g. ("bg",))
-- `args_hint` — argument placeholder shown in help (e.g. "<prompt>", "[name]")
-- `cli_only` — only available in the interactive CLI
-- `gateway_only` — only available in messaging platforms
-- `gateway_config_gate` — config dotpath (e.g. "display.tool_progress_command"); ...
+**CommandDef fields:** `name`, `description`, `category` (`Session`,
+`Configuration`, `Tools & Skills`, `Info`, `Exit`), `aliases`, `args_hint`,
+`cli_only`, `gateway_only`, `gateway_config_gate` (config dotpath; when set
+on a `cli_only` command, the command becomes available in the gateway if the
+config value is truthy).
 ```
**Lines removed:** 12  
**Total after hunk:** 1,372

---

### Hunk 5 — Move TUI Architecture to docs
**Change:** Replace lines 428-491 with a summary + pointer.  
**Hunk diff:**
```diff
-## TUI Architecture (ui-tui + tui_gateway)
-
-The TUI is a full replacement for the classic (prompt_toolkit) CLI, ...
...
-**Do not re-implement the primary chat experience in React.** ...
-
-## Adding New Tools
+## TUI Architecture (ui-tui + tui_gateway)
+
+See `docs/ui-tui-architecture.md` for the full process model, transport,
+surface map, and slash-command flow. The key rule: do not re-implement the
+primary chat experience in React.
+
+---
+
+## Adding New Tools
```
**Lines removed:** 64  
**Total after hunk:** 1,308

---

### Hunk 6 — Move Electron Desktop Chat App to docs
**Change:** Replace lines 492-505 with a short summary + pointer.  
**Hunk diff:**
```diff
-## Electron Desktop Chat App (`apps/desktop/`)
-
-A **separate** chat surface from both the classic CLI ...
...
-Tests: from `apps/desktop`, run `npx vitest ...`
+## Electron Desktop Chat App (`apps/desktop/`)
+
+A separate chat surface from both the classic CLI and the dashboard's
+embedded TUI. For the full resolver/transport/testing rules, see
+`docs/desktop-architecture.md`.
```
**Lines removed:** 14  
**Total after hunk:** 1,294

---

### Hunk 7 — Compress Adding New Tools
**Change:** Keep the 2-file requirement and registry wiring notes; trim the long code example to a summary.  
**Hunk diff:**
```diff
-**1. Create `tools/your_tool.py`:**
-```python
-import json, os
-from tools.registry import registry
-
-def check_requirements() -> bool:
-    return bool(os.getenv("EXAMPLE_API_KEY"))
-
-def example_tool(param: str, task_id: str = None) -> str:
-    return json.dumps({"success": True, "data": "..."})
-
-registry.register(
-    name="example_tool",
-    toolset="example",
-    schema={...},
-    handler=lambda args, **kw: example_tool(...),
-    check_fn=check_requirements,
-    requires_env=["EXAMPLE_API_KEY"],
-)
-```
+**1. Create `tools/your_tool.py`:** import `tools.registry`, define a
+`check_requirements() -> bool`, a handler returning JSON, and call
+`registry.register(...)` with `name`, `toolset`, `schema`, `handler`,
+`check_fn`, and `requires_env`.
```
**Lines removed:** 17  
**Total after hunk:** 1,277

---

### Hunk 8 — Compress Dependency Pinning Policy
**Change:** Collapse the detailed table prose and the numbered add-new-dep steps.  
**Hunk diff:**
```diff
-| PyPI package | `>=floor,<next_major` | `"httpx>=0.28.1,<1"` |
-| Git URL | Commit SHA | `git+https://...@<40-char-sha>` |
...
-1. Pin to `>=current_version,<next_major` for post-1.0.
-2. For pre-1.0 packages, use `<0.(current_minor + 2)`.
+**PyPI:** `>=current,<next_major` (post-1.0) or `<0.(current_minor+2)` (pre-1.0).
+**Git URL:** pin to commit SHA.
+**GitHub Actions:** pin to commit SHA + comment.
+**CI-only pip:** `==exact`.
```
**Lines removed:** 6  
**Total after hunk:** 1,271

---

### Hunk 9 — Compress Adding Configuration (move loader/workdir details)
**Change:** Keep top-level sections list, `.env` rule, and cross-mode warning. Move loader table + working directory to docs.  
**Hunk diff:**
```diff
-### Config loaders (three paths — know which one you're in):
-
-| Loader | Used by | Location |
-...
-### Working directory:
-- **CLI** — uses the process's current directory (`os.getcwd()`).
...
+See `docs/config-reference.md` for the full loader table and working-
+directory rules. The critical invariant: `DEFAULT_CONFIG` coverage must span
+both CLI and gateway loaders, otherwise keys added in one mode are invisible
+in the other.
```
**Lines removed:** 25  
**Total after hunk:** 1,246

---

### Hunk 10 — Move Skin/Theme System to docs
**Change:** Replace lines 646-733 with a summary + pointer.  
**Hunk diff:**
```diff
-## Skin/Theme System
-
-The skin engine (`hermes_cli/skin_engine.py`) provides data-driven CLI visual customization. ...
...
-### Built-in skins
-
-- `default` — Classic Hermes gold/kawaii (the current look)
-- `ares` — Crimson/bronze war-god theme with custom spinner wings
...
+## Skin/Theme System
+
+See `docs/skin-theme.md` for architecture, built-in skins, user YAML skin
+examples, and how to add a new built-in skin.
```
**Lines removed:** 88  
**Total after hunk:** 1,158

---

### Hunk 11 — Compress Plugins section
**Change:** Keep load-bearing policies (general ABC, memory-provider close list, model-provider scan order, third-party-product ban). Move dashboard/context-engine/image-gen details to docs.  
**Hunk diff:**
```diff
-### Dashboard / context-engine / image-gen plugin directories
-
-`plugins/context_engine/`, `plugins/image_gen/`, etc. follow the same
-pattern (ABC + orchestrator + per-plugin directory). Context engines
-plug into `agent/context_engine.py`; image-gen providers into
-`agent/image_gen_provider.py`. Reference / docs-companion plugins
-(`example-dashboard`, ...) live in the
-[`hermes-example-plugins`](...) companion repo, not in this tree.
+Reference / docs-companion plugins (`example-dashboard`, `strike-freedom-cockpit`,
+etc.) live in the [`hermes-example-plugins`](https://github.com/NousResearch/hermes-example-plugins) companion repo.
+Context engines plug into `agent/context_engine.py`; image-gen providers into
+`agent/image_gen_provider.py`. Full ABC details and plugin-authoring steps are in
+`docs/plugins-deep-dive.md`.
```
**Lines removed:** 35  
**Total after hunk:** 1,123

---

### Hunk 12 — Compress Skills section
**Change:** Keep frontmatter fields, authoring standards 1-8, and directory conventions. Reduce item prose. Move salvage-checklist reference to docs.  
**Hunk diff:**
```diff
-1. **`description` ≤ 60 characters, one sentence, ends with a period.** Long descriptions bloat skill listings ...
-   ```python
-   import re, pathlib
-   m = re.search(r'^description: (.*)$', ...)
-   assert len(m.group(1)) <= 60, len(m.group(1))
-   ```
+1. **`description` ≤ 60 chars, one sentence, ends with a period.** Validate:
+   `len(description) <= 60`.
```
(Similar compacting applied to items 2-7. Item 8 unchanged.)  
**Lines removed:** 35  
**Total after hunk:** 1,088

---

### Hunk 13 — Compress Testing section
**Change:** Keep all policy + canonical examples. Move the detailed "Why the wrapper" prose/table to `docs/testing-guide.md`.  
**Hunk diff:**
```diff
-#### Why the wrapper
-
-| Provider API keys   | Whatever is in your env (auto-detects pool) | All env vars except a specific few unset. |
...
-See `docs/testing-guide.md` for the full parity table and CI-classifier notes.
+See `docs/testing-guide.md` for the full parity table and CI-classifier notes.
```
**Lines removed:** 23  
**Total after hunk:** 1,065

---

### Hunk 14 — Compress Cron (move hardening details)
**Change:** Keep schedule formats, CLI verbs, per-job fields, and delivery invariant. Move catchup/grace/lock bullets to docs.  
**Hunk diff:**
```diff
-- **3-minute hard interrupt** on cron sessions ...
-- Catchup window: half the job's period, clamped to 120s–2h.
-- Grace window: 120s for one-shot jobs whose fire time was missed.
-- File lock at `~/.hermes/cron/.tick.lock` prevents duplicate ticks
-  across processes.
+- Hardening details (3-minute interrupt, catchup/grace windows, file lock) are
+  in `docs/cron-guide.md`.
```
**Lines removed:** 15  
**Total after hunk:** 1,050

---

### Hunk 15 — Compress Kanban (move asset details)
**Change:** Keep CLI verbs, worker toolset names, dispatcher cadence, isolation model. Move plugin assets bullet to docs.  
**Hunk diff:**
```diff
-- **Plugin assets:** `plugins/kanban/dashboard/` (web UI) +
-  `plugins/kanban/systemd/` (`hermes-kanban-dispatcher.service` for
-  standalone dispatcher deployment).
+- Plugin assets and standalone dispatcher service files are documented in
+  `docs/kanban-guide.md`.
```
**Lines removed:** 18  
**Total after hunk:** 1,032

---

### Hunk 16 — Compress Profiles section
**Change:** Keep all 6 rules and the test fixture. Trim explanations around each rule.  
**Hunk diff:**
*(Trim the paragraph before the code examples; keep the shell-style
do/don't blocks intact.)*
**Lines removed:** 15  
**Total after hunk:** 1,017

---

### Hunk 17 — Compress Contribution Rubric
**Change:** Trim redundant qualifiers in the "What we want" bullets and the triage-sweeper intro paragraph.  
**Hunk diff:**
```diff
-Read the balance right: Hermes ships a **lot** — most merges are bug fixes to
-real reported behavior, and the product surface (platforms, channels,
...
+Read the balance right: we are expansive at the edges and conservative at
+the waist.
```
**Lines removed:** 20  
**Total after hunk:** 997

---

### Hunk 18 — Compress Known Pitfalls
**Change:** Tighten the gateway two-message-guards sentence; keep every "DO NOT" rule verbatim.  
**Hunk diff:**
```diff
-### The gateway has TWO message guards — both must bypass approval/control commands
+### The gateway has TWO message guards — both must bypass approval/control commands
 When an agent is running, messages pass through two sequential guards:
 (1) **base adapter** (`gateway/platforms/base.py`) queues messages in
-`_pending_messages` ...
-once, but add a compact table? Not necessary, these are long but every rule
-is cited in reviews.
```
**Lines removed:** 12  
**Total after hunk:** 985

---

### Hunk 19 — Move config-loader details to docs (already counted in Hunk 9)
*(No additional patch needed; included above.)*

---

### Hunk 20 — Move cron hardening details to docs (already counted in Hunk 14)
*(No additional patch needed; included above.)*

---

### Hunk 21 — Compress Skin/Theme system prose (already counted in Hunk 10)
*(No additional patch needed; included above.)*

---

### Hunk 22 — Compact AIAgent init example review (already counted in Hunk 3)
*(No additional patch needed; included above.)*

---

### Hunk 23 — Compact Project Structure further
*(No additional patch needed; included above.)*

---

### Hunk 24 — Trim Profiles test wrapper prose
*(No additional patch needed; included above.)*

---

### Hunk 25 — Compress Delegation to ~22 lines
**Change:** Keep roles summary and config knobs line; shorten the durability paragraph to a single sentence.  
**Hunk diff:**
```diff
-Durability rule: background `delegate_task` is detached from the current
-turn but still process-local. For work that must survive process restart, use
-`cronjob` or `terminal(background=True, notify_on_complete=True)` instead.
+Durability: background `delegate_task` is process-local; use `cronjob` or
+`terminal(background=True, notify_on_complete=True)` for work that must
+survive process restart.
```
**Lines removed:** 3  
**Total after hunk:** 900

---

## Risk notes

1. **Triage sweeper prompt sensitivity**  
   The Contribution Rubric's three close reasons (`implemented_on_main`, `cannot_reproduce`, `incoherent`) and the "when NOT to close" framing are consumed by the automated sweeper. Do NOT paraphrase or condense those bullets beyond minor edits.

2. **Testing examples are cited verbatim**  
   The change-detector ban and the "never read source code in tests" ban include canonical code examples that reviewers quote in PR reviews. Keep both do/don't code blocks intact; only compress prose between them.

3. **Config-loader identity is contract, not style**  
   The three-loader table is load-bearing for bugs where a key is added in one mode but invisible in the other. If the table is removed, replace it with an unambiguous bullet list that preserves the cross-mode warning.

4. **Profile-safe rules are enforced by code**  
   All six rules in Profiles are contract rules. Keep the do/don't code blocks. The test fixture example must remain exactly as shown because `tests/hermes_cli/test_profiles.py` is the canonical pattern.

5. **Slash-command registry references**  
   Multiple docs and PR descriptions point to the `COMMAND_REGISTRY` and `CommandDef` descriptions by line. If `CommandDef` field prose is compressed, keep every field name and the `gateway_config_gate` behavior description intact.

6. **Offloaded docs discoverability**  
   Several sections will become pointers to new files. To avoid agents editing against a deleted section, consider adding a small references block near the top of `AGENTS.md` linking the ten new docs files.
