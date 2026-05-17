# Centralized Model Registry + Dashboard Model Management Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Refactor Hermes model selection so every runtime path resolves models through one centralized method/class using a stable `provider/model` model ID, then overhaul the dashboard so users can reorder the main/fallback model chain by drag-and-drop and add provider models from `/models`-style discovery, including auxiliary/vision assignment.

**Architecture:** Add a single backend model registry/resolver that normalizes old config shapes and new `provider/model` references into a `ResolvedModel` object. Then migrate gateway, API server, cron, auxiliary, delegation, CLI picker, and dashboard APIs to call that resolver instead of hand-parsing `{provider, model}`. Finally, extend the web dashboard Models page with a configured chain editor, drag/drop reorder, provider model discovery/add flow, and auxiliary assignment UI.

**Tech Stack:** Python backend (`agent/`, `gateway/`, `hermes_cli/`, `cron/`, `tools/`), YAML config, pytest; React/TypeScript dashboard (`web/src`), Vite build; native HTML5 pointer/drag events unless a dependency is explicitly approved.

---

## Current Context / Findings

Existing relevant files:

- Model resolution is scattered:
  - `gateway/run.py` — `_resolve_gateway_model`, `_load_fallback_model`, `_resolve_turn_agent_config`
  - `gateway/platforms/api_server.py` — `_resolve_model_name`, `_runtime_model_context`, request-time bounded models
  - `agent/agent_init.py` / `run_agent.py` — AIAgent init and fallback chain normalization
  - `agent/auxiliary_client.py` — auxiliary task provider/model resolution
  - `cron/scheduler.py` — cron model/fallback handling
  - `tools/cronjob_tools.py` — cron model override parsing
  - `tools/delegate_tool.py` — delegation model/provider inheritance and overrides
  - `hermes_cli/model_switch.py`, `hermes_cli/models.py`, `hermes_cli/inventory.py` — curated provider/model lists and picker payloads
- Existing model dashboard code:
  - `web/src/pages/ModelsPage.tsx`
  - `web/src/components/ModelPickerDialog.tsx`
  - `web/src/lib/api.ts`
  - `hermes_cli/web_server.py` endpoints:
    - `GET /api/model/options`
    - `GET /api/model/auxiliary`
    - `POST /api/model/set`
    - `GET /api/analytics/models?days=N`
- Existing fallback CLI helper:
  - `hermes_cli/fallback_cmd.py` already has `_read_chain()` and `_write_chain()` for `fallback_providers`.
- Existing design reference:
  - `~/.hermes/skills/autonomous-ai-agents/hermes-agent/references/centralized-model-registry-design.md`

Important constraint: preserve backward compatibility for current configs using:

```yaml
model:
  provider: openrouter
  default: anthropic/claude-sonnet-4
fallback_providers:
  - provider: gemini
    model: gemini-2.5-flash
auxiliary:
  vision:
    provider: openrouter
    model: openai/gpt-4o-mini
custom_providers: [...]
```

New canonical reference should be a model ID string:

```text
provider/model
custom:<name>/model
```

where model may itself contain slashes, so parsing must split only at the first `/` after a provider prefix.

---

## Target UX

### Dashboard: configured model chain

Models page should show a configuration-focused panel above analytics:

1. **Primary model** row.
2. **Fallback providers** list in tried order.
3. Drag handle on each fallback row; mouse drag/drop reorders list.
4. Buttons:
   - `Change main`
   - `Add fallback`
   - `Remove fallback`
   - `Assign to Vision`
   - `Assign to Auxiliary...`
5. Saving writes to `config.yaml` and applies to new sessions.
6. UI must clearly state existing sessions/gateway may need restart/new session if applicable.

### Dashboard: add model for provider

User flow:

1. User clicks `Add model` or `Add fallback`.
2. UI selects provider.
3. Backend returns provider models from the same source as `/models` / `model.options`.
4. User picks a model.
5. Backend adds it into the model registry/config if missing.
6. User can assign it to:
   - main
   - fallback chain
   - vision
   - any auxiliary task

### `/models` compatibility

The dashboard must reuse the same backend discovery as `/models` / `model.options`, not maintain a separate hardcoded frontend list.

---

## Proposed Backend Model Registry

Create `agent/model_registry.py`.

Core types:

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass(frozen=True)
class ModelRef:
    provider: str
    model: str

    @property
    def id(self) -> str:
        return f"{self.provider}/{self.model}"

@dataclass(frozen=True)
class ResolvedModel:
    id: str
    provider: str
    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    api_mode: Optional[str] = None
    context_length: Optional[int] = None
    capabilities: Dict[str, Any] = field(default_factory=dict)
    provider_config: Dict[str, Any] = field(default_factory=dict)
    model_config: Dict[str, Any] = field(default_factory=dict)
```

Main class:

```python
class ModelRegistry:
    def __init__(self, config: dict, *, env: Mapping[str, str] | None = None): ...

    def parse_ref(self, value: str | dict | None, *, default_provider: str | None = None) -> ModelRef: ...

    def resolve(self, value: str | dict | None, *, role: str = "main", default_provider: str | None = None) -> ResolvedModel: ...

    def main(self) -> ResolvedModel: ...

    def fallback_chain(self) -> list[ResolvedModel]: ...

    def auxiliary(self, task: str) -> ResolvedModel: ...

    def to_legacy_agent_kwargs(self, resolved: ResolvedModel) -> dict: ...
```

Key rules:

- Accept string IDs: `openrouter/anthropic/claude-sonnet-4`.
- Accept legacy dicts: `{provider: openrouter, model: anthropic/claude-sonnet-4}`.
- Accept current `model` block as main.
- Preserve `custom:<name>` provider IDs exactly.
- Resolve provider runtime credentials via existing `hermes_cli.runtime_provider.resolve_runtime()` / `gateway.run.resolve_runtime_provider()` wrappers where needed; do not duplicate auth logic.
- Preserve existing `fallback_providers` config format initially to avoid a huge migration.
- Add a model registry view in config, but make it optional/backward compatible.

Candidate config addition:

```yaml
model_registry:
  providers:
    openrouter:
      models:
        - id: anthropic/claude-sonnet-4
          context_length: 200000
          capabilities:
            vision: false
            tools: true
    custom:ollama-cloud:
      base_url: https://ollama.com/v1
      models:
        - id: gpt-oss:20b
```

Do **not** make this mandatory in first PR. First PR should centralize resolution and provide registry APIs while still reading old config.

---

## Implementation Tasks

### Task 1: Protect worktree and create feature branch

**Objective:** Start from latest `main` and isolate the work.

**Files:** none.

**Commands:**

```bash
cd /root/.hermes/hermes-agent
git status --short
git checkout main
git pull upstream main
git checkout -b feat/centralized-model-registry-dashboard
```

**Verification:**

```bash
git branch --show-current
```

Expected: `feat/centralized-model-registry-dashboard`.

**Commit:** none.

---

### Task 2: Add failing unit tests for model ref parsing and legacy config compatibility

**Objective:** Lock down the new `provider/model` ID behavior before implementation.

**Files:**

- Create: `tests/agent/test_model_registry.py`
- Create/modify later: `agent/model_registry.py`

**Test cases:**

```python
def test_parse_ref_splits_provider_once():
    reg = ModelRegistry({})
    ref = reg.parse_ref("openrouter/anthropic/claude-sonnet-4")
    assert ref.provider == "openrouter"
    assert ref.model == "anthropic/claude-sonnet-4"
    assert ref.id == "openrouter/anthropic/claude-sonnet-4"


def test_parse_custom_provider_ref():
    reg = ModelRegistry({})
    ref = reg.parse_ref("custom:ollama-cloud/gpt-oss:20b")
    assert ref.provider == "custom:ollama-cloud"
    assert ref.model == "gpt-oss:20b"


def test_resolve_main_from_legacy_model_block():
    cfg = {"model": {"provider": "gemini", "default": "gemini-2.5-flash"}}
    resolved = ModelRegistry(cfg).main()
    assert resolved.id == "gemini/gemini-2.5-flash"
    assert resolved.provider == "gemini"
    assert resolved.model == "gemini-2.5-flash"


def test_resolve_fallback_chain_from_legacy_list():
    cfg = {"fallback_providers": [
        {"provider": "openrouter", "model": "openai/gpt-4o-mini"},
        {"provider": "gemini", "model": "gemini-2.5-flash"},
    ]}
    chain = ModelRegistry(cfg).fallback_chain()
    assert [m.id for m in chain] == [
        "openrouter/openai/gpt-4o-mini",
        "gemini/gemini-2.5-flash",
    ]
```

**Run:**

```bash
python -m pytest tests/agent/test_model_registry.py -q
```

Expected: FAIL because `agent.model_registry` does not exist yet.

---

### Task 3: Implement `agent/model_registry.py`

**Objective:** Provide the single class/method all model resolution will converge on.

**Files:**

- Create: `agent/model_registry.py`

**Implementation notes:**

- Keep implementation dependency-light to avoid import cycles.
- No direct disk reads inside the registry; pass config in.
- Include helper functions:
  - `model_id(provider, model)`
  - `split_model_id(model_id)`
  - `legacy_entry_to_ref(entry)`
  - `resolved_to_legacy_dict(resolved)`
- Treat `auto` auxiliary provider as main model inheritance unless caller explicitly wants unresolved `auto`.
- Do not fetch network model metadata here. Capabilities can be optionally injected from `models.dev` later.

**Run:**

```bash
python -m pytest tests/agent/test_model_registry.py -q
```

Expected: PASS.

**Commit:**

```bash
git add agent/model_registry.py tests/agent/test_model_registry.py
git commit -m "feat: add centralized model registry"
```

---

### Task 4: Add tests for config write helpers: main/fallback/auxiliary

**Objective:** Define backend API semantics for setting main, fallback order, adding models, and assigning auxiliary tasks.

**Files:**

- Modify: `tests/hermes_cli/test_web_server_models.py` or create `tests/hermes_cli/test_dashboard_model_config.py`
- Modify later: `hermes_cli/web_server.py`

**Test cases:**

- `POST /api/model/set` still updates main legacy keys.
- New endpoint `GET /api/model/configured` returns:

```json
{
  "main": {"id": "openrouter/anthropic/claude-sonnet-4", "provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
  "fallbacks": [ ... ],
  "auxiliary": [ ... ]
}
```

- New endpoint `PUT /api/model/fallbacks` persists exact ordered list to `fallback_providers`.
- New endpoint `POST /api/model/register` adds provider/model to optional registry without clobbering existing config.

**Run:**

```bash
python -m pytest tests/hermes_cli/test_dashboard_model_config.py -q
```

Expected: FAIL until endpoints exist.

---

### Task 5: Add dashboard model config REST endpoints

**Objective:** Expose configured model chain and write operations for dashboard.

**Files:**

- Modify: `hermes_cli/web_server.py`
- Modify: `web/src/lib/api.ts` types later

**Endpoints:**

1. `GET /api/model/configured`
   - Reads `load_config()`.
   - Uses `ModelRegistry(cfg)`.
   - Returns main, fallback chain, auxiliary assignments, and registry entries.

2. `PUT /api/model/fallbacks`
   - Body:
     ```json
     {"fallbacks": [{"provider": "gemini", "model": "gemini-2.5-flash"}]}
     ```
   - Validates provider/model non-empty.
   - Writes `fallback_providers` in the same legacy list format.
   - Removes `fallback_model` legacy key.

3. `POST /api/model/register`
   - Body:
     ```json
     {"provider": "openrouter", "model": "openai/gpt-4o-mini", "capabilities": {"vision": true}}
     ```
   - Adds to `model_registry.providers[provider].models` if missing.
   - Does not automatically set main/fallback unless requested by separate call.

4. Extend `POST /api/model/set`
   - Keep current behavior.
   - Internally use `ModelRegistry` helper formatting where possible.

**Run:**

```bash
python -m pytest tests/hermes_cli/test_dashboard_model_config.py -q
```

Expected: PASS.

**Commit:**

```bash
git add hermes_cli/web_server.py tests/hermes_cli/test_dashboard_model_config.py
git commit -m "feat: expose dashboard model configuration endpoints"
```

---

### Task 6: Migrate API server bounded model resolution to `ModelRegistry`

**Objective:** Replace API-server-specific model parsing with central resolver.

**Files:**

- Modify: `gateway/platforms/api_server.py`
- Modify: `tests/gateway/test_api_server.py`

**Targets:**

- `_runtime_model_context()`
- `_resolve_model_name()`
- `_resolve_request_model()` if present on the current branch
- Any logic that builds allowed/exposed model list from main/fallback/auxiliary.

**Expected behavior:**

- `/v1/models` exposes exactly configured main + fallback models by stable ID, preserving deterministic request-time routing.
- Requests using an unknown model still return OpenAI-compatible 400.
- Requests using known fallback model route to the fallback provider/model.

**Run:**

```bash
python -m pytest tests/gateway/test_api_server.py -q
```

Expected: PASS.

**Commit:**

```bash
git add gateway/platforms/api_server.py tests/gateway/test_api_server.py
git commit -m "refactor(api-server): use centralized model registry"
```

---

### Task 7: Migrate gateway runtime model loading to `ModelRegistry`

**Objective:** Make gateway main/fallback turn construction call the central resolver.

**Files:**

- Modify: `gateway/run.py`
- Modify/add: `tests/gateway/test_gateway_model_registry.py`

**Targets:**

- `_resolve_gateway_model(config)`
- `_load_fallback_model()`
- `_try_resolve_fallback_provider()` if still required
- `_resolve_turn_agent_config()` only if model/provider parsing duplicates registry logic.

**Important:** preserve current runtime-provider credential resolution behavior. The registry should produce model identity; existing runtime provider code should still resolve API key/base URL where needed.

**Run:**

```bash
python -m pytest tests/gateway/test_gateway_model_registry.py tests/gateway/test_api_server.py -q
```

Expected: PASS.

**Commit:**

```bash
git add gateway/run.py tests/gateway/test_gateway_model_registry.py
git commit -m "refactor(gateway): centralize main and fallback model resolution"
```

---

### Task 8: Migrate auxiliary, cron, and delegation call sites

**Objective:** Remove remaining duplicated provider/model parsing from background and subagent paths.

**Files:**

- Modify: `agent/auxiliary_client.py`
- Modify: `cron/scheduler.py`
- Modify: `tools/cronjob_tools.py`
- Modify: `tools/delegate_tool.py`
- Add/modify tests under:
  - `tests/agent/`
  - `tests/cron/`
  - `tests/tools/`

**Rules:**

- `auxiliary.<task>.provider: auto` resolves to main model.
- Explicit auxiliary provider/model resolves via `ModelRegistry.auxiliary(task)`.
- Cron model override accepts both dict and new string model ID.
- Delegation explicit provider/model remains backward compatible.

**Run:**

```bash
python -m pytest tests/agent tests/cron tests/tools -q
```

Expected: PASS for touched subsets.

**Commit:**

```bash
git add agent/auxiliary_client.py cron/scheduler.py tools/cronjob_tools.py tools/delegate_tool.py tests/agent tests/cron tests/tools
git commit -m "refactor: route auxiliary cron and delegation models through registry"
```

---

### Task 9: Update CLI/model picker inventory to expose registered + discovered models

**Objective:** Ensure `/models`, TUI `model.options`, REST `/api/model/options`, and dashboard all share provider model discovery.

**Files:**

- Modify: `hermes_cli/model_switch.py`
- Modify: `hermes_cli/inventory.py`
- Modify: `hermes_cli/models.py` only if needed
- Add/modify: `tests/hermes_cli/test_inventory.py`

**Behavior:**

- `build_models_payload(load_picker_context())` should merge:
  1. authenticated providers from existing auth/runtime checks,
  2. curated `_PROVIDER_MODELS`,
  3. `models.dev` where existing code already supports it,
  4. user-added `model_registry.providers.<provider>.models`.
- Deduplicate models while preserving order:
  - current model first if present,
  - curated/discovered list,
  - user-added list.

**Run:**

```bash
python -m pytest tests/hermes_cli/test_inventory.py -q
```

Expected: PASS.

**Commit:**

```bash
git add hermes_cli/model_switch.py hermes_cli/inventory.py tests/hermes_cli/test_inventory.py
git commit -m "feat(models): include registered models in picker inventory"
```

---

### Task 10: Add web API client types

**Objective:** Let React consume new model configuration endpoints with strict types.

**Files:**

- Modify: `web/src/lib/api.ts`

**Types:**

```ts
export interface ConfiguredModelEntry {
  id: string;
  provider: string;
  model: string;
  capabilities?: {
    supports_vision?: boolean;
    supports_tools?: boolean;
    supports_reasoning?: boolean;
    context_window?: number;
  };
}

export interface ConfiguredModelsResponse {
  main: ConfiguredModelEntry | null;
  fallbacks: ConfiguredModelEntry[];
  auxiliary: AuxiliaryTaskAssignment[];
}
```

**Methods:**

```ts
getConfiguredModels()
setFallbackModels(fallbacks)
registerModel(body)
```

**Run:**

```bash
cd web && npm run build
```

Expected: type errors until UI uses or exports correctly; fix in following task.

---

### Task 11: Build fallback chain drag/drop component

**Objective:** Add mouse reorder UI for fallback providers.

**Files:**

- Modify: `web/src/pages/ModelsPage.tsx`
- Optionally create: `web/src/components/ModelChainEditor.tsx`

**Do not add a drag/drop dependency initially.** Implement simple native drag/drop:

- Each fallback row has `draggable`.
- Track `draggedIndex` state.
- On `dragOver`, call `preventDefault()`.
- On `drop`, reorder local state.
- `Save order` calls `api.setFallbackModels()`.
- Include keyboard-accessible fallback buttons as safety:
  - Move up
  - Move down
  - Remove

**UI rows:**

```text
☰  #1 gemini · gemini-2.5-flash       [Vision?] [↑] [↓] [Remove]
☰  #2 openrouter · openai/gpt-4o-mini [Tools]   [↑] [↓] [Remove]
```

**Run:**

```bash
cd web && npm run build
```

Expected: PASS.

**Commit:**

```bash
git add web/src/pages/ModelsPage.tsx web/src/lib/api.ts
git commit -m "feat(dashboard): add draggable fallback model chain"
```

---

### Task 12: Add dashboard model registration flow

**Objective:** Let users add new models for a provider from the `/models`/picker source.

**Files:**

- Modify: `web/src/components/ModelPickerDialog.tsx`
- Modify: `web/src/pages/ModelsPage.tsx`
- Modify: `web/src/lib/api.ts`

**UX:**

- Add button in settings panel: `Add model`.
- Reuse `ModelPickerDialog` in standalone mode.
- On provider/model confirm:
  1. call `api.registerModel({provider, model, capabilities?})`,
  2. then ask/offer actions:
     - Set as main
     - Add as fallback
     - Assign to Vision
     - Assign to Auxiliary task
- Keep minimal first implementation: register + show action buttons in same modal or after success banner.

**Capability defaults:**

- If `ModelPickerDialog` can expose metadata from backend, use it.
- Otherwise infer only from existing `models.dev` capabilities in backend response; do not guess vision capability from model name.
- User can explicitly assign to vision even if metadata is missing; warn if model is not known vision-capable.

**Run:**

```bash
cd web && npm run build
```

Expected: PASS.

**Commit:**

```bash
git add web/src/components/ModelPickerDialog.tsx web/src/pages/ModelsPage.tsx web/src/lib/api.ts
git commit -m "feat(dashboard): register provider models from picker"
```

---

### Task 13: Add auxiliary assignment improvements for Vision and tasks

**Objective:** Make assigning a newly added model to vision/auxiliary straightforward.

**Files:**

- Modify: `web/src/pages/ModelsPage.tsx`
- Potentially modify: `hermes_cli/web_server.py` if endpoint body needs batch assignment.

**UX additions:**

- Add `Assign to Vision` quick action on model chain rows and analytics cards.
- Keep existing Auxiliary Tasks modal, but allow entry points from a selected model.
- If assigning vision:
  - call `POST /api/model/set` with `{scope: "auxiliary", task: "vision", provider, model}`.

**Run:**

```bash
cd web && npm run build
python -m pytest tests/hermes_cli/test_dashboard_model_config.py -q
```

Expected: PASS.

**Commit:**

```bash
git add web/src/pages/ModelsPage.tsx hermes_cli/web_server.py tests/hermes_cli/test_dashboard_model_config.py
git commit -m "feat(dashboard): add quick auxiliary model assignment"
```

---

### Task 14: Update docs

**Objective:** Document the new single model ID, fallback ordering UI, and model registration behavior.

**Files:**

- Modify: `website/docs/user-guide/configuring-models.md`
- Modify: `website/docs/user-guide/features/fallback-providers.md`
- Modify: `website/docs/user-guide/features/api-server.md` if model IDs exposed by API server changed/clarified.
- Modify: `website/docs/developer-guide/provider-runtime.md`

**Docs must include:**

- `provider/model` model ID format.
- Backward compatible legacy config still works.
- Dashboard model changes apply to new sessions.
- Fallback order is tried top-to-bottom.
- Model registration adds entries to config but does not guarantee provider access/credentials.

**Run:**

```bash
python -m pytest tests/website -q || true
cd website && npm run build
```

If website tests/build are too slow or not available, at least run targeted docs lint if available and record limitation.

**Commit:**

```bash
git add website/docs/user-guide/configuring-models.md website/docs/user-guide/features/fallback-providers.md website/docs/user-guide/features/api-server.md website/docs/developer-guide/provider-runtime.md
git commit -m "docs: document centralized model ids and dashboard model management"
```

---

### Task 15: Regression tests and build validation

**Objective:** Prove no major runtime path broke.

**Commands:**

```bash
cd /root/.hermes/hermes-agent
python -m pytest tests/agent/test_model_registry.py -q
python -m pytest tests/gateway/test_api_server.py -q
python -m pytest tests/hermes_cli/test_inventory.py tests/hermes_cli/test_dashboard_model_config.py -q
python -m pytest tests/cron tests/tools -q
cd web && npm run build
```

Then run broader suite if feasible:

```bash
cd /root/.hermes/hermes-agent
python -m pytest tests/ -o 'addopts=' -q
```

**Expected:** all targeted tests pass. Full suite may be long; if it fails, triage and fix regressions before push.

---

### Task 16: Manual dashboard smoke test

**Objective:** Verify actual browser UI behavior, not just build.

**Commands:**

Start web/dashboard according to project scripts. If backend is the Hermes dashboard service, use the existing dev command or service path discovered in docs.

Manual checks:

1. Open Models page.
2. Confirm main model row displays current config.
3. Add two fallback entries.
4. Drag fallback #2 above #1 with mouse.
5. Refresh page; order persists.
6. Use move up/down fallback buttons; order persists.
7. Add a provider model through picker.
8. Assign it to Vision.
9. Confirm `~/.hermes/config.yaml` contains expected `fallback_providers` order and `auxiliary.vision` assignment.

Automated browser smoke if feasible:

- Use Playwright or browser tool against local dashboard.
- Capture screenshot evidence if a visual bug is fixed.

---

### Task 17: Final commit, rebase, push

**Objective:** Land clean branch with tests.

**Commands:**

```bash
cd /root/.hermes/hermes-agent
git status --short
git fetch upstream
git rebase upstream/main
# rerun targeted tests if rebase changed files
python -m pytest tests/agent/test_model_registry.py tests/gateway/test_api_server.py tests/hermes_cli/test_dashboard_model_config.py -q
cd web && npm run build
cd ..
git push -u origin feat/centralized-model-registry-dashboard
```

If branch already exists remotely after rebase:

```bash
git push --force-with-lease origin feat/centralized-model-registry-dashboard
```

---

## Risks / Tradeoffs

- **Import cycles:** `agent/model_registry.py` must not import high-level gateway/CLI modules at import time. Keep credential/runtime resolution outside or lazily imported.
- **Provider/model ID ambiguity:** Providers can be `custom:<name>`; models often contain `/`. Split only once at first `/`.
- **Backward compatibility:** Do not force a config migration in the first PR. Read old config and write old-compatible keys (`model.provider`, `model.default`, `fallback_providers`) while introducing optional registry metadata.
- **Dashboard dependency creep:** Avoid adding drag/drop dependency unless native implementation is too brittle. Native drag/drop + keyboard move buttons is enough.
- **Live sessions:** Config updates should be documented as applying to new sessions; current chat runtime may not hot-swap.
- **Capabilities truth:** Do not infer vision/tool support from names. Use `models.dev` metadata where available or user assignment with warning.
- **API server compatibility:** `/v1/models` and bounded routing recently changed; preserve deterministic model acceptance/routing behavior.

---

## Open Questions

1. Should canonical config eventually become `models:` or `model_registry:`? Plan uses `model_registry:` to avoid colliding with existing `/models` concepts and to keep migration optional.
2. Should fallback order include the primary model in one unified chain, or keep primary separate from fallback list? Plan keeps primary separate to match current Hermes semantics.
3. Should dashboard save immediately on drop or require `Save order`? Plan recommends explicit save to prevent accidental config writes, but instant-save is also acceptable if UI shows status and rollback.
4. Should registered user models appear before or after curated models in picker? Plan recommends current model first, curated/discovered next, user-added last unless already current.

---

## Acceptance Criteria

- One central backend class/method can resolve any configured model reference into `ResolvedModel`.
- Existing configs continue to work.
- Main model, fallback providers, auxiliary tasks, cron overrides, delegation, and API server model routing use or are covered by the central resolver.
- Dashboard can reorder fallback chain with mouse drag/drop.
- Dashboard can add/register a model for an authenticated provider from picker/discovery data.
- Dashboard can assign a model to Vision and other auxiliary tasks.
- Targeted Python tests pass.
- `web && npm run build` passes.
- Changes are committed and pushed to a feature branch after implementation.
