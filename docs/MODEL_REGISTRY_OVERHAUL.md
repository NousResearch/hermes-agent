# Centralized Model Registry & Dashboard Overhaul

## Summary

This PR implements a centralized model registry for Hermes Agent, replacing scattered model resolution logic across the codebase with a single, consistent source of truth. It also adds a complete fallback chain management UI to the dashboard.

## Changes

### Backend (`agent/model_registry.py`)
- **New `ModelRegistry` class**: Centralizes all model resolution logic
  - `main()` - resolves main model with legacy fallback
  - `fallback_chain()` - resolves fallback provider chain
  - `auxiliary()` - resolves auxiliary task assignments
- **`ModelRef` / `ResolvedModel` dataclasses**: Consistent model identity
- **`parse_ref()`**: Handles string IDs, legacy dicts, and `None`
- **Backward compatible**: Maintains full support for legacy config format

### Backend (`hermes_cli/web_server.py`)
- **New REST endpoints**:
  - `GET /api/model/configured` - returns main + fallbacks + auxiliary with full resolution
  - `PUT /api/model/fallbacks` - reorders/clears fallback chain, persists to config
  - `POST /api/model/register` - registers models in `model_registry.providers`, 409 on conflict

### Migration Points
All model resolution points now use `ModelRegistry`:
- `gateway/run.py` - `_resolve_gateway_model()`, `_try_resolve_fallback_provider()`, `_load_fallback_model()`
- `agent/auxiliary_client.py` - `_read_main_model()`, `_read_main_provider()`
- `agent/image_routing.py` - `_explicit_aux_vision_override()`
- `cli.py` - CLI fallback chain resolution
- `hermes_cli/inventory.py` - `load_picker_context()`

### Frontend (`web/src/pages/ModelsPage.tsx`)
- **New Fallback Chain UI section**:
  - Add button (opens model picker)
  - Save button (persists to config)
  - Move up/down buttons for reordering
  - Remove button for each fallback item
  - Error display for save failures
- **All UI elements have `data-testid` attributes** for Playwright tests

### Frontend (`web/src/lib/api.ts`)
- **New TypeScript interfaces**: `ModelInfo`, `AuxiliaryTaskAssignment`, `ConfiguredModelsResponse`, `SetFallbacksResponse`, `RegisterModelResponse`
- **New API client methods**: `getConfiguredModels()`, `setFallbackChain()`, `registerModel()`

### Tests
- **Backend tests**: 47 passing (model_registry + dashboard endpoints + gateway resolution)
- **Playwright UI tests**: New test suite in `tests/ui/test_models_page_fallback_chain.py`
  - Fallback chain section visibility
  - Add button visibility
  - Save button visibility
  - Remove button visibility
  - Move up/down button visibility
  - API integration tests

## Running the Tests

### Backend Tests
```bash
python3 -m pytest tests/agent/test_model_registry.py tests/hermes_cli/test_dashboard_model_config.py tests/gateway/test_run_model_resolution.py -v
```

### Playwright UI Tests
```bash
# Start the dashboard first
sudo systemctl restart hermes-dashboard

# Run tests
python3 -m pytest tests/ui/test_models_page_fallback_chain.py -v
```

### Full Regression Tests
```bash
python3 -m pytest tests/ -v --ignore=tests/plugins --ignore=tests/integration
```

## API Examples

### Get Configured Models
```bash
curl -H "X-Hermes-Session-Token: <token>" \
  http://127.0.0.1:9119/api/model/configured
```

### Update Fallback Chain
```bash
curl -X PUT -H "X-Hermes-Session-Token: <token>" \
  -H "Content-Type: application/json" \
  -d '[{"provider":"copilot","model":"gpt-5.4"},{"provider":"openrouter","model":"anthropic/claude-sonnet-4"}]' \
  http://127.0.0.1:9119/api/model/fallbacks
```

### Register a New Model
```bash
curl -X POST -H "X-Hermes-Session-Token: <token>" \
  -H "Content-Type: application/json" \
  -d '{"id":"custom/my-model","provider":"custom","model":"my-model","base_url":"https://api.example.com/v1"}' \
  http://127.0.0.1:9119/api/model/register
```

## Migration Notes

### For Developers
- All model resolution now goes through `ModelRegistry`
- Legacy config format is still supported but new code should use the registry
- The `ModelInfo` type replaces the old `dict`-based model references

### For Users
- No config changes required - all existing configs continue to work
- New fallback chain UI available at `/models` page in the dashboard
- Fallback providers are now stored in a list format (`fallback_providers`) instead of a single dict (`fallback_model`)

## Related
- Plan: `.hermes/plans/2026-05-16_model-registry-and-dashboard-overhaul.md`
