# Phase 5.5 Model Metadata Baseline

Base architecture: Phase 5.3 closed at `257b258a6a`.
Phase 5.5 shared type/reducer additions do not touch any pre-existing capability consumer path.

## Focused gate

```text
python -m pytest -q   tests/agent/test_models_dev.py   tests/agent/test_model_metadata.py   tests/hermes_cli/test_custom_provider_context_length.py   tests/hermes_cli/test_openrouter_reasoning_metadata.py   tests/hermes_cli/test_nous_reasoning_metadata.py   tests/agent/test_image_routing.py   tests/agent/test_custom_providers_vision.py   tests/hermes_cli/test_managed_vision_capability.py
```

Result before capability ownership migration:

```text
291 passed, 3 failed
```

The deterministic baseline failures are:

1. `tests/agent/test_model_metadata.py::TestMoAContextLength::test_moa_resolves_from_aggregator`
   - MoA preset context resolves to the generic 256K fallback instead of the aggregator's 1M window.

2. `tests/agent/test_image_routing.py::TestExtractImageRefs::test_finds_absolute_path`
   - Windows absolute image paths are not extracted.

3. `tests/agent/test_image_routing.py::TestExtractImageRefs::test_finds_home_relative_path`
   - Windows HOME-relative image paths are not extracted.

These failures are outside the Phase 5.5 ownership migration and are not acceptance blockers unless the migration introduces additional failures or changes their failure mode.

## Phase 5.5 acceptance rule

- No new failures relative to this focused baseline.
- Capability/metadata ownership must move to `models.metadata`.
- Existing known failures are not repaired opportunistically in this phase.

## Integrated Phase 5.5 result

Post-integration focused gate:

```text
292 passed, 2 failed
```

The MoA aggregator context case now passes. The only remaining failures are the two
pre-existing Windows image-path extraction cases listed above; their failure mode is unchanged.

### Closeout verification

- `tests/models`: **76 passed**
- model-metadata ownership guard: **7 passed**
- reasoning/context/catalog integration shard: **130 passed**
- Codex/Astra/Bedrock/MoA ownership-focused shard: **106 passed**
- Bedrock provider-confirmed restart persistence: **4 passed**
- `git diff --check`: clean
