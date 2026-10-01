# Phase 5.3 — Model Identity and Normalization Baseline

## Baseline

Branch:

`refactor/phase5-3-model-identity`

Parent:

`591df2fac49a11ecba29223055d122b965417a38` — `Close provider registry authority boundary`

Phase 5.3 starts from the completed Phase 5.2 provider-registry boundary. This document freezes the model-identity migration surface before ownership moves begin. It records current responsibilities and their target phase; it does not introduce a compatibility layer or change runtime behaviour.

## Governing boundary

Phase 5.3 owns:

> What model does this identifier refer to, once provider context is known or explicitly stated?

It does not own catalogue discovery, capability metadata, runtime route policy, model selection, or credentials.

The hard-cut target remains one final model-identity owner with consumers migrated directly to it. No forwarding module, dual parser, synchronized alias table, or old/new normalization bridge is planned.

## Current Phase 5.3 ownership debt

### `hermes_cli/models.py`

Current model-identity responsibilities:

- `parse_model_input()` — parses Hermes `provider:model` syntax and named `custom:<name>:<model>` identities.
- `_known_provider_names()` / `_configured_custom_provider_ids()` — feed qualified-model parsing.
- `_resolve_static_model_alias()` — interprets the shared short model-family aliases against static candidates.
- `_resolve_provider_prefix()` — interprets an explicitly configured `provider/model` prefix.
- `normalize_provider()` — duplicate model-side provider normalization surviving after provider identity moved to `providers`.
- local vendor-prefix stripping helpers used partly as identity interpretation.

These move or are deleted during 5.3 except where the operation is actually catalogue or later selection policy.

### `hermes_cli/model_normalize.py`

This file is currently the primary model-ID normalization implementation.

It owns:

- vendor detection;
- aggregator vendor qualification;
- matching provider-prefix stripping;
- provider-specific dot/hyphen and case normalization;
- retired DeepSeek aliases;
- Copilot/OpenCode normalization delegation;
- catalogue-assisted prefix repair;
- `normalize_model_for_provider()`;
- `suggest_prefixed_model_id()`.

It currently calls upward into CLI-owned provider/catalogue code. Phase 5.3 moves the identity semantics to the final model domain and passes any catalogue candidates in from outside rather than allowing the identity owner to discover them.

The file is scheduled for deletion once its consumers migrate.

### `hermes_cli/model_switch.py`

Identity responsibilities mixed into the switching workflow:

- `ModelIdentity`;
- `MODEL_ALIASES`;
- `DirectAlias.provider` + `DirectAlias.model`;
- direct-alias provider/model parsing;
- `resolve_startup_model_route()` qualified-input interpretation;
- `_provider_identity()`;
- `resolve_alias()` alias-matching semantics;
- `_configured_provider_identity()`;
- `_resolve_named_custom_model_id()`;
- `_entry_aliases()`.

The surrounding switching workflow is not all 5.3. Candidate acquisition, provider choice, credential gating, endpoint application and persistence remain with their later owners.

### ACP

`acp_adapter/model_catalog.py` currently implements a second model-reference encoding/decoding surface:

- `_choice_provider()`;
- `encode_model_choice()`;
- custom-provider longest-prefix recovery.

`acp_adapter/server.py` imports `hermes_cli.models.parse_model_input()` to decode an ACP model selection.

This is a cross-process behavior boundary. Phase 5.3 must preserve the existing `provider:model` choice-ID shape while moving its semantics to the canonical model-identity owner.

### Runtime imports of CLI model normalization

Current direct imports of `hermes_cli.model_normalize`:

- `agent/agent_init.py`;
- `agent/auxiliary_client.py`;
- `agent/chat_completion_helpers.py`;
- `agent/error_classifier.py`;
- `agent/fallback_cooldown.py`;
- `agent/turn_recovery.py`.

These are explicit upward dependencies to remove in 5.3.

Additional model-side consumers still import `hermes_cli.models.normalize_provider` even though canonical provider identity already lives in `providers`, including ACP, agent initialization and models.dev handling.

## Deferred ownership — not 5.3

### Phase 5.4 — model catalogue

Remain catalogue concerns:

- `_PROVIDER_MODELS` and static catalogue data;
- `model_ids()`;
- `provider_model_ids()`;
- live provider `/models` fetches;
- models.dev merge/discovery;
- provider catalogue caches;
- OpenRouter/AI Gateway/Copilot/etc. catalogue acquisition;
- external-process model-list acquisition;
- candidate lists used for unambiguous normalization.

Identity may receive candidate IDs as input. It must not own their acquisition.

### Phase 5.5 — capabilities and metadata

Remain metadata concerns:

- `agent/model_metadata.py`;
- context lengths;
- reasoning capabilities;
- vision/tool metadata;
- fast-mode capability decisions;
- model-specific metadata caches and endpoint metadata discovery.

The similarly named metadata alias helpers in `agent/model_metadata.py` are cache lookup mechanics, not Phase 5.3 model-family alias authority.

### Phase 5.6 — runtime route / API-mode policy

Remain route concerns:

- `hermes_cli/runtime_provider.py` API-mode and endpoint resolution;
- `hermes_cli/runtime_provider_custom.py` custom endpoint routing;
- `DirectAlias.base_url`;
- `model_switch._PROVIDER_API_MODE_OVERRIDES`;
- `model_switch.model_derived_api_mode()`;
- endpoint/protocol/client-construction decisions.

### Phase 5.7 — model selection

Remain selection concerns:

- `detect_static_provider_for_model()`;
- `detect_provider_for_model()`;
- `_detection_candidates()`;
- current-provider catalogue preference;
- configured-provider precedence;
- OpenRouter fallback;
- model-switch routing order;
- ambiguity presentation and user-facing selection policy.

These paths may consume canonical model identity after 5.3, but provider choice is not identity.

### Phase 6 — credentials

Remain credential concerns:

- `DirectAlias.api_key` / `DirectAlias.key_env`;
- `direct_alias_api_key()`;
- model-switch credential resolution;
- `models_detect.provider_has_credentials()`;
- auth stores, OAuth, token refresh and credential pools.

Selection may consult credential availability, but credential ownership does not move into the model domain.

## Compatibility boundaries that must remain behaviorally stable

The hard cut applies to internal ownership, not persisted or cross-process user contracts.

Preserve:

1. User-facing `provider:model` model-selection syntax.
2. Named custom-provider form `custom:<name>:<model>`, including longest configured-name matching.
3. Provider-native model IDs containing `:`, such as Ollama-style tags, without treating every colon as a Hermes provider delimiter.
4. Slash-bearing aggregator model IDs such as `anthropic/claude-...` unless the caller has explicitly established a configured-provider prefix interpretation.
5. Existing `config.yaml` `model_aliases:` and `model.aliases:` persisted shapes.
6. ACP model choice IDs encoded as `provider:model`.
7. Exact provider/model route identity for named custom providers; `custom:<name>` must not collapse to generic `custom`.

## Focused baseline gate

The pre-migration gate covers:

- model normalization;
- model switch parsing;
- `provider/model` prefix routing;
- startup model routing;
- named/custom-provider switching;
- configured-provider routing;
- external-process model aliases;
- user-provider slug preservation;
- canonical custom identity;
- Copilot custom model IDs;
- ACP named-provider choice-ID round-trip;
- agent provider fallback normalization;
- fallback API-mode preservation;
- fallback normalization after timeout/429.

Command:

```text
python -m pytest -q \
  tests/hermes_cli/test_model_normalize.py \
  tests/hermes_cli/test_model_switch_parsing.py \
  tests/hermes_cli/test_model_prefix_routing.py \
  tests/hermes_cli/test_startup_model_routing.py \
  tests/hermes_cli/test_model_switch_custom_providers.py \
  tests/hermes_cli/test_model_switch_configured_provider_routing.py \
  tests/hermes_cli/test_model_switch_external_process_alias.py \
  tests/hermes_cli/test_model_switch_user_provider_slug_verbatim.py \
  tests/hermes_cli/test_custom_provider_identity.py \
  tests/hermes_cli/test_canonical_custom_identity.py \
  tests/hermes_cli/test_copilot_custom_model_ids.py \
  tests/acp_adapter/test_named_provider_catalogs.py \
  tests/agent/test_provider_fallback.py \
  tests/agent/test_fallback_api_mode_preservation.py \
  tests/agent/test_fallback_429_after_timeout.py
```

Result: **217 passed, 2 failed in 83.94s**.

Baseline failures:

- `tests/hermes_cli/test_model_switch_custom_providers.py::test_switch_to_bare_custom_from_another_provider_resolves_the_configured_endpoint`
  - expected the configured custom base URL but retained the current OpenRouter URL;
- `tests/hermes_cli/test_model_switch_custom_providers.py::test_switch_to_bare_custom_ignores_an_openrouter_mirror`
  - expected the configured custom mirror URL in the configured branch but retained the current Anthropic URL.

Both failures are endpoint/base-URL adoption behavior in the model-switch route path. They are classified as existing **Phase 5.6 route-policy debt**, not Phase 5.3 identity failures. An isolated rerun of exactly these two tests reproduced both failures (`2 failed in 2.93s`), so they are pinned as deterministic baseline failures rather than test-order noise. Phase 5.3 must preserve this baseline unless the later route-policy phase intentionally fixes it.

The gate is capped at 600 seconds. Later 5.3 sub-phases must introduce no additional failures relative to this result.

## Red-window declaration

5.3.1 itself changes no architecture and should remain green relative to the baseline above.

The deliberate hard-cut migration window begins when the final model-identity owner starts replacing the CLI implementation. During that window, targeted coherent-subgraph verification is authoritative. Whole-repository greenness returns at the 5.3 integration gate; no temporary compatibility scaffolding should be added merely to keep intermediate commits globally green.
