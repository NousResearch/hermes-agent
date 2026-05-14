# Hermes Provider Routing and Custom Provider Slugs — Implementation Plan

> **For Hermes:** keep this PR small, fit it to the existing codebase, and do not add new abstractions unless the current ones already expect them

**Goal:** fix custom provider routing so Hermes preserves the real transport/api_mode, supports stable custom-provider slugs via `provider_key`, and honors custom provider context lengths during compression feasibility checks

**Architecture:**
This PR stays inside the current Hermes provider/config pipeline. It does not introduce new provider types or new runtime features. It only makes existing custom-provider resolution more faithful to config, improves slug stability for named endpoints, and threads existing custom-provider metadata into the compression feasibility path

**Tech Stack:** Python, pytest, Hermes config YAML, existing provider/model-switch runtime

---

## Scope

### In scope
- `hermes_cli/config.py`: allow `provider_key` and `description` in custom provider config entries
- `hermes_cli/providers.py`: resolve custom providers with the correct transport from `api_mode` / `transport`
- `hermes_cli/model_switch.py`: prefer `provider_key` when building custom-provider slugs
- `run_agent.py`: pass resolved custom providers into compression context-length checks
- `tests/*`: add or keep focused coverage for custom provider routing and compression feasibility

### Out of scope
- No new provider backends
- No new chat features
- No broad compression auto-sync on `/model --global`
- No changes to unrelated gateway behavior
- No schema migration beyond the existing config parser acceptance

---

## Current English audit summary

The uncommitted work is mostly coherent and fits Hermes' existing architecture, but the earlier version was too broad in one place: it tried to auto-rewrite compression config when switching the global model. That is risky because it silently changes user intent. The safer PR is the smaller provider-routing fix plus compression feasibility context support

### Safe pieces
- custom provider `api_mode` / `transport` is now preserved instead of being hardcoded to `openai_chat`
- `provider_key` gives stable identifiers for custom providers, which helps `/model` and config round-tripping
- compression feasibility can now see custom provider context-length overrides

### Risk to avoid
- do not couple `/model --global` to compression config rewriting unless there is an explicit opt-in flag or a separate PR

---

## Implementation tasks

### Task 1: Keep custom provider config schema compatible

**Objective:** let Hermes accept `provider_key` and `description` on custom provider entries without treating them as unknown noise

**Files:**
- Modify: `hermes_cli/config.py`
- Test: `tests/hermes_cli/test_provider_config_validation.py`

**Verification:**
- `pytest tests/hermes_cli/test_provider_config_validation.py -q`

---

### Task 2: Preserve real transport for custom providers

**Objective:** map custom providers to the correct runtime transport instead of forcing `openai_chat`

**Files:**
- Modify: `hermes_cli/providers.py`
- Test: `tests/hermes_cli/test_runtime_provider_resolution.py` or existing custom-provider model-switch tests

**Verification:**
- `pytest tests/hermes_cli/test_runtime_provider_resolution.py -q`
- `pytest tests/hermes_cli/test_custom_provider_model_switch.py -q`

---

### Task 3: Prefer explicit provider keys in model picker output

**Objective:** make stable custom-provider slugs win when present, so config and runtime stay aligned

**Files:**
- Modify: `hermes_cli/model_switch.py`
- Test: `tests/hermes_cli/test_model_switch_custom_providers.py`

**Verification:**
- `pytest tests/hermes_cli/test_model_switch_custom_providers.py -q`

---

### Task 4: Honor custom provider context lengths in compression checks

**Objective:** let compression feasibility use the same custom-provider metadata Hermes already knows about

**Files:**
- Modify: `run_agent.py`
- Test: `tests/run_agent/test_compression_feasibility.py`

**Verification:**
- `pytest tests/run_agent/test_compression_feasibility.py -q`

---

## PR checklist

- [ ] Keep the diff focused on existing Hermes plumbing
- [ ] Avoid accidental compression-config rewrites
- [ ] Run targeted tests for provider routing and compression feasibility
- [ ] Check `git diff --stat` before commit
- [ ] Write a conventional commit message with a detailed body
- [ ] Push branch and open PR against upstream Hermes
- [ ] Watch GitHub checks and fix only what CI reports

---

## Suggested commit shape

- `fix: preserve custom provider routing metadata`
- body should mention:
  - custom providers now keep their transport/api_mode
  - `provider_key` becomes the stable slug when present
  - compression feasibility now sees custom provider context lengths
  - no new runtime provider abstractions were introduced

---

## Notes for implementation

- Keep `run_agent.py` changes tightly scoped to the compression feasibility path
- Do not reintroduce compression config auto-sync in `/model --global`
- Prefer existing test files over inventing new coverage locations
- If a test needs small fixture adjustments, keep those changes local and mechanical
