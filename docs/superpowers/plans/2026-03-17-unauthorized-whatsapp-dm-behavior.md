# Unauthorized WhatsApp DM Behavior Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let Hermes ignore unauthorized WhatsApp DMs while preserving the current pairing flow as the default behavior elsewhere.

**Architecture:** Add a gateway-level unauthorized-DM behavior setting with a global default and a per-platform override stored in gateway config. Resolve the effective behavior in `GatewayRunner` before sending pairing replies, and keep the default as `pair` so existing platforms do not change unless configured.

**Tech Stack:** Python, pytest, gateway config/runtime, YAML-backed user config

---

## Chunk 1: Config Shape And Coverage

### Task 1: Add config round-trip coverage

**Files:**
- Modify: `tests/gateway/test_config.py`
- Modify: `gateway/config.py`

- [ ] **Step 1: Write the failing config tests**

Add tests covering:
- `GatewayConfig` round-trips a global `unauthorized_dm_behavior`
- `PlatformConfig.extra` round-trips a per-platform `unauthorized_dm_behavior`
- `load_gateway_config()` bridges a `whatsapp.unauthorized_dm_behavior` entry from `config.yaml`

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/gateway/test_config.py -q`
Expected: FAIL on missing unauthorized-DM behavior support

- [ ] **Step 3: Write minimal config implementation**

Update `gateway/config.py` so:
- `GatewayConfig` stores a global unauthorized-DM behavior string with default `pair`
- `from_dict()` / `to_dict()` preserve it
- `load_gateway_config()` bridges `gateway.unauthorized_dm_behavior` and `whatsapp.unauthorized_dm_behavior` from `config.yaml` into runtime config

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/gateway/test_config.py -q`
Expected: PASS

## Chunk 2: Runner Behavior

### Task 2: Add unauthorized DM behavior tests

**Files:**
- Create: `tests/gateway/test_unauthorized_dm_behavior.py`
- Modify: `gateway/run.py`

- [ ] **Step 1: Write the failing runner tests**

Add tests covering:
- default behavior still sends a pairing code for an unauthorized DM
- WhatsApp with `unauthorized_dm_behavior=ignore` sends nothing
- global `ignore` suppresses pairing replies for unauthorized DMs

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/gateway/test_unauthorized_dm_behavior.py -q`
Expected: FAIL because `GatewayRunner` always pairs unauthorized DMs today

- [ ] **Step 3: Write minimal runner implementation**

Update `gateway/run.py` so:
- it resolves the effective unauthorized-DM behavior from global config plus per-platform override
- unauthorized DMs only get pairing replies when the effective behavior is `pair`
- unauthorized DMs are silently ignored when the effective behavior is `ignore`

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/gateway/test_unauthorized_dm_behavior.py -q`
Expected: PASS

## Chunk 3: Docs And Regression Verification

### Task 3: Document and verify the feature

**Files:**
- Modify: `website/docs/user-guide/security.md`
- Modify: `website/docs/user-guide/messaging/whatsapp.md`

- [ ] **Step 1: Update docs**

Document:
- the new behavior options (`pair`, `ignore`)
- recommended WhatsApp configuration for private numbers
- that pairing remains the default unless overridden

- [ ] **Step 2: Run targeted verification**

Run: `source .venv/bin/activate && python -m pytest tests/gateway/test_config.py tests/gateway/test_unauthorized_dm_behavior.py tests/gateway/test_whatsapp_connect.py -q`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add gateway/config.py gateway/run.py tests/gateway/test_config.py tests/gateway/test_unauthorized_dm_behavior.py website/docs/user-guide/security.md website/docs/user-guide/messaging/whatsapp.md docs/superpowers/plans/2026-03-17-unauthorized-whatsapp-dm-behavior.md
git commit -m "feat: support ignoring unauthorized gateway DMs"
```
