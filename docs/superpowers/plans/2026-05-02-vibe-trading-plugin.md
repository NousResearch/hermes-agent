# Vibe-Trading Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a versioned Hermes user plugin that lets Hermes Agent call the existing Vibe-Trading API from Feishu conversations.

**Architecture:** Keep Hermes core unchanged. Add `plugins/vibe-trading` with a `plugin.yaml`, a focused Python registration module, and tests that exercise HTTP request construction, error handling, and registered tool handlers. Deploy by syncing the plugin directory to `/root/.hermes/plugins/vibe-trading` and enabling it in `/root/.hermes/config.yaml`.

**Tech Stack:** Hermes plugin API, Python stdlib `urllib.request`, `pytest`, Docker-deployed Hermes Agent v0.12.0, Vibe-Trading REST API at `http://192.168.1.58:8899`.

---

### Task 1: Tests

**Files:**
- Create: `tests/plugins/test_vibe_trading_plugin.py`

- [ ] Write tests that import `plugins/vibe-trading/__init__.py` by path.
- [ ] Verify `_request_json()` sends GET and POST requests to the configured base URL.
- [ ] Verify HTTP or JSON errors are returned as JSON error payloads instead of raising.
- [ ] Verify `register(ctx)` registers the expected tool names.

### Task 2: Plugin

**Files:**
- Create: `plugins/vibe-trading/plugin.yaml`
- Create: `plugins/vibe-trading/__init__.py`

- [ ] Add plugin metadata for `vibe-trading`.
- [ ] Implement a small HTTP helper with `VIBE_TRADING_BASE_URL` and `VIBE_TRADING_TIMEOUT_SECONDS`.
- [ ] Register conservative first-version tools: health, skills, swarm presets, swarm run, swarm result, sessions, session message, run result, run list.
- [ ] Return JSON strings from every handler and never raise tool exceptions.

### Task 3: Verify, Deploy, Publish

**Files:**
- Modify: remote `/root/.hermes/config.yaml`
- Copy: `plugins/vibe-trading/` to remote `/root/.hermes/plugins/vibe-trading/`

- [ ] Run focused pytest.
- [ ] Sync plugin to `root@192.168.1.63`.
- [ ] Enable plugin next to `tencent-news`.
- [ ] Restart Hermes container.
- [ ] Verify `vibe_health` can reach `http://192.168.1.58:8899/health`.
- [ ] Commit only the new plugin, tests, and docs.
- [ ] Push commit to the `fork` remote.
