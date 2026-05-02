# Vibe-Trading Natural Language Forwarding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let Hermes forward stock-related Feishu questions to Vibe-Trading Agent in natural language, defaulting A-share analysis to free AKShare-backed workflows.

**Architecture:** Extend the existing `plugins/vibe-trading` user plugin, not Hermes core. Add high-level tools that create a Vibe session, send the user's original question, poll session messages until the assistant answer is written, then return that report to Hermes. Keep Tushare/QVeris out of phase 1.

**Tech Stack:** Hermes plugin API, Python stdlib `urllib.request`, Vibe-Trading session REST API, pytest.

---

### Task 1: Natural-Language Forwarding Tests

**Files:**
- Modify: `tests/plugins/test_vibe_trading_plugin.py`

- [x] Add a test for `vibe_ask` that verifies it creates a session, sends the raw user question, polls messages, and returns the assistant report.
- [x] Add a test for `vibe_ask_ashare` that verifies the prompt includes the user's raw question plus an A-share/AKShare instruction.
- [x] Add a registration test expectation for `vibe_ask` and `vibe_ask_ashare`.

### Task 2: Plugin Implementation

**Files:**
- Modify: `plugins/vibe-trading/__init__.py`
- Modify: `plugins/vibe-trading/plugin.yaml`

- [x] Add `VIBE_TRADING_AGENT_TIMEOUT_SECONDS` and `VIBE_TRADING_AGENT_POLL_SECONDS` config helpers.
- [x] Add `_vibe_agent_ask(question, title, instruction)` helper.
- [x] Add `vibe_ask(question)` for generic Vibe-Trading Agent forwarding.
- [x] Add `vibe_ask_ashare(question)` for stock/A-share forwarding with free AKShare-first instructions.
- [x] Register both tools with clear schema descriptions so Hermes can choose them for natural language stock questions.

### Task 3: Verify and Deploy

**Files:**
- Copy: `plugins/vibe-trading/` to `192.168.1.63:/root/.hermes/plugins/vibe-trading/`

- [x] Run focused pytest.
- [x] Copy updated plugin to the Hermes host.
- [x] Restart Hermes container.
- [x] Verify `vibe_ask_ashare` can return a Vibe Agent report for a small A-share prompt.
- [ ] Commit and push to `fork`.
