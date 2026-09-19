# Operator Test Recipe — Telegram Rich Messages (Bot API 10.1+)

> **Audience**: Operators / users with real `TELEGRAM_BOT_TOKEN` muốn verify rich messages end-to-end trên bot của mình.
>
> **Pre-requisite**: Hermes Agent với Telegram gateway enabled (`display.platforms.telegram.streaming: true`).
>
> **Spec ref**: https://core.telegram.org/bots/api#rich-messages · Skill `~/.hermes/skills/devtools/telegram-rich-messages/SKILL.md` v1.1.0.

---

## 1. Pre-flight checks

### 1.1 Verify Bot API 10.1 live (no real token needed)

```bash
python3 scripts/dry_run_send_rich.py
# Expected output:
#   sendRichMessage: HTTP 401
#   sendRichMessageDraft: HTTP 401
# ✅ All methods DEPLOYED (401 = method exists, dummy token rejected)
```

If you see HTTP 404 → your Telegram client lib / server is <10.1. Upgrade.

### 1.2 Verify Hermes has rich messages wired

```bash
grep -n "_try_send_rich\|sendRichMessage\|rich_messages" \
    ~/.hermes/hermes-agent/plugins/platforms/telegram/adapter.py | head -10
# Expected: ~5-10 hits. If 0 → Hermes not at a recent commit with rich messages.
```

Verify your commit is recent enough:

```bash
git -C ~/.hermes/hermes-agent log --oneline plugins/platforms/telegram/adapter.py | head -5
# Expected: top commits mention "rich", "sendRichMessage", or PR #44780.
```

### 1.3 Opt-in to rich messages (currently OFF by default)

The rich path is opt-in because Telegram clients can render Bot API 10.1 messages as blank/unsupported bubbles. Verify your client supports rich messages first, then opt-in:

```yaml
# ~/.hermes/profiles/default/config.yaml
display:
  platforms:
    telegram:
      streaming: true
      extra:
        rich_messages: true   # Opt-in (default off per commit 6183e8ce1)
```

Restart the gateway: `hermes gateway restart` (or restart your process).

---

## 2. Manual test sequence

Run these in order. Each step has clear success criteria.

### 2.1 Test 1 — Plain rich paragraph

Send to your bot via DM:

```
/ping
```

**Bot should respond** with a short rich paragraph (the welcome message is usually rich-formatted).

**Verify in Telegram mobile/desktop**:
- ✅ Message shows formatted text (bold, italic, or code spans as expected)
- ✅ NOT shown as blank bubble
- ✅ Can be **copied as plain text** (long-press → Copy). If Copy doesn't work → your client doesn't support rich messages yet.

### 2.2 Test 2 — Table render

In DM, ask the bot a question that should produce tabular output:

```
What's the current price of BTC, ETH, SOL?
```

**Expected**: A table with columns Symbol / Price / 24h Δ, rows for each coin.

**Verify in Telegram**:
- ✅ Columns align correctly
- ✅ Cell formatting (bold for highlight, code for numbers) renders
- ✅ On TDesktop: hover row → see "Copy as plain text" option

### 2.3 Test 3 — Streaming draft animation

In DM, ask a complex question that triggers agent reasoning (≥3 sec):

```
Walk me through the architecture of Telegram Rich Messages Bot API 10.1
```

**Expected during streaming**:
- ✅ You see a "Thinking..." chip / animated placeholder for ~1-2 sec
- ✅ Text appears incrementally (not all-at-once)
- ✅ Final message is the complete rich answer (no missing parts)
- ✅ No duplicate final messages (the bug fixed in PR #46009)

### 2.4 Test 4 — Edit upgrade (progressive enhancement)

In DM, send a prompt that triggers a "loading" reply:

```
Generate a 200-word summary of quantum computing
```

**Expected**:
- ✅ Bot sends initial short "Generating..." text
- ✅ Bot edits the message in-place with the full rich summary (table + headings)
- ✅ Message timestamp reflects the **edit time**, not the original send time
- ✅ No duplicate "old version" + "new version" visible

### 2.5 Test 5 — CJK fallback (if you use Chinese / Japanese / Korean)

In DM:

```
Write a haiku in Chinese about Telegram
```

**Expected**:
- ✅ Bot uses **legacy MarkdownV2** path (not rich messages) — due to commit `ea056b055 fix(telegram): avoid rich messages for CJK text`
- ✅ Chinese characters render correctly without garbling
- ✅ Can copy as plain text

If you see garbled CJK → your client doesn't render rich messages cleanly for CJK; report at Issue #44428.

### 2.6 Test 6 — Code block with syntax highlight

In DM:

```
Write a Python hello world
```

**Expected**:
- ✅ Code rendered with syntax highlighting (Python keywords colored)
- ✅ Language label visible ("python" or similar)
- ✅ Copy preserves whitespace

---

## 3. Failure modes + fixes

| Symptom | Root cause | Fix |
|---------|-----------|-----|
| Bot shows blank bubble | Client <Bot API 10.1 | Upgrade Telegram client |
| Rich text rendered, but Copy-as-text broken | TDesktop edge case | Report at Issue #44428 |
| Streaming draft stays "Thinking..." forever | LLM timeout >30s | Reduce prompt size; check `streaming_overflow_limit` |
| Final message has duplicate old version | Pre-#46009 bug | Update to a recent Hermes commit (>=a59d5e37e) |
| CJK garbled | Pre-ea056b055 | Update Hermes; CJK path auto-falls-back to MarkdownV2 |
| `rich_messages: true` ignored | Config typo / yaml reload issue | `hermes config validate` then restart gateway |

---

## 4. CI integration (optional)

Add to your CI pipeline to catch regressions early:

```yaml
# .github/workflows/telegram-rich-smoke.yml
name: Telegram Rich Messages smoke
on: [push, pull_request]
jobs:
  smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Dry-run verify
        run: python3 scripts/dry_run_send_rich.py
        # Expected: exit 0 if methods deployed, exit 1 if not
```

If `dry_run_send_rich.py` exits 1 → fail the build. Telegram may have reverted an API method.

---

## 5. Where to file issues

| Concern | GitHub Issue |
|---------|--------------|
| Feature request: richer streaming UX | #44428 (canonical) or #45864 (duplicate) |
| Bug: rich rendering broken | #46009 (closed — verify fix landed first) |
| Bug: edit_message destroys rich | #46009 (verify fix) |
| CJK fallback broken | New issue, tag `platform/telegram` + `comp/gateway` |
| TDesktop edge case | Comment on #44428 |

---

## 6. References

- **Bot API spec**: https://core.telegram.org/bots/api#rich-messages
- **Bot API 10.1 changelog**: https://core.telegram.org/bots/api-changelog#june-11-2026
- **Skill**: `~/.hermes/skills/devtools/telegram-rich-messages/SKILL.md` v1.1.0
- **Cook session**: `~/.hermes/plans/reports/cook-gateway-rich-20260627-090803/`
- **Initial ship**: commit `a59d5e37e feat(telegram): make rich messages always on` (2026-06-13)
- **Opt-in default**: commit `6183e8ce1 fix(telegram): make Bot API 10.1 rich messages opt-in (default off)` (2026-06-21)
- **CJK fallback**: commit `ea056b055 fix(telegram): avoid rich messages for CJK text`
- **Edit fix**: PR #46009 (closed, fix landed)