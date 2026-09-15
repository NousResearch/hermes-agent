# TTS Debugging Guide for Feishu Platform

## Problem: TTS voice doesn't match AI written text (e.g., says "smiling eyes" instead of actual reply)

### Root Cause Chain

1. **AI response contains image markdown** → `extract_images()` in `base.py` processes it
2. **alt-text leaks into TTS input** → stripped but not fully removed, contaminating speech text
3. **`_strip_markdown_for_tts()` removes too much** → leaves behind unexpected tokens
4. **TTS reads leftover emoji/alt-text names** → produces garbled voice output

### Debugging Steps

1. **Check which TTS path fires**: Add debug logs to BOTH paths:

   In `gateway/run.py` `_send_voice_reply()`:
   ```python
   logger.info("=== RUN.PY TTS INPUT (raw) === text=%r | after_strip=%r", text[:500], tts_text[:500])
   ```

   In `gateway/platforms/base.py` auto-TTS block:
   ```python
   logger.info("=== TTS AUTO-FIRE for %s chat=%s text_content=%r ===", self.name, event.source.chat_id, text_content[:300])
   ```

2. **Send test message** and grep gateway logs for `TTS INPUT` or `TTS AUTO-FIRE`

3. **Check raw text vs stripped text**: If they differ significantly, the issue is in `_strip_markdown_for_tts()` or `extract_images()`

### Fixes Applied (2026-05-07)

1. `tts_tool.py`: `want_opus = platform in ("telegram", "feishu")` — ensures opus output for feishu
2. `base.py` `extract_images()`: completely discard alt-text instead of preserving it
3. Gateway-level: `want_opus` routing updated for feishu

### Key Files to Check

- `/gateway/run.py` → `_send_voice_reply()` (line ~8460) and `_should_send_voice_reply()` (line ~8413)
- `/gateway/platforms/base.py` → `extract_images()` and auto-TTS block (~line 2772)
- `/tools/tts_tool.py` → `_strip_markdown_for_tts()` (line ~1901) and `want_opus` (line ~1588)
- `/gateway/platforms/feishu.py` → `send_voice()` for audio delivery
