# Chinese Localization for Hermes Agent CLI

## Background

Hermes Agent has a growing user base in China and other Chinese-speaking regions. Many users (including myself) would benefit from having the CLI interface localized to Chinese, especially for:

- Compression feedback messages
- Session token usage display
- Voice mode status
- Approval prompts and choices
- Timeout warnings
- MCP server reload messages

## Proposed Solution

I've implemented a lightweight i18n solution in my fork: https://github.com/gzsiang/hermes-agent

### Key Features

1. **Configurable language switching** via `approvals.language` in `~/.hermes/config.yaml`:
   ```yaml
   approvals:
     language: zh  # or 'en' for English
   ```

2. **Zero breaking changes** - defaults to English when no language is set

3. **Minimal code changes**:
   - Added `tools/i18n.py` (~250 lines) with `format_zh()` function
   - Wrapped user-facing strings in `cli.py` with `format_zh()` calls
   - No external dependencies (pure Python)

4. **Easy to extend** - translation dictionary can be expanded for more strings or additional languages

### Example Usage

```python
# Before
print("🗜️ Compressing {count} messages...")

# After
print(f"🗜️ {format_zh('Compressing {count} messages', count=count)}...")
```

Output with `language: zh`:
```
🗜️ 正在压缩 10 条消息...
```

Output with `language: en` (default):
```
🗜️ Compressing 10 messages...
```

## Questions for the Maintainers

1. **Is there an official i18n plan** for Hermes Agent? If so, I'd love to align my implementation with it.

2. **Would you be open to a PR** for this lightweight i18n approach? I'm happy to:
   - Improve code style to match the project
   - Add unit tests
   - Expand translations based on feedback
   - Refactor if there's a preferred pattern

3. **If not ready for i18n**, I'll continue maintaining this in my fork for Chinese users. No pressure!

## Notes

- Current implementation only covers high-frequency CLI strings (~20 locations)
- Translation quality could be improved with native speaker review
- The approach is designed to be non-intrusive and easy to remove if not adopted

Thanks for building such an amazing tool! 🙏
