# TTS Auto-Fire Emoji Stripping

## Problem
When auto-TTS fires on AI text replies, emoji characters and `:smile:`-style alt-text get synthesized as speech (e.g., "smiling face", "egg"), making the audio output weird and unprofessional.

## Solution
Strip emojis and alt-text from `speech_text` before calling TTS in `base.py` auto-TTS block (~line 2787).

### Regex Pattern (Python)
```python
# After markdown cleanup, before TTS call:
speech_text = re.sub(
    "[\u2600-\u27BF\uFE0F\u200D"
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "]|:[A-Za-z0-9_.+-]+:",
    '', speech_text
)
speech_text = re.sub(r'\s+', ' ', speech_text).strip()
```

### Unicode Ranges Covered
| Range | Description | Examples |
|-------|-------------|----------|
| `\u2600-\u27BF` | Symbol misc + dingbats | ⚡☕❤️♻️ |
| `\U0001F600-\U0001F64F` | Emoticons | 😂😊😎🤔 |
| `\U0001F300-\U0001F5FF` | Misc symbols & pictographs | 🌍🎉🎵📱 |
| `\U0001F680-\U0001F6FF` | Transport/maps | 🚗✈️🏠🚀 |
| `\U0001F900-\U0001F9FF` | Supplemental | 🤖🦾🧩 |
| `\U0001FA00-\U0001FA6F` | Chess/symbols | ♟️ |
| `\uFE0F\u200D` | Variation selector + ZWJ | Combining emoji sequences |

### Alt-text Pattern
`:word:` — matches Slack/Discord/Feishu style emoji shortcodes like `:smile:`, `:wave:`, `:thumbsup:`

### Key Pitfalls
- **`\U` in raw strings doesn't work for regex** — `\U0001F600` in `r'...'` is literal `\U0001F600` (8 chars), NOT a unicode codepoint. Use non-raw strings with actual escape sequences or chr() based construction.
- **`\-` in character class causes SyntaxWarning** — In `[A-Za-z0-9_+.-]`, the `-` must be at start/end or escaped as `\-`. But `\-` triggers Python 3.12+ warnings. Use `.` before `-` (literal dot then hyphen) or put `-` at end: `[A-Za-z0-9_.+-]`.
- **Always clean up whitespace after emoji removal** — Removing emojis leaves double/triple spaces. Run `re.sub(r'\s+', ' ', text).strip()` afterward.

### Verification
```python
text = '你好世界 😂 算了算了 🥚 :smile:'
result = re.sub(pattern, '', text).strip()
# → '你好世界 算了算了'

text2 = 'Hello 🌍 world 😊 :wave: 你好世界 🎉'
result2 = re.sub(pattern, '', text2).strip()
# → 'Hello world 你好世界'
```
