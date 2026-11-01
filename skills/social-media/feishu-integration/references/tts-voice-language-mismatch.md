# TTS Voice Language Mismatch: Partial Audio Output

## Symptom
TTS voice output only contains pronunciation for some parts of the text — typically English words and emoji alt-text, but **no Chinese (or other non-English) pronunciation**. The audio file is shorter than expected.

## Root Cause
The Edge TTS `voice` config value is a **monolingual** voice that doesn't support the language in the input text.

```yaml
# WRONG — en-US-AriaNeural is English-only
tts:
  provider: edge
  edge:
    voice: en-US-AriaNeural   # ← pure English, returns NoAudioReceived for Chinese
```

When Edge TTS receives Chinese text with `en-US-AriaNeural`, it silently produces **zero audio** (not an error in the output file — the file is empty/missing). Tested:
```python
edge_tts.Communicate('你好世界', voice='en-US-AriaNeural')
→ NoAudioReceived: No audio was received
```

## Fix
Change to a bilingual or Chinese-appropriate voice:

```yaml
tts:
  provider: edge
  edge:
    voice: zh-CN-XiaoxiaoNeural   # ← supports both English and Chinese
```

### Available zh-CN Neural voices (all support mixed EN/CN):
| Voice | Gender | Style |
|-------|--------|-------|
| zh-CN-XiaoxiaoNeural | Female | News, Novel — Warm |
| zh-CN-XiaoyiNeural | Female | Cartoon, Novel — Lively |
| zh-CN-YunjianNeural | Male | Sports, Novel — Passionate |
| zh-CN-YunxiNeural | Male | Novel — Lively, Sunshine |
| zh-CN-YunxiaNeural | Male | Cartoon, Novel — Cute |
| zh-CN-YunyangNeural | Male | News — Professional, Reliable |

## Debugging Steps
1. Check `config.yaml` → `tts.edge.voice` value
2. Verify it supports the languages in your typical AI responses
3. Test with: `edge-tts --voice <VOICE> --write-media /tmp/test.mp3 -` (paste mixed text)
4. If `NoAudioReceived` error appears, switch to a bilingual voice

## Related
- `tts-debugging-guide.md` — general TTS debugging
- `auto-tts-all-replies.md` — enabling auto-TTS for all replies
