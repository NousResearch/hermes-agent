# Phonetic Captions — Pipeline Roles (Whisper vs Kimi)

## Tool Responsibilities

| Tool | Role |
|---|---|
| **faster-whisper** | Audio → raw text + timestamps only. Local, no internet. |
| **Kimi K2.5** (NVIDIA NIM) | Fix garbled Vietnamese diacritics, classify each segment as `"en"` or `"vi"`, generate English phonetic guide for VI segments |
| **ASS builder** (`build_ass`) | Format into 2-line (VI: main + phonetic) or 1-line (EN) styled subtitles |
| **FFmpeg** | Burn ASS subtitles into video pixels |

## Key Insight: Whisper + Kimi are a two-pass system

Whisper runs with `language=None` (auto-detect). It will pick up both English and Vietnamese in the same audio, but:
- Vietnamese diacritics often garbled into English-looking gibberish ("Fo" for "Phở", "thhem moy" for "thêm muối")
- No language classification — all segments come back with `lang: ""` and `phonetic: ""`

Kimi is the post-processing pass that:
1. Recognises garbled Vietnamese (prompt explicitly names patterns: "Hong biet", "zoe", "un" etc.)
2. Rewrites with correct diacritics
3. Labels each segment `"en"` or `"vi"`
4. Writes a simple English-letter phonetic guide for VI segments (e.g. `[tem mwoy]`)

## Segment lifecycle

```
transcribe()         → {id, start, end, text (raw Whisper), lang: "", phonetic: "", words: [...]}
generate_phonetics() → {id, start, end, text (corrected), lang: "en"|"vi", phonetic: "[...]"|""}
build_ass()          → writes .ass file; VI segments get 2 ASS Dialogue lines (MAIN + PHONETIC)
burn()               → FFmpeg bakes .ass into video
```

## ASS layout rule

- `lang == "vi"` with phonetic → **two lines**: Vietnamese (MAIN style, 48pt bold) + phonetic guide (PHONETIC style, 36pt italic, 60% alpha)
- `lang == "en"` or no phonetic → **one line**: English text only (MAIN style)

## NVIDIA NIM quirk

Kimi output sometimes lands in `reasoning_content` not `content`. The code checks both:
```python
raw = getattr(choice.message, "content", None) or ""
if not raw.strip():
    raw = getattr(choice.message, "reasoning_content", None) or ""
```

## Graceful degradation

If `NVIDIA_API_KEY` is not set, `generate_phonetics()` falls back: all segments treated as English, no phonetics, no VI correction. Video still works, just without the bilingual layer.
