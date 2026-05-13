# Hermes Agent Learnings

## 2026-05-13 — CJK Script Mixing Bug

### What happened
When drafting a LinkedIn DM reply, I mixed Chinese characters (联系我, 薪酬台帳) into Japanese output, and produced bilingual Japanese+Chinese hybrid text.

### Root causes

1. **Wrong memory framing** — I had "Japanese-speaking" in my memory. Shanewas is Bengali, works in Japan, receives Japanese but reads English only.

2. **Han Unification confusion** — All CJK characters (Chinese hanzi, Japanese kanji, Korean hanja) share the **same Unicode block (U+4E00–U+9FFF)**. Unicode cannot distinguish them. This means I was seeing identical code points and guessing wrong about which language they belonged to.

3. **No detection gate** — I was jumping straight into drafting without running any language identification check on either the sender's message or my own output.

### What I learned

**Unicode ranges that ARE distinguishable:**
- Hiragana (U+3040–U+309F) → Only Japanese
- Katakana (U+30A0–U+30FF) → Only Japanese
- Hangul (U+AC00–U+D7AF) → Only Korean

**Unicode ranges that OVERLAP (CJK Unified Ideographs):**
- U+4E00–U+9FFF → Chinese + Japanese + Korean (all identical code points)
- Cannot distinguish kanji from hanzi from hanja by Unicode alone

**Practical detection strategy (3-layer):**
1. **Unicode ranges** — if hiragana/katakana/hangul present → language identified
2. **Signature words** — の/ます → Japanese; 的/联系/薪酬 → Chinese; no overlap possible
3. **langdetect** — statistical model for ambiguous CJK-only text

### What I built

**`scripts/cjk_detect.py`** — main detection tool
- `python3 scripts/cjk_detect.py "text" -v` — full analysis
- `python3 scripts/cjk_detect.py "text" --verify english` — purity check for English output
- Uses Unicode ranges + signature word scan + langdetect

**`skills/san-japanese-communication/`** (v2) — skill updated
- Shanewas ALWAYS writes in English
- Workflow: detect sender language → draft in English → verify output with cjk_detect.py
- Table of common CJK mixing errors

### Memory updates
- Updated memory: LinkedIn DM rule → Shanewas ALWAYS writes in English
- Added CJK Detect tool note to memory
- CJK skill v2 created

### Files created/modified
- `/usr/local/lib/hermes-agent/scripts/cjk_detect.py` — new
- `/root/.hermes/skills/san-japanese-communication/SKILL.md` — rewritten v2
- Memory entries — updated

### Tags
#cjk #unicode #japanese #chinese #korean #linkedin #bug #mixing
