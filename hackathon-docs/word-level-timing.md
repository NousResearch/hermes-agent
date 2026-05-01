## Assessment: Word-Level Timing for Phonetic Captions

### What the code currently does

In `transcribe()` (video_caption.py), faster-whisper is called with `word_timestamps=False`. This means you get **segment-level** timing only — one `{start, end, text}` per logical phrase. Whisper's VAD grouping can bundle multiple short words into one segment.

### What's actually feasible

faster-whisper already supports `word_timestamps=True` — it's a single flag change. When enabled, each segment gains a `.words` list: `[{word, start, end, probability}]`. This is the most accurate timing you can get (it's what Whisper already computed internally).

### Three approaches, assessed

**A — Auto-split at transcription time (recommended)**
- Enable `word_timestamps=True`
- After transcription, run a pass that splits a segment wherever the gap between word N's end and word N+1's start exceeds a threshold (e.g. 0.4s)
- Result: `"không biết"` at 0–2s with a 0.5s pause in the middle automatically becomes two segments `"không"` (0–0.75s) and `"biết"` (1.25–2s)
- **Pro**: fully automatic, zero extra UI, works on every new job
- **Con**: might over-split at short pauses — mitigated by a tunable threshold

**B — Store word data + "Split at word boundary" in dashboard editor**
- Enable `word_timestamps=True`, store `words: [{word, start, end}]` in job JSON alongside each segment
- Add a "Split" button in the editor that shows the word breakdown with actual timestamps and lets user click where to cut
- **Pro**: precise and user-controlled; the timestamp challenge disappears because Whisper provides the times
- **Con**: more UI work; user must act manually on every problematic segment

**C — Fully manual user-added segments (not recommended)**
- Users type start/end timestamps and text from scratch
- Your instinct is correct: **this is not practical**. Without word timestamps shown to the user, they'd be guessing times. Not worth implementing.

### Recommendation

Do **A + B together** — they're cheap and complementary:

1. **`word_timestamps=True`** — one-line change in `transcribe()`
2. **Auto-split** on a configurable pause threshold (default ~0.4s, configurable via `caption.word_split_threshold` in `config.yaml`) — ~30 lines of new logic
3. **Store `words`** array per segment in the job JSON (free bonus data)
4. **"Split" button** in the dashboard that uses the stored word timestamps to let users manually fix any case the auto-split got wrong — shows actual word timings so no guessing

This means the common case (clear pause between taught Vietnamese words) is handled automatically, and the editor becomes the escape hatch for edge cases.
