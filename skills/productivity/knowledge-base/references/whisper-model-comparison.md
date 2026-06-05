# Whisper Model Comparison — French Multi-Speaker

Benchmarks from a real 30s Instagram Reel with 4 speakers, AAC 44kHz stereo,
converted to WAV 16kHz mono. Pyannote diarization run first (28 segments, 4 speakers),
then each whisper model on the same audio.

| Model | Time (30s) | Segments | Key quality diff |
|-------|-----------|----------|------------------|
| small int8 | 0.8s | 12 | "Attends des pouces", "La bobe l'éponge", "répondir" |
| medium int8 | 0.4s | 12 | "Attendez, pause", "Bob l'éponge", "rebondir" |
| large-v3 int8 | 0.6s | **15** | "devant moi", "le quart d'heure", "Je suis mort." ✓ |

**Winner: large-v3** — catches 3 extra segments (nuanced phrases "Je suis mort.",
corrects idioms "le quart d'heure sur la veste"), fixes proper nouns ("Bob"),
restores natural French syntax ("devant moi" pas "dans moi").

**Trade-off for pipeline:**
- small: ~500 MB RAM, ~0.7x realtime → 6h for 9h video
- medium: ~1.5 GB RAM, ~1.5x realtime → 13h for 9h video
- large-v3: ~3 GB RAM, ~2-3x realtime → 18-27h for 9h video

**Recommendation:** Use `large-v3` for ALL video content (YouTube + Instagram Reels).\nDecision taken 2026-05-24: quality over speed, no exceptions.\n`medium` and `small` not used in the pipeline.

Times shown for 30s clip are misleading (cache effects on tiny files).
Use the ~0.7x / ~1.5x / ~2-3x ratios for pipeline estimation.
