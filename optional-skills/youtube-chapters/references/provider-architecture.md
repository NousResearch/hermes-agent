# Transcript Provider Architecture

Providers must return chronological segments containing `start`, `end`, and `text`.
Normalize provider output before chapter generation.

## Priority 1: Public Captions

Use the isolated `utils.provider` adapter backed by `youtube-transcript-api` when dependencies
and network access are allowed. Invoke it through `scripts/fetch_transcript.py`. The MVP does
not require a browser or YouTube API key.

Handle unavailable, disabled, malformed, or rate-limited captions as provider failures.
Do not silently convert untimestamped text into precise segments.

## Priority 2: Official YouTube Captions API

Keep this optional. Use only when the user owns the video, OAuth is configured, caption
download permission exists, and quota cost is acceptable. It is not a general solution for
arbitrary public video transcripts.

## Future: Audio Transcription

Audio transcription is not implemented in this MVP. A future integration may use
timestamp-preserving ASR after caption retrieval fails.

Requirements:

- Preserve segment-level timestamps.
- Do not generate precise chapters when the ASR result has no timestamps.

## Adapter Contract

The live provider adapter exposes behavior equivalent to:

```python
def get_transcript(video_id: str) -> list[dict]:
    """Return timestamped segments or raise a provider-specific retrieval error."""
```

Provider-specific failures are converted into stable structured errors. No fallback provider is
configured in this MVP.
