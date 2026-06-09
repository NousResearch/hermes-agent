# YouTube Chapters Skill

Generates paste-ready YouTube chapter markers from a YouTube video URL, video ID, or
timestamped transcript.

## Expected Input

Typical prompts:

```text
Generate chapters for this YouTube video: <url>
Create YouTube timestamps from this video: <url>
Make a paste-ready chapter list for: <url>
```

The current MVP retrieves public captions through the optional `youtube-transcript-api`
dependency. It does not require YouTube Data API credentials.

## Expected Output

```text
00:00 Introduction
01:42 Project setup
04:18 Hermes Agent architecture
08:35 Transcript processing
12:10 Final result
```

## Transcript Strategy

1. Try public YouTube captions/transcripts through `youtube-transcript-api`.
2. For a user-owned video, optionally use the official YouTube Captions API when OAuth,
   caption permission, and acceptable quota are available.
3. If captions are unavailable, return a structured error instead of inventing
   timestamps.

The official YouTube Data API is not the sole MVP transcript method because downloading
captions for arbitrary public videos is permission/OAuth dependent and may incur quota cost.
Audio download and Whisper/ASR fallback are planned future work and are not included.

## Install the Optional Live Provider

From the skill directory:

```text
python -m pip install -r requirements.txt
```

## Fetch a Transcript Manually

From the skill directory:

```text
python scripts/fetch_transcript.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

Specify preferred languages in descending preference order. The default is `tr,en`:

```text
python scripts/fetch_transcript.py "VIDEO_ID" --languages tr,en
```

Inspect the available transcript tracks:

```text
python scripts/fetch_transcript.py "VIDEO_ID" --list-transcripts
```

The CLI accepts a cookie-file option for forward compatibility:

```text
python scripts/fetch_transcript.py "VIDEO_ID" --cookies path/to/cookies.txt
```

The currently supported `youtube-transcript-api` provider does not support cookie-based
retrieval and returns `CookiesUnsupported`. Cookies are private credentials. Never commit,
share, or include cookie files in a pull request.

Successful output is normalized JSON containing `start`, `end`, and cleaned `text`.
Missing or disabled captions produce a structured JSON error and a non-zero exit code:

```json
{
  "error": "TranscriptUnavailable",
  "message": "No captions or transcript could be retrieved for this video."
}
```

Provider failures include a short `detail` containing only the provider exception class name.
No traceback or private request data is returned.

Invalid YouTube URLs or video IDs are rejected before transcript retrieval:

```json
{
  "error": "InvalidYouTubeURL",
  "message": "The provided input is not a valid YouTube URL or video ID."
}
```

## Hermes Workflow

1. Parse the YouTube URL or video ID.
2. Run `scripts/fetch_transcript.py` and read its JSON output.
3. Group the returned timestamped segments and generate transcript-supported chapters.
4. Validate the chapters and return paste-ready markers.
5. If the script returns an error, explain that captions are unavailable and ASR fallback is
   not implemented yet.

`fetch_transcript.py` only fetches and normalizes transcript data. It does not call an LLM.
Hermes uses the timestamped transcript with its configured LLM to generate chapter markers.

## YouTube shows a transcript, but the CLI returns TranscriptUnavailable

The YouTube UI and an unauthenticated transcript provider do not always have the same access.
The transcript may not exist in the requested language, may require browser/session context,
or may be blocked for automated requests from the current IP or environment. Region, age, and
account restrictions can also prevent access. In some cases, the provider cannot access the
same transcript track displayed by the YouTube UI.

Recommended troubleshooting:

1. Try `--list-transcripts`.
2. Try `--languages tr,en`.
3. Try another captioned public video.
4. If a future provider version supports cookies, try `--cookies path/to/cookies.txt`. Treat
   cookies as private credentials and never commit them.
5. If captions still cannot be retrieved, a future ASR fallback using Whisper and `yt-dlp`
   may be needed.

## Validation

The deterministic utilities verify timestamp syntax, a `00:00` first chapter, strictly
increasing timestamps, non-empty titles, duplicate rejection, minimum chapter count when
appropriate, and known-duration bounds.

Run focused offline tests from the repository root:

```text
python -m unittest discover -s optional-skills/youtube-chapters/tests -v
```

## Limitations

- Does not guarantee transcripts for every YouTube video.
- Does not bypass YouTube restrictions or download private videos.
- Does not include Whisper, `yt-dlp`, ffmpeg, audio download, or ASR fallback.
- Does not invent precise timestamps without a timestamp source.
- Live retrieval depends on YouTube availability and the optional `youtube-transcript-api`
  dependency.
