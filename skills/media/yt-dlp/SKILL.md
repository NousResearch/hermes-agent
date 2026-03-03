---
name: yt-dlp
description: Use yt-dlp to fetch YouTube metadata, transcripts, and media, plus summarize videos by default when given only a YouTube URL. Use for channel stats, playlist inspection, subtitle extraction, audio or video downloads, and consistent temp-folder workflows with yt-dlp.
version: 1.0.0
author: community
license: MIT
metadata:
  hermes:
    tags: [YouTube, yt-dlp, Video, Transcripts, Subtitles, Download, Metadata, Summarize]
---

# yt-dlp

Use a dedicated temp folder for **all yt-dlp work** so you never spray files around the workspace. The default temp root is `TMP/yt-dlp` **inside the workspace root** (for Hermes this means a relative path `TMP/yt-dlp` from the current working directory). Always create it if missing, and keep intermediate JSON, subtitles, and media in that folder unless the user explicitly asks for a different destination. Prefer writing outputs into temp and then moving or copying to a final location if needed. If you create helper scripts, put them under `workspace/scripts/yt-dlp` and delete one-off scripts after use. Before fetching channel stats or user-specific data, check TOOLS.md for usernames and tool variables.

When constructing commands, **never run yt-dlp in random directories**; always either:

- `cd` into `TMP/yt-dlp` first, **or**
- pass `--paths TMP/yt-dlp` / `-P TMP/yt-dlp` so all output files land there.

On Windows / PowerShell (your setup), use **this pattern only**:

```powershell
New-Item -ItemType Directory -Force -Path "TMP/yt-dlp" | Out-Null
yt-dlp --paths "TMP/yt-dlp" ...
```

## Plain YouTube URL (default: summarize)

When given a **plain YouTube URL with no extra instruction**, treat it as:  
“Give me a **thorough written explanation** of this video as if I will *not* watch it.”

Default flow (run **at most once per URL**; if any yt-dlp step fails, show the error and explain it instead of looping retries):

1. **Prep temp folder**
   - Ensure `TMP/yt-dlp` exists as above.
2. **Metadata JSON**
   - Run yt-dlp with `--dump-single-json` into temp, e.g.:
     - PowerShell:
       ```powershell
       yt-dlp --dump-single-json --paths \"TMP/yt-dlp\" -o \"%(title)s [%(id)s].%(ext)s\" <URL> > \"TMP/yt-dlp\\meta.json\"
       ```
     - POSIX:
       ```bash
       yt-dlp --dump-single-json --paths TMP/yt-dlp -o '%(title)s [%(id)s].%(ext)s' <URL> > TMP/yt-dlp/meta.json
       ```
3. **Subtitles first (preferred)**
   - Try captions with **no media download**:
     - `--write-auto-subs` or `--write-subs`
     - `--sub-langs en` (or the user’s language)
     - `--sub-format vtt`
     - `--skip-download`
     - Always keep output in `TMP/yt-dlp` via `--paths` / `-P`.
   - Then use `read_file` to open the `.vtt` from `TMP/yt-dlp` and summarize **from the transcript**.
4. **If no captions exist**
   - Extract audio into `TMP/yt-dlp` with `--extract-audio` and an audio format (e.g. `mp3`), then summarize from the audio (or via Whisper if available).
5. **Summary style**
   - Write a **detailed, self-contained explanation** as if the user will **never watch the video**:
     - Who is speaking / main actors
     - Main claims, arguments, and conclusions
     - Important evidence, examples, or stories
     - Timeline / structure (intro, key sections, ending)
     - Any notable quotes or numbers (summarized, not just copied)
   - Do **not** just say “here are subtitles”; actually digest and explain.
   - Keep the summary in the response text (not as a file) unless the user explicitly asks for a file.

## Channel or playlist stats

Use `--dump-single-json` with `--flat-playlist` for speed, then parse fields like `channel`, `channel_id`, `uploader`, `uploader_id`, `channel_follower_count`, and entry counts. If view counts are missing from yt-dlp, say so and offer a browser scrape as a follow up.

## Media downloads

Use `-F` for formats, then `-f` to select, or `--extract-audio` with `--audio-format` and `--audio-quality` for audio. Use `-o` to control output names, and `--paths` to set final destinations. Use `--download-archive` to avoid duplicates for batch pulls.

## Subtitles

Use `--list-subs` first when accuracy matters, then `--write-subs` or `--write-auto-subs` with `--sub-langs` and `--sub-format`. Use `--convert-subs` if you need srt.

## Metadata and assets

Use `--write-info-json`, `--write-thumbnail`, or `--write-all-thumbnails`, and `--embed-metadata` or `--embed-thumbnail` when producing final media.

## Auth or member-only access

Use `--cookies-from-browser` or `--cookies`. Keep cookie paths private and never print them.

## JavaScript runtime warnings

If yt-dlp warns about missing JavaScript runtimes, note it and continue unless extraction fails. If it fails, suggest adding a JS runtime or switching to browser extraction.

## Whisper and Windows

When transcribing with Whisper, prefer CUDA if available by using `--device cuda`, and set UTF-8 output (for example set `PYTHONUTF8=1` or use an output format like srt/json) to avoid Windows UnicodeEncodeError when writing files.

## General

Prefer short, direct commands. Avoid listing huge inventories of formats unless explicitly asked.

If a yt-dlp command fails or you’re unsure about flags, **it is always allowed to run**:

```bash
yt-dlp --help
```

and then adjust the command based on the documented `Usage: yt-dlp [OPTIONS] URL [URL...]` and options. Never keep retrying the same failing command blindly; check `--help` or the error message once, fix the command, and then try again **at most one more time**.
