---
name: vidscript
description: Use when the user provides or uploads a video and wants it downloaded, transcribed, visually inspected for on-screen text, and translated into Chinese. Downloads/saves the video, extracts both spoken narration and visible prompts/subtitles/labels, translates them to natural Chinese, and returns the complete script package.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [video, transcription, translation, download, media]
    related_skills: [youtube-content]
---

# Video Script Extraction and Chinese Translation

## Overview

Use this skill whenever the user sends a video URL or uploads a video and asks for the script, transcript, captions, subtitles, or Chinese translation.

Goal: produce a complete, useful result in one pass:

1. Download or locate the video file.
2. Extract/transcribe the spoken narration.
3. Inspect the video frames for visible on-screen text: subtitles, prompts, labels, titles, CTAs, product text, UI text, and other meaningful information.
4. Save separate spoken transcript and visual-text/OCR files beside the video.
5. Merge the spoken narration and visible text into a complete script package.
6. Translate the complete package into natural Chinese.
7. Reply with the file paths plus the Chinese translation and, when short enough, the original transcript and visual text.

Prefer acting immediately over asking questions. Only ask if the source is inaccessible or the desired language/format is genuinely ambiguous.

## When to Use

Use when the user:

- Sends a URL to Instagram, YouTube, TikTok, X/Twitter, Vimeo, or another video page and asks to download/extract the script.
- Uploads a local video file and asks for the script/transcript/subtitles.
- Asks in Chinese: “提取脚本”, “转文字”, “生成字幕”, “翻译成中文”, “下载视频并提取文案”.
- Wants both the original spoken text and a Chinese version.
- Sends short-form content where important information is shown visually on screen, such as prompt cards, lists, captions, product labels, overlay subtitles, or call-to-action text.

Do not use for:

- Pure text translation with no video/audio.
- Summarizing a video without transcription, unless transcription is needed first.
- Copyright-sensitive redistribution requests beyond saving the user-provided/link-accessible media for personal processing.

## Workflow

### 1. Prepare output folders

Use a stable local folder so files are easy to find:

```bash
mkdir -p "$HOME/Downloads/video-scripts" "$HOME/Downloads/video-scripts/transcripts"
```

For platform-specific prior work, it is also OK to use an existing folder such as:

```bash
$HOME/Downloads/instagram
```

### 2. Download or identify the video

If the user provides a URL, use `yt-dlp` first:

```bash
yt-dlp -o "$HOME/Downloads/video-scripts/%(title).80s-%(id)s.%(ext)s" "VIDEO_URL"
```

If the user uploaded a file or provides a local path, do not re-download it. Use that file directly.

After downloading, verify that the output exists and note its path and size:

```bash
python3 - <<'PY'
from pathlib import Path
p = Path('VIDEO_PATH')
print(p)
print('exists=', p.exists())
print('size_mb=', round(p.stat().st_size/1024/1024, 2) if p.exists() else None)
PY
```

### 3. Inspect media streams

Check duration and whether audio exists:

```bash
ffprobe -v error -show_entries format=duration:stream=index,codec_type,codec_name -of json "VIDEO_PATH"
```

If there is no audio stream, explain that no spoken script can be extracted and offer OCR/frame analysis only if useful.

### 4. Transcribe audio

Use Whisper when available. For short videos, `small` is a good default. For faster/lower-resource processing, use `base`. If the language is unknown, omit `--language`; if it is clearly English, use `--language en`.

English example:

```bash
whisper "VIDEO_PATH" --model small --language en --task transcribe --output_format all --output_dir "$HOME/Downloads/video-scripts/transcripts"
```

Unknown-language example:

```bash
whisper "VIDEO_PATH" --model small --task transcribe --output_format all --output_dir "$HOME/Downloads/video-scripts/transcripts"
```

Prefer `--output_format all` when practical so `.txt`, `.srt`, `.vtt`, `.tsv`, and `.json` are available.

If Whisper CLI is missing but Python package is installed, use Python Whisper. If both are missing, install or ask permission only if the environment requires elevated/network setup.

### 5. Extract visible on-screen text

Do not treat the spoken transcript as the whole script. Many short videos contain the real value in visual text: prompt cards, list items, subtitles, labels, CTAs, product names, UI text, or before/after annotations. Always inspect the video visually after transcription.

Sample frames across the whole video. A practical default is 1 frame/second for short clips:

```bash
mkdir -p "$HOME/Downloads/video-scripts/frames/<video-id>"
ffmpeg -y -i "VIDEO_PATH" -vf fps=1 "$HOME/Downloads/video-scripts/frames/<video-id>/frame_%03d.jpg"
```

Create a contact sheet so vision/OCR can review the full clip efficiently:

```bash
python3 - <<'PY'
from pathlib import Path
from PIL import Image, ImageDraw
frames = sorted(Path('FRAME_DIR').glob('*.jpg'))
imgs = []
for i, p in enumerate(frames, 1):
    im = Image.open(p).convert('RGB')
    im.thumbnail((270, 480))
    canvas = Image.new('RGB', (300, 540), 'white')
    canvas.paste(im, ((300-im.width)//2, 30))
    d = ImageDraw.Draw(canvas)
    d.text((10, 5), f'frame {i:03d} / ~{i-1}s', fill=(255, 0, 0))
    imgs.append(canvas)
cols = 4
rows = (len(imgs) + cols - 1) // cols
sheet = Image.new('RGB', (cols*300, rows*540), 'white')
for idx, im in enumerate(imgs):
    sheet.paste(im, ((idx % cols) * 300, (idx // cols) * 540))
sheet.save(Path('FRAME_DIR') / 'contact_sheet.jpg', quality=90)
PY
```

Use `vision_analyze` on the contact sheet. Ask it to extract all visible text exactly, grouped by frame/time and category, and to mark uncertain words.

For higher reliability, use a two-pass visual extraction pipeline:

1. **Overview pass:** inspect the contact sheet to identify which frames contain important text and which categories/cards appear.
2. **Detail pass:** inspect important frames individually at full resolution, especially frames with long prompt text, dense subtitles, small fonts, or anything marked uncertain.

When available, combine traditional OCR with vision-model reading:

- Traditional OCR is useful for repeatable raw text extraction.
- Vision analysis is useful for structure: category titles, which prompt belongs to which card, and whether the OCR text is semantically plausible.
- Prefer the merged result when both agree.
- If they disagree, inspect the original frame again and mark uncertainty rather than guessing.

Optional local OCR checks, if tools are installed:

```bash
# Tesseract, if installed
tesseract "FRAME_PATH" stdout --psm 6

# OCRmyPDF/PaddleOCR/EasyOCR may also be used if present, but do not require installation unless needed.
```

For short videos with visible prompt cards or small text, do not rely only on the contact sheet. Save or inspect the individual high-resolution frames that contain the text.

Save the visual extraction as:

```text
$HOME/Downloads/video-scripts/transcripts/<video-name>.visual.txt
```

### 6. Read, clean, and merge transcript

Read the generated spoken `.txt` and visual `.visual.txt` with `read_file`. Clean obvious line breaks while preserving meaning.

Create a complete script package with at least these sections:

- Spoken narration / 口播脚本
- Visible on-screen text / 画面文字
- Extracted prompts or key visual lists / 提示词或重点列表, if present

For short clips, include the spoken transcript and visual text in the final reply. For long clips, provide saved file paths and a concise excerpt or summary.

### 7. Translate into Chinese

Translate the full spoken transcript and all meaningful visual text into natural, fluent Simplified Chinese. Keep technical brand/tool names in English unless there is a common Chinese form.

Translation style:

- Natural Chinese, not word-by-word literal translation.
- Preserve steps and sequence.
- Keep important numbers, URLs, product names, commands, and calls to action.
- If the transcript has uncertain words, mark them lightly, e.g. “疑似：...”, instead of pretending certainty.

### 8. Save Chinese translation

Save the Chinese translation beside the transcript, for example:

```text
$HOME/Downloads/video-scripts/transcripts/<video-name>.zh.txt
```

Use `write_file` rather than shell heredocs when writing the translation file.

### 9. Final response format

Respond in Chinese unless the user requested otherwise.

Include:

1. Video file path.
2. Spoken transcript file path.
3. Visual text/OCR file path, if visual text exists.
4. Chinese translation file path.
5. The Chinese translation inline.
6. The spoken transcript and extracted visual text inline if short enough.

Example final structure:

```text
已完成。

视频文件：
/path/to/video.mp4

英文脚本：
/path/to/transcript.txt

中文翻译：
/path/to/transcript.zh.txt

中文翻译如下：
...
```

## Commands Quick Reference

Check tools:

```bash
yt-dlp --version
whisper --help
ffprobe -version
```

Download:

```bash
yt-dlp -o "$HOME/Downloads/video-scripts/%(title).80s-%(id)s.%(ext)s" "URL"
```

Transcribe:

```bash
whisper "VIDEO_PATH" --model small --task transcribe --output_format all --output_dir "$HOME/Downloads/video-scripts/transcripts"
```

Inspect:

```bash
ffprobe -v error -show_entries format=duration:stream=index,codec_type,codec_name -of json "VIDEO_PATH"
```

## Common Pitfalls

1. **Assuming there is audio.** Always inspect or handle Whisper failures. Some videos are silent or music-only.

2. **Returning only the spoken transcript.** The user expects the complete video script. Always inspect frames and extract visible text/prompts/subtitles, especially in short-form videos.

3. **Forgetting to save the Chinese translation.** Save it as a `.zh.txt` file for reuse.

4. **Forgetting to save visual text.** Save meaningful on-screen text as `.visual.txt`; this is where long prompt cards, subtitles, labels, and CTAs belong.

5. **Over-literal translation.** Make the Chinese read like a real script while preserving all details.

6. **OCR uncertainty.** If frame text is small or blurry, mark uncertain words instead of inventing them. Use a two-pass approach: first contact sheet for overview, then individual high-resolution frames for detail. When possible, cross-check vision-model extraction with traditional OCR such as Tesseract, PaddleOCR, EasyOCR, or OCRmyPDF if already available.

7. **Platform login/cookies.** Some Instagram/TikTok/X URLs may require cookies or login. If `yt-dlp` fails due to auth, report the error and ask the user to provide an accessible link, cookies, or the video file.

8. **Huge transcripts.** If the transcript is too long for the final message, provide file paths and a concise summary, then offer to split the full translation into parts.

## Verification Checklist

Before finalizing:

- [ ] Video file exists locally or the uploaded file path is valid.
- [ ] Audio stream exists, or lack of audio is clearly reported.
- [ ] Transcript file exists and contains plausible spoken text.
- [ ] Sampled frames/contact sheet were inspected for visible on-screen text.
- [ ] Important text-heavy frames were inspected individually at higher resolution when contact-sheet text is small, dense, or uncertain.
- [ ] Traditional OCR was used as a cross-check when available and helpful.
- [ ] Visual text is saved to a `.visual.txt` file when meaningful on-screen text exists.
- [ ] Chinese translation includes both spoken narration and visible text/prompts.
- [ ] Chinese translation is saved to a `.zh.txt` file.
- [ ] Final reply includes the key file paths, Chinese translation, and any extracted visual prompts/lists.
