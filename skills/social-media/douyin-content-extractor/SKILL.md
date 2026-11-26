---
name: douyin-content-extractor
description: "Use when extracting and analyzing Douyin/TikTok China video or image-post content from shared links. Covers metadata, video/audio download, local STT transcription, image extraction/OCR, summary, and reusable Markdown output. Triggers: watch this Douyin, extract Douyin content, transcribe Douyin video, summarize Douyin, download Douyin."
version: 1.0.0
author: Community
license: MIT
metadata:
  hermes:
    tags: [douyin, tiktok, video-transcription, social-media, media-extraction, stt]
    related_skills: [social-media-content-extractor]
---

# Douyin Content Extractor

Use this skill to turn a Douyin share into usable text and media artifacts: metadata, video/audio, transcript, image descriptions/OCR, and a compact Markdown report.

The reliable pattern is **webpage extraction + media download + local speech recognition**, not a single official API. Douyin pages change often, so keep the workflow layered: try the simple parser first, then browser/API fallback, then media/STT.

Detailed fallback recipe: see `references/browser-aweme-detail-fallback.md` for the proven browser `performance` → `/aweme/v1/web/aweme/detail/` → `play_addr.url_list` extraction path.

## Safety and scope

- Work read-only on public/shared content.
- Do not log into Douyin, post, comment, like, follow, or modify any account without explicit user confirmation.
- Store temporary artifacts under `/tmp/douyin_<aweme_id>/` or another `/tmp/` path; do not clutter the project workspace.
- Tell the user when transcript text is **machine-transcribed and not fully human-proofread**.

## Required tools

Typical successful path uses:

- `browser` — load Douyin page and inspect in-page API/resource URLs.
- `terminal` — redirects, curl download, ffmpeg, STT CLI.
- `vision` — analyze downloaded image posts/screenshots when needed.
- `file` — read/write final Markdown artifacts.

Useful local commands:

```bash
ffmpeg
# mlx_whisper: Apple Silicon local STT. Install once via:
#   pip install mlx-whisper     # adds `mlx_whisper` to your PATH
#   # or: uv tool install mlx-whisper
mlx_whisper                                # local Apple Silicon STT
uvx --from yt-dlp yt-dlp                   # optional fallback; Douyin often requires fresh cookies
mcporter                                    # optional, only works if a douyin MCP server is configured
```

If `mlx_whisper` is not on `PATH`, locate it with `which mlx_whisper` or `python3 -m mlx_whisper --help` and adjust the script or invocation accordingly.

## Workflow

### 1. Normalize the shared link

Extract the first `https://v.douyin.com/.../` or `https://www.douyin.com/...` URL from user text.

For short links, follow redirects:

```bash
python3 - <<'PY'
import requests
url='https://v.douyin.com/SHORT/'
r=requests.get(url, allow_redirects=False, timeout=20, headers={'User-Agent':'Mozilla/5.0'})
print(r.status_code, r.headers.get('location'))
PY
```

Common redirect chain:

```text
v.douyin.com/... -> www.iesdouyin.com/share/video/<aweme_id>/... -> www.douyin.com/video/<aweme_id>
```

Capture the `aweme_id`, e.g. `7646495323627588864`.

### 2. Try simple parser path

If available, try agent-reach/mcporter first:

```bash
mcporter call 'douyin.parse_douyin_video_info(share_link: "https://v.douyin.com/xxx/")'
mcporter call 'douyin.get_douyin_download_link(share_link: "https://v.douyin.com/xxx/")'
```

If it returns `Unknown MCP server 'douyin'` or similar, do not stop. Use the browser fallback.

`yt-dlp` can also be tried, but Douyin often returns `Fresh cookies are needed`:

```bash
uvx --from yt-dlp yt-dlp --dump-json 'https://www.douyin.com/video/<aweme_id>'
```

### 3. Browser/API fallback: extract `aweme_detail`

Open the canonical page with the browser:

```text
https://www.douyin.com/video/<aweme_id>
```

In browser console, inspect page text and API resources:

```javascript
document.body.innerText.slice(0, 5000)
```

Find `aweme/detail` calls:

```javascript
performance.getEntriesByType('resource')
  .map(e => e.name)
  .filter(n => n.includes('/aweme/v1/web/aweme/detail/'))
```

Fetch the same API from the page context so cookies/signature parameters are reused:

```javascript
(async()=>{
  const u = performance.getEntriesByType('resource')
    .map(e=>e.name)
    .find(n=>n.includes('/aweme/v1/web/aweme/detail/'));
  const j = await (await fetch(u,{credentials:'include'})).json();
  const a = j.aweme_detail || j.aweme_list?.[0] || j;
  return {
    aweme_id: a.aweme_id,
    desc: a.desc,
    create_time: a.create_time,
    author: a.author && {nickname:a.author.nickname, uid:a.author.uid, sec_uid:a.author.sec_uid, signature:a.author.signature},
    stats: a.statistics,
    duration_ms: a.video?.duration,
    video_urls: [
      ...(a.video?.play_addr?.url_list || []),
      ...((a.video?.bit_rate || []).flatMap(b => b.play_addr?.url_list || []))
    ],
    image_urls: (a.images || []).flatMap(img => img.url_list || img.download_url_list || []),
    raw_keys: Object.keys(a).slice(0,80)
  };
})()
```

Save useful metadata into the final report.

### 4. Download media

Create a per-video temp directory:

```bash
AWEME_ID='<aweme_id>'
mkdir -p "/tmp/douyin_${AWEME_ID}"
```

For videos, choose a direct `video_urls` item or `/aweme/v1/play/` URL and download with referer/user-agent:

```bash
curl -L --fail --retry 2 --connect-timeout 20 \
  -A 'Mozilla/5.0' \
  -e 'https://www.douyin.com/' \
  '<play_url>' \
  -o "/tmp/douyin_${AWEME_ID}/video.mp4" \
  -w '\nhttp=%{http_code} size=%{size_download} type=%{content_type}\n'

file "/tmp/douyin_${AWEME_ID}/video.mp4"
```

For image posts, download each image URL:

```bash
curl -L --fail -A 'Mozilla/5.0' -e 'https://www.douyin.com/' '<image_url>' \
  -o "/tmp/douyin_${AWEME_ID}/image_01.jpg"
```

Important for Douyin image URLs: **keep the full signed query string** (`x-expires`, `x-signature`, `lk3s`, etc.). Do not shorten the URL or remove query parameters — stripped image URLs often download as tiny 403 HTML files. If image URLs are not visible after reload or carousel movement, re-open the canonical note page and extract current signed URLs from the DOM:

```javascript
[...document.querySelectorAll('img')]
  .map((i, idx) => ({ idx, src: i.currentSrc || i.src, alt: i.alt, w: i.naturalWidth, h: i.naturalHeight }))
  .filter(x => x.src && x.src.includes('tplv-dy-aweme-images'))
```

Then use `vision_analyze` for descriptions/OCR when the image content matters.

### 5. Extract audio and transcribe video

Verify video duration:

```bash
ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 \
  "/tmp/douyin_${AWEME_ID}/video.mp4"
```

Extract mono 16k WAV for STT:

```bash
ffmpeg -y -i "/tmp/douyin_${AWEME_ID}/video.mp4" \
  -vn -ac 1 -ar 16000 -c:a pcm_s16le \
  "/tmp/douyin_${AWEME_ID}/audio.wav"
```

Prefer MLX Whisper on Apple Silicon. A reusable wrapper is shipped at `scripts/transcribe_video.sh` (it picks up `mlx_whisper` from `PATH`; override with the `WHISPER_BIN` env var if it lives elsewhere):

```bash
# Recommended: use the bundled script
scripts/transcribe_video.sh \
  "/tmp/douyin_${AWEME_ID}/video.mp4" \
  "/tmp/douyin_${AWEME_ID}" \
  transcript_small

# Or invoke mlx_whisper directly (assumes it is on PATH)
mlx_whisper \
  "/tmp/douyin_${AWEME_ID}/audio.wav" \
  --model mlx-community/whisper-small-mlx \
  --language zh \
  --output-dir "/tmp/douyin_${AWEME_ID}" \
  --output-name transcript_small \
  --output-format txt \
  --verbose False

# If the small model download/auth fails, fall back:
mlx_whisper \
  "/tmp/douyin_${AWEME_ID}/audio.wav" \
  --model mlx-community/whisper-tiny-mlx \
  --language zh \
  --output-dir "/tmp/douyin_${AWEME_ID}" \
  --output-name transcript_tiny \
  --output-format txt
```

Fallback to `mlx-community/whisper-tiny` if model download/auth fails. Mention that tiny is faster but less accurate.

### 6. Clean transcript lightly, but label uncertainty

Machine transcripts may contain common Chinese ASR errors such as:

- 券商 → 圈商/劝商/全商
- 研报 → 眼报/延报
- 评级 → 平级
- 估值 → 孤职
- 折现率 → 折线率

Light cleanup is fine for readability, but do not pretend it is a human-verified transcript. If a phrase affects investment/medical/legal decisions, mark it as needing manual verification.

### 7. Output format

Create a Markdown file under `/tmp/douyin_<aweme_id>/douyin_<aweme_id>_extract.md` with this structure:

```markdown
# 抖音内容提取：[标题]

来源：<original/canonical URL>
作者：<nickname>
发布时间：<if available>
时长/类型：<video duration or image count>
提取方式：浏览器 aweme_detail + 媒体下载 + 本地 STT/视觉分析
质量说明：机器转写/机器视觉，未完全人工校对

## 核心摘要

## 主要观点/信息点

## 结构化要点

## 小飞判断 / 对用户场景的意义

## 原始元数据

## 机器转写正文或图片 OCR/描述
```

When replying on WeChat, keep the message compact and attach the file with:

```text
MEDIA:/tmp/douyin_<aweme_id>/douyin_<aweme_id>_extract.md
```

## Troubleshooting

- `mcporter` says `Unknown MCP server 'douyin'`: expected on some installs; use browser fallback.
- `yt-dlp` says fresh cookies needed: use browser/API fallback instead of trying to log in.
- Browser loads but no metadata: wait a few seconds, scroll/click play, then inspect `performance` again.
- Direct MP4 URL expires: re-fetch `aweme_detail` from the browser page and download immediately.
- Downloaded file is tiny/HTML: wrong URL or expired signature; check `file` and retry with another `url_list` item.
- Long video transcription times out: use `terminal(background=true, notify_on_complete=true)` or split audio into chunks.
- `mlx_whisper` not on `PATH`: install with `pip install mlx-whisper` (or `uv tool install mlx-whisper`), or set `WHISPER_BIN=/path/to/mlx_whisper` when calling `scripts/transcribe_video.sh`.
- Image posts/video frames have text in screenshots: use `vision_analyze` with a direct question: "提取图中文字并总结观点". If `vision_analyze` fails because the configured vision provider token is invalid/expired, fall back to the local `zai` CLI vision path. For multi-image Douyin notes, loop over all downloaded images and append output to one OCR file:

```bash
cd /tmp/douyin_${AWEME_ID}
for i in 01 02 03 04 05; do
  echo "===== IMAGE $i =====" | tee -a ocr.txt
  zai vision -f image_${i}.webp \
    '请提取这张图片中的所有中文/英文文字，尤其是标题、表格、仓库名、URL、框架名；保持原文，识别不确定处用 [?] 标记。' \
    2>&1 | tee -a ocr.txt
  echo | tee -a ocr.txt
done
```

For a single image:

```bash
zai vision -f /tmp/douyin_${AWEME_ID}/contact.jpg \
  '请提取画面中的字幕、项目名称和关键信息；不确定请标注。' --json
```

## Quality checklist before final answer

- [ ] Canonical URL or aweme_id captured.
- [ ] Metadata captured: title/desc, author, time, duration/type when available.
- [ ] Media download or image extraction verified with `file`, `ls -lh`, or `ffprobe`.
- [ ] Transcript/image OCR exists and is labeled as machine-generated.
- [ ] Final Markdown report written under `/tmp/`.
- [ ] User-facing response includes concise summary and `MEDIA:` attachment if a file was produced.
