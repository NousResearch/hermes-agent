---
name: youtube-content
description: Turn a YouTube transcript into chapters, summaries, X threads, blog drafts, and quote lists using the youtube_content tool.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [YouTube, Transcript, Chapters, Summary, Content, Marketing]
    related_skills: [obsidian, notion]
---

# YouTube Content Tool

Use the `youtube_content` tool when the user shares a YouTube video and wants structured written outputs derived from the transcript.

This tool fetches the transcript, then transforms it into one of several formats:
- chapter timestamps
- summary
- chapter summaries
- X/Twitter thread
- blog post draft
- quote extraction
- all formats in one response

## When To Use

Use this skill when the user asks for things like:
- "Bu videoya chapter cikar"
- "Bu YouTube videosunu thread'e cevir"
- "Transcriptten blog yazisi olustur"
- "Onemli alintilari ve timestamp'leri ver"
- "Videonun tum icerik paketini hazirla"

Do not use this skill for:
- Non-YouTube URLs
- Videos without accessible transcripts (private, disabled transcript)
- Tasks that require frame-by-frame visual analysis (use browser/vision tools instead)

## Quick Usage

### Basic chapter extraction

```python
youtube_content(
    url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    output_format="chapters"
)
```

### Turkish output (if transcript language supports it)

```python
youtube_content(
    url="https://youtu.be/dQw4w9WgXcQ",
    output_format="thread",
    language="tr"
)
```

### Try Turkish then English transcript fallback

```python
youtube_content(
    url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    output_format="summary",
    language="tr,en"
)
```

### Generate everything in one call

```python
youtube_content(
    url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    output_format="all"
)
```

## Output Formats

### 1. `chapters`

Returns timestamped chapters in plain lines:

```text
00:00 Giris
00:55 Bitcoin Dongu Analizi
02:05 Ayi Sezonu Belirtileri
04:10 Risk Yonetimi
06:42 Sonuc ve Ozet
```

Use this when:
- User wants YouTube chapter markers
- User will paste timestamps into video description
- User wants a quick structure map before deeper content work

### 2. `summary`

Returns a 5-10 sentence summary of the video.

Use this when:
- User wants a quick understanding before watching
- User is triaging multiple videos
- User wants a briefing version

### 3. `chapter_summaries`

Returns chapter timestamps plus exactly short summaries per chapter.

Example shape:

```text
00:00 Giris
Konusmaci videonun amacini tanimliyor ve hangi sorulara cevap verecegini acikliyor.
Odak noktasi piyasa dongusunu dogru okumak ve riskleri erken fark etmek.

00:55 Bitcoin Dongu Analizi
Onceki dongulerle mevcut piyasa yapisi karsilastiriliyor ve tekrar eden sinyaller anlatiliyor.
Ozellikle hacim davranisi ve momentum zayiflamasinin onemi vurgulaniyor.
```

Use this when:
- User wants study notes
- User wants newsletter-ready section summaries
- User plans to repurpose into carousels or scripts

### 4. `thread`

Returns a numbered X/Twitter thread draft.

Use this when:
- User wants social content from a long video
- User wants hook + takeaways + CTA format

### 5. `blog`

Returns a titled, sectioned blog post draft.

Use this when:
- User wants SEO/content-marketing repurposing
- User needs long-form written content from an interview/podcast/video

### 6. `quotes`

Returns timestamped quotes + why they matter.

Example shape:

```text
01:22 "The market does not top when everyone is scared; it tops when bad news stops mattering." - Why it matters: explains sentiment divergence as a late-cycle signal.
05:48 "Most people lose because they size for upside and ignore survival." - Why it matters: summarizes the video's risk-management thesis.
```

Use this when:
- User wants clips/highlights
- User is making quote cards or short posts
- User needs speaker soundbites with timestamps

### 7. `all`

Returns all formats in one response, grouped into sections:
- Chapters
- Summary
- Chapter Summaries
- Thread
- Blog
- Quotes

Use this when:
- User wants a complete repurposing pack in one go
- User is building a content pipeline from a single video

## Practical Workflow Examples

### Workflow A: YouTube chapters first, then thread

1. Call `youtube_content(..., output_format="chapters")`
2. Review chapter segmentation quality
3. Call `youtube_content(..., output_format="thread", language="tr")`
4. Optionally ask for refinements in a desired tone

### Workflow B: Full content pack for a creator

1. Call `youtube_content(..., output_format="all", language="tr")`
2. Extract blog section into `write_file`
3. Save quote list to content backlog (`notion`, `obsidian`, or local file)
4. Reformat thread manually to match user's posting style

## Error Handling Notes

Common failure cases:
- Invalid URL / unsupported YouTube URL format
- Transcript disabled by uploader
- Private or unavailable video
- No transcript in requested language

If transcript retrieval fails:
1. Tell the user exactly why (private / no transcript / disabled)
2. Ask for another video URL or a pasted transcript
3. If the user provides transcript text manually, use normal writing tools instead

## Tips

- `language="auto"` is best when you are unsure of the transcript language.
- Use `language="tr,en"` (or similar) when the user explicitly wants a transcript language fallback order.
- `all` is convenient but longer/more expensive than requesting only the needed format.
- For highly technical videos, `chapter_summaries` is often the most useful first output before `thread` or `blog`.
