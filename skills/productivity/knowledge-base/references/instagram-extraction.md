# Instagram Reel Extraction via SEO Metadata

When Firecrawl/web_extract is unavailable or out of credits, Instagram Reels can still be extracted
via curl with a search-engine user-agent. Instagram serves full SEO metadata (og:title, og:description,
author, like/comment counts) to Googlebot.

## Single command

```bash
curl -sL "https://www.instagram.com/reel/REEL_ID/" \
  -H "User-Agent: Googlebot/2.1" 2>&1 | python3 -c "
import sys, re, html
content = sys.stdin.read()

og_title = re.search(r'<meta property=\"og:title\" content=\"([^\"]+)\"', content)
og_desc = re.search(r'<meta property=\"og:description\" content=\"([^\"]+)\"', content)

if og_title: print('TITLE:', html.unescape(og_title.group(1)))
if og_desc: print('DESC:', html.unescape(og_desc.group(1))[:500])
"
```

## What you get

- **TITLE**: Full caption text (Instagram puts the entire caption in og:title)
- **DESC**: Author + date + likes/comments counts + caption preview
- **LIKES/COMMENTS**: Extractable from og:description with regex `(\d+[\d,.]*)\s*likes`

## Limitations

- No video content (images, audio, visual) — only text metadata
- No comments thread — only aggregate count
- Single reel only, not profile feeds
- Instagram may throttle heavy scraping

## GitHub raw content alternative

For GitHub markdown files (README, docs), prefer raw.githubusercontent.com over web_extract:

```bash
curl -sL "https://raw.githubusercontent.com/OWNER/REPO/BRANCH/PATH"
```

This avoids the extract backend entirely and works with curl (free, unlimited).
