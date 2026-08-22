---
name: rss-reader
description: Fetch and parse any RSS or Atom feed without external dependencies. Read AI/ML news, GitHub releases, blog updates, and more — combine with Hermes cron for automated daily digests.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [RSS, Atom, Feeds, News, Monitoring, Cron]
    related_skills: [blogwatcher, arxiv]
---

# RSS / Atom Feed Reader

Fetch and parse any RSS or Atom feed using only `curl` and Python's standard library. No installation, no API key, no external dependencies.

## Quick Reference

| Action | Method |
|--------|--------|
| Fetch + print feed | Inline Python snippet below |
| Get N most recent items | `--max N` in the helper snippet |
| Filter by keyword | `--grep WORD` in the helper snippet |
| List an OPML file | See OPML section |
| Daily digest via cron | See Cron section |

---

## Reading a Feed

```bash
curl -sL --max-time 10 "https://hnrss.org/frontpage" | python3 -c "
import sys, xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

def parse_date(s):
    if not s: return ''
    try: return parsedate_to_datetime(s).strftime('%Y-%m-%d')  # RSS RFC 2822
    except ValueError: return s[:10]  # Atom ISO-8601 fallback

feed = sys.stdin.read()
# Python 3.8+ stdlib ET does not expand external entities by default.
# For production use with fully untrusted feeds, pip install defusedxml
# and replace ET with defusedxml.ElementTree for Billion Laughs protection.
try:
    root = ET.fromstring(feed)
except ET.ParseError as e:
    sys.exit(f'Feed parse error: {e}')
ns = {'atom': 'http://www.w3.org/2005/Atom'}

# Detect RSS vs Atom
items = root.findall('.//item')  # RSS
if not items:
    items = root.findall('atom:entry', ns) or root.findall('{http://www.w3.org/2005/Atom}entry')  # Atom

MAX = 10
for i, item in enumerate(items[:MAX]):
    get = lambda tag, fb='': (item.findtext(tag) or item.findtext(f'atom:{tag}', namespaces=ns) or fb).strip()
    title = get('title', '(no title)')
    link  = get('link') or (item.find('{http://www.w3.org/2005/Atom}link') or item.find('link') or type('', (), {'get': lambda s, k, d='': d})()).get('href', '')
    date  = parse_date(get('pubDate') or get('published') or get('updated', ''))
    print(f'{i+1:2}. [{date}] {title}')
    if link: print(f'     {link}')
    print()
"
```

---

## AI / ML Feeds Worth Monitoring

These feeds are directly relevant to LLM research, agent development, and open-source AI:

| Source | Feed URL |
|--------|----------|
| Hacker News (AI filter) | `https://hnrss.org/newest?q=AI+OR+LLM+OR+agent` |
| arXiv cs.AI (latest) | `https://export.arxiv.org/rss/cs.AI` |
| arXiv cs.LG (latest) | `https://export.arxiv.org/rss/cs.LG` |
| arXiv cs.CL (latest) | `https://export.arxiv.org/rss/cs.CL` |
| Hugging Face Blog | `https://huggingface.co/blog/feed.xml` |
| OpenAI Blog | `https://openai.com/blog/rss.xml` |
| Anthropic Blog | `https://www.anthropic.com/rss.xml` |
| Google DeepMind Blog | `https://deepmind.google/blog/rss.xml` |
| Nous Research Blog | `https://nousresearch.com/feed/` |
| LessWrong (AI Safety) | `https://www.lesswrong.com/feed.xml?view=community-questions&karmaThreshold=30` |
| The Gradient | `https://thegradient.pub/rss/` |

### GitHub Release Feeds

Any GitHub repo exposes a releases feed at `https://github.com/<owner>/<repo>/releases.atom`:

```bash
# Track Hermes Agent releases
curl -sL "https://github.com/NousResearch/hermes-agent/releases.atom" | python3 -c "..."

# Track other tools you depend on
curl -sL "https://github.com/astral-sh/uv/releases.atom" | python3 -c "..."
```

---

## Multi-Feed Digest

Fetch several feeds at once and merge into a single digest:

```bash
python3 << 'EOF'
import sys, urllib.request, urllib.error, xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

def parse_date(s):
    if not s: return '0000-00-00'
    try: return parsedate_to_datetime(s.strip()).strftime('%Y-%m-%d')
    except ValueError: return s.strip()[:10]

def safe_url(url):
    """Reject non-http(s) schemes and loopback/link-local hosts."""
    p = urlparse(url)
    if p.scheme not in ('http', 'https'):
        raise ValueError(f"Blocked scheme: {p.scheme}")
    host = p.hostname or ''
    blocked = ('localhost', '127.0.0.1', '0.0.0.0', '::1', '169.254.169.254')
    if host in blocked or host.startswith('192.168.') or host.startswith('10.'):
        raise ValueError(f"Blocked host: {host}")
    return url

def fetch_feed(url):
    req = urllib.request.Request(safe_url(url), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.read()

def parse_feed(raw):
    # stdlib ET (Python 3.8+) does not expand external entities.
    # For untrusted feeds in production, use defusedxml.ElementTree instead.
    try:
        return ET.fromstring(raw)
    except ET.ParseError as e:
        raise ET.ParseError(f"Feed parse error: {e}") from e

FEEDS = [
    ("arXiv cs.AI",   "https://export.arxiv.org/rss/cs.AI"),
    ("HN AI",         "https://hnrss.org/newest?q=LLM+OR+agent&count=5"),
    ("HF Blog",       "https://huggingface.co/blog/feed.xml"),
    ("Nous Research", "https://nousresearch.com/feed/"),
]

MAX_PER_FEED = 5
items = []

for name, url in FEEDS:
    try:
        root = parse_feed(fetch_feed(url))
        entries = root.findall('.//item') or root.findall('{http://www.w3.org/2005/Atom}entry')
        for entry in entries[:MAX_PER_FEED]:
            get = lambda t, fb='': (entry.findtext(t) or entry.findtext(f'{{http://www.w3.org/2005/Atom}}{t}') or fb).strip()
            title = get('title', '(no title)')
            link  = get('link')
            if not link:
                a = entry.find('{http://www.w3.org/2005/Atom}link')
                link = a.get('href', '') if a is not None else ''
            date = parse_date(get('pubDate') or get('published') or get('updated', ''))
            items.append((date, name, title, link))
    except (urllib.error.URLError, ET.ParseError, ValueError, OSError) as e:
        print(f"[{name}] fetch error: {e}")

items.sort(reverse=True)
print(f"\n=== Feed Digest — {datetime.now(timezone.utc).strftime('%Y-%m-%d')} ===\n")
for date, source, title, link in items:
    print(f"[{source}] {date}")
    print(f"  {title}")
    if link: print(f"  {link}")
    print()
EOF
```

---

## Keyword Filtering

Filter a feed for items matching a keyword (useful for high-volume feeds like arXiv):

```bash
# Keyword passed as sys.argv[1] — not interpolated into code, safe from injection
KEYWORD="reinforcement learning"
curl -sL --max-time 10 "https://export.arxiv.org/rss/cs.AI" | python3 -c "
import sys, xml.etree.ElementTree as ET
kw = sys.argv[1].lower()
try:
    root = ET.fromstring(sys.stdin.read())
except ET.ParseError as e:
    sys.exit(f'Parse error: {e}')
for item in root.findall('.//item'):
    title = (item.findtext('title') or '').strip()
    desc  = (item.findtext('description') or '').strip()
    link  = (item.findtext('link') or '').strip()
    if kw in title.lower() or kw in desc.lower():
        print(f'  {title}')
        print(f'  {link}')
        print()
" "${KEYWORD}"
```

---

## OPML Import

If you have an OPML file exported from Feedly, Inoreader, or another reader:

```bash
python3 << 'EOF'
import xml.etree.ElementTree as ET, urllib.request

with open("subscriptions.opml") as f:
    root = ET.parse(f).getroot()

feeds = [(o.get('text') or o.get('title', 'Feed'), o.get('xmlUrl'))
         for o in root.iter('outline') if o.get('xmlUrl')]

print(f"Found {len(feeds)} feeds:")
for name, url in feeds:
    print(f"  - {name}: {url}")
EOF
```

---

## Cron: Daily AI Digest

Schedule a morning digest delivered to your preferred platform (Telegram, Discord, etc.):

```bash
hermes cron create "0 8 * * *" \
  "Run the multi-feed digest script to fetch the latest AI/ML news from arXiv cs.AI, HuggingFace Blog, and Hacker News. Summarize the 3 most interesting items in 2 sentences each. Keep the digest under 500 words." \
  --name "morning-ai-digest" \
  --deliver telegram
```

Or use a context file to keep feed URLs out of the cron prompt:

```bash
# ~/.hermes/context/feeds.md  — add your feed list here
hermes cron create "0 8 * * *" \
  "Fetch feeds from my feeds context file, pick the 5 most interesting items across all sources, and send a short digest." \
  --name "morning-ai-digest" \
  --deliver telegram
```

---

## Notes

- RSS and Atom are both supported by the inline parser above; RSS uses `<item>` elements, Atom uses `<entry>`.
- Some sites block default `curl` user-agents — add `-H "User-Agent: Mozilla/5.0"` if you get 403s.
- arXiv RSS feeds update once per day; fetching more frequently won't return new results.
- GitHub release feeds are Atom; the inline parser handles both formats transparently.
- For persistent read/unread tracking across sessions, use the `blogwatcher` skill in `research/` (requires `blogwatcher-cli` installation).
- Feed URLs sometimes move — if a feed stops working, check the site's `<link rel="alternate" type="application/rss+xml">` tag in its HTML source.
