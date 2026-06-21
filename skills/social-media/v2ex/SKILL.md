---
name: v2ex
description: "Use when the user wants to browse or search V2EX (Chinese developer community) — hot topics, node browsing, topic details with replies, user profiles. Zero-auth public REST API."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [v2ex, chinese, developer-community, forum, tech, zero-auth]
    related_skills: [reddit, bilibili]
---

# V2EX Skill

V2EX is a Chinese developer/tech forum at https://www.v2ex.com. Its public REST
API requires no authentication, no API key, and no setup — just curl. All
endpoints return JSON.

## When to Use

- User asks about trending tech topics in the Chinese developer community
- User shares a V2EX URL (https://www.v2ex.com/t/...) and wants to read it
- User wants to browse a specific node (python, tech, jobs, etc.)
- User asks about a V2EX user's profile

Don't use for: searching V2EX (their API lacks a search endpoint — use
`web_search` with `site:v2ex.com` instead), or any write operations.

## Quick Reference

```bash
# Hot topics
curl -s "https://www.v2ex.com/api/topics/hot.json" -H "User-Agent: hermes/1.0"

# Node topics (node_name: python, tech, jobs, qna, programmers, etc.)
curl -s "https://www.v2ex.com/api/topics/show.json?node_name=NODE&page=1" -H "User-Agent: hermes/1.0"

# Topic detail (ID from URL like https://www.v2ex.com/t/1234567)
curl -s "https://www.v2ex.com/api/topics/show.json?id=TOPIC_ID" -H "User-Agent: hermes/1.0"

# Topic replies (first page, up to 100)
curl -s "https://www.v2ex.com/api/replies/show.json?topic_id=TOPIC_ID&page=1" -H "User-Agent: hermes/1.0"

# User profile
curl -s "https://www.v2ex.com/api/members/show.json?username=USERNAME" -H "User-Agent: hermes/1.0"
```

Always include `-H "User-Agent: hermes/1.0"` — V2EX rejects requests without a
User-Agent header.

## Available Endpoints

### Hot Topics
`GET https://www.v2ex.com/api/topics/hot.json`
Returns array of topics with `id`, `title`, `url`, `content` (HTML), `replies`
(count), `node` (name + title), `member` (username).

### Nodes
Browse all nodes at https://www.v2ex.com/planes. Common nodes: `python`,
`tech`, `jobs`, `qna`, `programmers`, `create`, `share`, `macos`, `linux`.

### Node Topics
`GET .../api/topics/show.json?node_name=NODE&page=1`
Same structure as hot topics, filtered by node.

### Topic Detail
`GET .../api/topics/show.json?id=TOPIC_ID`
Full topic with `content` (HTML), `replies` count, node info, author info.

### Topic Replies
`GET .../api/replies/show.json?topic_id=TOPIC_ID&page=1`
Array of replies: `content` (HTML), `member` (username), `created` (timestamp).

### User Profile
`GET .../api/members/show.json?username=USERNAME`
Returns: `id`, `username`, `website`, `twitter`, `github`, `location`, `bio`,
`avatar_large`, `created`.

## Procedure

1. Determine the user's intent (hot topics, specific topic, node browse, user lookup).
2. For topic URLs like `https://www.v2ex.com/t/1234567`, extract the numeric ID.
3. Fetch the API endpoint with curl + User-Agent header.
4. Content fields contain HTML — strip `<p>`, `<br>`, `<a>` tags for readability.
5. Present results. For topics with many replies, only show the most relevant ones.

## Python Helper (optional, for structured processing)

```python
import json, urllib.request

def v2ex_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "hermes/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())
```

## Common Pitfalls

1. **No User-Agent → empty response.** Always include the header.
2. **GFW / network issues.** V2EX is hosted outside China. If unreachable, use
   a proxy or fall back to `web_extract` on the page URL.
3. **Search not available via API.** Use `web_search` with `site:v2ex.com`.
4. **Content is HTML.** Strip tags before presenting to the user.
5. **Rate limiting.** V2EX's API is free and has no documented limits, but
   don't hammer it with rapid requests.

## Verification Checklist

- [ ] `curl -s "https://www.v2ex.com/api/topics/hot.json" -H "User-Agent: hermes/1.0"` returns a JSON array
- [ ] User-Agent header is included in every request
- [ ] Content fields are stripped of HTML tags before display
