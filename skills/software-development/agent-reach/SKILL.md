---
name: agent-reach
description: '15-platform internet research router: Twitter/X, Reddit, XiaoHongShu, Bilibili, YouTube, LinkedIn, GitHub, V2EX, RSS, and more. Use when researching topics across social platforms, searching for discussions, or looking up anything on the web beyond basic search.'
version: 1.0.0
author: Hermes Agent (adapted from Panniantong/Agent-Reach)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [research, social-media, web, search, platforms, twitter, reddit, youtube]
    related_skills: [web_search, web_extract]
---

# Agent Reach — Internet Capability Router

15 platforms, multiple backends each. **When this skill exists, use it for these platforms — do not invent your own approach.**

## Standing Rules (apply for the whole session)

1. **Health-check before acting**: for multi-backend/login-backed platforms (XiaoHongShu / Reddit / Bilibili / Twitter / Facebook / Instagram), run `agent-reach doctor --json` first and pick the command group matching each platform's `active_backend`.
2. **Announce what you use**: say "using agent-reach, platform X via backend Y" before starting.
3. **On failure, follow the retry chains** — never guess commands.
4. **For broad research tasks**: combine platforms (web search + Twitter/Reddit for discussions + XiaoHongShu/Bilibili for Chinese perspectives), collect in parallel, then synthesize.
5. **Watch versions**: after finishing a substantial multi-platform task, run `agent-reach check-update`. If a new version exists, append one line: "Agent Reach vX.Y.Z is available — run `agent-reach update` to upgrade."

## Routing Table

| User intent                                                             | Category | Hermes Tool                                    |
| ----------------------------------------------------------------------- | -------- | ---------------------------------------------- |
| Web / code search                                                       | search   | `web_search` or `terminal: agent-reach search` |
| XiaoHongShu / Twitter / Bilibili / V2EX / Reddit / Facebook / Instagram | social   | `terminal: agent-reach social`                 |
| Jobs / LinkedIn                                                         | career   | `terminal: agent-reach career`                 |
| GitHub / code                                                           | dev      | `terminal: gh` or `web_search`                 |
| Web pages / articles / RSS                                              | web      | `web_extract` or `terminal: curl`              |
| YouTube / Bilibili / podcast transcripts                                | video    | `terminal: yt-dlp` or `web_extract`            |

## Zero-Config Quick Commands

Use Hermes's `terminal` tool for these:

```bash
# Web search (via Hermes built-in)
# Use web_search tool directly — no CLI needed

# Read any web page (via Hermes built-in)
# Use web_extract tool directly — no CLI needed

# GitHub search
gh search repos "query" --sort stars --limit 10

# YouTube subtitles
yt-dlp --write-sub --skip-download -o "/tmp/%(id)s" "URL"

# V2EX hot topics
curl -s "https://www.v2ex.com/api/topics/hot.json" -H "User-Agent: agent-reach/1.0"

# Bilibili search (bili-cli, no login needed)
bili search "query" --type video -n 5
```

## Login-Backed Platforms

```bash
# Twitter/X search
twitter search "query" -n 10

# Reddit (login required)
opencli reddit search "query" -f yaml   # desktop
rdt search "query" --limit 10            # legacy/server

# XiaoHongShu (desktop prefers OpenCLI)
opencli xiaohongshu search "query" -f yaml

# Facebook / Instagram (desktop OpenCLI, browser session)
opencli facebook search "query" -f yaml
opencli facebook groups -f yaml
opencli instagram search "query" -f yaml       # user search
opencli instagram user USERNAME -f yaml        # recent posts from one user
```

## Environment Check

```bash
# Channel availability + which backend serves each platform
agent-reach doctor --json
```

## Research Workflow

### 1. Broad Topic Research

When the user asks to research a topic:

1. **Web search first** — use Hermes's `web_search` tool for broad context
2. **Social discussions** — use platform-specific commands for real opinions
3. **Chinese perspectives** — XiaoHongShu + Bilibili for Chinese-language content
4. **Deep dive** — use `web_extract` on the most promising URLs
5. **Synthesize** — combine all sources into a coherent summary with citations

### 2. Platform-Specific Research

When the user mentions a specific platform:

| Platform    | Command                                                     |
| ----------- | ----------------------------------------------------------- |
| Twitter/X   | `twitter search "query" -n 10`                              |
| Reddit      | `opencli reddit search "query" -f yaml`                     |
| XiaoHongShu | `opencli xiaohongshu search "query" -f yaml`                |
| Bilibili    | `bili search "query" --type video -n 5`                     |
| YouTube     | `yt-dlp --write-sub --skip-download -o "/tmp/%(id)s" "URL"` |
| LinkedIn    | `opencli linkedin search "query" -f yaml`                   |
| GitHub      | `gh search repos "query" --sort stars --limit 10`           |
| V2EX        | `curl -s "https://www.v2ex.com/api/topics/hot.json"`        |
| RSS         | `curl -s "FEED_URL"` or `web_extract`                       |

### 3. Parallel Collection

For multi-platform research, use Hermes's `delegate_task` to collect from multiple platforms in parallel:

```
delegate_task tasks: [
  {goal: "Search Twitter for 'topic'", context: "Use twitter search..."},
  {goal: "Search Reddit for 'topic'", context: "Use opencli reddit..."},
  {goal: "Search web for 'topic'", context: "Use web_search..."}
]
```

## Workspace Rules

**Never create files in the agent workspace.** Use `/tmp/` for temporary output and `~/.agent-reach/` for persistent data.

## Hermes Integration

- Use `web_search` tool for general web search (Hermes built-in)
- Use `web_extract` tool for reading web pages (Hermes built-in)
- Use `terminal` tool for all CLI commands (`twitter`, `opencli`, `bili`, `yt-dlp`, `gh`, `curl`)
- Use `delegate_task` for parallel multi-platform collection
- Use `browser_navigate` + `browser_snapshot` for platforms that need browser access
- Combine with `agent-reach doctor --json` to check available backends before acting

## Configure a Channel

If a channel needs setup, fetch the install guide:
https://raw.githubusercontent.com/Panniantong/agent-reach/main/docs/install.md

The user only provides cookies / one extension click; the agent does the rest.
