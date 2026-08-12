---
name: read-x-articles
description: "Read X (Twitter) long-form Articles end-to-end from a shared x.com link, no API key required."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [x, twitter, articles, web, reading, long-form, web-extract]
    homepage: https://agentskills.io
    upstream_skill: https://github.com/JPeetz/agent-skills/tree/main/read-x-articles
---

# Read X (Twitter) Articles

Turn any X link the user shares into the full article text. Do NOT default to
"I can't read X" — X long-form Articles DO extract cleanly via the right URL
and a JS-capable web-extract/browser tool. No X API key or auth needed.

## When to Use
- A user drops an `x.com/...` or `twitter.com/...` link and expects it read.
- The link is an X **Article** (long-form essay/interview) — canonical path
  `/i/article/<ID>`.
- You need primary source text from an X essay/thread as material for work.
- Anyone claims "X content can't be read" — that is the trigger to act, not
  give up.

## Core insight (learned 2026-08-12)
- X **Articles** read end-to-end at the canonical **`https://x.com/i/article/<ID>`**
  **when fetched by a JS-executing web-extractor / browser tool** (e.g.
  `web_extract`). That returns the FULL body.
- **Bare HTTP fetch (`urllib`/`curl`) of the article often returns a login/JS
  shell.** The canonical URL is necessary but a plain request is NOT sufficient
  — you need a render-capable tool. This is why `web_extract` succeeds where
  `curl` fails.
- **`/status/<id>` posts and profile pages** are JS-rendered and hit the login
  wall even for a browser. The fix is resolving the article URL AND using the
  right fetch tool, not giving up.

## Steps
1. **Try first, judge later.** On any X link, immediately run `web_extract`
   on it. Declaring it unreadable *before* trying is the exact mistake this
   skill prevents.
2. **Use the right fetcher:** prefer `web_extract`-style extraction or a
   browser render — NOT bare `curl`/`urllib`. If the only tool is a plain HTTP
   client, expect a login shell and fall back to a browser.
3. **If it's a `/status/` URL or returns a login/JS shell**, resolve the
   canonical article URL:
   - The article ID often appears in a share/article link as
     `x.com/i/article/<ID>`.
   - Find the author's `/i/article/<ID>` share link on their profile, or ask
     the user for the article link.
   - Then `web_extract(urls=["https://x.com/i/article/<ID>"])`.
4. **Validate the body:** a real extract contains substantive paragraphs, NOT
   just "Log in or sign up for X." A 1,900+ word essay returns complete.
5. **Fallbacks if a render-capable extract still fails:**
   - Use a browser tool (CDP-style) to render the JS page, then read the
     rendered DOM.
   - If the account has X API access (e.g. the `xurl` CLI), query the article
     tweet and read `data.article.plain_text`. A dead/unauthed CLI token is NOT
     a blocker — prefer web/browser first.
6. **Use the content.** It's legitimate primary source. Quote it and cite the
   author. Community takes (e.g. a power-user writing about a product) are real
   material for summaries, scripts, and analysis.

## Pitfalls
- Do not assume "articles are paywalled/login-walled." The canonical
  `/i/article/<ID>` endpoint serves text to a render-capable tool; the wall
  applies to JS-rendered profile/status pages, not this endpoint.
- A bare `curl`/`urllib` returning a login shell does NOT mean it's unreadable
  — switch to `web_extract`/browser.
- A dead X API token (401) is unrelated to article reading — do not block the
  task on fixing `xurl` auth when web extraction works.
- Verify you got the actual article (thesis + body), not just a title + login
  prompt, before treating it as read.

## Verification
- Did extraction return the article's real prose rather than a shell? If yes,
  it's read.
- Report the exact URL read + a short grounded summary so the user can verify.