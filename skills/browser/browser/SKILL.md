---
name: browser
description: Interactive browser automation using Hermes native browser tools. Navigate pages, interact with elements, extract content, and analyze pages visually. Designed for JS-heavy sites, form interaction, authentication flows, and dynamic content scraping.
version: 1.0.0
author: Hermes Community
license: MIT
metadata:
  hermes:
    tags: [Browser, Automation, Web, Scraping, Forms, Vision, Interaction]
    related_skills: [duckduckgo-search, screenshot-website]
---

# Browser

Interactive browser automation using Hermes native browser tools. All tools are built-in — no external dependencies required. Backed by local Chromium by default, or Browserbase/Browser Use in cloud mode.

## When to Use

| Situation | Recommended tool |
|-----------|-----------------|
| Keyword search, quick facts | `web_search` |
| Static HTML page, no JavaScript needed | `web_extract` |
| One-shot full-page screenshot | `screenshot-website` skill |
| JavaScript-rendered content | **browser tools** |
| Form filling, button clicks, navigation | **browser tools** |
| Login / authentication flow | **browser tools** |
| Visual analysis, CAPTCHA, complex layout | **browser tools** + `browser_vision` |

## Quick Reference

| Tool | What it does |
|------|-------------|
| `browser_navigate(url)` | Open a URL |
| `browser_snapshot()` | Read the page as a text accessibility tree |
| `browser_click(ref)` | Click an element by ref (e.g. `@e5`) |
| `browser_type(ref, text)` | Type into an input field |
| `browser_press(key)` | Press a keyboard key |
| `browser_scroll(direction)` | Scroll the page up or down |
| `browser_back()` | Go back in browser history |
| `browser_close()` | Close the session |
| `browser_console()` | Read JS console output and errors |
| `browser_get_images()` | List all images on the page |
| `browser_vision(question)` | Screenshot + AI visual analysis |

## Core Workflow

```
browser_navigate(url)
  → browser_snapshot()          # read page, get element refs (@e1, @e2…)
  → browser_click(@eN)          # interact with elements
  → browser_type(@eN, text)     # fill in fields
  → browser_press("Enter")      # submit or navigate
  → browser_snapshot()          # verify the result
  → browser_close()             # always clean up
```

## Element Refs

`browser_snapshot()` returns an accessibility tree. Interactive elements are tagged with refs (`@e1`, `@e2`, etc.). Pass these refs to `browser_click` and `browser_type`.

```
# Snapshot excerpt example:
# [input @e2] placeholder="Search…"
# [button @e4] "Search"

browser_type("@e2", "my query")
browser_press("Enter")
```

**Tips:**
- Use `browser_snapshot(full=True)` for the complete tree (verbose, use when elements are missing)
- Use `browser_snapshot(user_task="what I'm looking for")` for a task-focused summary

## Common Patterns

### Scraping a JS-rendered page

```
browser_navigate("https://example.com/listings")
browser_scroll("down")                              # trigger lazy-loaded content
browser_snapshot(user_task="extract all prices and titles")
browser_close()
```

### Form submission

```
browser_navigate("https://example.com/search")
browser_snapshot()                                  # find input refs
browser_type("@e2", "my search query")
browser_press("Enter")
browser_snapshot()                                  # read results
browser_close()
```

### Login flow

```
browser_navigate("https://example.com/login")
browser_snapshot()                                  # find username/password refs
browser_type("@e1", "user@example.com")
browser_type("@e2", "password")
browser_click("@e3")                                # submit button
browser_snapshot()                                  # verify login succeeded
# ... continue with authenticated session
browser_close()
```

### Visual analysis and CAPTCHAs

```
browser_navigate("https://example.com")
browser_vision("What is on this page? Is there a CAPTCHA or challenge?")
browser_vision("Describe the CAPTCHA and what value I should enter", annotate=True)
```

### Debugging JavaScript errors

```
browser_navigate("https://example.com/app")
browser_snapshot()
browser_console()                                   # inspect JS logs and errors
browser_console(clear=True)                         # clear buffer after reading
```

## Tool Reference

### `browser_navigate(url)`
Opens a URL. Initializes the browser session on first call with stealth features enabled.

### `browser_snapshot(full=False, user_task=None)`
Returns the page content as a text-based accessibility tree.
- `full=True` — complete tree (use when elements seem missing in compact view)
- `user_task="..."` — returns a task-aware summarized view instead of the raw tree

### `browser_click(ref)`
Clicks an element. `ref` is an element reference from the snapshot (e.g. `"@e5"`).

### `browser_type(ref, text)`
Types text into an input element. `ref` from snapshot.

### `browser_press(key)`
Presses a keyboard key. Common values: `"Enter"`, `"Tab"`, `"Escape"`, `"ArrowDown"`, `"Space"`.

### `browser_scroll(direction)`
Scrolls the page. `direction`: `"up"` or `"down"`. Essential for triggering lazy-loaded content.

### `browser_back()`
Navigates back in browser history.

### `browser_close()`
Closes the browser session and frees resources. **Always call this when done.**

### `browser_console(clear=False)`
Returns JavaScript console messages (log, warn, error, info) and uncaught exceptions.
- `clear=True` — clears the message buffer after reading

### `browser_get_images()`
Returns all images on the current page as a list of `{src, alt}` objects.

### `browser_vision(question, annotate=False)`
Takes a screenshot of the current page and analyzes it with vision AI.
- `question` — what you want to understand visually
- `annotate=True` — overlays numbered labels on interactive elements (useful for CAPTCHA or complex UI)
- Returns: text analysis + `screenshot_path`
- To share the screenshot in your response: `MEDIA:<screenshot_path>`

## Notes

- **Session isolation** — sessions are scoped per `task_id`, safe for parallel and concurrent jobs
- **Backend selection** — local Chromium by default; set `BROWSERBASE_API_KEY` for Browserbase (cloud + stealth) or `BROWSER_USE_API_KEY` for Browser Use
- **Bot detection** — some sites block headless browsers; use `browser_vision` to detect challenges and diagnose failures
- **Always close** — call `browser_close()` at the end of every task to avoid leaked sessions
