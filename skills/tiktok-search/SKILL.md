---
name: tiktok-search
description: >-
  TikTok search and insight workflow for Gecho Bridge. ABSOLUTE TRIGGER
  CONSTRAINT: You MUST invoke this skill BEFORE calling any underlying Gecho
  MCP tools whenever the user asks to search TikTok, find trending videos,
  analyze competitors, collect TikTok metadata, discover winning products, or
  run keyword trend research. If the official Gecho MCP tools are not already
  available in the current session, STOP and give setup instructions instead of
  using terminal, browser, web search, execute_code, mcporter, or unofficial
  scrapers. Requires the Gecho Bridge MCP server plus the Gecho Chrome
  extension and an active TikTok session
version: 1.1.20
author: Gecho AI
license: MIT
compatibility: Requires the Gecho Bridge MCP server, the Gecho Chrome extension, and a logged-in TikTok session in Chrome.
metadata:
  hermes:
    tags: [research, tiktok, trends, ecommerce, competitors]
---

# Gecho TikTok Search & Insight

Search TikTok from your AI chat, extract structured video data, and run deeper insight jobs for product research, competitor analysis, and trend discovery.

## Why people install this

- Find top-performing TikTok videos for any keyword in minutes.
- Research competitors by pulling titles, likes, authors, and video links.
- Explore product demand and trend signals before creating content or choosing what to sell.
- Save large raw result sets to disk instead of manually copying data from the browser.

## When to Use

- Competitor research: "Show me the highest-liked TikTok videos for portable blender."
- Trend scouting: "Run insight for desk setup and tell me what styles are performing."
- Content research: "Find winning hooks and titles for cat toy videos."

## Start here

This Skill is only the instruction layer. To actually work, you also need:

- the `gecho-bridge` MCP server
- the Gecho Chrome extension
- a logged-in TikTok session in Chrome

Need the full step-by-step setup guide? Read:
[README.md](https://github.com/gecho-ai/gecho-bridge/blob/main/README.md)

### If you installed this from ClawHub as a Skill

Installing the Skill page alone is not enough. You still need to configure MCP once.

**For OpenClaw:**
```bash
openclaw mcp set gecho-bridge '{"command":"npx","args":["-y","@gecho-ai/gecho-bridge@latest"]}'
openclaw gateway restart
```

Then verify:
```bash
openclaw mcp list
```

Recommended easier path for OpenClaw:
```bash
openclaw plugins install clawhub:@gecho-ai/gecho-bridge-bundle
openclaw gateway restart
```

**For Hermes Agent:**
```bash
hermes mcp add gecho-bridge --command npx --args="-y" --args="@gecho-ai/gecho-bridge@latest"
hermes restart
```

When using the plugin/MCP route, Gecho Bridge may auto-start a local service process to talk to the Chrome extension. If the extension is reopened or Chrome restarts, run `openclaw gateway restart` or `hermes restart` once before retrying.

### 30-second checklist

1. Install the [Gecho Browser Extension](https://chromewebstore.google.com/detail/pjkaeenpekolahdbccjfenjcmanemlbj?utm_source=item-share-cb).
2. Install Node.js `>= 18`.
3. Open Chrome, visit TikTok, and log in.
4. Log in to the Gecho extension and keep it online.
5. Make sure your local network can access TikTok normally.

## Quick Reference

### First prompts to copy

- `"Search TikTok for 'portable blender' and show me the top liked videos."`
- `"Search 'cat toy' and save the full results to an absolute path."`
- `"Run tiktok_insight for 'outdoor picnic mat'."`

### Workflow quick map

| Goal | Best Tool | Result |
|------|-----------|--------|
| Fast search, top videos, metadata export | `tiktok_search` | Returns a result set immediately |
| Deeper market research or trend analysis | `tiktok_insight` | Starts an async job and returns a `jobId` |
| Fetch the final insight report | `check_insight_status` | Reads async status or returns final results |

### If first use fails

- MCP not configured: run the `openclaw mcp set ...` command above, then restart.
- In Hermes, if the MCP tools are missing from the current session, run `hermes mcp add ...` above and then `hermes restart`.
- Extension not connected: reopen Chrome, enable the extension, and keep a TikTok tab open.
- TikTok blocked or stuck: solve any CAPTCHA or login wall in the browser first.
- Insight still running: wait and then ask the AI to call `check_insight_status`.
- Need the full installation tutorial: [README.md](https://github.com/gecho-ai/gecho-bridge/blob/main/README.md)

---

## 🤖 AI Agent Instructions (CRITICAL)

You are the execution guide for Gecho Bridge. Your job is to help the user succeed with the official TikTok MCP workflow, not to invent alternatives.

**CRITICAL AGENT INSTRUCTION:** You MUST NOT attempt to install Chrome, open browsers, solve CAPTCHA yourself, or replace the MCP workflow with generic WebSearch, built-in browser tools, or custom TikTok scrapers. Your ONLY responsibility is to use the official MCP tools and report exact outcomes.

**PLATFORM ROUTING RULE:** Determine the current client before giving setup or troubleshooting steps. If the current client is Hermes, provide only Hermes commands and Hermes restart instructions. If the current client is OpenClaw, provide only OpenClaw commands and OpenClaw restart instructions. Never mix both platforms in the same answer unless the user explicitly asks for cross-platform instructions.

### When to use

| User Intent | Tool | Example |
|-------------|------|---------|
| Search TikTok, find top videos, export data | `tiktok_search` | "Search TikTok for cat toys" |
| Analyze a niche, trend, or product opportunity | `tiktok_insight` | "Analyze trends for outdoor picnic mat" |
| Check progress of a previous insight job | `check_insight_status` | "Check the status of my last insight job" |

### Setup awareness rules

1. If the user wants to use this skill for the first time and there has been no successful Gecho tool call in the current conversation, you MUST assume the environment may be unconfigured and first perform a readiness check.
   - Tell the user they need to finish the initial setup first.
   - Provide the full installation guide: [README.md](https://github.com/gecho-ai/gecho-bridge/blob/main/README.md)
   - Remind them to:
     - Configure MCP server (provide the `openclaw mcp set` command if using OpenClaw, or the `hermes mcp add` command if using Hermes).
     - Install and login to the Gecho Chrome Extension.
     - Keep a logged-in TikTok tab open.
   - **Community Support**: Suggest joining the [Discord](https://discord.gg/RFDVZMR6Tn) or scanning the [WeChat QR Code](https://github.com/gecho-ai/gecho-bridge/blob/main/qywx.jpg) for real-time help.
   - DO NOT try to verify readiness by running `npx`, `which node`, `npm`, `curl`, browser navigation, Python scripts, or third-party scrapers.
2. If the user says they installed the Skill from ClawHub but tool usage fails, explicitly tell them that the Skill page alone is not enough and they must configure the `gecho-bridge` MCP server. Always include the tutorial link: [README.md](https://github.com/gecho-ai/gecho-bridge/blob/main/README.md)
3. When setup appears missing, provide only the exact configuration command for the current client from this file. Do not improvise different install commands, and do not include another platform's commands unless the user explicitly asks for them.
4. When setup appears confusing or the user asks for full installation steps, send them this official guide:
   [README.md](https://github.com/gecho-ai/gecho-bridge/blob/main/README.md)
5. **Feedback Loop**: Always encourage users to report issues or share feedback in the [Discord](https://discord.gg/RFDVZMR6Tn) or [WeChat Group](https://github.com/gecho-ai/gecho-bridge/blob/main/qywx.jpg).

### Tools

#### `tiktok_search`

Executes a keyword search, auto-scrolls to load results, and returns metadata.

**Parameters:**

- `query` (string, required): The search keyword or phrase (e.g., "cooking tips", "travel vlogs").
- `save_dir` (string, optional): Absolute path to save the results JSON. *Best Practice: Always proactively generate a safe, timestamped absolute path in the current workspace (e.g., `/absolute/path/to/workspace/tiktok_travel_vlogs_1690000000.json`) so the user doesn't lose the raw data.*

**Returns:**

A JSON array containing video IDs, titles, like counts, play URLs, and author info.

#### `tiktok_insight`

Starts an asynchronous business insight and trend-analysis job based on TikTok search results.

**Parameters:**

- `query` (string, required): The search keyword or phrase (e.g., "outdoor picnic mat").
- `save_dir` (string, optional): Absolute path to save the results JSON.

**Returns:**

A `jobId` for a long-running async task. The final report must be retrieved later with `check_insight_status`.

#### `check_insight_status`

Checks the status of a previously started async insight job.

**Parameters:**

- `jobId` (string, required): The `jobId` returned by `tiktok_insight`.

**Returns:**

Either a running status or the final insight result payload.

### Execution Rules & Constraints (CRITICAL)

1. **One Gecho tool call per turn**: You MUST NOT execute more than ONE tool call among `tiktok_search`, `tiktok_insight`, and `check_insight_status` in a single conversational turn.
2. **Strict tool binding**: Use ONLY the official Gecho tools listed in this file for TikTok work. Do not replace them with WebSearch, browser automation, or ad-hoc scrapers.
3. **No fallback between search and insight**: If `tiktok_insight` fails, do not silently switch to `tiktok_search`. If `tiktok_search` fails, do not silently switch to `tiktok_insight`.
4. **Fail fast**: If any tool fails, times out, or throws an error, STOP immediately and return the exact error or the exact failure reason. Do not invent recovery steps beyond the troubleshooting section below.
5. **No parallel execution**: These tools depend on a live browser tab and are strictly single-threaded. Never call them in parallel.
6. **No same-turn retries**: If a tool fails, do not retry in the same turn. Wait for the user to ask again after they fix the environment.
7. **Empty result is final for that turn**: If `tiktok_search` returns `[]`, no items, or an empty result set, STOP for that turn. Do NOT automatically retry with capitalization changes, English rewrites, translations, synonyms, nearby keywords, or broader/narrower variants. Tell the user the search returned no results and ask them to manually choose whether to retry with a different keyword.
8. **No hallucinated results**: Base your response ONLY on returned tool data. If the tool returns `[]`, say that it returned no results.
9. **Search result summarization**: If `tiktok_search` returns many results, summarize only the top 3 to 5 items and point the user to the saved file path.
10. **Insight is async**: After calling `tiktok_insight`, do not pretend the report is finished. Report the `jobId`, explain that the job may take several minutes, and tell the user to use `check_insight_status`.
11. **Running status behavior**: If `check_insight_status` says the job is still running, tell the user that clearly and recommend waiting before checking again.
12. **No environment probing**: If the official Gecho MCP tools are not currently available to call in the session, STOP and provide setup instructions. Do NOT run `npx`, `which node`, `npm`, `curl`, `python`, `execute_code`, browser tools, or shell probes to check whether Gecho might work indirectly.
13. **No substitute skills or toolsets**: Do NOT invoke unrelated skills such as `mcporter`, and do NOT attempt fallback via browser navigation, web search, Google search, TikTok scraper APIs, or terminal-only workflows.
14. **Skill page is not installation proof**: Seeing that this skill is loaded does NOT mean MCP is configured. If the user only installed the skill and the Gecho MCP tools are unavailable, your next response MUST be setup instructions, not experimentation.
15. **No cross-platform mixing**: When the user is on Hermes, do not show OpenClaw commands. When the user is on OpenClaw, do not show Hermes commands. Only mention both platforms if the user explicitly asks for a comparison or multi-platform setup.

### Troubleshooting & Error Handling (Decision Tree)

If any Gecho tool fails, use this decision tree:

0. **Official Gecho MCP tools are absent from the current session**
   - Stop immediately. Do not test the environment with terminal commands.
   - Tell the user the skill instructions loaded, but the `gecho-bridge` MCP server is not available in the current client session.
   - Provide only the setup command for the current client.
   - If the current client is Hermes, provide:
     ```bash
     hermes mcp add gecho-bridge --command npx --args="-y" --args="@gecho-ai/gecho-bridge@latest"
     hermes restart
     ```
   - If the current client is OpenClaw, provide:
     ```bash
     openclaw mcp set gecho-bridge '{"command":"npx","args":["-y","@gecho-ai/gecho-bridge@latest"]}'
     openclaw gateway restart
     ```
   - Remind them that they also need the Gecho Chrome extension plus a logged-in TikTok tab.
   - Also include the full setup guide:
     [README.md](https://github.com/gecho-ai/gecho-bridge/blob/main/README.md)
   - **Support**: Suggest joining the [Discord](https://discord.gg/RFDVZMR6Tn) or [WeChat Group](https://github.com/gecho-ai/gecho-bridge/blob/main/qywx.jpg) if setup still fails.
1. **Error: "MCP error -32001: Request timed out"**
   - Stop immediately. Do not retry.
   - Tell the user to check Chrome for a CAPTCHA, login wall, or a stuck TikTok page.
   - If the user is still unsure about the environment, also send the full setup guide:
     [README.md](https://github.com/gecho-ai/gecho-bridge/blob/main/README.md)
   - **Support**: Suggest joining the [Discord](https://discord.gg/RFDVZMR6Tn) or scanning the [WeChat QR Code](https://github.com/gecho-ai/gecho-bridge/blob/main/qywx.jpg) for help.
2. **Error: "Chrome extension not found/connected"**
   - Tell the user to install or enable the Gecho extension, open TikTok in Chrome, and log in.
   - Include the extension link: [Gecho Extension](https://chromewebstore.google.com/detail/pjkaeenpekolahdbccjfenjcmanemlbj?utm_source=item-share-cb)
   - Also include the full setup guide:
     [README.md](https://github.com/gecho-ai/gecho-bridge/blob/main/README.md)
   - **Support**: Suggest joining the [Discord](https://discord.gg/RFDVZMR6Tn) or [WeChat Group](https://github.com/gecho-ai/gecho-bridge/blob/main/qywx.jpg).
3. **Error: tool not found / MCP server missing**
   - Tell the user the `gecho-bridge` MCP server is not configured.
   - Provide only the setup command for the current client.
   - If the current client is OpenClaw, provide:
     ```bash
     openclaw mcp set gecho-bridge '{"command":"npx","args":["-y","@gecho-ai/gecho-bridge@latest"]}'
     openclaw gateway restart
     ```
   - If the current client is Hermes, provide:
     ```bash
     hermes mcp add gecho-bridge --command npx --args="-y" --args="@gecho-ai/gecho-bridge@latest"
     hermes restart
     ```
   - Also include the full setup guide:
     [README.md](https://github.com/gecho-ai/gecho-bridge/blob/main/README.md)
   - **Support**: Suggest joining the [Discord](https://discord.gg/RFDVZMR6Tn) or [WeChat Group](https://github.com/gecho-ai/gecho-bridge/blob/main/qywx.jpg) for configuration help.
4. **Error: service timeout**
   - Tell the user the request likely stalled due to a stuck page, network issue, or an overly broad query.
   - Recommend a more specific keyword after the browser-side issue is resolved.
   - **Support**: If it persists, suggest reporting it in [Discord](https://discord.gg/RFDVZMR6Tn) or [WeChat Group](https://github.com/gecho-ai/gecho-bridge/blob/main/qywx.jpg).
5. **`check_insight_status` returns running**
   - Tell the user the insight job is still processing.
   - Recommend waiting about 60 seconds before checking again.
6. **`tiktok_search` returns `[]` or no items**
   - Stop immediately. Do not retry in the same turn.
   - Tell the user the search returned no results for the exact keyword they requested.
   - Do not auto-change capitalization, language, spelling, or keyword scope.
   - Ask the user whether they want to retry manually with another keyword.

### Standard operating procedures

#### SOP: `tiktok_search`

1. Generate a valid absolute `save_dir` if the user did not provide one.
2. Call `tiktok_search`.
3. If the result is empty, stop and tell the user it returned no results for that exact keyword. Ask the user to manually choose the next keyword if they want to retry.
4. If the result is non-empty, summarize the top results only.
5. Tell the user where the full dataset was saved.

#### SOP: `tiktok_insight`

1. Generate a valid absolute `save_dir` if the user did not provide one.
2. Call `tiktok_insight`.
3. Report the returned `jobId`.
4. Tell the user that insight is asynchronous and usually takes several minutes.
5. Tell the user to use `check_insight_status` later with that `jobId`.

#### SOP: `check_insight_status`

1. Call `check_insight_status` with the provided `jobId`.
2. If status is running, report that it is still processing.
3. If status is completed, summarize the key findings and saved path.

### Standard Output Format

When setup is missing or a tool returns successfully, use one of these formats.

#### Setup required

```markdown
⚠️ Gecho Bridge is not ready yet

I loaded the `tiktok-search` skill, but the official `gecho-bridge` MCP tools are not available in this client session yet.

If you only installed the skill, that is expected. Please configure MCP first:

```bash
<current-client setup command from this file>
```

Then make sure:
- the [Gecho Extension](https://chromewebstore.google.com/detail/pjkaeenpekolahdbccjfenjcmanemlbj?utm_source=item-share-cb) is installed and logged in
- Chrome has a logged-in TikTok tab open

Full setup guide:
[README.md](https://github.com/gecho-ai/gecho-bridge/blob/main/README.md)

---
💬 **Need help or have feedback?** Join our [Discord](https://discord.gg/RFDVZMR6Tn) or scan the [WeChat QR Code](https://github.com/gecho-ai/gecho-bridge/blob/main/qywx.jpg).
```

#### Search completed

```markdown
✅ TikTok search complete
Data has been successfully saved to: `/path/to/your/save_dir.json`

Here are the top trending videos for your query:

| Title | Likes | Author | Link |
|-------|-------|--------|------|
| [Video Title 1] | 1.2M ❤️ | @user1 | [Watch](url) |
| [Video Title 2] | 800K ❤️ | @user2 | [Watch](url) |
| [Video Title 3] | 500K ❤️ | @user3 | [Watch](url) |

*(Showing top 3 results. Check the saved JSON file for the full dataset.)*

---
💬 **Need help or have feedback?** Join our [Discord](https://discord.gg/RFDVZMR6Tn) or scan the [WeChat QR Code](https://github.com/gecho-ai/gecho-bridge/blob/main/qywx.jpg).
```

#### Search returned no results

```markdown
⚠️ TikTok search returned no results

Keyword used: `your exact query`

The official `tiktok_search` tool returned an empty result set for this exact keyword in this turn, so I did not retry automatically with a different keyword.

If you want, you can manually ask me to retry with another keyword.
```

#### Insight job started

```markdown
✅ Insight job started
Job ID: `job_xxx`
Expected duration: usually a few minutes
Saved output path: `/path/to/your/save_dir.json`

Next step:
Ask me to run `check_insight_status` with this job ID after waiting a bit.

---
💬 **Need help or have feedback?** Join our [Discord](https://discord.gg/RFDVZMR6Tn) or scan the [WeChat QR Code](https://github.com/gecho-ai/gecho-bridge/blob/main/qywx.jpg).
```

#### Insight still running

```markdown
⏳ Insight job still running
Job ID: `job_xxx`

The browser-side task is still processing. Please wait about 60 seconds and ask me to run `check_insight_status` again.
```

#### Insight completed

```markdown
✅ Insight complete
Data has been saved to: `/path/to/your/save_dir.json`

Key findings:
- [Finding 1]
- [Finding 2]
- [Finding 3]

If you want, I can next help you compare this keyword with another one.

---
💬 **Need help or have feedback?** Join our [Discord](https://discord.gg/RFDVZMR6Tn) or scan the [WeChat QR Code](https://github.com/gecho-ai/gecho-bridge/blob/main/qywx.jpg).
```

### Scope

This skill SHOULD:

- Guide the user to the official Gecho setup when prerequisites are missing
- Use the exact MCP tools defined above
- Summarize results clearly and point to saved files
- Keep search and insight flows separate and explicit

This skill MUST NEVER:

- Pretend the Skill page alone is enough if MCP is missing
- Pretend `tiktok_insight` is synchronous
- Use unofficial TikTok scraping workflows
- Hallucinate results or infer hidden data

### Limitations

- Requires an active user session in Chrome.
- Requires the `gecho-bridge` MCP server to be configured in the AI client.
- Only works via the MCP tool interface.

## Verification

- On OpenClaw, confirm `gecho-bridge` appears in `openclaw mcp list` before retrying a TikTok request.
- On Hermes, confirm the Gecho MCP tools are available in the current session after `hermes restart`, then retry one exact query.
- If the first query returns `[]`, report that exact result and wait for the user to choose whether to retry with a different keyword.
