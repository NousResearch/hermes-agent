---
name: visual-web-research
description: Leverages the Playwright MCP server to autonomously browse the web, bypass paywalls visually, extract dynamic charts, and navigate complex JS-heavy financial/academic terminals (e.g., Bloomberg, Interactive Brokers, complex PDF readers).
version: 1.0.0
author: Hermes Agent
license: MIT
dependencies: [mcp-playwright]
metadata:
  hermes:
    tags: [Research, Browser, Playwright, MCP, Vision]
    related_skills: [source-discovery]
---

# Visual Web Research (Playwright MCP)

Some research requires direct browser navigation. Standard text scrapers (`web_extract`) often fail on:
1. Paywalled academic sites that require clicking "Accept Cookies" or waiting for JS redirects.
2. Financial charts (TradingView, Bloomberg) that render dynamically on canvas.
3. Interactive interactive data visualizations.

By utilizing the existing `.playwright-mcp` service, you can direct Hermes to visually navigate these pages.

## Usage

Ask Hermes:
- "Open the Federal Reserve's interactive data portal, navigate to the M2 money supply chart, and take a screenshot."
- "Go to the interactive Nature article on AlphaFold3. Extract the key chart parameters visually and save to my research notes."
- "Log into my active TradingView session and summarize the 4-hour MACD trend for gold."

## Core Capabilities (Via Playwright)

- **Execution:** Full Chromium/JS execution capability.
- **Vision Models:** Can take DOM screenshots and use the `Vision` agentic capabilities to interpret charts that cannot be OCR'd.
- **Session Re-use:** Re-uses an active session for persisting logins.

## Under the Hood

To activate this for a complex deep-dive, Hermes simply switches tools from `web_extract` over to the `mcp-playwright` tool array defined in your workspace root (`.playwright-mcp`). 

*Note: For large-scale batch academic API scraping, prefer `arxiv` or Semantic Scholar. Use Visual Web Research specifically for visual-only or interaction-heavy targets.*
