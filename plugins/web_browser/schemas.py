"""JSON schemas + Hebrew operator labels for the web_browser plugin tools.

All ten tools share two extra fields beyond the standard OpenAI function-tool
schema:

* ``label_he`` — Hebrew label used by the Hermes /api/tools registry surface
  per [[hermes-agent-202606-001]]. Operator-facing UI reads
  this verbatim.
* ``category`` — coarse grouping (always ``"web"`` for this plugin).
"""

from __future__ import annotations

from typing import Any, Dict

CATEGORY = "web"


def _tool(name: str, label_he: str, description: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Build a tool definition with the extra Hermes label_he + category fields."""
    return {
        "name": name,
        "label_he": label_he,
        "category": CATEGORY,
        "description": description,
        "parameters": params,
    }


BROWSER_NAVIGATE_SCHEMA = _tool(
    "browser_navigate",
    "ניווט באתר",
    "Open a URL in the headless agent-browser session. Returns the page title, "
    "final URL after redirects, and a compact accessibility snapshot with @e refs "
    "for use with browser_click / browser_type. Wraps `agent-browser open <url>`.",
    {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Full URL to open (https:// or http://)."},  # windows-footgun: ok  (string literal, not an open() call)
            "task": {
                "type": "string",
                "description": "Optional natural-language description of what the agent wants to accomplish; passed to `--task` so cloud browser providers can show it in their dashboards.",
            },
            "basic_auth": {
                "type": "string",
                "description": "Optional HTTP basic-auth credential in `user:password` form (for protected staging sites).",
            },
        },
        "required": ["url"],
        "additionalProperties": False,
    },
)

BROWSER_SCREENSHOT_SCHEMA = _tool(
    "browser_screenshot",
    "צילום מסך",
    "Take a PNG screenshot of the current page (or of a single selector). Returns "
    "the file path on disk; the agent can attach it back to the conversation. "
    "Wraps `agent-browser screenshot`.",
    {
        "type": "object",
        "properties": {
            "selector": {
                "type": "string",
                "description": "Optional CSS selector or @ref. When set, screenshot only that element; otherwise full viewport.",
            },
            "path": {
                "type": "string",
                "description": "Optional output file path. Default: agent-browser picks a temp path and returns it.",
            },
        },
        "additionalProperties": False,
    },
)

BROWSER_SNAPSHOT_SCHEMA = _tool(
    "browser_snapshot",
    "תמונת מצב נגישות",
    "Return the accessibility-tree snapshot of the current page with @e refs "
    "(e.g. @e5) so the agent can click / type via those refs. Wraps "
    "`agent-browser snapshot`.",
    {
        "type": "object",
        "properties": {
            "full": {
                "type": "boolean",
                "description": "Default false (compact, interactive elements only). Set true for the complete page tree.",
            },
        },
        "additionalProperties": False,
    },
)

BROWSER_CLICK_SCHEMA = _tool(
    "browser_click",
    "לחיצה",
    "Click an element by CSS selector or @ref from a recent snapshot. Wraps "
    "`agent-browser click <selector>`.",
    {
        "type": "object",
        "properties": {
            "selector": {"type": "string", "description": "CSS selector or @ref (e.g. '@e5')."},
        },
        "required": ["selector"],
        "additionalProperties": False,
    },
)

BROWSER_FILL_SCHEMA = _tool(
    "browser_fill",
    "מילוי שדה",
    "Clear an input and fill it with `value`. Wraps `agent-browser fill "
    "<selector> <value>`.",
    {
        "type": "object",
        "properties": {
            "selector": {"type": "string", "description": "CSS selector or @ref of the input."},
            "value": {"type": "string", "description": "Text to fill in."},
        },
        "required": ["selector", "value"],
        "additionalProperties": False,
    },
)

BROWSER_TYPE_SCHEMA = _tool(
    "browser_type",
    "הקלדה",
    "Type text into a focused input without clearing first (appends). Wraps "
    "`agent-browser type <selector> <text>`.",
    {
        "type": "object",
        "properties": {
            "selector": {"type": "string", "description": "CSS selector or @ref."},
            "text": {"type": "string", "description": "Text to type."},
        },
        "required": ["selector", "text"],
        "additionalProperties": False,
    },
)

BROWSER_PRESS_SCHEMA = _tool(
    "browser_press",
    "לחיצת מקש",
    "Press a keyboard key (Enter, Tab, Escape, Control+a, etc.). Wraps "
    "`agent-browser press <key>`.",
    {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "Key name or chord. E.g. 'Enter', 'Tab', 'Control+a'."},
        },
        "required": ["key"],
        "additionalProperties": False,
    },
)

BROWSER_GET_SCHEMA = _tool(
    "browser_get",
    "חילוץ מידע",
    "Extract data from the page — supported fields: text, html, value, title, "
    "url, count, box, styles, attr. Wraps `agent-browser get <what> [selector]`.",
    {
        "type": "object",
        "properties": {
            "what": {
                "type": "string",
                "enum": ["text", "html", "value", "title", "url", "count", "box", "styles", "attr"],
                "description": "What to read from the page.",
            },
            "selector": {"type": "string", "description": "Optional CSS selector or @ref; omit for full-page where applicable."},
            "attr": {"type": "string", "description": "Only used when what='attr' — attribute name to read."},
        },
        "required": ["what"],
        "additionalProperties": False,
    },
)

BROWSER_FIND_SCHEMA = _tool(
    "browser_find",
    "חיפוש אלמנט",
    "Find an element by role / text / label / placeholder / alt / title / testid "
    "and optionally act on it. Wraps `agent-browser find <locator> <value> "
    "<action> [text]`.",
    {
        "type": "object",
        "properties": {
            "locator": {
                "type": "string",
                "enum": ["role", "text", "label", "placeholder", "alt", "title", "testid", "first", "last", "nth"],
                "description": "Locator strategy.",
            },
            "value": {"type": "string", "description": "Locator value (e.g. role name, text content, testid)."},
            "action": {
                "type": "string",
                "enum": ["click", "type", "fill", "press", "get-text", "exists"],
                "description": "What to do with the matched element. Default: get-text.",
            },
            "text": {"type": "string", "description": "Extra arg for action=type/fill/press."},
        },
        "required": ["locator", "value"],
        "additionalProperties": False,
    },
)

BROWSER_CLOSE_SCHEMA = _tool(
    "browser_close",
    "סגירת דפדפן",
    "Close the active agent-browser session (or every session with all=true). "
    "Wraps `agent-browser close`.",
    {
        "type": "object",
        "properties": {
            "all": {"type": "boolean", "description": "When true, closes every active session."},
        },
        "additionalProperties": False,
    },
)
