"""Tool schemas for cdp-browser — what the LLM sees."""

CDP_LIST = {
    "name": "cdp_list",
    "description": (
        "List open browser tabs via CDP (default port 9333, Brave with "
        "--remote-debugging-port). Returns targetId | title | url per page tab. "
        "Use before cdp_run to pick a --tab target id."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "port": {
                "type": "integer",
                "description": "CDP port (default 9333)",
            },
        },
    },
}

CDP_RUN = {
    "name": "cdp_run",
    "description": (
        "Run a composed one-pass CDP steps script against one browser tab. "
        "Steps is a JSON array of ops executed in ONE websocket connection — "
        "no agent round-trip between steps (ego-lite 'code-base' pattern). "
        "Ops: open_tab url | navigate url | snapshot max | eval expr ret | "
        "focus sel | type text | click sel | click_coord x y | upload sel file | "
        "wait ms | capture out | close | echo msg. Returns per-step results. "
        "~26 semantic els on Gemini vs ~190 AX refs (7x cheaper snapshots)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "steps": {
                "type": "string",
                "description": (
                    "JSON array of step objects, e.g. "
                    '[{"op":"snapshot","max":15},{"op":"eval","expr":"document.title"}]'
                ),
            },
            "tab": {
                "type": "string",
                "description": (
                    "Target tab: 'auto' (gemini tab if present else first page), "
                    "'gemini', 'new' (open fresh tab), or exact targetId from cdp_list"
                ),
            },
            "port": {
                "type": "integer",
                "description": "CDP port (default 9333)",
            },
        },
        "required": ["steps"],
    },
}

CDP_SPACES = {
    "name": "cdp_spaces",
    "description": (
        "Run N named browser tabs concurrently, each with its own steps script "
        "(ego-lite 'spaces' pattern — parallel isolated contexts in one browser). "
        "spaces is a JSON object: {\"spaces\":[{\"name\":\"a\",\"tab\":\"new\","
        "\"url\":\"https://...\",\"steps\":[...]}, ...]}. tab: 'new' opens a "
        "fresh tab (url optional), 'gemini'/'auto'/targetId attach existing. "
        "Returns per-space step results."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "spaces": {
                "type": "string",
                "description": (
                    "JSON spaces spec, e.g. "
                    '{"spaces":[{"name":"a","tab":"new","url":"https://example.com","steps":[{"op":"snapshot"}]}]}'
                ),
            },
            "port": {
                "type": "integer",
                "description": "CDP port (default 9333)",
            },
        },
        "required": ["spaces"],
    },
}
