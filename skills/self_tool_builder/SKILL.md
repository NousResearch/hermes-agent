---
name: self_tool_builder
version: 1.0.0
description: Teaches Hermes how to research, write, test, and register its own new tools autonomously — zero human code required.
triggers:
  - "write me a tool for"
  - "create a new tool that"
  - "I need a tool to"
  - "build a tool for"
author: community
tags: [meta, self-extending, tool-generation, autonomous]
---

# Self Tool Builder Skill

This skill enables Hermes to autonomously extend its own capabilities by writing new Python tools from scratch, testing them, and registering them — without any human writing code.

## When to Use This Skill

Use this skill when the user says things like:
- "I need a tool that can do X"
- "Write me a tool for Y"
- "Can you add Z capability to yourself?"

---

## Step-by-Step Process

### Step 1 — Understand the Requirement
Before writing anything, clarify:
- What inputs does the tool need?
- What should it return?
- Does it require an external API? If so, is there a key available?
- Are there similar existing tools to avoid duplication?

Use `skills_list` to check for existing tools first.

### Step 2 — Research the API / Approach
Use `web_search` and `web_extract` to:
- Find the best API or library for the job
- Get example code and documentation
- Identify rate limits, auth requirements, error formats

### Step 3 — Write the Tool File
Create the tool at `tools/{tool_name}.py` using `write_file`.

**Tool Template:**
```python
"""
{tool_name} — {one line description}
Generated autonomously by Hermes Self Tool Builder skill.
"""

import os
import httpx
from typing import Any

TOOL_NAME = "{tool_name}"
TOOL_DESCRIPTION = "{clear description of what this tool does}"
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "param1": {
            "type": "string",
            "description": "Description of param1"
        }
    },
    "required": ["param1"]
}


async def run(param1: str, **kwargs) -> dict[str, Any]:
    try:
        api_key = os.getenv("{API_KEY_ENV_VAR}", "")
        if not api_key:
            return {"success": False, "error": "Missing {API_KEY_ENV_VAR} in environment"}

        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(
                "https://api.example.com/endpoint",
                params={"q": param1},
                headers={"Authorization": f"Bearer {api_key}"}
            )
            response.raise_for_status()
            data = response.json()

        return {"success": True, "result": data}

    except httpx.HTTPStatusError as e:
        return {"success": False, "error": f"HTTP {e.response.status_code}: {e.response.text}"}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

### Step 4 — Write the Test File
Create `tests/test_{tool_name}.py` using `write_file`.

### Step 5 — Run the Tests
Use `terminal` to execute:
```bash
python tests/test_{tool_name}.py
```
If tests fail — fix with `patch`, re-run until green.

### Step 6 — Register the Tool
Use `patch` to add the tool name to `toolsets.py`.

### Step 7 — Save a Skill for This Tool
Use `skill_manage` to create `skills/{tool_name}/SKILL.md`.

### Step 8 — Report to User
Tell the user what tool was created, what file it lives in, and what env var is needed.

---

## Important Rules

- Never hardcode API keys — always use `os.getenv()`
- Always handle exceptions — return `{"success": False, "error": "..."}` not raw exceptions
- Use `httpx` for HTTP — it's already available in Hermes
- Test before registering — broken tools hurt the agent
- Keep tools focused — one tool, one job

---

## Example Session

**User:** "I need a tool that can look up the weather for any city"

**Hermes does:**
1. Searches for free weather APIs → finds Open-Meteo (no key needed!)
2. Writes `tools/weather.py`
3. Writes `tests/test_weather.py`
4. Runs tests → passes
5. Adds `"weather"` to toolsets.py
6. Creates `skills/weather/SKILL.md`
7. Reports: "Done! Try: 'What's the weather in Istanbul?'"
