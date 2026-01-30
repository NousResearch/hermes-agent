# Tools

Tools are functions that extend the agent's capabilities. Each tool is defined with an OpenAI-compatible JSON schema and an async handler function.

## Tool Structure

Each tool module in `tools/` exports:
1. **Schema definitions** - OpenAI function-calling format
2. **Handler functions** - Async functions that execute the tool

```python
# Example: tools/web_tools.py

# Schema definition
WEB_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    }
}

# Handler function
async def web_search(query: str) -> dict:
    """Execute web search and return results."""
    # Implementation...
    return {"results": [...]}
```

## Tool Categories

| Category | Module | Tools |
|----------|--------|-------|
| **Web** | `web_tools.py` | `web_search`, `web_extract`, `web_crawl` |
| **Terminal** | `terminal_tool.py` | `terminal` (local/docker/singularity/modal backends) |
| **Browser** | `browser_tool.py` | `browser_navigate`, `browser_click`, `browser_type`, etc. |
| **Vision** | `vision_tools.py` | `vision_analyze` |
| **Image Gen** | `image_generation_tool.py` | `image_generate` |
| **Reasoning** | `mixture_of_agents_tool.py` | `mixture_of_agents` |
| **Skills** | `skills_tool.py` | `skills_categories`, `skills_list`, `skill_view` |

## Tool Registration

Tools are registered in `model_tools.py`:

```python
# model_tools.py
TOOL_SCHEMAS = [
    *WEB_TOOL_SCHEMAS,
    *TERMINAL_TOOL_SCHEMAS,
    *BROWSER_TOOL_SCHEMAS,
    # ...
]

TOOL_HANDLERS = {
    "web_search": web_search,
    "terminal": terminal_tool,
    "browser_navigate": browser_navigate,
    # ...
}
```

## Toolsets

Tools are grouped into **toolsets** for logical organization (see `toolsets.py`):

```python
TOOLSETS = {
    "web": {
        "description": "Web search and content extraction",
        "tools": ["web_search", "web_extract", "web_crawl"]
    },
    "terminal": {
        "description": "Command execution",
        "tools": ["terminal"]
    },
    # ...
}
```

## Adding a New Tool

1. Create handler function in `tools/your_tool.py`
2. Define JSON schema following OpenAI format
3. Register in `model_tools.py` (schemas and handlers)
4. Add to appropriate toolset in `toolsets.py`
5. Update `tools/__init__.py` exports

## Stateful Tools

Some tools maintain state across calls within a session:

- **Terminal**: Keeps container/sandbox running between commands
- **Browser**: Maintains browser session for multi-step navigation

State is managed per `task_id` and cleaned up automatically.

## Skills Tools (Progressive Disclosure)

Skills are on-demand knowledge documents. They use **progressive disclosure** to minimize tokens:

```
Level 0: skills_categories()     → ["mlops", "devops"]           (~50 tokens)
Level 1: skills_list(category)   → [{name, description}, ...]   (~3k tokens)
Level 2: skill_view(name)        → Full content + metadata       (varies)
Level 3: skill_view(name, path)  → Specific reference file       (varies)
```

Skill directory structure:
```
skills/
└── mlops/
    └── axolotl/
        ├── SKILL.md           # Main instructions (required)
        ├── references/        # Additional docs
        └── templates/         # Output formats, configs
```

SKILL.md uses YAML frontmatter:
```yaml
---
name: axolotl
description: Fine-tuning LLMs with Axolotl
tags: [Fine-Tuning, LoRA, DPO]
---
```
