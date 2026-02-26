

# 🔧 Self Tool Builder — Hermes Extends Itself

**A skill that lets Hermes autonomously write, test, and register new Python tools. Zero human code required.**

## The Problem

Every time you need a new capability in Hermes, someone has to:
1. Find the right API
2. Write the tool file
3. Write tests
4. Register it in toolsets.py
5. Document it

**What if Hermes did all of that itself?**

## The Solution

`skills/self_tool_builder/SKILL.md` — a procedural memory document that chains Hermes's existing built-in tools into a full self-extension loop:

```
User: "I need a tool that checks HN top stories"
         ↓
Hermes: web_search("Hacker News API documentation")
         ↓
Hermes: write_file("tools/hackernews.py")
         ↓
Hermes: write_file("tests/test_hackernews.py")
         ↓
Hermes: terminal("python tests/test_hackernews.py")
         ↓  [if tests fail → patch → re-run]
Hermes: patch("toolsets.py", add "hackernews")
         ↓
Hermes: skill_manage(create "skills/hackernews/SKILL.md")
         ↓
User: ✅ New permanent tool installed
```

## What's Included

| File | Description |
|------|-------------|
| `skills/self_tool_builder/SKILL.md` | The core skill — 8-step autonomous tool creation loop |
| `tools/weather.py` | Demo: weather lookup tool (no API key, Open-Meteo) |
| `tests/test_weather.py` | 5 tests covering happy path + error cases |
| `docs/ADVENTURE.md` | Full adventure log |

## Quick Start

```bash
# 1. Install the skill
cp -r skills/self_tool_builder ~/.hermes/skills/

# 2. Run the demo tool tests
python tests/test_weather.py

# 3. Ask Hermes to build something new
hermes -q "I need a tool that monitors my website uptime"
```

## Why This Is Different

Most contributions add one integration. This adds the **meta-capability** to add any integration — turning Hermes into a truly self-extending 

