# neuro-skill-router

Route 332+ skills through 5-signal Reciprocal Rank Fusion — the LLM only
sees the top-3 most relevant skills for each query. Runs as a Hermes
`pre_llm_call` hook: the router fires *before* the LLM prompt is assembled,
so routing is deterministic — not dependent on the model choosing to invoke
a tool.

## Why This Plugin Exists

Hermes with 332 skills has a discovery problem: the LLM sees 332 names
+ descriptions and must guess which one(s) to use. With 56% of skill
invocations being skipped (Vercel, Jan 2026), most installed skills are
never activated.

neuro-skill-router solves this at the **host level** — not by asking the
LLM to call a routing tool (MCP), but by injecting only the top-3 skills
before the LLM ever sees the query. The LLM never sees the other 329.

## Architecture

```
User query
  |
Hermes Agent (receive message)
  |
[ pre_llm_call hook ] ← neuro-skill-router
  +-- router.query(user_message, top_k=3)    (6-10ms, 4-signal RRF)
  +-- return {"context": "[Top 3 skills]"}    ← Hermes injects this
  |
LLM sees [Top 3 skills for this query:] + original message
  |
LLM reasons within the 3-skill context
```

## Installation

**Prerequisite**: neuro-skill Python package installed.

```bash
pip install neuro-skill
neuro-skill hermes install
```

Then restart Hermes. The plugin is auto-discovered from
`HERMES_HOME/plugins/neuro-skill-router/`.

## Verified Configuration

| Component | Detail |
|-----------|--------|
| Hermes | v0.16.0 (Windows Desktop GUI) |
| Model | DeepSeek V4 Flash |
| Skills | 332 (ECC skill set) |
| Plugin hooks | `on_session_start` + `pre_llm_call` |
| Routing | 4-signal RRF: BM25 + Cosine + Graph + CF |
| Latency | 6-10ms per query, 2.3s one-time startup at session start |

## Skills Directory Auto-Detection

The plugin scans both **Claude Code** and **Hermes** skill directories:

```
~/.claude/skills, ~/.claude/agents, ~/.claude/.agents/skills
~/.hermes/skills, %LOCALAPPDATA%/hermes/skills
%LOCALAPPDATA%/hermes-agent/skills, $HERMES_HOME/skills
```

## Manual Route Injection

When the router can't find a clear match, you can still manually invoke:

```bash
# Query via CLI
neuro-skill query "check Python code for SQL injection"

# Add a priority rule
neuro-skill rule add "cs.*安全" "csharp-reviewer"

# Fuzzy correction — no need to remember exact skill names
neuro-skill correct "检索cs安全检查" "C# 代码审查工具"
```

## Links

- **Repository**: https://github.com/wuykjl/neuro-skill
- **PyPI**: `pip install neuro-skill`
- **Tests**: 102 passed, 1.8s
