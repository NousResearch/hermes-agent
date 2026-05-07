# Hermes CodeAct Mode — Design Specification

**Status:** Draft v0.2 — All design questions resolved  
**Date:** 2026-05-04  
**Branch context:** feat/qwen-aware-compaction  

> All seven open design questions (Q-1 through Q-7) have been answered and are now incorporated 
> as confirmed decisions throughout this document. The spec is ready for Phase 1 implementation.

---

## 1. Executive Summary

This document specifies the design for **Hermes CodeAct Mode**: a replacement primary tool-calling 
system that gives the agent a single persistent Python interpreter as its action space, instead of 
the current 74-tool JSON schema catalogue. The agent writes Python code that calls pre-defined tool 
functions; the interpreter executes it, and results flow back into context.

The primary motivation is small-model reliability. Qwen3.6 27B/35B and Gemma4 26B/31B consistently 
fail at multi-tool JSON calling due to schema fidelity requirements. CodeAct reduces tool invocation 
to Python function calls — a syntax these models have seen in abundance during pre-training and 
handle well. Secondary benefits include within-session state persistence (variables survive across 
turns), natural multi-tool composition (loops, conditionals, data pipelines in one block), and a 
direct integration path for Hermes's skill system to create "learned tools."

---

## 2. Goals and Non-Goals

### Goals

- Replace the 74-schema multi-tool API call with a single `run(code, stdin?)` tool for all 
  provider/model routes that explicitly advertise CodeAct capability
- Maintain a **persistent Python kernel per agent session** so variables defined in turn N are 
  available in turn N+1
- Expose **all registered Hermes tools** as callable Python functions in the kernel namespace 
  (not just the current 7)
- Provide a **lazy help system** so the model can query tool signatures and docstrings on demand 
  without loading all docs upfront
- Add a **structured output envelope** (`{"thoughts": "...", "code": "..."}`) for provider/model
  routes that advertise that capability
- Wire **skills as injectable Python functions**: relevant skills are loaded into the kernel 
  namespace at session start and can be called directly
- Provide a **skill promotion pipeline**: the autonomous curator can extract agent-generated 
  helper functions from session trajectories and promote them to persistent skills
- Maintain full backward compatibility: multi-tool JSON calling remains available and is the 
  default for provider/model routes that don't opt in

### Non-Goals

- Replacing or removing the existing `execute_code` tool (it becomes one of the tools callable 
  from within a CodeAct block, not the mechanism itself)
- Supporting `pip install` without sandboxing — pip install is enabled by default (see §5.5)
- Changing how trajectory compression, Atropos RL, or batch runner work (CodeAct trajectories 
  are compatible with existing compression)
- Modifying the messaging gateway platforms (CodeAct is a tool-calling mechanism, not a 
  platform concern)

---

## 3. Terminology

| Term | Definition |
|---|---|
| **CodeAct mode** | The new primary tool-calling mode described in this spec |
| **Hermes Kernel** | A persistent Python interpreter (subprocess) owned for the lifetime of one agent session |
| **Kernel namespace** | The shared `globals` dict maintained by the kernel across all code executions |
| **Tool stub** | A Python function in the kernel namespace that wraps a Hermes registry tool via IPC |
| **Skill function** | A Python function in the kernel namespace loaded from a Hermes skill file |
| **CodeAct turn** | One model response → code extraction → kernel execution → result cycle |
| **Structured envelope** | The `{"thoughts": "...", "code": "..."}` JSON wrapper around model output |
| **Tool registry** | The existing `tools/registry.py` singleton managing all 74 Hermes tools |
| **Multi-tool mode** | The existing tool-calling system using 74 JSON schemas (legacy/cloud default) |

---

## 4. High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Agent Session                        │
│                                                         │
│  User message                                           │
│       │                                                 │
│       ▼                                                 │
│  ┌─────────────┐    tool schema     ┌───────────────┐  │
│  │  run_agent  │ ──────────────────▶│  LLM Provider │  │
│  │  (AIAgent)  │◀────────────────── │  (1 tool:     │  │
│  └──────┬──────┘  model response    │   run_code)   │  │
│         │                           └───────────────┘  │
│         │ extract code                                  │
│         ▼                                               │
│  ┌─────────────────────────────────────────┐           │
│  │           CodeActDispatcher             │           │
│  │  • parses envelope (thoughts/code)      │           │
│  │  • validates code (AST lint)            │           │
│  │  • routes to HermesKernel               │           │
│  └──────────────────┬──────────────────────┘           │
│                     │                                   │
│         ┌───────────▼──────────────┐                   │
│         │      HermesKernel        │                   │
│         │  (persistent subprocess) │                   │
│         │                          │                   │
│         │  globals_dict: {         │                   │
│         │    web_search: <stub>,   │                   │
│         │    read_file: <stub>,    │                   │
│         │    ... (all 74 tools)    │                   │
│         │    help: <fn>,           │                   │
│         │    <skill_fn>: <fn>,     │                   │
│         │    result_1: ...,        │ ← user variables  │
│         │    df: ...,              │   persist here     │
│         │  }                       │                   │
│         └──────────────┬───────────┘                   │
│                        │ IPC (UDS socket)               │
│                        ▼                               │
│         ┌──────────────────────────┐                   │
│         │   Hermes Tool Registry   │                   │
│         │   (74 tools, existing)   │                   │
│         └──────────────────────────┘                   │
│                                                         │
│  ┌─────────────────────────────────────────┐           │
│  │         Skill Namespace Injector        │           │
│  │  Loads relevant skills as Python fns    │           │
│  │  at session start → kernel globals_dict │           │
│  └─────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────┘
```

### Mode routing

```
Provider/model capability
    │
    ├─ codeact_mode: false  →  multi-tool JSON (existing path, unchanged)
    │
    └─ codeact_mode: true
         │
         ├─ structured_envelope: false  →  raw code output, parse first ```python block
         │
         └─ structured_envelope: true
              │
              ├─ envelope_enforcement: "grammar"  →  constrained generation (vLLM/llama.cpp)
              └─ envelope_enforcement: "prompt"   →  prompt-only, JSON parse with fallback
```

---

## 5. The Hermes Kernel

### 5.1 Overview

The `HermesKernel` is a long-lived Python subprocess, one per agent session. It holds the 
persistent `globals_dict` that the agent's code executes against. Tool stub functions in the 
namespace make IPC calls back to the parent process's tool registry — the same UDS socket 
mechanism already used by `execute_code`, extended to all registry tools.

**Critically different from the current `execute_code` design:**

| Property | Current `execute_code` | `HermesKernel` |
|---|---|---|
| Process lifetime | Fresh per call | Lives for entire agent session |
| State between calls | None (subprocess dies) | Full Python globals persist |
| Tools available | 7 hardcoded | All registry tools |
| Tool discovery | Static stubs file | Dynamic from registry at session start |
| Reset semantics | N/A (always fresh) | Explicit `kernel.reset()` clears user vars |

### 5.2 Kernel lifecycle

```
Agent session starts
    │
    ▼
HermesKernel.__init__(session_id, registry, skill_loader)
    │
    ▼
HermesKernel.start()
    ├─ spawn subprocess: agent/codeact_kernel_process.py
    ├─ establish UDS socket
    ├─ send INIT message: {tool_stubs_source, skill_functions_source}
    └─ kernel process: exec(init_code, globals_dict)
           globals_dict now contains: all tool stubs + skill functions + help()
    │
    ▼
[Agent turns — kernel.execute(code) called for each CodeAct turn]
    │
    │  On each execute():
    │      parent sends: {type: "exec", code: "..."}
    │      kernel runs: exec(code, globals_dict)   # globals_dict persists
    │      stdout captured, returned to parent
    │      tool calls within code go through IPC back to parent's registry
    │
    ▼
[User resets context / starts new task]  — auto on compaction + idle/unrelated query
    │
    ├─ kernel.soft_reset()  →  remove user-defined vars, keep tool stubs + skills
    └─ kernel.hard_reset()  →  kill process, respawn fresh
    │
    ▼
Agent session ends
    └─ kernel.shutdown()  →  graceful process termination
```

### 5.3 Kernel process internals

**File:** `agent/codeact_kernel_process.py`

```python
# Runs as persistent subprocess
# Receives commands via UDS socket
# Maintains globals_dict across all exec() calls

import sys, json, traceback, io
from contextlib import redirect_stdout, redirect_stderr

globals_dict: dict = {}  # THE persistent state

def handle_init(payload: dict) -> None:
    # Execute tool stubs and skill functions into globals_dict
    exec(payload["tool_stubs_source"], globals_dict)
    exec(payload["skill_functions_source"], globals_dict)

def handle_exec(payload: dict) -> dict:
    code = payload["code"]
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # Execute in persistent globals_dict — state accumulates here
            compiled = compile(code, "<codeact>", "exec")
            exec(compiled, globals_dict)
        
        # If last statement is an expression, also eval it for display
        last_val = _eval_last_expr(code, globals_dict)
        
        return {
            "status": "ok",
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
            "last_value": repr(last_val) if last_val is not None else None,
        }
    except Exception:
        return {
            "status": "error",
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
            "traceback": traceback.format_exc(),
        }

def handle_soft_reset(payload: dict) -> dict:
    # Remove user-defined variables, preserve tool stubs + skills
    protected_keys = set(globals_dict.get("__protected__", []))
    for key in list(globals_dict.keys()):
        if key not in protected_keys and not key.startswith("__"):
            del globals_dict[key]
    return {"status": "ok"}

def main():
    sock = connect_to_parent()
    while True:
        msg = recv_json(sock)
        match msg["type"]:
            case "init":   send_json(sock, handle_init(msg))
            case "exec":   send_json(sock, handle_exec(msg))
            case "reset":  send_json(sock, handle_soft_reset(msg))
            case "quit":   sys.exit(0)

if __name__ == "__main__":
    main()
```

The `globals_dict["__protected__"]` set is populated at init time with all tool stub names and 
skill function names, ensuring `soft_reset()` never accidentally removes them.

### 5.4 Reset semantics

**Confirmed decision:** Soft reset (user vars cleared, tools + skills preserved) triggers on:
- Context compaction event (compacted variables are unreachable in new context anyway)
- Between unrelated queries (detected via topic boundary heuristic — see below)
- Explicit `/reset` slash command always triggers hard reset

Hard reset (kill + respawn) triggers on:
- Explicit `/reset` slash command
- `hermes --reset-kernel` flag  
- Kernel crash/timeout

**User-configurable override:** `codeact.kernel.auto_reset: false` disables automatic resets 
(both compaction-triggered and between-query). With this flag set, the kernel only resets on 
explicit `/reset`. This is opt-in — the default is automatic reset.

**Between-query detection heuristic:** A query is considered "unrelated" to the previous one 
if the user's message contains no references to variables, results, or subjects from the previous 
CodeAct turn's context. In practice: if the previous turn defined variables and the new user 
message has no semantic overlap with them, soft reset fires before the new turn executes.
For now, a simple implementation: soft reset between queries when the conversation has been 
idle for > 30 minutes, or when the user explicitly starts with keywords like "new task", 
"different question", etc. The more sophisticated semantic detection is a Phase 2 improvement.

### 5.5 Security model

The kernel subprocess inherits the same environment scrubbing as current `execute_code`:
- All env vars with `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `CREDENTIAL`, `AUTH` in name are 
  stripped before subprocess spawn
- Safe prefixes allowed: `PATH`, `HOME`, `USER`, `LANG`, `HERMES_*`, etc.
- `sys.passthrough_env` in config extends the allowlist

**Import policy (confirmed):** Allow all standard library imports. No module allowlist. 
Block only the narrow set of shell-escape patterns (`os.system()`, `subprocess` with 
`shell=True`, `eval()` on untrusted strings). The approval system handles dangerous tool 
calls; import restrictions are not the security boundary here.

**pip install (confirmed):** Enabled by default, since the kernel runs in a sandboxed 
subprocess environment. The agent may run `subprocess.run(["pip", "install", "package"])` 
or use the `terminal()` tool stub to install packages. This can be disabled via 
`codeact.kernel.allow_pip_install: false` for environments where package installation 
should be locked down.

The approval system (`tools/approval.py`) continues to gate destructive tool calls. When a 
tool stub executes a registry tool that has an approval check, the approval prompt appears 
in the parent process as normal.

---

## 6. Tool Namespace Generation

### 6.1 From registry to Python stubs

**File:** `agent/codeact_namespace.py`

At kernel init, `build_tool_namespace_source(registry, enabled_toolsets)` generates a Python 
source string that defines stub functions for every enabled tool:

```python
def build_tool_namespace_source(registry: ToolRegistry, enabled_toolsets: set[str]) -> str:
    """
    Generate the Python source for all tool stub functions.
    Result is exec()'d into the kernel's globals_dict at init.
    """
    lines = [
        "import json as _json",
        "from __hermes_ipc__ import _call_tool  # injected by kernel runner",
        "",
        "# === Hermes Tool Stubs ===",
        "# Call these functions to invoke Hermes tools.",
        "",
    ]
    
    for tool_name, entry in registry.iter_enabled(enabled_toolsets):
        stub = _generate_stub(tool_name, entry)
        lines.extend(stub)
        lines.append("")
    
    # Mark all tool names as protected (survive soft_reset)
    tool_names = [name for name, _ in registry.iter_enabled(enabled_toolsets)]
    lines.append(f"__protected__ = {set(tool_names) | {'help', '__protected__'}!r}")
    
    return "\n".join(lines)


def _generate_stub(tool_name: str, entry: ToolEntry) -> list[str]:
    """Generate one stub function from a tool registry entry."""
    schema = entry.schema  # OpenAI-format JSON schema
    params = schema.get("parameters", {}).get("properties", {})
    required = set(schema.get("parameters", {}).get("required", []))
    
    # Build Python signature
    sig_parts = []
    for param_name, param_schema in params.items():
        py_type = _json_type_to_python(param_schema.get("type", "Any"))
        default = "" if param_name in required else " = None"
        sig_parts.append(f"{param_name}: {py_type}{default}")
    
    signature = f"def {tool_name}({', '.join(sig_parts)}):"
    
    # One-line summary for inline doc
    short_desc = schema.get("description", "").split("\n")[0][:120]
    
    # Full param doc (lazy — stored in __doc__, not in system prompt)
    param_docs = []
    for param_name, param_schema in params.items():
        desc = param_schema.get("description", "")
        param_docs.append(f"    {param_name}: {desc}")
    
    full_doc = f'    """{short_desc}\n\n' + "\n".join(param_docs) + '\n    """'
    
    # Body: build kwargs dict, call IPC, return result
    body_lines = [
        "    _kwargs = {k: v for k, v in locals().items() if v is not None}",
        f"    return _call_tool({tool_name!r}, _kwargs)",
    ]
    
    return [signature, full_doc] + body_lines
```

### 6.2 What the generated stubs look like

For `web_search(query: str, limit: int = 5) -> str`:

```python
def web_search(query: str, limit: int = None):
    """Search the web for information. Returns formatted results string.

    query: Search query string
    limit: Max results to return (default 5)
    """
    _kwargs = {k: v for k, v in locals().items() if v is not None}
    return _call_tool("web_search", _kwargs)
```

For `edit_file(path: str, old_string: str, new_string: str, replace_all: bool = None)`:

```python
def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = None):
    """Replace text in a file. old_string must match exactly.

    path: File path relative to project root
    old_string: Exact text to find and replace
    new_string: Replacement text
    replace_all: If True, replace all occurrences (default False)
    """
    _kwargs = {k: v for k, v in locals().items() if v is not None}
    return _call_tool("edit_file", _kwargs)
```

### 6.3 IPC bridge (`_call_tool`)

`_call_tool` is not a stub — it's a real function injected at kernel init that sends IPC messages 
to the parent process's `handle_function_call()`:

```python
# Injected into kernel namespace at init (not exec'd from generated source)
def _call_tool(tool_name: str, kwargs: dict) -> str:
    """Send a tool call to the parent Hermes process and return the result."""
    request = json.dumps({"type": "tool_call", "name": tool_name, "args": kwargs})
    sock.sendall(request.encode() + b"\n")
    response_line = recv_line(sock)
    response = json.loads(response_line)
    if response.get("error"):
        raise RuntimeError(f"Tool {tool_name} error: {response['error']}")
    return response["result"]
```

This reuses the existing UDS socket mechanism. The parent's IPC handler dispatches to 
`handle_function_call()` exactly as before — so approval checks, rate limiting, result 
truncation, and all existing tool infrastructure work unchanged.

### 6.4 Dynamic regeneration

When the user runs `hermes tools enable web` or the toolset changes during a session, 
the `CodeActDispatcher` triggers a kernel `reload_namespace` message, which re-execs the 
stub source with the updated tool set. User variables in `globals_dict` are preserved; 
only the tool stubs are refreshed.

---

## 7. The Help System

### 7.1 Design

Tools are described at two levels of detail:

**Level 1 — System prompt (compact, always present):**

One line per tool, grouped by category, injected into the system prompt at session start:

```
# Available Tools (call as Python functions)
## Files
  read_file(path, offset=1, limit=500)     — Read lines from a file
  write_file(path, content)                — Write content to a file
  edit_file(path, old_string, new_string)  — Replace text in a file
  search_files(pattern, path=".")          — Search file contents

## Web
  web_search(query, limit=5)              — Search the web
  web_extract(urls)                        — Extract content from URLs
  web_crawl(url, depth=1)                  — Crawl a site

## Memory
  store_memory(note, category="general")  — Persist a fact
  search_memory(query)                     — Search past memory
  ...

# For full parameter docs: help("tool_name")
# For all tools: help()
```

Token estimate: ~74 tools × ~1.5 lines × ~8 tokens/line ≈ **~900 tokens** upfront. 
Compare to current 74-schema multi-tool format: **~8,000–12,000 tokens** of JSON schema. 
CodeAct mode reduces the tool description overhead by ~90%.

**Level 2 — On-demand via `help()` (lazy, in-kernel):**

```python
def help(tool_name: str = None) -> str:
    """
    Return full documentation for a tool, or list all tools.
    
    help()              → compact one-liner list of all tools
    help("web_search")  → full signature, description, all parameters
    help("memory")      → all tools whose name contains 'memory'
    """
    if tool_name is None:
        return _format_all_tools_compact()
    
    matches = [name for name in _TOOL_REGISTRY if tool_name in name]
    if not matches:
        return f"No tool matching '{tool_name}'. Try help() for full list."
    
    return "\n\n".join(_format_tool_full(name) for name in matches)
```

`_TOOL_REGISTRY` is a dict of `{tool_name: (full_doc_string, signature)}` injected at kernel 
init alongside the stubs. Full docs are stored there but never rendered into the system prompt 
unless `help()` is called.

### 7.2 Agent-clip "help" feature mapping

The agent-clip feature the user liked (where the agent types `help` or `help command_name` to 
get full documentation) maps directly to the `help()` function in the kernel namespace. The 
model can call:

```python
print(help("web_search"))
```

And the kernel returns the full docstring to stdout, which appears in the tool result, which 
goes back into context for the next turn. This is lazy introspection — pay the token cost only 
when the model needs it.

---

## 8. Structured Output Envelope

### 8.1 Format

```json
{
  "thoughts": "I need to find the context window limit for Qwen3.6 27B, then store it.",
  "code": "result = web_search(query='qwen3.6 27b context window tokens')\nstore_memory(note=result, category='model/qwen3')\nprint(result)"
}
```

The `thoughts` field is the model's reasoning before it acts. The `code` field is executable 
Python. Both are strings. This is the entire schema — two fields, both strings, always the same.

The JSON envelope replaces markdown code fence parsing. The parent extracts `response["code"]` 
and sends it to the kernel.

### 8.2 When to use the envelope

| Model profile | Envelope | Enforcement |
|---|---|---|
| Local Qwen3.6 27B/35B via llama.cpp | Yes | Grammar-guided (preferred) or prompt |
| Local Gemma4 26B/31B via llama.cpp | Yes | Grammar-guided (preferred) or prompt |
| Local small models (< 14B) | No | Parse first ```python block |
| Claude Sonnet/Opus (cloud, CodeAct mode) | Optional | Prompt-only |
| GPT-4 class (multi-tool mode) | N/A | N/A (uses existing multi-tool path) |

### 8.3 Enforcement mechanisms

**Grammar-guided (preferred for local models):**

llama.cpp, vLLM, and most local serving stacks support grammar-constrained sampling via GBNF 
or JSON schema constraints. For the envelope, the grammar is trivially simple:

```json
{
  "type": "object",
  "properties": {
    "thoughts": {"type": "string"},
    "code": {"type": "string"}
  },
  "required": ["thoughts", "code"]
}
```

This is passed as `response_format` in the API call for backends that support it. The model 
is physically incapable of producing malformed output.

**Confirmed:** The local Qwen/Gemma setup runs buun-llama-cpp (spiritbuun/buun-llama-cpp), a 
fork of llama.cpp with TCQ KV cache compression. It uses `llama-server` with a full 
OpenAI-compatible API, and inherits llama.cpp's complete grammar support:
- `grammar` parameter (raw GBNF grammar string)
- `json_schema` parameter (JSON schema → auto-converted to GBNF internally)  
- `response_format: {"type": "json_schema", "json_schema": {...}}` (OpenAI structured output API)

Grammar-guided enforcement is available without any server changes. The `response_format` 
JSON schema approach is preferred over raw GBNF — it's cleaner, uses the same API parameter 
as OpenAI, and the schema for the envelope is trivially simple (two string fields).

**Bonus:** buun-llama-cpp's TCQ KV cache compression (2-3x more context in same VRAM) is 
directly complementary to the qwen-aware compaction work — TCQ compresses KV entries in VRAM; 
compaction reduces token count in context. Both apply simultaneously and independently.

**Prompt-only (fallback):**

When constrained generation is unavailable, the system prompt instructs the model to output 
the JSON envelope format and the parser attempts `json.loads(response)`. On parse failure:
1. Try to extract first ```python block from raw response (graceful fallback)
2. If that also fails, return an error message asking the model to reformat
3. Log the parse failure for trajectory analysis

**Envelope extraction code (parent side):**

```python
def extract_code_from_response(response: str, mode: str) -> tuple[str, str]:
    """
    Returns (thoughts, code). On failure raises CodeActParseError.
    """
    if mode == "envelope":
        try:
            parsed = json.loads(response.strip())
            return parsed.get("thoughts", ""), parsed["code"]
        except (json.JSONDecodeError, KeyError):
            # Fallback: find first ```python block
            if match := re.search(r"```python\n(.*?)```", response, re.DOTALL):
                return "", match.group(1)
            raise CodeActParseError(f"Could not parse CodeAct response: {response[:200]}")
    
    elif mode == "raw":
        # No envelope — parse first ```python block
        if match := re.search(r"```python\n(.*?)```", response, re.DOTALL):
            return "", match.group(1)
        # Also accept bare code (no fence) if response looks like code
        if _looks_like_code(response):
            return "", response.strip()
        raise CodeActParseError("No code found in response")
```

---

## 9. Skill Namespace Injection

### 9.1 Design

Hermes skills (currently: JSON files in `~/.hermes/skills/`, injected as text into the system 
prompt) gain a new capability in CodeAct mode: **procedural skills become callable Python 
functions in the kernel namespace**.

A skill that describes how to do something becomes a function that does that thing. When the 
model writes `result = research_topic("qwen3 benchmarks")`, it calls a skill function that 
executes the multi-step research pattern defined in that skill — using the kernel's tool stubs 
for each sub-step.

### 9.2 Two types of skill functions

**Type A — Pre-written skill functions (from skill definition):**

Skills with a `codeact_fn` field contain Python source code that is injected directly:

```json
{
  "name": "research_topic",
  "domain": "research",
  "description": "Multi-step web research on a topic",
  "codeact_fn": "def research_topic(query: str, max_results: int = 5) -> str:\n    results = web_search(query=query, limit=max_results)\n    facts = extract_key_points(text=results)\n    store_memory(note=facts, category='research')\n    return facts\n"
}
```

**Type B — Promoted skill functions (generated by curator from trajectories):**

When the curator identifies a helper function the agent wrote repeatedly, it promotes it:

```json
{
  "name": "parse_benchmark_table",
  "domain": "data-science",
  "description": "Extract benchmark metrics from markdown table text",
  "codeact_fn": "def parse_benchmark_table(text: str) -> dict:\n    import re\n    rows = re.findall(r'\\|(.+?)\\|', text)\n    # ... extracted from agent trajectory\n    return metrics\n",
  "source": "promoted",
  "promotion_count": 4,
  "promoted_date": "2026-05-01"
}
```

### 9.3 Skill relevance selection

Not all skills are injected — that would bloat the namespace and risk naming collisions. The 
`SkillNamespaceInjector` selects skills at session start:

```python
class SkillNamespaceInjector:
    def select_skills_for_session(
        self, 
        task_description: str, 
        all_skills: list[Skill],
        max_skills: int = 20,
    ) -> list[Skill]:
        """
        Select skills most relevant to the current task.
        
        Strategy:
        1. Always include: skills explicitly loaded via /skill or config
        2. Semantic similarity to task_description via embeddings
        3. Recently used skills (from skill_usage.py tracking)
        4. Skills with codeact_fn only (others can't be injected as functions)
        """
```

**Confirmed:** Keep existing behavior for text-only skills. Skills with `codeact_fn` go into 
the kernel namespace as callable functions. Skills without `codeact_fn` continue to appear in 
the system prompt as reference text, exactly as today. The two mechanisms coexist cleanly.

### 9.4 Namespace collision handling

Tool stubs take priority over skill functions. If a skill function name conflicts with a tool 
stub name, the skill function is renamed `skill_{name}` and a warning is logged.

If two skill functions share a name, the more recently used one wins.

---

## 10. Skill Promotion Pipeline

### 10.1 What gets promoted

The autonomous curator (existing: `agent/curator.py`) gains a new analysis pass: **CodeAct 
trajectory mining**. After each session, the trajectory compressor (`trajectory_compressor.py`) 
produces a compressed trajectory. The curator scans this for:

1. **Repeated helper functions**: `def xyz(...)` patterns that appear in 3+ separate CodeAct 
   turns across 2+ different sessions
2. **Named utility patterns**: functions the agent defines and then calls, suggesting it 
   considers them reusable
3. **Explicitly promoted**: agent calls `promote_to_skill(fn_name, description)` — a new 
   built-in skill-creation function in the kernel namespace

### 10.2 Promotion flow

```
Trajectory → curator.analyze_codeact_trajectories()
    │
    ├─ Find candidate functions (3+ occurrences across sessions)
    │
    ▼
┌─────────────────────────────────────────────┐
│   Promotion Candidates                      │
│   name: "parse_benchmark_table"             │
│   occurrences: 4                            │
│   sessions: [sess_a, sess_b, sess_c]        │
│   source_code: "def parse_benchmark_table..." │
└─────────────────────────────┬───────────────┘
                              │
               [CONFIRMED: Flag for review first]
                              │
                  Option B (default): Flag for review
                  Curator report: "3 promotion candidates"
                  User runs: hermes curator promote
                              │
                  Option A: Auto-promote (after testing validation)
                  Enable via: auto_promote: true in config
                  Write skill file immediately, available next session

After promotion:
    │
    ▼
~/.hermes/skills/promoted/parse_benchmark_table.json
    │
    ▼
Next session: SkillNamespaceInjector may select it → kernel namespace
    │
    ▼
Agent can call: parse_benchmark_table(text) directly
```

### 10.3 The `promote_to_skill` built-in

A special function available in the kernel namespace that lets the agent explicitly request 
promotion:

```python
def promote_to_skill(
    fn_name: str, 
    description: str, 
    domain: str = "general",
    tags: list[str] = None
) -> str:
    """
    Promote a function from this session to a persistent Hermes skill.
    The function must be defined in the current kernel namespace.
    
    Example:
        def parse_table(text): ...
        promote_to_skill("parse_table", "Parse markdown benchmark tables", domain="data-science")
    """
```

This gives the agent itself agency over what becomes a persistent skill. It's the CodeAct 
analog of Hermes's existing `/skill save` command.

---

## 11. Integration with Hermes Architecture

### 11.1 New files

| File | Role |
|---|---|
| `agent/codeact_kernel.py` | `HermesKernel` class — manages the persistent subprocess |
| `agent/codeact_kernel_process.py` | The kernel subprocess entry point |
| `agent/codeact_namespace.py` | Stub generation, namespace building, `help()` |
| `agent/codeact_dispatcher.py` | `CodeActDispatcher` — envelope parsing, kernel routing |
| `agent/codeact_skill_injector.py` | `SkillNamespaceInjector` |
| `agent/codeact_tool.py` | The single `run_code` tool definition (schema + handler) |

### 11.2 Changes to existing files

**`run_agent.py`**

The main agent loop changes at two points:

**(A) Tool schema selection (session init):**
```python
# Before (multi-tool):
self.tools = get_tool_definitions(enabled_toolsets, disabled_toolsets)

# After (CodeAct mode, gated by provider/model capability):
if codeact_profile.get("codeact_mode"):
    self.tools = [build_codeact_tool_schema(registry, enabled_toolsets)]
    self._kernel = HermesKernel(session_id, registry, skill_injector)
    self._kernel.start()
else:
    self.tools = get_tool_definitions(enabled_toolsets, disabled_toolsets)
```

**(B) Tool call dispatch:**
```python
# Before (multi-tool):
result = handle_function_call(tool_call.function.name, tool_call.function.arguments)

# After (CodeAct mode):
if codeact_profile.get("codeact_mode"):
    thoughts, code = extract_code_from_response(tool_call.function.arguments, mode=envelope_mode)
    result = self._kernel.execute(code)
else:
    result = handle_function_call(tool_call.function.name, tool_call.function.arguments)
```

The `HermesKernel.execute()` call is the only change to the dispatch path — everything after 
(result → message append → next turn) is identical.

**`model_tools.py`**

Add `build_codeact_tool_schema(registry, enabled_toolsets) -> dict`:

```python
def build_codeact_tool_schema(registry: ToolRegistry, enabled_toolsets: set) -> dict:
    """
    Build the single 'run_code' tool schema for CodeAct mode.
    The description embeds the compact tool catalogue.
    """
    compact_catalogue = build_compact_tool_catalogue(registry, enabled_toolsets)
    
    return {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": (
                "Execute Python code in a persistent interpreter.\n"
                "Variables defined here persist across turns.\n"
                "Tool functions are available directly by name — no imports needed.\n\n"
                + compact_catalogue +
                "\n\nCall help() for full list, help('name') for full docs on a specific tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thoughts": {
                        "type": "string",
                        "description": "Your reasoning before writing the code (required)"
                    },
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    }
                },
                "required": ["thoughts", "code"]
            }
        }
    }
```

**`agent/context_compressor.py` (qwen-aware compaction)**

CodeAct turns produce a different message shape than multi-tool turns. The compressor needs 
to recognize CodeAct `run_code` tool calls and handle them correctly:

- A CodeAct tool call message looks like: `{role: "assistant", tool_calls: [{function: {name: "run_code", arguments: "{\"thoughts\": ..., \"code\": ...}"}}]}`
- The result message looks like: `{role: "tool", content: "<stdout output>"}`
- Dedup logic: identical `code` field in consecutive `run_code` calls → deduplicate (same as 
  existing `dedup_operations` for read/edit)
- Operation key for dedup: `hash(code)` (not tool name + path as for file ops)

**`agent/model_metadata.py`**

Add `codeact_mode` and `structured_envelope` fields to model capability metadata:

```python
# Example additions to existing model entries:
"qwen3.6-27b": {
    "context_window": 131072,
    "supports_tool_calling": True,
    "codeact_mode": True,
    "structured_envelope": True,
    "envelope_enforcement": "grammar",   # or "prompt"
    "qwen_aware": True,
},
"gemma4-27b": {
    "context_window": 131072,
    "supports_tool_calling": True,
    "codeact_mode": True,
    "structured_envelope": True,
    "envelope_enforcement": "grammar",
},
"claude-sonnet-4-6": {
    "context_window": 200000,
    "supports_tool_calling": True,
    "codeact_mode": False,  # uses multi-tool by default
    "structured_envelope": False,
},
```

### 11.3 The `run_code` tool registration

**`agent/codeact_tool.py`** registers `run_code` as a standard Hermes tool:

```python
from tools.registry import registry

registry.register(
    name="run_code",
    toolset="codeact",
    schema=None,              # dynamically built per session
    handler=_codeact_handler, # routes to HermesKernel.execute()
    check_fn=_check_kernel_available,
    emoji="🐍",
    max_result_size_chars=50_000,
)
```

This means the tool appears in `hermes tools` output, respects toolset enable/disable, and 
flows through the existing plugin hook system (`pre_tool_call`, `post_tool_call`).

---

## 12. Context Management Interaction

### 12.1 CodeAct and the qwen-aware compaction system

The `feat/qwen-aware-compaction` work (current branch) compacts conversation history to reduce 
re-processing cost for Qwen's non-cacheable KV architecture. CodeAct turns interact with 
compaction as follows:

**What a CodeAct conversation looks like in the message list:**

```
[system prompt + tool catalogue]
[user: "Research qwen3.6 benchmarks and summarize"]
[assistant: tool_call{run_code, thoughts="...", code="result = web_search(...)..."}]
[tool: "Search results: ..."]
[assistant: tool_call{run_code, thoughts="...", code="summary = summarize(result)..."}]
[tool: "Summary: ..."]
[assistant: "Here is the summary: ..."]
```

**Dedup logic for CodeAct:**

The existing `dedup_operations` feature (collapses consecutive identical tool calls on the same 
key) needs a CodeAct-specific operation key. Recommendation: use `hash(code_block)` as the key. 
Consecutive identical code blocks (which can happen if the model retries the same failing call) 
are collapsed.

**Anchor semantics:**

`anchor_first_assistant: true` applies to CodeAct turns as-is — the first assistant message 
(first `run_code` call) is never the compaction start point.

**Variable invalidation on compaction:**

When a compaction event occurs, any kernel variables that were defined in the compacted region 
are now referenced-but-undefined in the new context. The compressor should append a kernel 
`soft_reset()` call to the compaction event to clear stale variables. The agent will recompute 
what it needs.

**Confirmed default:** Cleared (soft reset) on compaction. This is the default behavior.

**Optional checkpoint/restore mode:** Configurable via `codeact.kernel.checkpoint_on_compaction: true`. 
When enabled, the kernel serializes its `globals_dict` (excluding un-picklable objects like 
open file handles and tool stubs) to `~/.hermes/sessions/<session_id>/kernel_checkpoint.pkl` 
before compaction, then restores it after. Tool stubs and skill functions are re-injected fresh 
from the registry (not pickled), while user-defined variables and data structures are restored.
Serialization uses `pickle` with a fallback to `json` for non-picklable values (logged as 
warnings). This mode trades simplicity for continuity on long multi-hour sessions.

### 12.2 Token budget

Approximate token costs per CodeAct turn (for compaction threshold calculations):

| Component | Tokens (approx) |
|---|---|
| System prompt + compact tool catalogue | ~1,500 |
| Structured envelope overhead per turn | ~50 |
| Typical code block | ~100–400 |
| Tool result (stdout) | ~100–2,000 |
| Thoughts field | ~50–200 |

Total per CodeAct turn: ~300–2,700 tokens, vs ~1,500–5,000 for multi-tool turns (due to 
individual tool schema repetition in context).

---

## 13. Configuration Reference

New keys in `~/.hermes/config.yaml`:

```yaml
codeact:
  # Enable CodeAct mode for routes that explicitly advertise support
  enabled: auto          # auto = use provider/model capability; true/false to force

  # Structured output envelope
  envelope:
    enabled: auto        # auto = use provider/model capability
    enforcement: auto    # auto / grammar / prompt
                         # For buun-llama-cpp: defaults to "grammar" (response_format json_schema)
  
  # Kernel settings
  kernel:
    timeout_seconds: 300          # max time for a single code execution
    max_tool_calls_per_block: 50  # same as current execute_code limit
    allow_pip_install: true       # pip install allowed by default (sandboxed subprocess)
    
    # Reset policy
    auto_reset: true              # false = require explicit /reset; true = auto-reset (default)
    soft_reset_on_compaction: true    # reset user vars when context is compacted
    soft_reset_between_queries: true  # reset user vars between unrelated queries
    idle_reset_minutes: 30            # reset after N minutes of session idle time
    
    # Checkpoint/restore on compaction (alternative to soft reset)
    checkpoint_on_compaction: false   # true = serialize/restore globals across compaction
                                      # false = soft reset on compaction (default)
    checkpoint_path: "~/.hermes/sessions/{session_id}/kernel_checkpoint.pkl"
    
  # Compact tool catalogue in system prompt
  catalogue:
    max_tokens: 1500     # truncate if more than this
    group_by_category: true
    
  # Skill injection
  skills:
    max_injected: 20           # max skill functions in kernel namespace
    inject_recently_used: true # always inject last N used skills
    recently_used_count: 5
    semantic_selection: true   # use embeddings to pick relevant skills
    
  # Skill promotion (Phase 5)
  promotion:
    min_occurrences: 3         # helper fn must appear 3+ times to be candidate
    min_sessions: 2            # across at least 2 different sessions
    auto_promote: false        # false = flag for review (default); true = automatic after validation
    review_command: "hermes curator promote"
```

---

## 14. Testing Strategy

### 14.1 Unit tests

- `test_codeact_namespace.py`: stub generation for all 74 tools, type hint correctness, 
  docstring completeness
- `test_codeact_kernel.py`: kernel lifecycle (start/execute/reset/shutdown), state persistence 
  across calls, soft reset behavior, tool call IPC
- `test_codeact_dispatcher.py`: envelope parsing (valid JSON, malformed JSON fallback, 
  raw code fallback), error handling
- `test_codeact_skill_injector.py`: skill selection, collision handling, namespace updates

### 14.2 Integration tests (extending existing benchmark harness)

The existing `tests/benchmarks/` (tiers 1–3) should add:

- **Tier 2 extension**: CodeAct multi-tool composition — tasks requiring 3+ tool calls 
  in sequence, verifying state persistence (result from call N used in call N+2)
- **Tier 3 extension**: Trajectory differential — same tasks on Qwen3.6 27B with multi-tool 
  JSON vs CodeAct mode, measuring success rate improvement

### 14.3 Regression tests

- Every existing multi-tool test should pass unchanged (CodeAct mode is opt-in per provider/model capability)
- The existing `execute_code` tool should continue working as a tool callable *from within* 
  a CodeAct block (the agent writes `result = execute_code(code="import pandas as pd...")`)

---

## 15. Migration Path

### Phase 1 — Infrastructure (no behavior change)
- Implement `HermesKernel`, `CodeActNamespace`, `CodeActDispatcher`
- Add `run_code` to registry with `codeact_mode: false` default
- All provider/model routes default to multi-tool (existing behavior unchanged)
- Tests pass

### Phase 2 — Local provider opt-in
- Set `codeact.enabled: true` on local provider/model capability entries
- Enable for local provider routes in default config
- Structured envelope with prompt-only enforcement (no grammar dependency)
- Measure success rate improvement on existing benchmarks

### Phase 3 — Grammar enforcement
- Add grammar-constrained generation support to relevant provider adapters
  (llama.cpp adapter, Ollama adapter)
- Enable `envelope_enforcement: grammar` for local provider/model routes
- Measure parse error reduction

### Phase 4 — Skill injection
- Implement `SkillNamespaceInjector`
- Add `codeact_fn` field to skill schema
- Migrate high-value research skills to include `codeact_fn`

### Phase 5 — Skill promotion
- Implement curator CodeAct trajectory mining
- Add `promote_to_skill()` built-in to kernel namespace
- Enable promotion pipeline (review mode first, auto-promote later if desired)

---

## 16. Confirmed Design Decisions

All design questions resolved 2026-05-04.

| ID | Decision | Config key |
|---|---|---|
| **Q-1** | `pip install` **enabled by default** in the sandboxed kernel subprocess | `codeact.kernel.allow_pip_install: true` (default) |
| **Q-2** | Soft reset fires on **compaction AND between unrelated queries** by default; disable with `auto_reset: false` (manual `/reset` only) | `codeact.kernel.auto_reset: true` (default) |
| **Q-3** | **Allow all stdlib**; block only `os.system()`, `subprocess shell=True` patterns; approval system handles dangerous tool calls | hardcoded in kernel, no config needed |
| **Q-4** | **Grammar-guided enforcement available** — buun-llama-cpp uses `llama-server` with full `response_format` JSON schema support; grammar mode is the default for local profiles | `codeact.envelope.enforcement: grammar` (default for local) |
| **Q-5** | **Text-only skills keep existing behavior** — appear in system prompt as reference text; skills with `codeact_fn` additionally get kernel namespace injection | no new config needed |
| **Q-6** | **Flagged for review first** (`auto_promote: false`); switch to automatic after testing validation passes | `codeact.promotion.auto_promote: false` (default) |
| **Q-7** | **Cleared by default** on compaction; **optional checkpoint/restore** available via `checkpoint_on_compaction: true` for long sessions | `codeact.kernel.checkpoint_on_compaction: false` (default) |

**Bonus finding (Q-4):** buun-llama-cpp's TCQ KV cache compression (2-3x more context in 
same VRAM) is complementary to the qwen-aware compaction work on this branch. The two systems 
stack independently: TCQ reduces per-token VRAM cost, compaction reduces the token count. 
Both apply to the same Qwen3.6/Gemma4 inference sessions.

---

## 17. Appendix: File Change Summary

```
NEW FILES:
  agent/codeact_kernel.py           HermesKernel class
  agent/codeact_kernel_process.py   Kernel subprocess entry point
  agent/codeact_namespace.py        Stub gen, help(), catalogue builder
  agent/codeact_dispatcher.py       Envelope parsing + kernel routing
  agent/codeact_skill_injector.py   Skill selection + namespace injection
  agent/codeact_tool.py             run_code tool registration
  tests/test_codeact_*.py           Test suite (6 files)

MODIFIED FILES:
  run_agent.py                      Tool schema selection + dispatch routing (~30 LOC)
  model_tools.py                    build_codeact_tool_schema() function (~60 LOC)
  agent/model_metadata.py           codeact_mode, structured_envelope fields
  agent/context_compressor.py       CodeAct dedup key + soft_reset on compaction
  agent/curator.py                  CodeAct trajectory mining pass
  tools/code_execution_tool.py      No changes (remains as a tool callable from CodeAct)
  skills/*.json                     Add optional codeact_fn field (backwards compatible)

UNCHANGED:
  All 74 existing tools             (called through IPC as before)
  All provider adapters             (send single tool schema instead of 74)
  All messaging platforms           (unaffected)
  trajectory_compressor.py         (CodeAct trajectories are compatible)
  batch_runner.py                   (unaffected)
```

---

*End of Draft v0.2 — All design questions resolved. Ready for Phase 1 implementation.*
