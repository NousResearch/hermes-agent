---
name: ai-agent-delegation
description: "Delegate coding tasks to external AI agents: Claude Code, OpenAI Codex, OpenCode — with PTY handling, CLI flags, and output interpretation."
version: 1.0.0
author: Hermes Agent + Teknium
license: MIT
metadata:
  hermes:
    tags: [Coding-Agent, Claude, Codex, OpenCode, Delegation, PTY, Automation, Code-Review, Refactoring]
    related_skills: [hermes-agent]
---

# AI Agent Delegation

Delegate coding tasks to external autonomous AI coding agents from the Hermes terminal. This skill covers three agents: **Claude Code** (Anthropic), **Codex** (OpenAI), and **OpenCode** (opencode.ai).

## Overview

| Feature | Claude Code | Codex | OpenCode |
|---------|-------------|-------|----------|
| **Install** | `npm install -g @anthropic-ai/claude-code` | `npm install -g @openai/codex` | `npm i -g opencode-ai@latest` |
| **Auth** | `claude auth login` / `ANTHROPIC_API_KEY` | `OPENAI_API_KEY` / OAuth | `opencode auth login` |
| **One-shot command** | `claude -p 'task'` | `codex exec 'task'` | `opencode run 'task'` |
| **Interactive TUI** | `claude` (with tmux) | `codex` | `opencode` |
| **PR review** | `--from-pr N` or pipe `git diff` | `codex review --base origin/main` | `opencode pr N` |
| **Model selection** | `--model <alias>` | `--model` (API provider) | `--model provider/model` |
| **Requires git repo** | No (optional) | **Yes** | Recommended |
| **Requires PTY** | Interactive only | Always (interactive) | TUI: yes; `run`: no |
| **Background mode** | Via tmux orchestration | `--full-auto` | Built-in `background=true` |
| **Output format** | `--output-format json/stream-json` | Plain text + events | `--format json` |

## Shared Patterns

### PTY Handling

All three agents are interactive TUI applications. Use `pty=true` when running them interactively:

- **Claude Code:** Requires tmux orchestration for interactive mode (print mode `-p` needs no PTY)
- **Codex:** Always interactive — `pty=true` required
- **OpenCode:** Interactive TUI needs `pty=true`; `opencode run` does NOT need PTY

### Background Process Monitoring

When running long tasks with `background=true`, monitor with the `process` tool:

```
process(action="poll", session_id="<id>")   # Check status + new output
process(action="log", session_id="<id>")    # Full output with pagination
process(action="submit", session_id="<id>", data="response")  # Answer prompts
process(action="kill", session_id="<id>")   # Terminate
```

### Session Continuation

- **Claude Code:** `--continue` or `--resume <id>` (print mode: `--resume <id>`)
- **Codex:** Not documented (each `exec` is a new invocation)
- **OpenCode:** `-c` (continue last session) or `-s <id>` (specific session)

### PR Review Workflows

All three agents can review pull requests:

```
# Claude Code — pipe diff
git diff main...branch | claude -p 'Review this diff'

# Claude Code — from PR number
claude -p 'Review this PR' --from-pr 42

# Codex — in a cloned repo
REVIEW=$(mktemp -d) && git clone $REPO $REVIEW && cd $REVIEW && gh pr checkout 42 && codex review --base origin/main

# OpenCode — built-in PR command
opencode pr 42
```

### Parallel Task Execution

Use separate workdirs or git worktrees to avoid collisions:

```
git worktree add -b fix/issue-78 /tmp/issue-78 main
git worktree add -b fix/issue-99 /tmp/issue-99 main

# Claude Code — independent tmux sessions
tmux new-session -d -s cc1
tmux new-session -d -s cc2

# Codex — background with process monitoring
terminal(command="codex exec 'Fix issue #78' --full-auto", workdir="/tmp/issue-78", background=true, pty=true)

# OpenCode — separate parallel runs
terminal(command="opencode run 'Fix issue #78 and commit'", workdir="/tmp/issue-78", background=true, pty=true)
```

---

## Model Routing — Critical: M2.7 vs M2.5 Fix

**CONFIRMED BUG (May 2026):** `minimax-m2.5-free` routes to M2.7, NOT M2.5.

When Claude Code or Hermes sends `minimax-m2.5-free` to OpenCode Zen, it resolves to `minimax/minimax-m2.7-20260318` — MiniMax M2.7, NOT M2.5. This causes:

- **M2.7 uses extended reasoning** — output goes to `reasoning` field, not `content`
- With default `max_tokens` (low), all tokens go to reasoning → **`content: null`** in response
- Claude Code sees `content: null` and retries → `Retrying in Xs...` loop → total failure

**APPLIED FIX (May 2026):**
- Claude Code `settings.json`: `"ANTHROPIC_MODEL": "minimax-m2.5"` (not `-free`)
- Claude Code `settings.json`: `"maxTokens": 2048`
- Hermes uses `minimax-m2.5-free` which resolves correctly (no M2.7 routing issue in Hermes)

### CONFIRMED BUG: minimax-m2.5-free resolves to M2.7
Full details, reproduction steps, and API response comparisons in `references/model-routing-quirks.md`.

**API response comparison:**

| Model | Content | Reasoning | Finish |
|-------|---------|-----------|--------|
| `minimax-m2.5` | Has text | null | stop |
| `minimax-m2.5-free` (→ M2.7) | **null** | Has text | length |

**Always test the actual resolved model** — check the `model` field in the API response, not just what you sent. The OpenCode Zen gateway may route to a different model than requested.

**OpenCode Zen model catalog** (as of May 2026): `minimax-m2.7`, `minimax-m2.5`, `minimax-m2.5-free`, `kimi-k2.6`, `qwen3.6-plus`, `deepseek-v4-flash-free`, `claude-opus-4-6`, `claude-sonnet-4-6`, `gpt-5.4`, etc. See `references/opencode-zen-models.md` for full list and quirks.

## Claude Code — WSL Invocation (Critical)

**Problem:** `claude` is a Windows binary, not in WSL's PATH. Invoking it from Hermes (WSL) requires routing through Windows CMD.

**Working pattern:**
```
/mnt/c/Windows/System32/cmd.exe /c "cd /d <WINDOWS_PATH> && claude -p 'task' --output-format json --verbose"
```

**Why `--verbose` is required:** `--output-format stream-json` requires `--verbose` flag. Without it, Claude Code exits with: `Error: When using --print, --output-format=stream-json requires --verbose`.

**Why stdin pipes FAIL:** Piping to `claude -p` drops the prompt content. Claude Code receives an empty prompt and asks for clarification. Use direct `-p 'task'` passing, not echo/pipes.

**Working directory:** Use Windows path in CMD (`cd /d C:\Users\...`). UNC paths fail — CMD defaults to Windows directory when UNC is detected.

**JSON output:** Parse with `--output-format json` (no `--verbose` needed for json). For stream-json, add `--verbose`.

**Confirmed working (May 2026):**
```bash
# From WSL — full delegation to Claude Code on Windows
/mnt/c/Windows/System32/cmd.exe /c "cd /d C:\Users\<username>\Desktop && claude -p \"create a file test.txt with content hello\" --output-format json"
```

**Note:** Claude Code defaults to Anthropic's API. To route through OpenCode Zen, configure `base_url` to point to OpenCode Zen's endpoint with the appropriate API key.

## Hermes as Orchestrator — Delegation Intelligence

### Core principle: 100% efficiency

I decide what to do myself vs delegate to Claude Code based on:
- **Complexity** — multi-file, framework-heavy, or architecturally complex → delegate to Claude Code
- **Error likelihood** — tasks with high chance of bugs or retries → let Claude Code handle (it has tool use, can self-correct)
- **Speed** — simple/single file → I do it directly (no delegation overhead)
- **Parallelization** — 5+ coding tasks → split: 2-3 to Claude Code, rest to me

Claude Code = coding specialist (executes code, self-corrects via tool use).
Hermes = planner/allocator (orchestrates, verifies, manages workflow).

### Delegation verification — CRITICAL

**After delegating, always verify via filesystem check. DO NOT pre-create files.**

Wrong pattern (passive orchestration failure):
1. User asks to create file via delegation
2. I create the file myself "to help"
3. File exists but Claude Code never ran

Right pattern (active orchestration):
1. User asks to create file
2. I delegate to Claude Code
3. Claude Code executes
4. I check filesystem for the file's existence
5. If missing → report failure, retry, or handle differently

**Pre-creating is orchestration failure.** Verification is the orchestrator's job.

### Permission system — cannot intercept from WSL

Claude Code has its own permission approval prompts (e.g., "I need permission to write to your Desktop"). Hermes **cannot intercept or approve these from WSL**. Options:
- User approves in their terminal session
- Pre-approve write permission once for the target directory
- Use a directory Claude Code already has write access to

### Global MCP servers (ruflo, ruv-swarm, flow-nexus)

Removing from project-level `settings.local.json` (setting it to `{}`) does NOT disable globally-installed MCPs. They still connect because they're npm-installed globally.

**To disable globally:** Need to find Claude Code's global config file (`%APPDATA%\Claude\settings.json`) or uninstall the npm packages globally. Project-level settings only affect project-level MCPs.

---

## Pitfall: CRLF Corruption on Windows Projects (WSL)

**CRITICAL for WSL→Windows workflows:** The Hermes `write_file` tool and bash heredocs (`cat > file << 'EOF'`) convert CRLF→LF when writing to Windows-mounted paths (`/mnt/c/...`). This silently corrupts `.tsx`, `.ts`, `.jsx`, and `.json` files — TypeScript/JSX parsers choke on LF-only line endings in mixed-CRLF projects.

**Symptoms:** Hundreds of cryptic `TS1109: Expression expected`, `TS1005: ';' expected`, `TS1160: Unterminated template literal` errors from `tsc` immediately after a file write. The file LOOKS fine when read back, but every line ending is wrong.

**Safe write methods (preserve CRLF):**

```bash
# 1. PowerShell Set-Content (best for Windows projects)
powershell.exe -Command "Set-Content -Path 'src\file.tsx' -Value \$content -NoNewline"

# 2. printf with explicit \r\n (for small files)
printf "line1\r\nline2\r\n" > src/file.tsx

# 3. Python binary mode (for complex content or targeted edits)
python3 -c "
with open('src/file.tsx', 'rb') as f:
    content = f.read()
content = content.replace(b'old', b'new')
with open('src/file.tsx', 'wb') as f:
    f.write(content)
"

# 4. Python binary mode via script (for multi-step edits)
cat > /tmp/fix.py << 'PYEOF'
with open('src/file.tsx', 'rb') as f:
    content = f.read()
# Multiple targeted replacements
content = content.replace(b'old1', b'new1')
content = content.replace(b'old2', b'new2')
with open('src/file.tsx', 'wb') as f:
    f.write(content)
PYEOF
python3 /tmp/fix.py

# 5. sed to convert LF→CRLF after writing
sed -i 's/$/\r/' src/file.tsx
```

**Unsafe methods (corrupt CRLF):**
- `write_file` (hermes tool) — always produces LF
- `write_file` from `hermes_tools` inside `execute_code` — also produces LF
- `cat > file << 'EOF'` — always produces LF
- `echo` with `>` redirect — always produces LF

**Verification:** Always check with `file src/file.tsx` — should say `with CRLF line terminators`, NOT just `ASCII text`.

**Also:** When editing existing files with the `patch` tool, verify the result with `file` command. The patch tool may or may not preserve line endings depending on the edit.

**TypeScript 5.9+ note:** After fixing CRLF corruption, you may still have TS6133 ("declared but never read") errors for unused variables. The `_` prefix convention only suppresses TS6133 for **function parameters**, NOT for local variables or const declarations in TS 5.9. Options: (1) actually remove the unused code, (2) disable `noUnusedLocals`/`noUnusedParameters` in tsconfig, (3) use `// @ts-expect-error` comments.

## Claude Flow — Multi-Agent Orchestration Layer

Claude Code on this machine has **Claude Flow** installed at `/mnt/c/Users/<username>/.claude/`. This is NOT the Claude Code binary — it's an orchestration layer that adds:
- Multi-agent routing (coder, planner, researcher, tester, reviewer)
- Hook system (pre-bash safety, post-edit tracking, session lifecycle)
- Intelligence layer (PageRank + trigram similarity over memory graph)
- SPARC 5-phase methodology (spec → pseudo → arch → refine → complete)

The agent personas and SPARC methodology have been extracted into Hermes skills: `agent-personas`, `sparc-methodology`, `self-sufficient-coding-agent`.

Full architecture details: `references/claude-flow-architecture.md`.

## Absorbing External Agent Knowledge

When the user asks you to "absorb" or internalize knowledge from an external agent system (Claude Code, Claude Flow, etc.), follow these principles:

### Audit Before Creating
1. **Check existing skills first** — for every pattern you find, ask: "Do I already have this capability?" If `github-code-review` exists, don't create `pr-review-swarm`.
2. **Merge into existing skills** — if a pattern enriches an existing skill, patch that skill. Don't create a new one.
3. **One skill per class** — not one skill per source. If Claude Flow has 5 security agents, that's one `security-scanning-patterns` skill, not 5.

### Trim Ruthlessly
4. **Skip distributed-systems infrastructure** — consensus protocols (Byzantine, Raft, Gossip), swarm coordinators, background workers. These are for multi-agent systems you're not running.
5. **Skip theoretical patterns** — PageRank memory ranking, HNSW vector indexing, trigram matching. If it describes internal infrastructure you can't implement natively, skip it.
6. **Skip "V3 specialized" or internal infrastructure agents** — these are typically the source system's internal plumbing, not portable knowledge.
7. **Only keep what directly applies to your actual work** — if the user isn't building distributed systems, don't absorb distributed systems knowledge.

### The Lean Principle
> A lean skill set that you actually use is better than 20 skills that add noise.

8. **Keep the useful core** — agent personas, methodology phases, security regex patterns. These are portable and actionable.
9. **Reference file for the rest** — put the "what was skipped and why" in a reference file, not in skills.
10. **Holographic memory for facts, skills for procedures** — store "Claude Code is installed at X" in memory, store "how to review code" in skills.

## References

- [Claude Code](./references/claude-code.md)
- [OpenAI Codex](./references/codex.md)
- [OpenCode](./references/opencode.md)
- [MiniMax Model Lineup](./references/minimax-models.md)
- [OpenCode Zen Models](./references/opencode-zen-models.md)
- [Claude Flow Architecture](./references/claude-flow-architecture.md)
- [Claude Code Directory Map](./references/claude-code-directory-map.md)