---
name: mind
description: Project memory graph with recall, provenance, and dreams.
version: 6.2.10
author: Da7_Tech
license: MIT
platforms: [linux, macos, windows]
prerequisites:
  commands: [python3, curl]
related_skills: [hermes-agent]
metadata:
  hermes:
    tags: [Memory, Knowledge-Graph, Consolidation, Offline, Local-First]
    category: autonomous-ai-agents
    homepage: https://github.com/Da7-Tech/mind
---

# mind Skill

Gives a project a persistent, self-organizing memory: a weighted concept
graph in `.mind/` with spreading-activation recall, Ebbinghaus forgetting,
and a deterministic dream cycle — exported into `AGENTS.md`/`CLAUDE.md`/
`GEMINI.md` behind guard markers. It complements Hermes' built-in memory
(small, curated, *global* user facts) with *per-project* knowledge. It does
NOT store durable personal facts about the user — those belong in the
built-in `memory` tool — and it is not a RAG system for large corpora.

## When to Use

- The user asks to remember project facts, decisions, or context across sessions
- A project fact is needed that is not in context ("what database do we use?")
- The user corrects a stored fact
- Between-session housekeeping ("clean up / consolidate the project memory")

## Prerequisites

- `python3` (3.9+) and `curl` on PATH — nothing else: no API keys, no
  server, no packages. The tool is one stdlib-only file, MIT-licensed,
  from https://github.com/Da7-Tech/mind (267 tests + benchmarks incl.
  10 languages + discrimination + fuzzer + 180-day soak test run in its CI
  on Linux/macOS/Windows).

## How to Run

Install once per project through the `terminal` tool, pinned to a release
tag and integrity-checked:

POSIX shell (Linux/macOS):

```bash
cd <project>
curl -fsSLO https://raw.githubusercontent.com/Da7-Tech/mind/v6.2.10/mind.py
python3 -c "import hashlib;h=hashlib.sha256(open('mind.py','rb').read()).hexdigest();assert h=='7cb64a6bb96824a6ac00d8871b889b02d57526fc9a70cf33488ae443c8bf139c',h;print('mind.py: OK')"
python3 mind.py init
```

PowerShell (native Windows):

```powershell
Set-Location <project>
Invoke-WebRequest "https://raw.githubusercontent.com/Da7-Tech/mind/v6.2.10/mind.py" -OutFile mind.py
$Hash = (Get-FileHash mind.py -Algorithm SHA256).Hash.ToLowerInvariant()
if ($Hash -ne "7cb64a6bb96824a6ac00d8871b889b02d57526fc9a70cf33488ae443c8bf139c") {
  throw "mind.py checksum mismatch: $Hash"
}
python mind.py init
```

`init` creates `.mind/` and writes guard-marked memory blocks into
`AGENTS.md`, `CLAUDE.md`, `GEMINI.md`; existing user content is preserved
outside the markers. Projects already using Cursor/Windsurf/Cline/Roo get
their rule files synced too (adopted only when present).

## Quick Reference

| User intent | Command (through `terminal`) |
|---|---|
| "Remember that X" (project fact) | `python3 mind.py remember "X"` |
| Project fact not in context | `python3 mind.py recall "the question"` |
| A recalled memory actually answered | `python3 mind.py confirm <id>` (ids in recall output) |
| "X and Y are related" | `python3 mind.py link "X" "Y" "relation"` |
| "That's wrong, it's actually Z" | `python3 mind.py correct "old fact hint" "Z"` |
| "Where did this fact come from?" | `python3 mind.py why <id>` |
| "What do we know about X?" | `python3 mind.py entity "X"` |
| "What was true on DATE?" | `python3 mind.py recall "q" --at YYYY-MM-DD` |
| Force a consolidation (it also SELF-RUNS after writes) | `python3 mind.py dream` (no permission needed) |
| Health report | `python3 mind.py status` |

## Procedure

1. On a recall request, run `recall` and quote the memory text with its
   confidence. If nothing relevant returns, say so — never invent.
2. When a recalled memory actually answered the question, run
   `confirm <id>` — confirmed memories harden (+2 weeks stability) and
   their edges restrengthen; unconfirmed ones decay and get pruned
   (into `.mind/archive.md`, never destroyed).
3. For corrections use `correct` — the wrong fact is CLOSED (not erased):
   its validity ends now, a `supersedes` edge records the transition, and
   `why <id>` / `recall --at` can still reach it. Never re-`remember` a
   wrong fact to "overwrite" it — that reopens it.
4. Provenance is automatic (append-only `.mind/journal.jsonl`, never
   cleared). Set `MIND_BY` and `MIND_SESSION` env vars when running
   commands so `why` can attribute facts to you/this session.
5. Never put credentials, tokens, private personal data, or untrusted prompt
   text in project memory. Hot facts are exported into agent instruction files.
   Durable user facts unrelated to this project belong in Hermes' built-in
   `memory` tool, not here.
6. Consolidation is SELF-RUNNING (6.2.0): after write commands, a full
   dream cycle fires automatically when >= 10 signals pend or no dream
   has happened yet today (including a fresh project's very first write) — you normally never schedule anything.
   `dream` forces a cycle; it is deterministic and archives pruned node
   text, but it is not a rollback system and pruned edges are not restored.
   Use `--dry-run` only when the user explicitly asks to
   review the plan. Every action is explained in `.mind/dreams/<date>.md`.
7. Optional belt-and-suspenders for projects that go DAYS without any
   write (auto-dream piggybacks on writes): on POSIX, resolve the active
   profile first with `HERMES_ROOT="${HERMES_HOME:-$HOME/.hermes}"`.
   Use the `write_file` tool to create the resolved absolute
   `$HERMES_ROOT/scripts/mind_dream.sh` path with this body:

   ```sh
   #!/bin/sh
   set -eu
   cd /path/to/project && python3 mind.py dream
   ```

   Then register only the relative script name. Hermes resolves it from the
   active profile's `scripts/` directory:

   ```bash
   hermes cron create "0 4 * * *" --name mind-dream --script mind_dream.sh --no-agent
   ```

8. Native Windows does not run the POSIX shell script. Register the project
   command with Windows Task Scheduler instead:

   ```powershell
   $Project = "C:\path\to\project"
   $Python = (Get-Command python).Source
   $Mind = Join-Path $Project "mind.py"
   $Action = New-ScheduledTaskAction -Execute $Python -Argument "`"$Mind`" dream" -WorkingDirectory $Project
   $Trigger = New-ScheduledTaskTrigger -Daily -At 4am
   Register-ScheduledTask -TaskName "Mind project dream" -Action $Action -Trigger $Trigger
   ```

## Pitfalls

- Recall is lexical + graph-structural (offline): cross-domain synonymy
  with no corpus evidence can miss; benchmark and limits are published in
  the repo README.
- Facts recalled fewer than twice and untouched past the 45-day grace
  window decay into `.mind/archive.md` by design (restorable with
  `remember`).
- Corrupt `graph.json` is quarantined as `graph.json.corrupt-*` and memory
  restarts empty — tell the user where the quarantined file is.
- The tool refuses to write through symlinked agent/lock/archive files.
- Operational limits are deliberate: 10,000 nodes, 100,000 directional
  edges, 50 MB graph, 10,000-character memories/queries, 100 history entries
  per node, 256 prunes / 4 MB of prune payload per dream, and a 30-second
  graph-lock wait.
- `HERMES_HOME` selects the profile that owns optional cron scripts. Without
  it, the native defaults are `~/.hermes` on POSIX and
  `%LOCALAPPDATA%/hermes` on Windows.

## Verification

POSIX:

```bash
tmp="$(mktemp -d)"
cd "$tmp"
curl -fsSLo mind.py https://raw.githubusercontent.com/Da7-Tech/mind/v6.2.10/mind.py
python3 -c "import hashlib;h=hashlib.sha256(open('mind.py','rb').read()).hexdigest();assert h=='7cb64a6bb96824a6ac00d8871b889b02d57526fc9a70cf33488ae443c8bf139c',h;print('OK')"
python3 mind.py init >/dev/null
python3 mind.py remember "the sky signal is 7413" >/dev/null
python3 mind.py recall "sky signal"
```

PowerShell:

```powershell
$Tmp = Join-Path $env:TEMP ("mind-" + [guid]::NewGuid())
New-Item -ItemType Directory $Tmp | Out-Null
Set-Location $Tmp
Invoke-WebRequest "https://raw.githubusercontent.com/Da7-Tech/mind/v6.2.10/mind.py" -OutFile mind.py
if ((Get-FileHash mind.py -Algorithm SHA256).Hash.ToLowerInvariant() -ne "7cb64a6bb96824a6ac00d8871b889b02d57526fc9a70cf33488ae443c8bf139c") {
  throw "checksum mismatch"
}
python mind.py init | Out-Null
python mind.py remember "the sky signal is 7413" | Out-Null
python mind.py recall "sky signal"
```

Expected: one result containing `7413` with a printed memory id.
