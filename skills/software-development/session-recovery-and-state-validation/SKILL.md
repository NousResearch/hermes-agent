---
name: session-recovery-and-state-validation
description: Recover an exact prior user prompt or intent from Hermes session history, then validate the present filesystem/host state so you can distinguish lost context from lost implementation.
tags: [sessions, recall, recovery, validation, ssh, git, mission-control]
---

# Session recovery and state validation

Use this when:
- The user says a PC/server reboot or crash may have lost prior instructions
- The user asks "can you still see my prompt?" or references earlier work that may be missing
- You need to confirm whether only chat context was interrupted, or whether scripts/config/repos were actually lost
- You need to recover a Mission Control / UI intent that another agent implemented incorrectly

## Goal
Recover the exact instruction or closest authoritative wording from prior sessions, then verify the current machine/repo state so the answer is grounded in evidence rather than memory alone.

## Recommended workflow

1. Search cross-session history first
- Use `session_search` with broad OR queries combining nouns from the user’s request.
- Example patterns:
  - `Pluto OR Henry OR scripts OR configwijzigingen OR prompt`
  - `"Mission Control" OR mission-control OR responsive OR device OR split mockup OR Cursor`
- Prefer `role_filter=user,assistant` when you only need human/assistant narrative.

2. Find the exact quoted prompt in raw session files
- If the summary is not precise enough, inspect `~/.hermes/sessions/session_*.json` directly.
- Use `search_files` on the session JSONs to find distinctive fragments.
- Then use `read_file` around the matching line range to recover the exact user wording.
- This is the most reliable way to answer “can you still see my prompt?” with proof.

3. Distinguish intent recovery from implementation recovery
- After recovering the prompt, explicitly separate:
  - recovered instruction/intent
  - current implementation state on disk/hosts
- Do not assume that because the prompt exists, the implementation was completed.

4. Validate the current host state with live checks
- For local/WSL state, use `terminal`.
- For remote Pluto/Henry state, use SSH via the known Hermes key.
- Check the exact folders/repos named in the recovered prompt.
- For Mission Control-style tasks, inspect:
  - repo presence
  - current branch
  - remotes
  - tracked/untracked changes
  - key source files (`page.tsx`, layout files, config files, package.json)

5. When checking whether an editor/tool is really installed, verify multiple signals
- Don’t rely on one signal.
- For Cursor on Linux/Pluto, check:
  - `command -v cursor`
  - desktop entries (`~/.local/share/applications`, `/usr/share/applications`)
  - install directories (`~/.local/opt/cursor`, `/usr/share/cursor`)
- For Windows from WSL, query PowerShell for:
  - `Get-Command`
  - installed app registry entries
  - expected EXE paths
- Report absence clearly if all checks are empty.

6. For UI misimplementation claims, inspect the code before concluding
- If the user says another agent misunderstood a responsive mockup/example, open the real UI source files.
- Look for evidence like:
  - manual layout toggles
  - separate layout variants (A / C / A+C)
  - literal split-screen rendering
  - missing device-aware breakpoints
- Then restate the intended behavior in plain language:
  - one responsive app
  - mobile shows mobile experience
  - desktop shows desktop experience
  - experimental variants are UX modes, not literal side-by-side mockups

7. Answer in three parts
Structure the response as:
- What I recovered from your prior prompt
- What I verified in the current environment
- My single recommendation / next action

## Useful command patterns

Recover exact prior prompt from session JSON:
- `search_files(pattern="Mission Control|telefon|desktop versie|split test", target="content", path="~/.hermes/sessions", file_glob="session_*.json")`
- `read_file(path="~/.hermes/sessions/session_<id>.json", offset=<near-match>, limit=40)`

Check Pluto Mission Control repo state:
- `ssh -i ~/.ssh/hermes_access_ed25519 -o BatchMode=yes sander@100.108.223.25 'cd ~/.openclaw/workspace/mission-control && git rev-parse --is-inside-work-tree && git status --short && git remote -v && git branch --show-current'`

Check Cursor on Pluto:
- `ssh -i ~/.ssh/hermes_access_ed25519 -o BatchMode=yes sander@100.108.223.25 'command -v cursor || true; find ~/.local/share/applications /usr/share/applications -maxdepth 1 -iname "*cursor*.desktop" 2>/dev/null; find ~/.local /opt /usr/share -maxdepth 3 \( -iname "*cursor*" -o -iname cursor \) 2>/dev/null | sed -n "1,40p"'`

Check Windows-side Cursor from WSL:
- `powershell.exe -NoProfile -Command '$cursor = Get-Command cursor -ErrorAction SilentlyContinue; $cursorExe = Get-ChildItem "$env:LOCALAPPDATA\Programs\Cursor\Cursor.exe" -ErrorAction SilentlyContinue; $cursorApp = Get-ItemProperty "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*" -ErrorAction SilentlyContinue | Where-Object { $_.DisplayName -like "*Cursor*" } | Select-Object DisplayName, DisplayVersion, InstallLocation; ...'`

## Pitfalls
- `session_search` summaries are excellent for orientation, but not enough when the user asks for the exact prompt text.
- Session JSON grep results can be noisy because tool schemas also contain words like `prompt`; narrow to distinctive user phrases when possible.
- `printf '---'` over SSH can fail because leading dashes are parsed as options in some shells; prefer `echo` for separators.
- A repo can be a git worktree but still have no remote configured. Always check both `git status` and `git remote -v`.
- When checking Windows apps from WSL, empty PowerShell output usually means “not detected”, not tool failure, if the exit code is 0.

## Success criteria
You should end with evidence-backed statements such as:
- the exact recovered user prompt
- whether the current implementation exists on disk
- whether the target tool/editor is actually installed
- whether the repo is properly connected to GitHub or still only local
- whether the current code matches the user’s intended outcome or reflects a misunderstanding