# Skill Authoring Standards

Every new or modernized skill — bundled, optional, or contributed — must meet these standards before merge. Reviewers reject PRs that violate them.

## 1. Description constraints
`description` must be ≤ 60 characters, one sentence, ends with a period. State the capability, not the implementation. No marketing words ("powerful", "comprehensive", "seamless", "advanced"). Don't repeat the skill name. Verify with:
```python
import re, pathlib
m = re.search(r'^description: (.*)$',
              pathlib.Path('skills/<cat>/<name>/SKILL.md').read_text(),
              re.MULTILINE)
assert len(m.group(1)) <= 60, len(m.group(1))
```

## 2. Tools referencing
Tools referenced in SKILL.md prose must be native Hermes tools or MCP servers the skill explicitly expects. Point at the proper tool by name in backticks (`` `terminal` ``, `` `web_extract` ``, `` `read_file` ``, `` `patch` ``, `` `search_files` ``, `` `vision_analyze` ``, `` `browser_navigate` ``, `` `delegate_task` ``, etc.). Do NOT name shell utilities the agent already has wrapped — `grep` → `search_files`, `cat`/`head`/`tail` → `read_file`, `sed`/`awk` → `patch`, `find`/`ls` → `search_files target='files'`. If the skill depends on an MCP server, name the MCP server and document the expected setup in `## Prerequisites`.

## 3. Platform gating
`platforms:` gating must be audited against actual script imports. Skills that use POSIX-only primitives (`fcntl`, `termios`, `os.setsid`, `os.kill(pid, 0)` for liveness, `/proc`, `/tmp` hardcoded, `signal.SIGKILL`, bash heredocs, `osascript`, `apt`, `systemctl`) must declare their supported platforms. Default posture: try to fix it cross-platform first — `tempfile.gettempdir`, `pathlib.Path`, `psutil.pid_exists`, Python-level filtering instead of `grep`.

## 4. Contributor credit
`author` credits the human contributor first. For external contributions, the contributor's real name + GitHub handle goes first; "Hermes Agent" is the secondary collaborator.

## 5. SKILL.md layout
SKILL.md body uses the modern section order:
1. `# <Skill> Skill` title
2. 2-3 sentence intro stating what it does and doesn't do
3. `## When to Use`
4. `## Prerequisites`
5. `## How to Run`
6. `## Quick Reference`
7. `## Procedure`
8. `## Pitfalls`
9. `## Verification`

Target ~200 lines for a complex skill, ~100 lines for a simple one. Cut redundant intro fluff, marketing prose, and re-explanations of env vars.

## 6. Directory structure
Scripts go in `scripts/`, references in `references/`, templates in `templates/`. Don't expect the model to inline-write parsers, XML walkers, or non-trivial logic every call — ship a helper script. Reference it from SKILL.md by path relative to the skill directory.

## 7. Tests
Tests live at `tests/skills/test_<skill>_skill.py` and use only stdlib + pytest + `unittest.mock`. No live network calls. Run via `scripts/run_tests.sh tests/skills/test_<skill>_skill.py -q`.

## 8. Env blocks
`.env.example` additions are isolated to a clearly delimited block. Don't touch the surrounding file.
