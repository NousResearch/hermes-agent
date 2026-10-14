# Public setup recipe and Hermes tool patterns

## Setup for a new user

1. Install Hermes Agent and enable the `terminal` and `process` tools.
2. Install and authenticate Claude Code (`npm install -g @anthropic-ai/claude-code`, then `claude doctor`).
3. Install and authenticate Codex CLI (`npm install -g @openai/codex`, then `codex --version`).
4. Put this skill under `~/.hermes/skills/autonomous-ai-agents/hermes-claude-codex-workstream/`, or install it from the Hermes skill library if available.
5. Ask Hermes to load this skill before build orchestration.
6. Start with one low-risk repo and one small slice; keep artefacts in the run directory; require local verification before reporting completion.

Example user prompt:

```text
Use the hermes-claude-codex-workstream skill. I want to fix the settings
page loading bug in this repo. Claude should plan/review, Codex should
implement in an isolated worktree, and Hermes should run final
verification before signoff.
```

## Hermes tool patterns

Claude print mode for bounded tasks (workdir is the repo or worktree the
task targets):

```python
terminal(
  command="claude -p \"Review the brief and return risks only\" --output-format json --max-turns 5",
  workdir="<absolute repo or worktree path>",
  timeout=180,
)
```

Codex needs a PTY; long bounded work runs in the background with completion
notification (never silently):

```python
terminal(
  command="codex exec --full-auto \"Implement the signed-off brief in <absolute RUN_DIR>/brief.md\"",
  workdir="<absolute WORKTREE path>",
  pty=True,
  background=True,
  notify_on_complete=True,
)
```

Poll bounded long work:

```python
process(action="poll", session_id="<session-id>")
process(action="log", session_id="<session-id>", limit=200)
process(action="wait", session_id="<session-id>", timeout=300)
```

## Auth notes

- Claude Code: browser OAuth, console auth, SSO, or `ANTHROPIC_API_KEY`.
- Codex CLI: Codex OAuth or `OPENAI_API_KEY`.
- Hermes itself may use a different provider configuration from the
  standalone CLIs. Do not assume Hermes auth proves CLI auth, or the
  reverse. Never print keys, tokens, or credential file contents.
