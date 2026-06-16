# /after Command — Precise Conversation Followup Targeting

## Overview

The `/after` command lets users inject context-override directives into the
agent's system prompt for precise conversational followups. It is designed for
scenarios where the user wants to reference a specific point in the recent
conversation history without repeating the full context.

## Usage

```
/after <N> <content>
```

- **`<N>`** — Number of turns back to reference (1 = most recent turn).
- **`<content>`** — The context or directive to inject.

Arguments can also be expressed as the alias:

```
/af <N> <content>
```

## Examples

```
/after 3 Focus on the error in the SQL query I showed
/after 1 Explain that docker-compose.yml more carefully
/af 5 What was the original requirement before the refactor?
```

## How It Works

1. **Storage.** When a user sends `/after <N> <content>`, the pair `(N, content)`
   is appended to an in-memory per-session queue (`_after_queue`) on the
   `GatewayRunner`. The current agent run (if any) is **not** interrupted.

2. **Injection.** On the next call to `_handle_message_with_agent`, the queue
   is drained and each entry is formatted into a system-level directive:
   ```
   [User directive: When responding, focus on what was said N turn(s) ago.
   Context reference: <content>]
   ```
   These directives are appended to `context_prompt` before it is passed to
   `_run_agent`.

3. **One-shot consumption.** Each `/after` entry is consumed exactly once —
   after injection there is no persistent state. If the user needs the
   reference again, they send another `/after` command.

## Cold Path (No Running Agent)

When no agent is currently running for the session, `/after` stores the entry
in `_after_queue` and rewrites the event text so the agent processes the turn
normally with the context directive injected into the context prompt.

## Hot Path (Agent Running)

When an agent is actively running, `/after` stores the entry and returns an
acknowledgment without interrupting. The directive arrives on the **next**
agent turn.

## Design Decisions

- **Turn boundary, not mid-run.** Unlike `/steer` (which lands between
  tool-call iterations), `/after` operates at the turn boundary — the content
  is injected into the system prompt for the next full agent processing cycle.
  This avoids disrupting in-progress tool chains while still providing precise
  targeting.

- **N-based indexing, not content search.** The numerical index (N turns back)
  is simpler, more reliable, and avoids the ambiguity of fuzzy content
  matching. Users who need content-level precision can include distinguishing
  keywords in the `<content>` portion.

- **In-memory only.** The `_after_queue` is not persisted across gateway
  restarts. This keeps the implementation lightweight and avoids complexity
  around serializing ephemeral user directives.

## Implementation

Files modified:

- `hermes_cli/commands.py` — Added `after` to `COMMAND_REGISTRY` with alias `af`.
- `gateway/run.py` — Added `_after_queue` dict, command handler in both hot-path
  and cold-path dispatch, and context injection in `_handle_message_with_agent`.
- `docs/after-command.md` — This documentation file.

Key locations:

| Component | Location |
|-----------|----------|
| Command registration | `hermes_cli/commands.py` (COMMAND_REGISTRY) |
| Queue storage | `GatewayRunner._after_queue` in `gateway/run.py` |
| Hot-path dispatch | `gateway/run.py` (running-agent guard section) |
| Cold-path dispatch | `gateway/run.py` (after steer handler) |
| Context injection | `gateway/run.py` (_handle_message_with_agent, after voice context) |
