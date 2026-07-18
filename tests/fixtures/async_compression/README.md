# Async-compression replay fixtures

Sanitized, read-only copies of real session transcripts used by
`scripts/replay_async_compression.py` to exercise the background-compression
candidate lifecycle against real conversation shapes — **without** sending any
message to a provider and **without** executing any tool present in the
history. The replay treats every message as opaque transcript data.

## Format

One `*.json` file per session: a JSON **list** of OpenAI-style message dicts
(at least 8 messages), e.g.

```json
[
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": null,
   "tool_calls": [{"id": "call_1", "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"}}]},
  {"role": "tool", "tool_call_id": "call_1", "name": "read_file",
   "content": "..."}
]
```

Only the fields the provider sees matter (`role`, `content`, `name`,
`tool_call_id`, `tool_calls`). Internal persistence metadata (underscore
keys, `id`, `timestamp`, `active`, …) is ignored by the canonical digest and
may be stripped.

## Exporting a real session (read-only)

Run against a **copy** of `state.db`, never the live file:

```bash
cp ~/.hermes/state.db /tmp/state-copy.db
python - <<'EOF'
import json, sys
sys.path.insert(0, "/path/to/hermes-agent")
from hermes_state import SessionDB
db = SessionDB(db_path="/tmp/state-copy.db")
rows = db.get_messages("SESSION_ID", include_inactive=False)
keep = ("role", "content", "name", "tool_call_id", "tool_calls")
out = [{k: r.get(k) for k in keep if r.get(k) is not None} for r in rows]
print(json.dumps(out, ensure_ascii=False, indent=1))
EOF
```

## Sanitization checklist (mandatory before committing)

- Remove or mask credentials, API keys, tokens, phone numbers and any
  personal identifiers (names, addresses, customer data).
- Remove absolute paths that reveal infrastructure layout when not needed
  for the scenario.
- Keep tool_call/result **pairs together** — the replay validates that no
  pair is ever split.
- Prefer long sessions (50+ messages) with mixed shapes: tool rounds,
  subagent delegation, previous `[CONTEXT COMPACTION]` summaries.

## Running

```bash
python scripts/replay_async_compression.py           # builtin + all fixtures
python scripts/replay_async_compression.py --json    # machine-readable
python scripts/replay_async_compression.py --scenario fixture_<name>
```

The structural gate requires 100% of the checks (suffix intact, tool pairs
valid, no duplicates, history archived + searchable, summary within budget,
sentinels preserved). Any failure blocks the canary rollout.
