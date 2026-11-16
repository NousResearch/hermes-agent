# Codex input ownership probe

Run in a fresh process from the checkout:

```sh
PYTHONPATH=. .venv/bin/python evals/codex_echo/probe.py
```

The executable fixture is an **offline JSON-RPC peer**, not a live Codex service.
It exercises production subprocess pipes, protocol handling, event projection,
message splice, real SQLite writes, and conversation replay in isolated homes.
No actual provider, Telegram account, or recipient is contacted.

## Independently repeated A/B

Base: `d8a07768c5ee59afae62e9fb51d45e1e30aa74da`.
Fix implementation: `fdaae2635b6df5202e8bb74eb4662159b7735f99`.
Each case accepts the same input twice. Both plain/platform-ID and rich/keyless
inputs run through every row of this matrix:

| Projection | Expected user rows | Base | Fix |
|---|---:|---:|---:|
| Leading exact echo | 2 | 4 | 2 |
| Assistant only | 2 | 2 | 2 |
| Leading different user event | 4 | 4 | 4 |
| Exact echo, assistant, later identical user event | 4 | 6 | 4 |

Base passes 4/8 cases; fix passes 8/8. This confirms the local ownership mechanism,
not provider fidelity or the issue's proposed universal gateway double-write
mechanism. Historical duplicates are not migrated. The probe checks final reply
text and durable user-row counts; the regression invariant additionally checks
exact replayed user content. The separate transport invariant checks that the
recorded submitted text equals the actual `turn/start` payload.

Independent review checked the sole splice caller, turn-start ownership,
rich-input coercion, no-echo/nonmatching controls, and preservation of later
identical events and independently repeated accepted inputs. No production
change was needed after review. Recovery work for #104691 remains separate in
#104857; its original macOS initiating failure is not established here.
