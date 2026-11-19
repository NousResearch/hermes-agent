# Completion delivery lifecycle probe

Run with a fresh HOME and HERMES_HOME using the repository virtualenv:

```sh
probe=$(mktemp -d)
HOME="$probe" HERMES_HOME="$probe/state" .venv/bin/python evals/gateway_completion/delivery_lifecycle.py
```

No remote model or Discord credentials are used. The probe starts actual terminal subprocesses,
process-reader threads and async delegation workers; it uses the actual DiscordAdapter and
runner lifecycle with only outbound transport and model boundary replaced locally. It checks
notify completion, idle watch matching, raw notice-only behavior, async completion delivery,
and unavailable raw API-owner retention. `HERMES_EVAL_REPO` selects another checkout for A/B.

On base 03f3b09222b: terminal notify control succeeds, idle watch stays queued with zero turns
and the probe fails. Fixed: idle watch wakes once, the later post-turn hook adds no duplicate,
normal async Discord completion is delivered, and missing API transport remains pending at zero
attempts. Durable restart evidence additionally used ten fresh subprocesses sharing SQLite:
base exhausted eight attempts and dropped; fixed stayed pending/zero then delivered both batch
siblings at one attempt, with one API delivery row and no model turn.
