# Approval Architecture

This doc maps the two approval systems Hermes uses to gate dangerous
agent behavior. The audit (F-I1) originally flagged this as "dual
approval systems bridged via ``submit_pending()`` — accidental
coupling." Closer reading showed the bridge is intentional and the two
systems have genuinely different responsibilities. This doc records
that so future refactors don't collapse them by mistake.

## The two systems

| Module | Gates | Fires on | Decision input |
|---|---|---|---|
| ``tools/approval.py`` | dangerous SHELL COMMANDS | subprocess strings, arg values | regex patterns in ``DANGEROUS_PATTERNS`` |
| ``agent/approval.py`` | dangerous TOOL CALLS | any tool marked ``requires_confirmation=True`` | tool flag, not content |

A single tool call can trigger either, both, or neither:

- ``terminal("rm -rf /")``        → both: ``agent/approval`` fires because
                                   ``terminal`` may be marked
                                   ``requires_confirmation`` in some
                                   profiles; ``tools/approval`` also fires
                                   because ``rm -rf`` matches the regex.
- ``terminal("ls")``              → neither, if ``terminal`` has
                                   ``requires_confirmation=False`` and
                                   ``ls`` is not dangerous.
- ``web_fetch(url)``               → ``agent/approval`` only (if the tool
                                   is flagged) — no shell, no regex gate.
- ``execute_code("subprocess.run([...])")`` → ``tools/approval`` only —
                                   the code runs a shell, but the tool
                                   itself has no requires_confirmation
                                   flag.

## The bridge

Exactly one call connects the two:

```
agent/approval.py::ApprovalManager.resolve_action()
    └─ if HERMES_GATEWAY_SESSION set:
           from tools.approval import submit_pending
           submit_pending(session_key, { command: "...", pattern_key: "..." })
```

Rationale: the gateway has an SSE approval stream the companion UI
already speaks (via ``tools/approval``'s state machine). When a
tool-flag approval needs to reach the user, reusing that pipe is
cheaper than duplicating it. The companion sees both as "a command
needing approval."

## Shared contract (do not break)

Both systems read:

- ``HERMES_SESSION_KEY`` env var — session identity for approval state
- ``HERMES_GATEWAY_SESSION`` env var — marker that we're in gateway mode
- ``tools.approval._approval_session_key`` contextvar — per-thread/task
  override of ``HERMES_SESSION_KEY`` (v0.7.0+)

Changing the name, semantic, or precedence of any of these breaks
cross-system coordination. The contextvar takes precedence over the env
var so concurrent gateway request handlers don't share state.

## When a refactor IS appropriate

- If you add a new approval gate kind (e.g. per-resource-quota), put it
  in its own module with its own decision surface. Do not fold it into
  either of these files.
- If you rename any of the three env-var/contextvar names above, grep
  for them across the tree first — there are callers in ``agent/core.py``,
  ``gateway/run.py``, and tests that must all move together.
- If you believe the two files should merge: confirm the bridge is the
  only coupling (grep for ``from tools.approval``, ``from agent.approval``),
  then extract a shared ``hermes.approval`` package with ``patterns.py``,
  ``tool_flags.py``, and ``gateway.py`` submodules. This was proposed as
  full F-I1 but deferred — the duplication is low and the bridge is
  well-contained.

## Audit status

- Flagged by audit 2026-04-15 as F-I1 (informational).
- Outcome: kept dual systems, added this doc + cross-reference docstrings
  in each module so the relationship is discoverable without reading
  both files end-to-end.
