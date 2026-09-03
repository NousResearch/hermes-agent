# Mutating Hook Taxonomy

This page covers the **mutating** hook category: callbacks that return a value
or raise a signal which changes control flow, as distinct from observer hooks
that are only notified after the fact and cannot alter behavior. See
[`docs/middleware/README.md`](../middleware/README.md) for the general
middleware contract (registration, payload shape, execution order) that this
page builds on.

Originally proposed in
[NousResearch/hermes-agent#64231](https://github.com/NousResearch/hermes-agent/issues/64231),
covering `#64662` (`llm_execution` block signal) and `#58524`
(`classify_api_error`).

## Two shapes, one bucket

Mutating hooks split into two patterns depending on what "mutating" means at
that call site.

### Shape A — Block signal

Used by `#64662` (`llm_execution`). A hook needs to unconditionally stop an
operation from proceeding. Raising a plain `Exception` doesn't work: the
middleware runner's general fallthrough handler catches it and continues with
the next middleware (or the terminal call) — correct for an unrelated
failure, but it defeats an intentional block. The fix is a purpose-built
exception subclass with an explicit re-raise in the runner, positioned
*before* the general fallthrough.

```python
class LLMExecutionBlocked(Exception):
    """Raised by llm_execution middleware to unconditionally prevent an LLM
    API call from proceeding. The runner re-raises this explicitly before
    the general fallthrough handler, so raising it from any middleware in
    the chain reliably propagates to the caller."""

    def __init__(self, reason: str, *, metadata: dict | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.metadata = metadata or {}
```

Usage:

```python
def budget_guard(request, next_call, session_id, **ctx):
    if cost_exceeded(session_id):
        raise LLMExecutionBlocked(
            "Session cost budget exceeded",
            metadata={"budget_usd": 5.0, "session_id": session_id},
        )
    return next_call(request)
```

`Exception`, not `BaseException`: using `BaseException` would conflate a
domain signal with true interpreter-level events (`SystemExit`,
`KeyboardInterrupt`). The explicit `except LLMExecutionBlocked: raise` line in
the runner is self-documenting — it states, at the point a reviewer is
reading the runner, "this must not be swallowed by the fallthrough path."

**The `checked_by` envelope.** Deny-path metadata (block/gate decisions, not
pure observers) may carry a small host-owned convention rather than growing
new first-class exception fields:

```python
raise LLMExecutionBlocked(
    "Session cost budget exceeded",
    metadata={
        "checked_by": "budget_guard",       # required on deny-path
        "decision": "block",
        "chain": ["budget_guard", "..."],   # optional, order that ran
    },
)
```

- `reason` stays the human/log-facing string; the envelope lives in `metadata`
  so existing raises and their tests don't churn.
- `checked_by` is **required on the deny-path**, but a plugin never has to set
  it itself: the runner backfills it from the raising callback's own
  registered name (`_run_execution_chain`'s `except LLMExecutionBlocked`
  site) if the plugin omitted it. The host fills the gap rather than failing
  closed on a missing key — a bare `raise LLMExecutionBlocked("...")` stays a
  valid, working call site.
- `decision` / `chain` are conventions a plugin may set for richer
  provenance; the runner does not require or backfill them.
- Pure observer hooks are not part of this envelope — it applies only to
  deny/gate paths like `llm_execution`.

### Shape B — First-valid-wins

Used by `#58524` (`classify_api_error`). A hook needs to supply an answer
that may or may not be given, and the first plugin to answer wins over the
built-in pipeline. Contract:

- Callback returns `None` to decline (defer to the next plugin, then the
  built-in pipeline), or a non-`None` classification to answer.
- **Dispatch is run-all, not short-circuit.** The runner calls *every*
  registered plugin for that kind, isolating each callback's failures so one
  plugin crashing or hanging can't silently block the others from running.
  It then picks the **first valid result** among everything returned, in
  registration order. `classify_api_error` does not stop calling later
  plugins once an earlier one answers — the first answer wins, but dispatch
  itself always runs to completion.
- If two plugins are both capable of answering, only the first-registered
  one's result is used. Registration order is the tie-break, worth
  documenting explicitly at the registration point rather than leaving it
  implicit — two silent-until-conflict plugins is exactly how #64714's
  "first-wins transform semantics" issue happened.
- A hook that specifically wants to *stop* dispatch after its own answer is a
  different, stricter contract than "first-valid-wins" and must say so
  explicitly — it is not implied by this shape. `classify_api_error` does not
  make that request.

(Earlier drafts of this page called this shape "value short-circuit." Renamed
to first-valid-wins because the dispatch mechanics — run every callback,
isolate failures, pick the first answer — are not a short-circuit, and the
old name implied otherwise.)

## Cross-cutting conventions (apply to both shapes)

1. **Keyword-only payload.** Every mutating hook callback takes keyword-only
   arguments, no positional payload.

2. **Privacy gate.** Any payload field that may carry raw user content or an
   unredacted provider dump (`error_body`, `error_message`, prompt/message
   text, etc.) gets called out explicitly in the hook's docstring with a
   `Privacy:` line. This doesn't try to enforce redaction in the runner — it
   makes the exposure visible and documented at the point a reviewer or
   auditor needs it.

3. **Hot-path signaling.** Each hook declares whether it fires on every call
   (hot) or only on a bounded trigger (cold). This varies *within* the
   bucket — `llm_execution` is hot (fires on every LLM call),
   `classify_api_error` is cold (fires only on API failure) — so it can't be
   inferred from "mutating" alone and needs to be explicit. Proposal: a
   `HOT_PATH: bool` class attribute or docstring field, giving a cost-guard CI
   check (flag a new hook missing one) something machine-checkable to key off.

4. **Schema versioning.** Follows whatever field-versioning convention lands
   in the taxonomy's schema-versioning work; not blocking on it here.

## Applying it to the two known members

| | `#64662` `llm_execution` | `#58524` `classify_api_error` |
|---|---|---|
| Shape | A — block signal | B — first-valid-wins |
| Payload | keyword-only | keyword-only |
| Privacy-flagged fields | request/prompt content | `error_body`, `error_message` |
| Hot/cold | **hot** — every LLM call | **cold** — API-failure-triggered only |
| Dispatch | n/a (raises to abort) | run-all, isolate failures, first valid wins |

## Open questions

- **Shared base class or per-site subclass for Shape A?** `LLMExecutionBlocked`
  is purpose-named for its call site, matching the existing
  `_DownstreamExecutionError` naming precedent in
  `hermes_cli/middleware.py`. Open to a shared base class if a future Shape A
  member wants a common catch point.
- **Does this bucket want a name in the `<subsystem>_<noun>_<verb-past>`
  grammar**, or does "mutating hook" stay a cross-cutting category above that
  grammar?
