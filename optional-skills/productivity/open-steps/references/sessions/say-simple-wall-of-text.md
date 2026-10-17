# Example 1 - a status update nobody outside engineering can use

Source: a real end-of-day message, anonymised. Two calls against the same
text.

## The text

> Rebased onto main and force-pushed after the upstream squash-merge
> invalidated the branch ancestry. CI green except the flaky e2e lane (known,
> tracked). Bumped starlette to close the transitive CVE; lockfile
> regenerated. The reconcile worker now short-circuits on idempotency-key
> collision instead of raising, so Monday's retry storm can't recur. Left the
> feature flag off pending the grants ceremony.

## `say-simple` mode

> Monday's failure can't happen again: when the same request is submitted
> twice, the system now keeps one copy instead of crashing. A security hole
> in one of our libraries is closed. All automatic checks pass except one
> that is known to be unreliable. None of this is live yet - it waits behind
> a switch, until the database-permissions step that only you can run.

## `say-simple mode 3`

> 1. Nothing is live yet - it all waits on the permissions step that is yours.
> 2. Monday's double-submission failure is fixed for good.
> 3. A known security hole in one library is closed.

## What the rewrites did

| Move | Where |
|---|---|
| Branch mechanics dropped entirely | rebase and force-push change nothing for this reader |
| The caveat survived every version | "not live yet" is bad news, and in the 3-point version it ranks first |
| The not-quite-green status survived | "except one known unreliable check" - shortening may not turn amber into green |
| Terms became what the reader would see | "retry storm" → "crashing on a doubled submission" |
| The claims stayed the source's | nothing here was verified by the rewrite - as true as the message, no truer (rule 3) |
