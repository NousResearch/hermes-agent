# Deferred residual lines — pullable patch set (onto v0.17.0)

The 37 split PRs cover all 137 changed files at the FILE level. A small set of post-snapshot
residual LINES are deferred per documented user policy. This branch makes them PULLABLE: each
`.patch` here is `git diff v0.16.0 -- <file>` for one deferred file; apply with `git apply`
(may need `--3way` against v0.17.0 due to upstream drift). Categories are separated per policy.

- private-overlay/  : [Hermes] v2026.6.5 update-overlay machinery — NOT for upstream (reference only).
- copilot-limits/   : account-sensitive caps — apply AFTER generalizing account values.
- cmx/              : CMX-touching — belongs in the single CMX-implementation PR (not yet opened).

Each is one `git apply` away from its content on v0.17.0; nothing is silently dropped.
