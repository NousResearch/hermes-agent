# Changelog

## [Unreleased]

### Changed

- **Codex OAuth singleton: root write-through + reuse-rescue (#87503).** When a
  profile refreshes a Codex OAuth grant it resolved from the global-root
  auth store, the rotated token chain is now written back to root (and, for
  owned-block callers, mirrored to root on every save) so sibling profiles no
  longer replay a consumed refresh token and die with `invalid_grant`. A
  relogin-required refresh now also attempts a *reuse-rescue* — adopting a
  fresher sibling chain already held by root — before falling back to
  `~/.codex` CLI recovery. This is **self-healing, not preventive**: the first
  failed POST that produced the relogin error still happens outside any lock,
  exactly as before; the change heals it, it does not serialize it. Concurrent
  same-chain callers may each burn one POST, and each loser's rescue succeeds
  afterward using the winner's chain.

  Persistence failures are classified into two durability classes and logged
  accordingly:

  - **CLASS-D (durable, self-healing):** any failure where the rotated chain
    already has a durable store copy (e.g. the profile store saved but the
    root write-through failed). Logged as a WARNING; the residual sync gap
    closes via the next-trigger resync, and the return-value contract is
    unchanged.
  - **CLASS-N (no durable copy):** any failure where no store durably recorded
    the rotated chain (e.g. a root-resolved caller whose direct root write
    failed, leaving root holding the consumed pre-refresh chain). Logged as
    CRITICAL after exactly three in-call persistence attempts with two fixed
    backoffs (0.5s then 1.0s); the refreshed tokens are still returned, and
    the next refresh naturally re-enters the relogin/rescue path. The message
    names `hermes model` as the remediation for a persistent failure.
