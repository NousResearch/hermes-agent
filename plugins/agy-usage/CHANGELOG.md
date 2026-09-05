# Changelog

## 2026-09-05 — Usage & Quota summary card

### Added

- Register the read-only AGY summary card in the built-in
  `usage-quota:providers` slot.
- Keep `/agy-usage` as the detailed model/diagnostic view and link to it from
  the summary card.
- Preserve explicit local-CLI scope and `agy /usage` provenance so the card is
  not confused with Hermes session analytics or another provider's quota.

## 2026-09-05 — v0.1.0 single-account diagnostic slice

### Verified

- Confirmed the target is Hermes Dashboard, not the Naiwinai workspace.
- Built a standalone user plugin; no Hermes core files are modified.
- Added an account-oriented `accounts[]` contract so future accounts can render
  as separate cards without changing the top-level response shape.
- Backend uses only no-generation commands, sequentially:
  - `agy -p /usage --print-timeout 30s`
  - `agy models`
- Backend uses `shell=False`, never adds
  `--dangerously-skip-permissions`, and allow-lists the browser payload.
- Backend does not read or return OAuth tokens, keyring payloads, raw logs,
  command stderr, or raw provider responses.
- Quota parser preserves provider scope, remaining percentage, and reset time;
  malformed/missing quota rows become unavailable rather than fabricated zero.
- Model parser preserves exact current CLI IDs and labels.
- UI includes account identity, quota windows, model list, source/fetched time,
  loading, refresh, partial, and unavailable states.
- Focused tests: 6 passed.
- Python compile, manifest JSON validation, JavaScript syntax check, safe-command
  static check, and `git diff --check` passed.
- Real status-only probe on the target Windows machine returned one connected
  account snapshot with four quota windows and fourteen models. The probe did
  not send a model-generation prompt.

### Not verified / intentionally deferred

- Plugin is installed at the canonical user path and enabled through Hermes'
  `plugins.enabled` config. The config was read back after writing.
- Live Dashboard HTTP verification passed: plugin discovery returned the user
  manifest, all three browser assets returned `200`, and the authenticated
  plugin endpoint returned one connected account with four quota windows and
  fourteen models. Email values were withheld from verification output.
- Native Preview visual verification is still blocked: the Preview surface
  resolved to `chrome-error://chromewebdata/` while the Dashboard listener and
  HTTP endpoint were healthy. No visual pass is claimed.
- Multi-account discovery, per-email rotation, quota bypass, token migration,
  and Hermes model-routing changes are out of scope for this version.
- The quota is reported by the local `agy /usage` diagnostic command. It is not
  inferred from Hermes analytics and is not represented as a bridge-local
  estimate, but the provider's Terms/policy compatibility for third-party
  dashboard wrappers remains a separate risk gate.
