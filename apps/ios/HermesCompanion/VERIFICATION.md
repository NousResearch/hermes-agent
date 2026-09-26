# Verification and evidence limits

## Prior personal deployment

The application and renewal implementation passed 17 Core tests, 36 feature
tests, 10 Python tests, an unsigned simulator build and a signed clean device
build. Two simulator UI tests covered login preview and tab navigation earlier
in implementation. The operator reported reaching the main tabs on a physical
iPhone after real private-HTTPS login. The paired desktop showed Gateway ready
and loaded the existing Kanban board.

An actual script-only signing renewal installed over local Wi-Fi without launch.
Signature and identity checks passed; the newly installed build's expiry advanced
with approximately seven days remaining. An unrelated app's profile was unchanged.
Unauthenticated HTTPS identity and WSS requests were denied. Device identifiers,
credentials, local cron IDs and private deployment receipts are not published.

## Public upstream copy

This copy is relocated under `apps/ios/HermesCompanion`, with generic setup
instructions and no dependency on another application repository. Source-surface
compatibility is checked against the upstream checkout, not inferred from the
personal backend. See the PR for exact commands and results on this branch.

The source-surface checker is diagnostic-only: it checks registered route/method
presence and known protocol versions without running the backend. It cannot
prove payload semantics, loaded runtime version, or physical-device acceptance.

## Not established

- Full desktop feature parity or compatibility with every future backend API.
- Acceptance of live prompts, tool execution, approval responses, job triggers
  or Kanban mutations; those require disposable runtime test resources.
- Full physical-device coverage of streaming, reconnection and all charts.
- Guaranteed unattended Apple authentication, trust or installation behavior.
- Official distribution, App Store approval or notarized Mac packaging.

Do not infer these outcomes from green mocks, simulator builds or login alone.
