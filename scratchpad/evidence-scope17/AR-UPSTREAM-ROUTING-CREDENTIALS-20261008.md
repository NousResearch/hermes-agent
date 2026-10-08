# Independent AR: upstream routing and credentials

Outcome: **NO FINDINGS** on the final three-fix slice. The independent read-only reviewer was `/root/sdk_upstream_routing_slice/routing_ar_retry`.

Base commit: `14703a2526597461653aa24e299f950dcccc6f5c`.
Reviewed implementation SHA-256: `7e54841d4217c70eb1f845c61cf452e584001fc234eebe30319cfa967cf18e29`.
Reviewed new test SHA-256: `2fa76f742b6a0c2b56bcc08da713baf0a3d517b26c7bfdf1ceaf248378fd336c`.
Committed reviewed code and tests: `7e3a74febf7b1d57c9c2218f096d22a24167ff66`.

The slice preserves configured Anthropic Messages wire mode after Nous credential refresh, classifies the router timeout shim for provider fallback, and prevents credential-pool rotation when `resolved_api_key` was explicitly pinned. Four regression cases failed on the parent, with three passing; temporary receipt `sdk-routing-red.log` SHA-256 `cb2f869867023c963670513a05ba28ee92c6b592f93fdef6e2a2021d8d8cacc5`. Final canonical run with retries disabled passed 18 tests across four files; temporary receipt `sdk-routing-final-green.log` SHA-256 `68c9791a3943f868970aa2683cccee43731a2bece1c30ca48c2629085e080410`.

Full `scripts/check --base HEAD` completed with 11 checks OK and health 0 blocking, 0 advisory; temporary receipt `sdk-routing-check-final.log` SHA-256 `7831a613a540927421eac2558f5f1576741f2fcdcb67d70afb0d44f0481c1b37`.

Limit: the existing HTTP auth-rung test reports 7 passed and 1 failed both on the parent and patched code. The failure occurs before the changed paths because `home_io_guard` rejects a bootstrap read of the production-root worktree update marker; parent receipt `sdk-routing-existing-parent.log` SHA-256 `e34d0e9cc6f1328425752c2c408cca9cc5e740133d4a7ed284c7e95c1afc526f`. No guard bypass or real provider call was made. This AR covers the routing/credential slice only, not upstream merge, installed runtime, provider execution, or Gordon activation.
