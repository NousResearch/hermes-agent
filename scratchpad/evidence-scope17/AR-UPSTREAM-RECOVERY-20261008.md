# Independent AR: upstream recovery controls

Outcome: **NO FINDINGS** on the final recovery slice.

Base commit: `cddf1809b31bc67647afad97baa9380055ebccde`.
Reviewed implementation diff SHA-256: `125b2789cf494e1b0632087477367e7f388fb3fef1cb7cc161861bb5ff95a119`.
Reviewed `agent/auxiliary_client.py` SHA-256: `412f9cf77cbe7b23de750c03a8c96c6939a4124faa0f52bd6748a8475ac50cf4`.
Reviewed `tests/agent/test_auxiliary_recovery_request_controls.py` SHA-256: `bff6efb774f0a135f41ea243928be18970617251157105e94ffc8134e48bd8c5`.
The worker committed those exact files as `14703a2526597461653aa24e299f950dcccc6f5c` after the review.

The independent reviewer verified the final code after a style-only trim. A canonical run of the seven new tests passed with retries disabled. Its temporary receipt `recovery-ar-canonical.log` has SHA-256 `4e8b2b558cb1d181a4994a433882575b817bea6971b570c55609b0e93d65424b`. The worker's five-file canonical batch passed 36 tests; temporary receipt `recovery-green-final.log` SHA-256 `9e1485e2fffc4a74d4189955f9b32b726a97895d057f8db74cfb7fe0c1249ffb`. The red baseline temporary receipt `recovery-red.log` SHA-256 is `371842fff6177cfa880a0741b9824c7225d033b30d6427dfea49438d6f09b2dc`.

The full `scripts/check` rerun completed with 11 checks OK, health 0, blocking 0, advisory 0. Temporary receipt `recovery-check-final.log` SHA-256: `1222a0fd088ebc631811106d26efc33176cca6a6bfeb53eb9ba9769d8d918fc3`.

Limit: the pre-existing HTTP auth end-to-end test could not run under the isolated worktree's HomeIOGuard because it touched the installed repository's common Git state. No guard bypass or installed-checkout mutation was performed. A preliminary reviewer `pytest` invocation was disclosed; the canonical seven-test rerun above is the independent validation used for this outcome. This AR covers the recovery slice only, not the later routing/credential slice, upstream merge, installed runtime, provider execution, or Gordon activation.
