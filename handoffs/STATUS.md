# TH1-900 — Plugin development-group isolation

Branch: `fix/TH1-900-plugin-runtime-groups`
Base: stable Hermes `71d34bad1045610252eda02ee79549436896e784`.

Goal: install jev-effort-router 0.3.0 across M5/M4/WSL without changing stable core dependency pins or weakening the plugin's own developer requirements.
Root cause: PM copies plugin-only dependency groups into its generated runtime uv workspace. uv resolves all declared groups even with no-default-groups, so plugin Ruff>=0.17 conflicts with core Ruff==0.15.10. Local-only developer groups are not published runtime package requirements.
Chosen fix: omit non-published dependency-group declarations and their group-only uv configuration from generated plugin runtime snapshots; preserve published runtime dependencies/extras and original source metadata. Core test environment remains separate.

Next: one offline real-uv regression red on base; implement/review; relevant existing PM suite once; backed-up local patch and real catalog update; functional plugin smoke; peer backups/rollout/verifications. Keep main/delegation routing unchanged.
No live source mutation yet. PR pending. Finish/stop with pushed handoff and Linear comment.
