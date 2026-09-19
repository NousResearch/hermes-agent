# hermes-agent — Roadmap

> Autonomously maintained by the roadmap sync (reliability-first). Items cite reproducible codebase signals; acceptance is proven by cited evidence.

**Vision**: A reliable, customer-friendly repository advanced by evidence-cited roadmap cycles owned by the autonomy loop

**Pillars**: reliability work outranks customer-experience work; every roadmap item cites reproducible codebase signals; acceptance is proven by cited evidence, never claimed

## Open items

### Add test coverage for 21 untested module(s)
- id: `rm-002` | track: reliability | priority: 100.0 | status: candidate
- signals: reliability.no_tests:*, reliability.no_tests:.github/scripts/run-workspace-checks.mjs, reliability.no_tests:.worktrees/t_3143b8b5/.github/scripts/run-workspace-checks.mjs, reliability.no_tests:.worktrees/t_3143b8b5/agent/api_error_summary.py, reliability.no_tests:.worktrees/t_3143b8b5/agent/api_request_hooks.py (+16 more)
- acceptance: Every module in ['*', '.github/scripts/run-workspace-checks.mjs', '.worktrees/t_3143b8b5/.github/scripts/run-workspace-checks.mjs', '.worktrees/t_3143b8b5/agent/api_error_summary.py', '.worktrees/t_3143b8b5/agent/api_request_hooks.py', '.worktrees/t_3143b8b5/agent/auxiliary_health.py', '.worktrees/t_3143b8b5/agent/auxiliary_wire.py', '.worktrees/t_3143b8b5/agent/chat_completion_helpers_relay.py', '.worktrees/t_3143b8b5/agent/chat_completion_stream_monitor.py', '.worktrees/t_3143b8b5/agent/compression_facade.py', '.worktrees/t_3143b8b5/agent/credential_pool_admin.py', '.worktrees/t_3143b8b5/agent/credential_pool_model_cooldowns.py', '.worktrees/t_3143b8b5/agent/lazy_forward.py', '.worktrees/t_3143b8b5/agent/reasoning_params.py', '.worktrees/t_3143b8b5/agent/terminal_approval_batch.py', '.worktrees/t_3143b8b5/agent/transcript_repair.py', '.worktrees/t_3143b8b5/agent/turn_empty_response.py', '.worktrees/t_3143b8b5/agent/turn_preflight_gate.py', '.worktrees/t_3143b8b5/agent/turn_response_intake.py', '.worktrees/t_3143b8b5/agent/turn_stop_gates.py', '.worktrees/t_3143b8b5/agent/turn_tool_round.py'] has a corresponding test file with at least one passing test
- evidence: CI: pytest collects the new test files and they pass

### Refactor 21 high-complexity function(s)
- id: `rm-001` | track: reliability | priority: 90.0 | status: candidate
- signals: reliability.complexity_hot:*, reliability.complexity_hot:hermes_cli/web_dist/assets/react-vendor-BoVnYuL4.js::L1, reliability.complexity_hot:hermes_cli/web_dist/assets/react-vendor-BoVnYuL4.js::L1, reliability.complexity_hot:hermes_cli/web_dist/assets/react-vendor-BoVnYuL4.js::L1, reliability.complexity_hot:hermes_cli/web_dist/assets/react-vendor-BoVnYuL4.js::L1 (+16 more)
- acceptance: Each flagged function is decomposed below the branch threshold with behavior locked by characterization tests
- evidence: ast-based branch-count check passes in CI

### Refresh stale top-level documentation
- id: `rm-003` | track: reliability | priority: 43.0 | status: candidate
- signals: reliability.stale_doc:README.md
- acceptance: Docs regenerated/updated; staleness detector reports 0 signals
- evidence: inference.stale_docs returns [] for the repo

<!-- managed by hermes-roadmap render; do not edit by hand -->
