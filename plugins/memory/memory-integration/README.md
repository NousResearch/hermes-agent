# memory-integration

Minimal read-only provider skeleton for reporting memory-integration configuration status.

This Workstream 2B slice exposes only `memory_integration_status`. It reports status without writing vault content or local state.

Status path reporting redacts absolute paths by default. Set `memory_integration.status.include_absolute_paths: true` in Hermes `config.yaml` to opt in to absolute path output.
