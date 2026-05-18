# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changes

- cli/tui: Replace `print()` with `_cprint()` in `process_command` handlers (`_handle_rollback_command`, `_handle_browser_command`, `_handle_tools_command` usage hints, `/reload` and `/plugins` inline handlers) so output is not swallowed when stdout is redirected or patched by prompt_toolkit's TUI (#20711)
- plugins/honcho: Strip `/v3` path prefix from `base_url` before passing it to the Honcho SDK constructor for local/self-hosted instances to prevent double-prefixing that causes 404 errors (#20688)
- run_agent: Pass `custom_providers` to `get_model_context_length()` inside `_check_compression_model_feasibility()` so per-model `context_length` overrides in `custom_providers` config are honoured for the compression model, preventing spurious "Auto-lowered threshold" warnings (#20608)
