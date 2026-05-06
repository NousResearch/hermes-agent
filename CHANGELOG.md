# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changes

- run_agent/compression: Pass `custom_providers` to `get_model_context_length()` in `_check_compression_model_feasibility()` so per-model `context_length` overrides defined in `custom_providers` config are honoured for the compression model, preventing spurious "Auto-lowered threshold" warnings at startup (#20608)
