# Hermes Agent v0.13.32

This release adds the next Tasker-style Android automation slice for the native Hermes app.

## Android Automation

- Added explicit saved `location` trigger support for `android_automation_tool`.
- Added `run_location_trigger` dispatch with latitude/longitude event validation.
- Added saved location filters for coordinate radius, provider, place/name, and maximum accuracy.
- Added Tasker-style location variables: `%LOC`, `%LAT`, `%LON`, `%LOCACC`, `%LOCPROVIDER`, `%LOCNAME`, plus `LOCATION_*` aliases.
- Updated native model tool instructions so local Gemma and other tool-calling models can create and dispatch location-triggered saved tasks.

## Boundaries

- Location triggers are explicit event dispatches in this release. Hermes does not silently collect background GPS or register a provider-backed observer yet.
- Shizuku remains user-started and permission-gated for privileged actions.

## Validation

- Focused unit and instrumentation tests cover location trigger creation, miss/match dispatch, variable expansion, and real app-workspace file writes.
