# Hermes Agent v0.13.68

## Android

- Adds saved Tasker-style variable append, add, subtract, and literal replace
  actions to the Android automation bridge.
- Imports safe Tasker XML Variable Add, Variable Subtract, and replace-enabled
  Variable Search Replace actions as disabled Hermes automation records.
- Keeps variable transform actions bounded to the Hermes automation variable
  store with the existing value size cap and runtime variable expansion.
