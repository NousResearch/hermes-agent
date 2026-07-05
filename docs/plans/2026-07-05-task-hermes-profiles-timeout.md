# TASK-HERMES-PROFILES-TIMEOUT

## Summary

Hermes Desktop can show profile-loading failures during cold startup when the backend is still warming. The profile list endpoint already uses a longer timeout, but `/api/profiles/active` and other boot reads were still using Electron's shorter default timeout.

## Fix Contract

- Treat profile, status, model-info, logs, and config reads as cold-start boot reads.
- Use the shared 60s boot timeout for those reads so slow startup does not produce false profile-load failures.
- Keep profile mutations and other explicit user actions on their existing timeout behavior unless they are proven to be cold-start reads.

## Verification

- Unit tests assert the longer timeout request shape for boot reads.
- Isolated dev Electron launch should run past the old 15s failure window without `/api/profiles/active` or status/config/model-info timeout errors.
