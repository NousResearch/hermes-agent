# Changelog

## 2026-11-03

- Added deterministic coverage proving the post-merge self-improvement check refuses to pass with a dirty backup repository.

## 2026-11-01

- Added deterministic regression coverage for modern MCP activity names, preserving the full tool suffix in emitted events.

## 2026-10-31

- Added regression coverage proving profile activity reporting remains best-effort when its endpoint refuses the connection.

## 2026-10-30

- Hardened the profile's Mnemosyne hourly maintenance check so unexpected OS-level subprocess failures are reported as a controlled failure instead of escaping as an uncaught exception.
