# Changelog

## 2026-10-31

- Added regression coverage proving profile activity reporting remains best-effort when its endpoint refuses the connection.

## 2026-10-30

- Hardened the profile's Mnemosyne hourly maintenance check so unexpected OS-level subprocess failures are reported as a controlled failure instead of escaping as an uncaught exception.
