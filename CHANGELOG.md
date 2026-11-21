# Changelog

## 2026-10-31

- Added profile regression coverage proving backup lock cleanup also runs when an interrupted backup raises `KeyboardInterrupt`; this cycle is test-only because the existing cleanup behavior is already correct.
