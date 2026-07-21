# Spec: Doctor secret-file permission check

## Problem

Hermes stores sensitive local secrets in files such as `$HERMES_HOME/.env`,
`$HERMES_HOME/auth.json`, and GitHub CLI `hosts.yml`. Existing writers try to
create new files with restrictive modes, but users can still end up with old or
manually-edited files that are group/world-readable. Joe's operating manual calls
out periodic permission verification as part of the system-layer protection
model.

## Goal

Add a small `hermes doctor` diagnostic that detects obviously over-permissive
secret files and gives safe, copy-pasteable remediation guidance without printing
secret contents.

## Non-goals

- Do not delete, rewrite, or rotate credentials automatically.
- Do not inspect secret values.
- Do not enforce POSIX modes on Windows, where mode bits are not reliable.
- Do not add network calls.

## Behavior

- On POSIX platforms, `hermes doctor` checks:
  - `$HERMES_HOME/.env`
  - `$HERMES_HOME/auth.json`
  - `$HOME/.config/gh/hosts.yml`
- Missing files are ignored.
- A checked file is OK when no group/other permission bits are present.
- A checked file warns when any group/other permission bit is present and adds a
  manual issue with a `chmod 600 <path>` suggestion.
- On Windows, the check is skipped with a clear informational line.

## Tests

- Unit-test the pure permission evaluator with secure and insecure files.
- Unit-test missing files are ignored.
- Unit-test the doctor rendering helper appends a manual issue for insecure
  files without printing file contents.
- Skip POSIX mode tests on Windows.
