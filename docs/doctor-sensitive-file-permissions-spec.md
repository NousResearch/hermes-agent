# Doctor Sensitive File Permissions Spec

## Problem

Hermes stores credentials in local files such as `$HERMES_HOME/.env`, `$HERMES_HOME/auth.json`, and GitHub CLI's `hosts.yml`. `hermes doctor` already checks whether some of these files exist, but it does not consistently flag when credential-bearing files are readable by group/other users on POSIX systems.

## Goal

Add a small, non-destructive doctor audit that warns when known sensitive credential files are broader than owner-only access.

## Non-goals

- Do not print secrets or file contents.
- Do not delete, rotate, or rewrite credential files.
- Do not fail Windows users where POSIX permission bits are not the security model.
- Do not add a new core model tool.

## Acceptance criteria

- `hermes_cli.doctor` exposes a helper that classifies missing, secure, and too-broad sensitive files.
- On POSIX, any existing sensitive file with group/other permission bits set is reported as a warning and adds a concrete remediation to the doctor action list.
- Existing owner-only files report OK.
- Missing optional files are skipped rather than reported as broken.
- Tests cover secure, insecure, missing, and Windows-skip behavior.
