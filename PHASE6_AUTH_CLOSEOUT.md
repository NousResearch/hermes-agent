# Phase 6.8 — Integration and closeout

Phase 6 is an independent authentication ownership cut based directly on One
Gateway `bc1b572e2c`, on `refactor/phase6-auth-credentials`. It does not depend
on Phases 1–5. See the preceding Phase 6 ownership notes for the implementation.

## Closeout changes

- Correct remaining Codex status/device-login and Nous shared-store test patches
  to target the defining modules. The loopback refresh test now patches the token
  endpoint in the canonical runtime provider rather than CLI presentation.
- Update the command-resolution allowlist for the Anthropic setup-token helper,
  whose browser/terminal presentation moved to `hermes_cli/auth_anthropic.py`.
- Verify every auth submodule imports without either CLI namespace.
- Update contributor and provider/pool documentation to identify canonical auth
  owners and explicit application-supplied environments.

## Validation

All Python regression runs use `scripts/run_tests.sh`, with a disposable PM-built
Python 3.14.7 test environment and isolated credential homes on native Windows.

The integration selection covers 445 files: all changed regression files, auth
and provider suites, credential/OAuth/profile-scope consumers, and packaging
checks. Its initial failures include 11 closeout regressions, all subsequently
fixed, 45 failures reproduced on the exact One Gateway baseline, and one webhook timing flake. The
654-test TUI server file exceeded the default 300-second file limit and was
rerun separately with a longer limit.

The focused final follow-up passes 153 tests across 16 files, with no failures.
It includes the corrected test files, managed runtime resolution, all auth tests,
refresh-stampede prevention, cloned-profile grant ownership, provider boundaries,
credential lifecycle and multiplex cloud credential isolation.

- Syntax compilation: all 8,348 tracked Python files pass.
- Wheel build: all 48 auth modules are present; retired implementations are absent.
  All 46 non-package auth modules import from an extracted fresh wheel, and both
  main CLI help and auth help succeed from that installation.
- Web, Ink TUI and Electron desktop source builds all pass.
- Structural ownership and fresh-process no-CLI-import gates pass.
- Compatibility audit: no in-tree use of any of the 2,084 existing compatibility
  pointers. External auth type identity and provider hook contracts pass in a
  fresh process.
- Documentation audit finds no retired implementation paths in website docs,
  skills or AGENTS files. `git diff --check` passes.

A separately selected 84-test TUI tail completes: 80 pass and four SSH-profile
assertions fail. All four reproduce on the baseline in two corresponding
selections (78 passes / one failure, and two passes / three failures).
The full file's timeout remains a validation limitation, not a passing result.
An extended retry was stopped after eight minutes without further progress at
`test_prompt_submit_releases_old_history_before_heap_trim`.
The full baseline file also fails to complete within its 340-second job budget.
The independent review is draft PR https://github.com/Lokee86/hermes-agent/pull/4
against One Gateway `bc1b572e2c`; the broad gate is not entirely green on this host.

## Existing baseline failures

These are tested baseline limitations, not unexplained exclusions. The initial
45 failing tests reproduce on `bc1b572e2c` using the same interpreter and runner:

| Area | Failing tests | Baseline cause |
| --- | ---: | --- |
| Gateway daemon/automation fixtures | 5 | Child environments omit Windows socket/home variables |
| Gateway native HTTP boundary | 1 | Local listener connection failure |
| CLI update/restore authority | 2 | Child environment cannot resolve the native Windows home |
| Agent write-credential safety | 2 | POSIX `pwd` imports on Windows |
| Honcho path presentation | 1 | Windows separators differ from the asserted POSIX string |
| MCP OAuth tests | 32 | Native Windows console detection bypasses mocked stdin |
| PM downloader tests | 2 | Windows path-length failure in temporary staging paths |

The baseline webhook-authority test also failed under the initial loaded run but
passed in the separate baseline run; its failure is recorded as a timing flake,
not silently counted as a Phase 6 regression.

## Compatibility and integration

The demonstrated external plugin boundary retains
`agent.credential_pool.load_pool`, `PooledCredential`, `AUTH_TYPE_OAUTH` and
`hermes_cli.auth_constants.AuthError`. It is intentionally limited to external
contracts; internal consumers import canonical owners directly. Existing provider
CLI action and runtime refresh hooks remain unchanged.

Shared auth implementations and obsolete internal forwarding paths are removed.
Provider/model routing and configuration ownership are unchanged. This baseline
has no `nous_cli`; its genuine CLI presentation moves only during subsequent
Phase 0 integration, as specified in `PHASE6_AUTH_PRESENTATION.md`.

The Python wheel check uses the existing Nix build marker to exercise component
packaging; it is not a full Nix distribution build. Frontend checks are source
builds, not an OS installer/signing test.
