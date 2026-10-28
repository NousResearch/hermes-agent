# Publishing and continuous integration

## Distribution layout

The standalone repository is [P2ppyJack/session-coord](https://github.com/P2ppyJack/session-coord).
The installable skill is `skills/multi-session-coordination/`; the repository-root
`install.py` additionally installs scripts and enrollment instructions. Keep the
board framework-independent. Optional host integration must use supported public
interfaces and must not rewrite the host application's source files.

## Pre-release checks

- Use an isolated branch and review the exact diff against its intended base.
- Run the commands in `.github/workflows/tests.yml`, including its linters,
  security checks, skill-contract checks, selftests, and installation checks.
- Verify fresh installation, upgrade, repeated installation, and opt-out behavior
  against temporary directories. Preserve existing databases and configuration.
- Exercise legacy data and new behavior through the real entry points. Keep
  compatibility fixtures and regression tests for persisted or serialized formats.
- Check the exact source package and submission diff for private paths, secrets,
  internal records, and unintended files. Use portable example resources.
- Document unrun platforms and skipped checks explicitly. Local test success is
  not proof of a successful GitHub Actions run.

## Cross-platform verification

The board uses Python's standard library. Shell-based verification and the cron
guard require a working POSIX shell. On Windows, probe a Git-for-Windows Bash
executable rather than assuming that a command named `bash` is usable: it may be
a WSL launcher without an installed distribution. Report unavailable verification
as skipped, not passed. Use Python's `sqlite3` module rather than requiring a
separate SQLite command-line program.

## Release artifacts

The stable board, repository helper, external native plugin, Hermes host ABI, and
joined-delegation helper have independent release states. A board release must not
claim that optional native continuation is supported until every required component
is published and a real integrated check passes. Configuration on disk is not proof
that an already-running receiver loaded the plugin; report
`activation=fresh_process_only` as restart required.

Keep unreleased work under `CHANGELOG.md` `[Unreleased]`. At release time keep the
skill version, changelog, installation documentation, and signed tag consistent.
Release notes describe technical changes, compatibility, tests, and limitations.
After authorized publication, verify the remote commit/tag, release contents, and
the actual Actions result. Preserve review history when updating an existing
contribution instead of opening a duplicate.

Public documentation, commit messages, PR descriptions, and release notes contain
technical information and required attribution only. Internal preparation records
and conversation narratives are not release assets.
