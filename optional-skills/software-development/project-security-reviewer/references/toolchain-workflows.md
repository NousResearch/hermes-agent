# Toolchain Workflows

Use this reference after `detect_project.sh` identifies a manifest. Inspect the repository's own scripts, CI configuration, and security policy before running commands. Do not install dependencies or scanners without approval.

## Foundry / Solidity

When `foundry.toml` exists, use an installed Foundry-specific review workflow. When `foundry-security-reviewer` is available, it runs Forge build/test/coverage/snapshot and uses Slither or Aderyn when available. Focus on authorization, upgrade safety, accounting, oracle use, reentrancy, and invariant coverage.

## Node.js / TypeScript

Read `package.json`, lockfiles, workspace configuration, and CI first. Use the package manager indicated by the lockfile and only run declared test, lint, and type-check scripts. Review request boundaries, authorization, deserialization, SSRF, injection, client-side trust assumptions, and dependency changes. Run an installed dependency or static scanner only when its configuration and network behavior are understood.

## Python

Read `pyproject.toml`, `requirements*.txt`, lockfiles, and test configuration. Prefer the project's existing `pytest`, lint, type-check, and dependency-audit commands when their tools are available. Review unsafe deserialization, command construction, path traversal, template injection, authentication, authorization, and secrets handling.

## Rust

Read `Cargo.toml`, `Cargo.lock`, workspace configuration, and CI. Run `cargo test` and `cargo clippy` when appropriate; use an installed `cargo-audit` or `cargo-deny` only when configured. Review unsafe blocks, integer conversions, authentication boundaries, input parsing, concurrency, and FFI.

## Go

Read `go.mod`, `go.sum`, build tags, and CI. Run `go test ./...` when the repository's configuration permits it; use installed vulnerability tooling only when available. Review HTTP input handling, authorization middleware, SQL construction, file paths, race conditions, and `context` cancellation.

## Java / Kotlin

Read Maven or Gradle build files, dependency locks, framework configuration, and CI. Run the project's declared test and static-analysis tasks. Review deserialization, expression-language injection, ORM/query construction, authorization annotations, cryptography, and dependency updates.

## Ruby / PHP and Unknown Toolchains

Start by reading the README, manifest, lockfile, CI, and test entrypoints. Run only documented local checks. For an unknown project, identify its entrypoints, data stores, external integrations, authentication model, and deployment path before attempting automated analysis.
