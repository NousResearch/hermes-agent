# Codebase Map

Create a compact, durable map of a repository before editing, reviewing, or delegating work. Use this skill when a task touches an unfamiliar or large codebase, when context is tight, or when handing work to subagents.

## When to Use

- Starting work in a repo you have not recently inspected.
- Preparing a subagent prompt that needs repo orientation without dumping files.
- Reviewing a PR and needing likely impact areas.
- Debugging where entry points, tests, or ownership are unclear.

## Outputs

Write the map to `CODEBASE_MAP.md` at the repository root or to a task-specific file under `.plans/` when the repo should not be modified.

A useful map includes:

1. **Purpose** — one paragraph summarizing what the project does.
2. **Primary languages and package managers** — from lockfiles and manifests.
3. **Entrypoints** — CLIs, servers, jobs, web apps, extension hosts.
4. **Important directories** — one-line responsibility for each.
5. **Test/build commands** — exact commands discovered from project files.
6. **Dependency seams** — model providers, databases, network clients, tool adapters.
7. **Likely change targets** — files/modules relevant to the current task.
8. **Risks** — security, migrations, generated files, side effects.
9. **Open questions** — only if the repository itself does not answer them.

## Fast Mapping Procedure

1. Inspect root files: `README*`, `AGENTS.md`, `CONTRIBUTING*`, `pyproject.toml`, `package.json`, `go.mod`, `Cargo.toml`, `.github/workflows/*`.
2. List top-level directories and identify source, tests, docs, examples, scripts, generated assets, and vendored code.
3. Search for entrypoints:
   - Python: `[project.scripts]`, `if __name__ == "__main__"`, `click`, `typer`, `fire`, `argparse`.
   - Node/TS: `package.json` scripts, `bin`, framework configs, extension manifests.
   - Go: `cmd/*`, `main.go`.
4. Search for tests and quality gates: `pytest`, `vitest`, `jest`, `ruff`, `ty`, `mypy`, `go test`, `cargo test`, `prettier`, `biome`.
5. Search for integration seams: `OpenAI`, `Anthropic`, `MCP`, `GitHub`, `Docker`, `SQLite`, `Postgres`, `Redis`, `VNC`, `browser`, `cron`.
6. Summarize. Do not paste massive file trees; prefer high-signal bullets.

## Subagent Prompt Template

```text
You are mapping this repository for future agents.
Repository: <path or URL>
Task focus: <feature/bug/review area>
Produce CODEBASE_MAP.md with: purpose, languages, entrypoints, key dirs, tests/build commands, dependency seams, likely change targets, risks, and open questions.
Keep it concise and cite file paths.
Do not modify functional code.
```

## Quality Bar

- Every claim names at least one supporting file path.
- Commands are copied from manifests or verified by running `--help`/dry-run where safe.
- Generated/vendor/build directories are excluded unless they are the subject of the task.
- The map is small enough to paste into a subagent prompt.

## Example Skeleton

```markdown
# Codebase Map

## Purpose

...

## Languages and Tooling

- Python package: `pyproject.toml`
- Node app: `website/package.json`

## Entrypoints

- CLI: `hermes_cli/main.py`
- Dashboard: `dashboard/` ...

## Key Directories

- `agent/` — ...
- `tools/` — ...

## Commands

- Tests: `pytest ...`
- Type checks: `ty check ...`

## Task-Relevant Targets

...

## Risks

...
```
