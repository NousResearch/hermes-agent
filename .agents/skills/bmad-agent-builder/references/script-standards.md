# Script Creation Standards

When building scripts for a skill, follow these standards to ensure portability and zero-friction execution. Skills must work across macOS, Linux, and Windows (native, Git Bash, and WSL).

## Python Over Bash

**Always favor Python for script logic.** Bash is not portable — it fails or behaves inconsistently on Windows (Git Bash is MSYS2-based, not a full Linux shell; WSL bash can conflict with Git Bash on PATH; PowerShell is a different language entirely). Python with `uv run` works identically on all platforms.

**Safe bash commands** — these work reliably across all environments and are fine to use directly:

- `git`, `gh` — version control and GitHub CLI
- `uv run` — Python script execution with automatic dependency handling
- `npm`, `npx`, `pnpm` — Node.js ecosystem
- `mkdir -p` — directory creation

**Everything else should be Python** — piping, `jq`, `grep`, `sed`, `awk`, `find`, `diff`, `wc`, and any non-trivial logic. Even `sed -i` behaves differently on macOS vs Linux. If it's more than a single safe command, write a Python script.

## Favor the Standard Library

Always prefer Python's standard library over external dependencies. The stdlib is pre-installed everywhere, requires no `uv run`, and has zero supply-chain risk. Common stdlib modules that cover most script needs:

- `json` — JSON parsing and output
- `pathlib` — cross-platform path handling
- `re` — pattern matching
- `argparse` — CLI interface
- `collections` — counters, defaultdicts
- `difflib` — text comparison
- `ast` — Python source analysis
- `csv`, `xml.etree` — data formats

Only pull in external dependencies when the stdlib genuinely cannot do the job (e.g., `tiktoken` for accurate token counting, `pyyaml` for YAML parsing, `jsonschema` for schema validation). **External dependencies must be confirmed with the user during the build process** — they add install-time cost, supply-chain surface, and require `uv` to be available.

## PEP 723 Inline Metadata (Required)

Every Python script MUST include a PEP 723 metadata block. For scripts with external dependencies, use the `uv run` shebang:

```python
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml>=6.0", "jsonschema>=4.0"]
# ///
```

For scripts using only the standard library, use a plain Python shebang but still include the metadata block:

```python
#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
```

**Key rules:**

- The shebang MUST be line 1 — before the metadata block
- Always include `requires-python`
- List all external dependencies with version constraints
- Never use `requirements.txt`, `pip install`, or expect global package installs
- The shebang is a Unix convenience — cross-platform invocation relies on `uv run scripts/foo.py`, not direct shebang execution

## Invocation in SKILL.md

How a built skill's SKILL.md should reference its scripts (bare path from the skill root, per the path conventions):

- **All scripts:** `uv run scripts/foo.py {args}` — consistent invocation regardless of whether the script has external dependencies

`uv run` reads the PEP 723 metadata, silently caches dependencies in an isolated environment, and runs the script — no user prompt, no global install. Like `npx` for Python.

## Graceful Degradation

Skills may run in environments where Python or `uv` is unavailable (e.g., claude.ai web). Scripts should be the fast, reliable path — but the skill must still deliver its outcome when execution is not possible.

**Pattern:** When a script cannot execute, the LLM performs the equivalent work directly. The script's `--help` documents what it checks, making this fallback natural. Design scripts so their logic is understandable from their help output and the skill's context.

In SKILL.md, frame script steps as outcomes, not just commands:

- Good: "Validate path conventions (run `scripts/scan-paths.py --help` for details)"
- Avoid: "Execute `uv run scripts/scan-paths.py`" with no context about what it does

## Script Interface Standards

- Implement `--help` via `argparse` (single source of truth for the script's API)
- Accept target path as a positional argument
- `-o` flag for output file (default to stdout)
- Diagnostics and progress to stderr
- Exit codes: 0=pass, 1=fail, 2=error
- `--verbose` flag for debugging
- Output valid JSON to stdout
- No interactive prompts, no network dependencies
- Tests in `scripts/tests/`
