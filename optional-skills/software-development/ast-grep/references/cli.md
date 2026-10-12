# CLI reference — `sg` / `ast-grep`

Compact reference for the underlying `sg` binary that the helper wraps. Use this when the helper isn't enough or when you want to invoke `sg` directly.

> **Binary name on Linux**: prefer `ast-grep` over `sg` because `sg` collides with `setgroups` from `util-linux`.

---

## `sg run` — one-shot search/rewrite

The default subcommand. `sg -p 'foo'` is shorthand for `sg run -p 'foo'`.

```bash
sg run [OPTIONS] --pattern <PATTERN> [PATHS...]
```

| Flag | Purpose |
|---|---|
| `-p, --pattern <P>` | AST pattern to match. **Always single-quote** in shell to prevent `$VAR` expansion. |
| `-r, --rewrite <R>` | Replacement pattern. Used with `-U` to apply. |
| `-l, --lang <LANG>` | Language. Inferred from path extension if omitted. |
| `--selector <KIND>` | When the pattern is ambiguous, extract only this AST kind. |
| `--strictness <S>` | `cst` \| `smart` (default) \| `ast` \| `relaxed` \| `signature` |
| `--debug-query[=<F>]` | Print parsed pattern. F: `pattern` \| `ast` \| `cst` \| `sexp` |
| `--stdin` | Read code from stdin instead of files. Lang must be set. |
| `--globs <G>` | Include/exclude glob (repeatable; prefix `!` to exclude). |
| `--follow` | Follow symlinks. |
| `--no-ignore <T>` | Disable a class of ignore: `hidden`, `dot`, `exclude`, `global`, `parent`, `vcs`. |
| `-i, --interactive` | Step through matches and confirm each rewrite. |
| `-U, --update-all` | Apply all rewrites without confirmation. **Mutually exclusive with `--json`** (silently). |
| `--json[=<S>]` | Emit JSON. S: `pretty` \| `stream` \| `compact` (compact is best for piping). |
| `--color <W>` | `auto` \| `always` \| `ansi` \| `never` |
| `--inspect <G>` | Detail level: `nothing` \| `summary` \| `entity` |
| `-A, -B, -C <N>` | Context lines after / before / around each match. |
| `-j, --threads <N>` | Thread count (default: heuristic; `0` = auto). |

### `--update-all` + `--json` — the trap

`sg` silently ignores `--update-all` when `--json` is set. To preview AND apply, run **two passes**:

```bash
# Pass 1: preview
sg run -p 'foo()' -r 'bar()' --json=compact src/

# Pass 2: apply
sg run -p 'foo()' -r 'bar()' --update-all src/
```

The `ast_grep_helper.py replace --apply` subcommand does this automatically.

### Examples

```bash
# Basic search
sg run -p 'console.log($MSG)' --lang ts src/

# Search with context lines
sg run -p 'eval($CODE)' --lang js -C 3 .

# Rewrite, dry-run preview as JSON
sg run -p 'console.log($MSG)' -r 'logger.info($MSG)' --json=compact --lang ts src/

# Rewrite, apply
sg run -p 'console.log($MSG)' -r 'logger.info($MSG)' --update-all --lang ts src/

# Pattern from stdin
echo 'console.log("x")' | sg run -p 'console.log($MSG)' --lang js --stdin

# Limit to specific files
sg run -p 'foo()' --lang ts --globs 'src/**/*.ts' --globs '!**/*.test.ts' .

# Debug a pattern that returns 0 matches
sg run -p 'def $F($$$):' --lang py --debug-query=ast --stdin <<< 'def foo(): pass'
```

---

## `sg scan` — YAML rule scanner

Run a configuration of YAML rules across files. Used for project-wide lints and codemods.

```bash
sg scan [OPTIONS] [PATHS...]
```

| Flag | Purpose |
|---|---|
| `-c, --config <C>` | Path to `sgconfig.yml` (default: walk up from cwd looking for one). |
| `-r, --rule <F>` | Run a **single** rule file. Mutually exclusive with `--config`. |
| `--inline-rules <Y>` | Pass YAML rule text inline. Use `---` to separate multiple rules. |
| `--filter <RE>` | Only run rules whose `id` matches this regex. |
| `--include-metadata` | Include rule `metadata` field in JSON output. |
| `-U, --update-all` | Apply fixes from `fix:` automatically. |
| `--report-style <S>` | `rich` \| `medium` \| `short` |
| `--format <F>` | `github` \| `sarif` (CI-friendly outputs). |
| `--error[=ID]`, `--warning[=ID]`, `--info[=ID]`, `--hint[=ID]`, `--off[=ID]` | Promote/demote severity. |
| `-i, --interactive` | Confirm each fix interactively. |
| `--json[=<S>]` | JSON output. |

### Examples

```bash
# Run all rules in sgconfig.yml-discovered ruleDirs
sg scan src/

# Run a single rule file (no sgconfig.yml needed)
sg scan -r rules/no-console.yml src/

# Inline rule (great for one-offs and CI)
sg scan --inline-rules '
id: no-todo
language: TypeScript
severity: warning
rule: { pattern: TODO }' src/

# Apply all auto-fixes
sg scan -U src/

# CI-friendly GitHub annotations
sg scan --format github src/

# SARIF for security scanners
sg scan --format sarif src/ > sarif.json
```

---

## `sg test` — run rule snapshot tests

```bash
sg test [OPTIONS]
```

| Flag | Purpose |
|---|---|
| `-c, --config <C>` | Path to `sgconfig.yml`. |
| `-t, --test-dir <D>` | Test directory. |
| `--snapshot-dir <D>` | Snapshot directory (default: `__snapshots__`). |
| `--skip-snapshot-tests` | Validate test code parses; don't compare snapshots. |
| `-U, --update-all` | Update all changed snapshots. |
| `-f, --filter <G>` | Filter test cases by glob on rule id. |
| `--include-off` | Include rules with severity `off`. |
| `-i, --interactive` | Step through changed snapshots and accept/reject each. |

A test directory looks like:

```
test/
├── no-console.yml          # `valid:` and `invalid:` snippets
└── no-console-test.yml     # alternative test file format
__snapshots__/
└── no-console-snapshot.yml # expected match locations
```

---

## `sg new` — scaffold

```bash
sg new <COMMAND> [NAME] [OPTIONS]
```

| Subcommand | Creates |
|---|---|
| `project` | `sgconfig.yml`, `rules/`, `utils/`, `__snapshots__/` directory tree |
| `rule` | A new YAML rule file in the first `ruleDirs` entry |
| `test` | A new test file in `testConfigs[0].testDir` |
| `util` | A new utility rule in the first `utilDirs` entry |

```bash
# New project in current dir
sg new project --yes

# New rule
sg new rule no-console --lang typescript

# New test
sg new test no-console --yes
```

---

## `sg lsp` — language server

```bash
sg lsp -c sgconfig.yml
```

Speak LSP over stdin/stdout. Configure your editor (VS Code extension, Neovim `nvim-lspconfig`, Helix `languages.toml`) to spawn this command for live diagnostics.

---

## `sg completions` — shell completions

```bash
sg completions bash >> ~/.bashrc
sg completions zsh > "${fpath[1]}/_sg"
sg completions fish > ~/.config/fish/completions/sg.fish
sg completions powershell >> $PROFILE
```

---

## Useful one-liners

```bash
# Count matches per file
sg run -p 'console.log($_)' --lang ts --json=compact . \
  | jq -r '.[].file' | sort | uniq -c | sort -rn

# Find all unique kinds in a file (great for figuring out kind names)
sg run -p '$_' --lang ts --debug-query=cst src/foo.ts \
  | grep -oE 'kind: [a-z_]+' | sort -u

# Rewrite only in a subset of files
sg run -p 'foo()' -r 'bar()' --update-all --globs 'src/**/*.ts' --globs '!src/legacy/**' .

# Apply fixes from many rules but only ones matching a pattern in their id
sg scan --filter 'no-' -U src/

# Use ast-grep as a linter in pre-commit
sg scan --format github src/ || exit 1
```

---

## See also

- `references/yaml-rules.md` — rule schema (`pattern`, `kind`, `regex`, `inside`, `has`, `all`, `any`, `not`, `matches`, `transform`, `fix`).
- `references/sgconfig.md` — project configuration.
- Official: <https://ast-grep.github.io/reference/cli.html>
