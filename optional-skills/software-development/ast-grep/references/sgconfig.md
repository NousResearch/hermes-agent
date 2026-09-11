# sgconfig.yml — project configuration

`sgconfig.yml` lives at your project root (the same place as `package.json`, `Cargo.toml`, `pyproject.toml`, etc.) and tells `sg scan`/`sg test` where to find rules and tests.

`sg` walks **upward** from the current directory until it finds an `sgconfig.yml`. You can also pass `--config <path>` explicitly.

---

## Minimal project layout

```
my-project/
├── sgconfig.yml
├── rules/
│   ├── no-console.yml
│   └── no-as-any.yml
├── utils/
│   └── is-literal.yml
├── tests/
│   ├── no-console.yml
│   └── __snapshots__/
│       └── no-console-snapshot.yml
└── src/
    └── ...
```

```yaml
# sgconfig.yml
ruleDirs:
  - rules

testConfigs:
  - testDir: tests
    snapshotDir: __snapshots__

utilDirs:
  - utils
```

That's it. `sg scan src/` will load every `.yml` in `rules/`, find every `.ts`/`.py`/whatever matching the rule's `language`, and report violations.

---

## Full schema

```yaml
# Rule directories — required
ruleDirs:
  - rules
  - team-rules
  - vendor/sg-rules

# Test directories — optional
testConfigs:
  - testDir: tests
    snapshotDir: __snapshots__
  - testDir: integration-tests

# Utility rule directories — optional
# Files here become global utilities accessible via `matches: <id>` from any rule.
utilDirs:
  - utils
  - team-utils

# Override file-extension -> language mapping — optional
# Useful when your code uses non-standard extensions.
languageGlobs:
  html:
    - '*.vue'
    - '*.svelte'
    - '*.astro'
  json:
    - '.eslintrc'
    - '.prettierrc'
  cpp:
    - '*.c'                  # treat C as C++
  tsx:
    - '*.ts'                 # treat all .ts as TSX (so TSX rules work everywhere)

# Custom tree-sitter languages (experimental) — optional
customLanguages:
  mojo:
    libraryPath: tree-sitter-mojo.so
    extensions: [mojo, '🔥']
    expandoChar: _           # Replace $ in patterns when language uses $ syntactically
    languageSymbol: tree_sitter_mojo

# Language injection — embedded code in another language (experimental) — optional
# Example: CSS inside styled-components template literals.
languageInjections:
  - hostLanguage: js
    rule:
      pattern: 'styled.$TAG`$CONTENT`'
    injected: css
```

---

## Field-by-field

### `ruleDirs` (required)

`Array<string>` — directories containing rule YAML files. Resolved relative to `sgconfig.yml`.

Each `.yml`/`.yaml` file in these directories is loaded as a rule. One file can contain multiple rules separated by `---`.

### `testConfigs`

`Array<TestConfig>` where each entry has:

- `testDir` (required): directory of test YAML files.
- `snapshotDir` (optional, default `__snapshots__`): directory for snapshots.

Each test file looks like:

```yaml
id: no-console
valid:
  - 'logger.info("hi")'
invalid:
  - 'console.log("hi")'
```

`sg test` runs every test, compares matches against the snapshot, and fails on diff. Snapshots are created on first run with `-U`.

### `utilDirs`

`Array<string>` — directories with global utility rules. Each util file must have `id` and `language`. Utils become referenceable via `matches: <id>` from any rule in the project.

### `languageGlobs`

`HashMap<string, Array<string>>` — override which extensions map to which language. Takes precedence over the built-in defaults.

Useful for:

- Custom file extensions (`.eslintrc` is JSON).
- Force-treating `.ts` files as TSX (so JSX-shaped patterns work).
- Vue/Svelte/Astro files (HTML host language).

### `customLanguages` (experimental)

Register a tree-sitter parser that ast-grep doesn't ship with. Requires:

- `libraryPath`: path to a built `.so` / `.dylib` / `.dll` containing the grammar.
- `extensions`: file extensions to recognize.
- `languageSymbol`: the C symbol exported by the grammar (typically `tree_sitter_<name>`).
- `expandoChar` (optional): character to substitute for `$` in patterns when the host language uses `$` syntactically (PHP, jQuery, etc.).

This is **rarely needed** — ast-grep already supports 25 languages out of the box.

### `languageInjections` (experimental)

Match patterns inside embedded languages. Example: CSS inside JS template literals (styled-components, emotion).

```yaml
languageInjections:
  - hostLanguage: js
    rule:
      pattern: 'styled.$TAG`$CONTENT`'
    injected: css
```

After this, a `css` rule with pattern `color: $C` will match `$CONTENT` strings.

---

## Common configurations

### Monorepo with shared rules

```
monorepo/
├── sgconfig.yml          # root config — applies to entire monorepo
├── shared-rules/
│   ├── no-todo.yml
│   └── no-as-any.yml
└── packages/
    ├── frontend/
    │   ├── sgconfig.yml  # extends root with frontend-specific rules
    │   └── rules/
    └── backend/
        ├── sgconfig.yml  # extends root with backend-specific rules
        └── rules/
```

Each package's `sgconfig.yml` references both the package-local rules and the shared ones:

```yaml
# packages/frontend/sgconfig.yml
ruleDirs:
  - rules
  - ../../shared-rules
```

### Single rule file (no project)

For one-offs, skip `sgconfig.yml` entirely:

```bash
sg scan -r path/to/single-rule.yml src/
```

### Inline rule (no file)

```bash
sg scan --inline-rules '
id: no-todo
language: TypeScript
severity: warning
rule: { pattern: TODO }' src/
```

Multiple rules separated by `---`:

```bash
sg scan --inline-rules '
id: no-todo
language: TypeScript
rule: { pattern: TODO }
---
id: no-fixme
language: TypeScript
rule: { pattern: FIXME }' src/
```

---

## Editor integration

VS Code / Neovim / Helix detect `sgconfig.yml` automatically and surface diagnostics from every rule. Without `sgconfig.yml`, the LSP runs without any rules loaded.

To enable schema validation in your editor, add a header to each rule file:

```yaml
# yaml-language-server: $schema=https://raw.githubusercontent.com/ast-grep/ast-grep/main/schemas/rule.json
id: no-console
language: TypeScript
rule:
  pattern: console.log($_)
```

---

## See also

- `references/yaml-rules.md` — rule schema (atomic / relational / composite / transform / fix).
- `references/cli.md` — `sg scan`, `sg test`, `sg new project`.
- Official: <https://ast-grep.github.io/reference/sgconfig.html>, <https://ast-grep.github.io/guide/project/project-config.html>
