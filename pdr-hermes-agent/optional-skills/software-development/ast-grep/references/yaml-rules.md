# YAML rule reference — atomic, relational, composite, transform, fix

Use this when you outgrow inline `sg run -p ...` patterns and need a reusable, testable rule. A YAML rule is the unit of work for `sg scan`. Drop one or more files in `ruleDirs/` (configured via `sgconfig.yml`) and they get loaded automatically.

This page is the practical reference. The full upstream docs live at:

- <https://ast-grep.github.io/reference/yaml.html>
- <https://ast-grep.github.io/reference/rule.html>
- <https://ast-grep.github.io/cheatsheet/rule.html>

---

## Skeleton

A single YAML file can hold multiple rules separated by `---`.

```yaml
id: no-console
language: TypeScript
severity: warning
message: "Avoid console.* in production"
note: |
  Use a proper logger so we can route logs to stderr in production
  and silence them in tests.
url: https://internal.docs/rules/no-console

rule:
  pattern: console.$METHOD($$$ARGS)

fix: logger.$METHOD($$$ARGS)

constraints:
  METHOD:
    not:
      regex: '^(error|warn)$'

files:
  - 'src/**/*.ts'
ignores:
  - 'src/**/*.test.ts'

metadata:
  category: logging
```

---

## Top-level fields

| Field | Required | Description |
|---|---|---|
| `id` | yes | Unique identifier. Use `kebab-case`. |
| `language` | yes | One of: `Bash`, `C`, `Cpp`, `CSharp`, `Css`, `Elixir`, `Go`, `Haskell`, `Html`, `Java`, `JavaScript`, `Json`, `Kotlin`, `Lua`, `Nix`, `Php`, `Python`, `Ruby`, `Rust`, `Scala`, `Solidity`, `Swift`, `TypeScript`, `Tsx`, `Yaml`. **Capitalized PascalCase** is canonical, but lowercase often works. |
| `rule` | yes | The matching logic. Object containing one or more atomic / relational / composite rules. |
| `constraints` | no | Filter on captured single-metavariables (`$VAR`, not `$$$`). |
| `utils` | no | Local utility rules referenced by `matches:` in this file. |
| `transform` | no | Manipulate metavariable strings before `fix`. |
| `fix` | no | String or `FixConfig` for auto-rewrite. |
| `rewriters` | no | Rewriter rules for the `rewrite` transform. |
| `severity` | no | `hint` \| `info` \| `warning` \| `error` \| `off` (default: `hint`). |
| `message` | no | Concise lint message. May reference `$VAR` capture text. |
| `note` | no | Detailed markdown explanation (no `$VAR` interpolation). |
| `labels` | no | Custom diagnostic highlighting per-metavariable. |
| `files` | no | Glob include list. |
| `ignores` | no | Glob exclude list. |
| `url` | no | Doc link shown in editor diagnostics. |
| `metadata` | no | Free-form data ignored by `sg`, useful for external tooling. |

---

## Atomic rules — match a single node

### `pattern`

Match by structural pattern. The most common rule.

```yaml
# String form
rule:
  pattern: console.log($MSG)

# Object form (when context is needed)
rule:
  pattern:
    context: 'class C { $FIELD = $INIT }'
    selector: field_definition
    strictness: relaxed   # optional, default: smart
```

### `kind`

Match by AST node type name. Tree-sitter grammar-specific.

```yaml
rule:
  kind: call_expression
```

ast-grep 0.39+ supports limited ESQuery selectors:

```yaml
rule:
  kind: call_expression > identifier   # direct child
  kind: call_expression + identifier   # next sibling
  kind: call_expression ~ identifier   # following sibling
  kind: call_expression identifier     # descendant
```

To find the right `kind`, parse a known-good file:

```bash
sg run -p '$_' --lang ts --debug-query=cst src/foo.ts | head -40
```

### `regex`

Match node text against a Rust regex. Whole-text match (no partial). Always combine with `kind` or `pattern` for performance.

```yaml
rule:
  all:
    - kind: identifier
    - regex: '^[A-Z][a-z]+$'   # PascalCase
```

Inline flags work: `(?i)apple`, `(?m)^foo`. No look-around, no backreferences.

### `nthChild`

Match by 1-based index among **named** siblings. Inspired by CSS `:nth-child`.

```yaml
rule:
  nthChild: 1                # first sibling

# Functional form
rule:
  nthChild: 2n+1             # odd siblings

# With reverse and ofRule
rule:
  nthChild:
    position: 1
    reverse: true            # last
    ofRule:
      kind: function_declaration
```

### `range`

Match by character range. Useful for tooling that pinpoints a known location.

```yaml
rule:
  range:
    start: { line: 0, column: 0 }
    end:   { line: 0, column: 11 }
```

---

## Relational rules — match by relation to other nodes

All four take a sub-rule object plus optional `stopBy` and (for `inside`/`has`) `field`.

### `inside` — target is inside parent/ancestor matching sub-rule

```yaml
rule:
  pattern: this.$PROP
  inside:
    kind: class_body
    stopBy: end                # walk up to file root, default: neighbor
```

### `has` — target has child/descendant matching sub-rule

```yaml
rule:
  kind: function_declaration
  has:
    kind: throw_statement
    stopBy: end
```

### `precedes` — target appears before sibling matching sub-rule

```yaml
rule:
  kind: import_statement
  precedes:
    kind: function_declaration
```

### `follows` — target appears after sibling matching sub-rule

```yaml
rule:
  pattern: super($$$)
  follows:
    pattern: $X = $Y
```

### `stopBy`

| Value | Behavior |
|---|---|
| `"neighbor"` (default) | Stop at immediate parent/child/sibling. |
| `"end"` | Walk all the way to root / leaf / sequence boundary. |
| Rule object | Stop when sub-rule matches (inclusive). |

### `field`

Specify the semantic role of the target inside its parent (e.g. `name`, `body`, `value`, `key`).

```yaml
rule:
  kind: pair
  has:
    field: key
    regex: '^password$'
```

---

## Composite rules — combine sub-rules

| Rule | Meaning |
|---|---|
| `all` | All sub-rules must match the same target node. Metavariables from all sub-rules merge. |
| `any` | At least one sub-rule must match. Only metavars from the matched branch survive. |
| `not` | Inverse: target must NOT match the sub-rule. |
| `matches` | Reference a utility rule by id. |

```yaml
rule:
  all:
    - kind: call_expression
    - pattern: $FN($$$ARGS)
    - inside:
        kind: function_declaration
        stopBy: end

rule:
  any:
    - pattern: console.log($X)
    - pattern: console.warn($X)
    - pattern: console.error($X)

rule:
  all:
    - pattern: $E.unwrap()
    - not:
        inside:
          kind: function_item
          has:
            kind: result_type
            stopBy: end

rule:
  matches: is-react-component
```

> Composites apply to a **single** target. To express "node X has BOTH a number child AND a string child," use two relational rules at the top level, not `all` inside `has`. See `references/pitfalls.md` §10.

---

## Implicit `all` — multiple rule fields

A rule object with multiple fields is treated as an implicit `all`:

```yaml
# These two are equivalent
rule:
  pattern: this.$PROP
  inside: { kind: class_body }

rule:
  all:
    - pattern: this.$PROP
    - inside: { kind: class_body }
```

Use the explicit `all` array when capture order matters (rare, but possible with downstream `transform`).

---

## `constraints` — post-match metavariable filtering

After the main `rule` matches, additional checks on captured single metavariables:

```yaml
rule:
  pattern: function $NAME($$$P) { $$$B }

constraints:
  NAME:
    regex: '^[a-z][a-zA-Z0-9]*$'   # camelCase only
    not:
      regex: '^_'                   # not starting with _
```

Constraints **only apply to single metavars** (`$VAR`), not multi (`$$$VAR`).

---

## `utils` — local reusable sub-rules

```yaml
utils:
  is-literal:
    any:
      - kind: number
      - kind: string
      - kind: 'true'
      - kind: 'false'

rule:
  all:
    - pattern: $X = $Y
    - has:
        matches: is-literal      # references utils.is-literal
```

For utils accessible across multiple rule files, use `utilDirs` in `sgconfig.yml` and put each util in its own YAML file with `id` and `language`.

---

## `transform` — manipulate captures before `fix`

Operations: `replace`, `substring`, `convert`, `rewrite`.

### `replace` — regex search/replace on a captured string

```yaml
rule:
  pattern: $OLD_FN($$$A)
constraints:
  OLD_FN:
    regex: '^debug_'
transform:
  NEW_FN:
    replace:
      source: $OLD_FN
      replace: '^debug_'
      by: 'release_'
fix: $NEW_FN($$$A)
```

### `substring` — character slicing (negative indices supported)

```yaml
transform:
  INNER:
    substring:
      source: $WRAPPED
      startChar: 1
      endChar: -1
```

### `convert` — case conversion

```yaml
transform:
  KEBAB:
    convert:
      source: $CAMEL
      toCase: kebabCase   # camelCase | snakeCase | kebabCase | pascalCase | upperCase | lowerCase | capitalize
      separatedBy: [underscore]   # optional: dash | dot | space | slash | underscore | caseChange
```

### `rewrite` — apply other rewriter rules (experimental)

```yaml
rewriters:
  - id: stringify
    rule: { pattern: "'' + $A" }
    fix: "String($A)"

rule:
  pattern: stringify-all($EXPR)
transform:
  REWRITTEN:
    rewrite:
      source: $EXPR
      rewriters: [stringify]
      joinBy: "\n"
fix: $REWRITTEN
```

### Transforms can chain

Later transforms can reference variables produced by earlier ones:

```yaml
transform:
  KEBABED:
    convert: { source: $X, toCase: kebabCase }
  PREFIXED:
    replace:
      source: $KEBABED
      replace: '^'
      by: 'css-'
fix: $PREFIXED
```

---

## `fix` — auto-rewrite

### String form

```yaml
fix: logger.log($$$ARGS)

# Empty string deletes the match
fix: ""
```

### FixConfig form (for list-item deletion that needs to expand the range)

When deleting one item from a comma-separated list, you also need to remove the trailing comma. Use `expandEnd`:

```yaml
rule:
  kind: pair
  has:
    field: key
    regex: '^password$'

fix:
  template: ''
  expandEnd:
    regex: ','
```

`expandStart` and `expandEnd` accept `regex` matching characters that should be absorbed into the rewrite range.

---

## `rewriters` — sub-rule library for `rewrite` transform

Top-level field defining one or more named rewriters:

```yaml
rewriters:
  - id: nullable-to-optional
    rule: { pattern: $X | null }
    fix: '$X | undefined'

  - id: stringify
    rule: { pattern: "'' + $A" }
    fix: 'String($A)'
```

Used inside `transform` via the `rewrite` operation (see above).

---

## `labels` — custom diagnostic highlighting

```yaml
rule:
  pattern: $FN($$$ARGS)

labels:
  FN:
    style: primary
    message: "this function shouldn't be called"
  ARGS:
    style: secondary
    message: "with these arguments"
```

Editor extensions render the diagnostic with these labels. Defaults are usually fine.

---

## `files` and `ignores` — file selection per-rule

```yaml
files:
  - 'src/**/*.ts'
  - 'lib/**/*.ts'

ignores:
  - 'src/**/*.test.ts'
  - '**/__generated__/**'
```

If omitted, the rule runs on every file matching its `language`. These globs override `sgconfig.yml`-level globs for this rule only.

Object form (rare):

```yaml
files:
  - pattern: 'src/**/*.ts'
    case_sensitive: true
```

---

## See also

- `references/recipes.md` — copy-paste rules by language.
- `references/cli.md` — `sg scan`, `sg test`.
- `references/sgconfig.md` — project-level configuration.
- Official rule reference: <https://ast-grep.github.io/reference/rule.html>
- Cheat sheets: <https://ast-grep.github.io/cheatsheet/rule.html>, <https://ast-grep.github.io/cheatsheet/yaml.html>
