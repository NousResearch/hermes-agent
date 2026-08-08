# Pattern syntax — meta-variables and how patterns parse

ast-grep is **not regex**. Patterns are written in the **same syntax as the target language** (TypeScript, Python, Go, etc.), and ast-grep matches them against the AST of every file. The wildcards are called **meta-variables**.

This page is the canonical primer. If a pattern fails, 90% of the time it is one of the issues on this page.

---

## The three meta-variables

| Syntax | Matches | Capture |
|---|---|---|
| `$VAR` | exactly **one** AST node | yes, by name |
| `$$$` | **zero or more** AST nodes (a list) | no (anonymous) |
| `$$$VAR` | zero or more AST nodes | yes, by name |
| `$_` | one AST node | no (anonymous) |

A meta-variable always replaces a **whole AST node**, never a substring of a node. `$VAR` cannot match the first three characters of an identifier, only an entire identifier (or expression, or statement, depending on context).

### Naming rules

- Must start with `$`.
- Then uppercase letters `A-Z`, digits, or underscores.
- **Valid**: `$X`, `$VAR`, `$VAR_1`, `$_`, `$_VAR`, `$ARG1`.
- **Invalid**: `$lower`, `$kebab-case`, `$1` (digit first), `$$single` (use `$_` for anonymous).

### Same-name = same content

Two occurrences of the same metavariable in a pattern must capture **identical text**:

```ts
// Pattern
$X === $X

// Matches
a === a
foo.bar === foo.bar

// Does NOT match
a === b
foo === foo.bar
```

Useful for finding redundant comparisons, double assignments, etc.

### `$$$` is greedy

When you write `foo($$$A, b, $$$C)`, the matcher does **not** backtrack or try every possible split. It greedily fills `$$$A` until the pattern can match `b`, then everything left goes into `$$$C`.

```ts
// Pattern
foo($$$A, b, $$$C)

// Input
foo(a, c, b, b, c)

// Capture
$$$A = [a, c]
$$$C = [b, c]
```

If you need a different split, restructure the pattern (e.g. add a constraint).

---

## Patterns must be valid code

The pattern itself must parse with the target language's grammar. ast-grep treats `$VAR` and `$$$` as identifiers/argument lists during parsing, then matches structurally.

### What goes wrong

| Bad pattern | Why it fails | Fix |
|---|---|---|
| `function $NAME` | Function declaration without body — not a valid AST node in JS/TS/Go/Rust. | `function $NAME($$$) { $$$ }` |
| `def $FN($$$):` | Trailing colon. ast-grep parses as a complete function definition; the colon makes it a statement. | `def $FN($$$)` |
| `class Foo:` | Same — Python class without body. | `class Foo($$$)` |
| `fn $NAME` | Rust fn without signature. | `fn $NAME($$$) -> $RET { $$$ }` |
| `if x` | Incomplete `if` — most languages require the body. | `if x { $$$ }` (curly-brace languages) or `if x: $$$` (Python uses `pattern.context`/`selector` instead) |
| `"key": "$VAL"` | JSON pattern — a key/value pair on its own isn't valid JSON. | Use `pattern: { context: '{"key": "$VAL"}', selector: pair }` |

### When a sub-expression isn't valid on its own

Sometimes you want to match an *expression* that the language only allows inside a larger context. Use the `pattern` object form:

```yaml
pattern:
  context: 'class A { $FIELD = $INIT }'
  selector: field_definition
```

This says: parse `class A { $FIELD = $INIT }` as a whole, then keep only the `field_definition` sub-tree as the actual pattern.

---

## Strictness levels

When CST nodes don't match exactly (extra whitespace, different unnamed punctuation), ast-grep can be more or less forgiving. Pass `--strictness <LEVEL>` on the CLI, or set it in a YAML rule:

| Level | Matches |
|---|---|
| `cst` | Every node, including unnamed (commas, parens, etc.) |
| `smart` (default) | All except unnamed nodes in the **target** that aren't in the pattern |
| `ast` | Only named AST nodes |
| `relaxed` | Named AST nodes, ignoring comments |
| `signature` | Only node kinds — text and unnamed nodes ignored |

`smart` is almost always what you want. Reach for `signature` when you want to match "any function called `foo`" regardless of arguments.

---

## Testing a pattern

Two tools help you confirm a pattern parses the way you expect:

```bash
# Print the AST of the pattern itself
sg run -p 'console.log($MSG)' --lang ts --debug-query=ast

# Print the parsed CST of a file (great for figuring out kind names)
sg run -p '$_' --lang ts --debug-query=cst src/example.ts | head -40
```

`--debug-query=ast` shows the named AST nodes only (cleaner). `--debug-query=cst` shows everything including punctuation. Both go to stderr, so they don't interfere with stdout JSON.

The web playground is also fast: <https://ast-grep.github.io/playground.html>.

---

## When ast-grep is the wrong tool

If your pattern is fundamentally text-shaped, switch to `grep` / `rg`:

- Match across multiple files **for any text** → `rg`
- Cross-language regex with alternation → `rg -e foo -e bar`
- Match comments only → `rg --type ts '^\s*//.*TODO'`
- Match URLs, emails, license headers → `rg`

ast-grep is for **code structure**: function shapes, call patterns, control flow, type annotations, imports, error handling. If your "pattern" only depends on the bytes of the file and not on the syntax, regex is the right tool.

---

## See also

- `references/pitfalls.md` — concrete regex anti-patterns and language-specific traps.
- `references/recipes.md` — copy-paste-ready patterns for TS/JS/Py/Go/Rust.
- `references/yaml-rules.md` — `kind`, `regex`, `inside`, `has`, `all`, `any`, `not`, `matches`.
- Official: <https://ast-grep.github.io/guide/pattern-syntax.html>
