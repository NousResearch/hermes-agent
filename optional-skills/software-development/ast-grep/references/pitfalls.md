# Pitfalls — what breaks patterns and how to fix them

This is the failure-mode field guide. The `scripts/ast_grep_helper.py validate` subcommand mechanically checks for the items in §1 before calling `sg`; the rest are lower-frequency but still common.

---

## 1. Regex syntax does not work

ast-grep does **not** interpret regex inside patterns. The following all fail:

| Bad | Why | Use instead |
|---|---|---|
| `foo\|bar` | `\|` is regex alternation. ast-grep does not alternate. | Two separate calls, OR `any: [pattern: foo, pattern: bar]` in a YAML rule, OR `rg -e foo -e bar`. |
| `foo.*bar` | `.*` is a regex wildcard. | `foo($$$) bar` if the gap is a list of nodes; otherwise switch to `rg`. |
| `\w+`, `\d+`, `\s` | Regex character classes. | `$VAR` to capture any identifier. For digits-only, use `kind: number_literal`. |
| `[a-z]+` | Regex character class. | No AST equivalent — switch to `rg`. |
| `^foo$` | Regex anchors. | Anchor by AST: use `kind: program > expression_statement` or use `inside`/`not has`. |

**Why this happens**: LLMs default to regex thinking. The mental switch is "ast-grep patterns are *code*, not *strings*."

When you genuinely need regex, use the `regex` rule field in YAML (matches node text with Rust regex):

```yaml
rule:
  all:
    - kind: identifier
    - regex: '^[A-Z][a-z]+$'   # CamelCase identifiers only
```

Note: `regex` matches the **whole node text** — no partial matches. Combine with `kind` or `pattern` for performance.

---

## 2. Incomplete AST nodes

Patterns must be valid code that the parser accepts as a complete node. Common mistakes:

```text
# JS/TS
function foo                              ❌ no params, no body
function $NAME($$$) { $$$ }               ✅

async function $NAME                      ❌
async function $NAME($$$) { $$$ }         ✅

# Python
def foo:                                  ❌ trailing colon makes it a statement
def $FN($$$)                              ✅
class Foo:                                ❌
class $C($$$)                             ✅

# Go
func foo                                  ❌
func $NAME($$$) { $$$ }                   ✅

# Rust
fn foo                                    ❌
fn $NAME($$$) -> $RET { $$$ }             ✅
fn $NAME($$$) { $$$ }                     ✅ (-> () inferred)

# Java
public void foo                           ❌
public void $NAME($$$) { $$$ }            ✅
```

If a pattern returns 0 matches and looks correct, run `sg run -p '<pattern>' --lang <lang> --debug-query=ast --stdin <<< 'echo'` and see what the parser thinks the pattern is. If it returns an `ERROR` node, the pattern is malformed.

---

## 3. Pattern parses as the wrong kind

A class field initializer `a = 123` *also* parses as an assignment expression. If you want only field definitions, you must disambiguate:

```yaml
# WRONG — pattern parses as assignment_expression, not field_definition
pattern: a = 123
kind: field_definition

# CORRECT — use pattern object with context + selector
pattern:
  context: 'class C { a = 123 }'
  selector: field_definition
```

`kind` and `pattern` are **independent constraints**, not modifiers of each other. ast-grep does not change *how* it parses based on `kind`.

---

## 4. The `|` ambiguity

A bare `|` in a pattern is interpreted as bitwise-or in most languages, **not** alternation. So:

```yaml
pattern: foo | bar          # parses as: foo bitwise-or'd with bar
```

…matches expressions like `x | y`, not "either foo or bar". To get alternation, use `any`:

```yaml
rule:
  any:
    - pattern: foo
    - pattern: bar
```

In TypeScript union types (`A | B`), `|` is part of the type syntax — `pattern: A | B` correctly parses as a union type and matches that.

---

## 5. Same-name metavars collide

```ts
// Pattern: $X = $X
// Captures only when both sides are TEXTUALLY identical.

// Matches:
a = a
foo.bar = foo.bar

// Does NOT match:
a = b
let x = compute()      // because $X needs to bind once and re-use
```

If you actually want two independent captures, name them differently: `$X = $Y`.

---

## 6. `$$$` is greedy then commits

`$$$` does **not** backtrack. It captures as much as possible, then commits. If your pattern needs a non-greedy match, structure it differently:

```ts
// You want "match foo($X), where $X is any single arg"
// BAD:  foo($$$X)        // matches foo(a), foo(a, b), foo(a, b, c) - too broad
// GOOD: foo($X)          // matches only single-arg calls

// You want "match foo() with at least one arg"
// BAD:  foo($$$X)        // also matches foo()
// GOOD: foo($X, $$$REST) // forces at least one arg
```

---

## 7. `kind` names depend on tree-sitter grammar

`kind: function_declaration` works for JavaScript, but Python uses `function_definition`, Rust uses `function_item`, Go uses `function_declaration` (same as JS by coincidence). To find the right name, parse a known-good file:

```bash
sg run -p '$_' --lang python --debug-query=cst path/to/example.py | grep -i function
```

Or open <https://ast-grep.github.io/playground.html> and click on a node to see its `kind`.

---

## 8. `inside` / `has` defaults to `stopBy: neighbor`

```yaml
inside:
  kind: function_declaration   # only checks the IMMEDIATE parent
```

If you want "anywhere inside a function (any depth)":

```yaml
inside:
  kind: function_declaration
  stopBy: end                  # walks up to the file root
```

Same for `has` (descendants):

```yaml
has:
  kind: return_statement
  stopBy: end                  # walks down the whole subtree
```

Without `stopBy: end`, `has` only matches direct children.

---

## 9. CLI silently ignores `--update-all` when `--json` is set

This is the single biggest gotcha when scripting ast-grep. If you run:

```bash
sg run -p 'foo()' -r 'bar()' --json=compact --update-all .
```

…you get the JSON output but **no files are mutated**. ast-grep silently drops `--update-all` when `--json` is on. To both preview and apply, run **two passes**:

```bash
# Pass 1: preview as JSON
sg run -p 'foo()' -r 'bar()' --json=compact .

# Pass 2: actually apply
sg run -p 'foo()' -r 'bar()' --update-all .
```

`scripts/ast_grep_helper.py replace` does this automatically when `--apply` is set.

---

## 10. Composite rules apply to a single node

`all` and `any` evaluate against **one target node** at a time:

```yaml
# WRONG — wants "node has BOTH a number child AND a string child"
has:
  all:
    - kind: number       # impossible: one node cannot be both at once
    - kind: string

# CORRECT
all:
  - has: { kind: number }
  - has: { kind: string }
```

Lift relational rules out of composites when the relation is "the surrounding node has X children matching Y."

---

## 11. Field order is not guaranteed

When a rule object has multiple fields:

```yaml
rule:
  pattern: $X = compute()
  has: { kind: number }
```

…ast-grep evaluates them as an implicit `all`, but the **order** in which metavariables are captured is not guaranteed. If your `transform` or `fix` depends on capture order, use an explicit `all` array:

```yaml
rule:
  all:
    - pattern: function $F() { $$$ }
    - has: { pattern: $F() }     # $F captured by pattern first; here we just check
```

---

## 12. `regex` without `kind` is slow

`regex` alone scans every node text in the file. On large repos this is noticeably slow. Always combine:

```yaml
# Slow
rule:
  regex: '^TODO'

# Fast
rule:
  all:
    - kind: comment
    - regex: '^//\s*TODO'
```

---

## 13. No scope / type / data-flow analysis

ast-grep is a **structural** matcher. It does NOT know:

- Whether two `foo` references point to the same variable.
- Whether a variable is shadowed.
- Whether a function is async, throws, returns a Promise.
- Whether a value flows from input to output.

For those questions, use a real type-aware tool: TypeScript LSP, Pyright, Semgrep with type inference, CodeQL, etc.

ast-grep is great when *the syntactic shape* is what you care about: "find every call to `eval(...)`", "find every `as any`", "find every empty catch block." It is weak for "find every variable that's never used."

---

## 14. Pattern testing is the fastest debugger

When a pattern returns 0 matches and you can't see why:

1. Open <https://ast-grep.github.io/playground.html>.
2. Paste your code into the left pane, your pattern into the top-right.
3. The bottom-right shows the parsed AST and which nodes matched (highlighted) or failed.

Or locally:

```bash
sg run -p '<pattern>' --lang <lang> --debug-query=ast --stdin <<< '<sample-code>'
```

stderr shows the parsed pattern; stdout shows the JSON match result. If the pattern shows up as `ERROR (XXX)`, it doesn't parse.

---

## See also

- `references/patterns.md` — meta-variables, strictness, naming rules.
- `references/recipes.md` — known-good patterns by language.
- `references/cli.md` — `--debug-query`, `--strictness`, `--update-all`.
