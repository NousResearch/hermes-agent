# Recipes — copy-paste patterns by language

Every pattern in this file has been verified against the canonical syntax. They are starting points; tweak metavariable names and constraints to fit your case.

Use them with the helper:

```bash
ast-grep-helper search '<PATTERN>' --lang <LANG> [path]
ast-grep-helper replace '<PATTERN>' '<REWRITE>' --lang <LANG> [path]   # dry-run
ast-grep-helper replace '<PATTERN>' '<REWRITE>' --lang <LANG> [path] --apply
```

Or directly:

```bash
sg run -p '<PATTERN>' --lang <LANG> [path]
sg run -p '<PATTERN>' -r '<REWRITE>' --update-all --lang <LANG> [path]
```

---

## TypeScript / TSX / JavaScript

### Find structural patterns

```typescript
// Every function declaration
function $NAME($$$PARAMS) { $$$BODY }

// Every async function
async function $NAME($$$PARAMS) { $$$BODY }

// Every arrow function (any param shape)
($$$PARAMS) => $$$BODY

// Every method on a class
class $C { $$$ $METHOD($$$P) { $$$B } $$$ }

// Every import statement
import { $$$NAMES } from '$MOD'
import $DEFAULT from '$MOD'
import * as $NS from '$MOD'

// Every console.* call
console.$METHOD($$$ARGS)

// Every JSX element of a given name
<MyComponent $$$PROPS>$$$CHILDREN</MyComponent>

// Every try/catch
try { $$$BODY } catch ($E) { $$$HANDLER }

// Every throw
throw $EXPR

// Every new expression
new $CLASS($$$ARGS)

// Every type assertion to any (anti-pattern!)
$EXPR as any
$EXPR as unknown as $T
```

### Common rewrites

```bash
# console.log -> logger.info
sg run -p 'console.log($$$A)' -r 'logger.info($$$A)' --lang ts --update-all .

# require -> import (one-arg case)
sg run -p 'const $V = require($M)' -r 'import $V from $M' --lang ts --update-all .

# .then(callback) -> await on the same line (use with caution; needs async function context)
sg run -p '$P.then($CB)' -r 'const $TMP = await $P; $CB($TMP)' --lang ts --update-all .

# Strip `as any`
sg run -p '$E as any' -r '$E' --lang ts --update-all .

# Rename a function call site
sg run -p 'oldName($$$A)' -r 'newName($$$A)' --lang ts --update-all .
```

---

## Python

> **Reminder**: never end a Python pattern with `:`. Patterns parse as a complete statement, so `def foo($$$):` is invalid.

```python
# Every function definition
def $FN($$$PARAMS)

# Every class definition
class $C($$$BASES)

# Every decorator usage
@$DEC
def $FN($$$P)

# Every print call (Python 3)
print($$$ARGS)

# Every f-string
f"$STR"

# Every with-statement
with $CTX as $VAR: $$$BODY

# Every try/except
try: $$$BODY
except $EXC: $$$HANDLER

# Every list comprehension
[$EXPR for $VAR in $ITER]

# Every async def
async def $FN($$$PARAMS)

# Type hints — Optional[X]
Optional[$T]

# Type hints — X | None (PEP 604)
$T | None
```

### Common rewrites

```bash
# print(...) -> logger.info(...)
sg run -p 'print($$$A)' -r 'logger.info($$$A)' --lang py --update-all .

# Optional[X] -> X | None
sg run -p 'Optional[$T]' -r '$T | None' --lang py --update-all .

# from typing import List -> remove (built-in list works in 3.9+)
sg run -p 'from typing import List' -r 'from typing import List  # TODO: remove, use list' --lang py --update-all .
```

---

## Go

```go
// Every function
func $NAME($$$PARAMS) $$$RET { $$$BODY }

// Every method
func ($RECV $TYPE) $NAME($$$PARAMS) $$$RET { $$$BODY }

// The classic err nil-check
if err != nil { $$$BODY }

// Every fmt.Println / fmt.Printf / fmt.Sprintf
fmt.$METHOD($$$ARGS)

// Every defer
defer $EXPR

// Every goroutine
go $EXPR

// Every channel send/recv
$CH <- $VAL
$VAL := <-$CH

// Every type assertion
$EXPR.($TYPE)
```

### Common rewrites

```bash
# fmt.Println -> log.Println
sg run -p 'fmt.Println($$$A)' -r 'log.Println($$$A)' --lang go --update-all .

# Add error wrapping
sg run -p 'return $ERR' -r 'return fmt.Errorf("operation failed: %w", $ERR)' --lang go --update-all .
```

---

## Rust

```rust
// Every fn
fn $NAME($$$PARAMS) -> $RET { $$$BODY }
fn $NAME($$$PARAMS) { $$$BODY }       // no return type

// Every async fn
async fn $NAME($$$PARAMS) -> $RET { $$$BODY }

// Every method on impl
impl $TYPE { fn $METHOD($$$P) -> $R { $$$B } }

// Every trait impl
impl $TRAIT for $TYPE { $$$ITEMS }

// Every match expression
match $EXPR { $$$ARMS }

// Every Result-returning fn that uses ?
fn $N($$$P) -> Result<$T, $E> { $$$ }

// .unwrap() / .expect() (anti-patterns)
$EXPR.unwrap()
$EXPR.expect($MSG)

// Every println!/eprintln!/format!
println!($$$ARGS)
format!($$$ARGS)
```

### Common rewrites

```bash
# unwrap() -> ? in Result-returning fns (caution: needs context)
sg run -p '$E.unwrap()' -r '$E?' --lang rust --update-all .

# eprintln! -> log::error!
sg run -p 'eprintln!($$$A)' -r 'log::error!($$$A)' --lang rust --update-all .
```

---

## Java

```java
// Every public class
public class $NAME { $$$BODY }

// Every method (any modifier)
$$$MOD $RET $NAME($$$P) { $$$BODY }

// Every System.out.println / System.err.println
System.$STREAM.println($$$ARGS)

// Every try-with-resources
try ($$$RES) { $$$BODY } catch ($EXC $E) { $$$HANDLER }

// Every annotation usage
@$ANNOTATION
$DECL
```

---

## C / C++

```cpp
// Every printf-family call
printf($$$ARGS)
sprintf($$$ARGS)
fprintf($$$ARGS)

// Every malloc / free pair (find-only — pairing requires data flow)
malloc($SIZE)
free($PTR)

// Every for-loop
for ($INIT; $COND; $POST) { $$$BODY }

// C++ smart pointer make
std::make_shared<$T>($$$ARGS)
std::make_unique<$T>($$$ARGS)
```

### Rewrites

```bash
# malloc(N * sizeof(T)) -> calloc(N, sizeof(T)) - safer
sg run -p 'malloc($N * sizeof($T))' -r 'calloc($N, sizeof($T))' --lang c --update-all .
```

---

## CSS

```css
/* Every rule with a specific property */
{ $$$ color: $VAL; $$$ }

/* Every @media query */
@media $QUERY { $$$BODY }

/* Every var() reference */
var($NAME)
```

---

## HTML

```html
<!-- Every img without alt -->
<img $$$ />

<!-- Every script tag -->
<script $$$>$$$BODY</script>

<!-- Every link to stylesheet -->
<link rel="stylesheet" href=$URL />
```

---

## Bash / Shell

```bash
# Every for-loop
for $VAR in $$$LIST; do $$$BODY; done

# Every if-statement
if $$$COND; then $$$BODY; fi

# Every function definition
$NAME() { $$$BODY }

# Every subshell call
$( $$$CMD )
```

---

## YAML rule recipes (for `sg scan`)

These are full YAML rules you can drop in `rules/*.yml` and run via `sg scan`. See `references/yaml-rules.md` for the full schema.

### no-console (TypeScript)

```yaml
id: no-console
language: TypeScript
severity: warning
message: "Avoid console.* in production"
rule:
  pattern: console.$METHOD($$$ARGS)
fix: logger.$METHOD($$$ARGS)
```

### no-as-any (TypeScript)

```yaml
id: no-as-any
language: TypeScript
severity: error
message: "`as any` defeats type safety. Use a proper type."
rule:
  pattern: $EXPR as any
fix: $EXPR
```

### empty-catch (JavaScript)

```yaml
id: empty-catch
language: JavaScript
severity: error
message: "Empty catch swallows errors silently."
rule:
  all:
    - pattern: try { $$$T } catch ($E) { $$$H }
    - has:
        kind: catch_clause
        has:
          kind: statement_block
          not:
            has:
              kind: statement
              stopBy: end
```

### print-to-logger (Python)

```yaml
id: print-to-logger
language: Python
severity: hint
message: "Use logger.info instead of print"
rule:
  pattern: print($$$ARGS)
fix: logger.info($$$ARGS)
```

### no-unwrap (Rust)

```yaml
id: no-unwrap
language: Rust
severity: warning
message: "Avoid .unwrap() in production code; propagate or handle the error."
rule:
  pattern: $EXPR.unwrap()
```

---

## See also

- `references/patterns.md` — meta-variable rules.
- `references/yaml-rules.md` — full YAML rule schema (atomic / relational / composite / transform / fix).
- `references/cli.md` — `sg run`, `sg scan`, `sg test`, `sg new`.
- Official catalog: <https://ast-grep.github.io/catalog/> (community-maintained, browse by language).
