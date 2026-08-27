# Rust Monolith Split Technique (2026-06-20)

Mechanical technique for splitting a large Rust `lib.rs` (3,000-5,000+ lines)
into submodules without changing any public API. Tested on aidens-tool-kit
(3,396 → 54 lines across 7 modules) and aidens-cli (4,996 → ~2,013 lines
across 5 extracted modules).

## When to use

- A crate's `lib.rs` exceeds 1,500-2,000 lines and needs module extraction
  for reviewability
- The file has clear section boundaries (groups of functions/structs that
  belong together)
- You want to preserve the exact public API (no breaking changes)

## The technique

### Step 1: Identify boundaries

Read the full file and identify line ranges for each section. Look for:
- `pub struct` / `pub enum` / `pub trait` declarations
- `impl` blocks
- Free function groups (executors, helpers, descriptors)
- `#[cfg(test)] mod tests {` — this is the test module boundary

### Step 2: Extract sections with sed

```bash
# Extract from the git-committed version (avoids working-tree corruption)
git show HEAD:path/to/lib.rs > /tmp/original.rs

# Extract each section by line range
sed -n '33,630p' /tmp/original.rs | cat -s > registry.rs
sed -n '632,1004p' /tmp/original.rs | cat -s > dispatcher.rs
# ... etc

# For non-contiguous sections, concatenate:
{ sed -n '1006,1617p' /tmp/original.rs; echo ""; sed -n '1929,1974p' /tmp/original.rs; } | cat -s > executors.rs
```

### Step 3: Add `use super::*;` to each module

```bash
for f in registry.rs dispatcher.rs executors.rs; do
  { echo "use super::*;"; echo ""; cat "$f"; } > /tmp/${f}.tmp && mv /tmp/${f}.tmp "$f"
done
```

### Step 4: Make lib.rs imports `pub(crate)`

This is THE KEY INSIGHT. `use super::*;` in submodules only brings in items
that are visible from the parent. If lib.rs has `use aidens_contracts::Foo;`,
submodules can't see `Foo` unless it's `pub(crate) use`:

```rust
// In lib.rs — change from:
use aidens_contracts::{ToolDescriptorV1, ...};
// To:
pub(crate) use aidens_contracts::{ToolDescriptorV1, ...};
```

### Step 5: Make cross-module items `pub(crate)`

Items defined in one submodule but referenced by another need `pub(crate)`:

```rust
// In registry.rs — make visible to dispatcher.rs
pub(crate) enum ToolExecutorV1 { ... }
pub(crate) fn validate_tool_input_with_canonical_runtime(...) { ... }

// In sandbox.rs
pub(crate) fn canonical_sandbox_root(...) { ... }
```

### Step 6: Add explicit cross-module imports

`use super::*;` brings in lib.rs's `pub(crate)` items, but NOT items from
sibling modules. Add explicit imports for cross-module references:

```rust
// In dispatcher.rs — needs items from registry.rs and executors.rs
use super::registry::ToolExecutorV1;
use super::executors::*;
use super::sandbox::canonical_sandbox_root;
```

### Step 7: Write the new thin lib.rs

```rust
pub(crate) use aidens_contracts::{...};  // pub(crate) imports
pub(crate) use std::collections::BTreeMap;

pub mod canonical_stack;  // existing sub-modules
mod exposure;
mod registry;    // new
mod dispatcher;  // new
mod executors;   // new
mod sandbox;     // new
mod patch;       // new
mod descriptors; // new
#[cfg(test)]
mod tests;

pub use registry::ToolRegistryV1;  // re-exports preserve public API
pub use dispatcher::ToolDispatcher;
```

### Step 8: Extract inline tests to tests.rs

If tests were inline (`#[cfg(test)] mod tests { ... }`), extract them:

```bash
# Find the test module start line
test_line=$(grep -n '#\[cfg(test)\]' lib.rs | head -1 | cut -d: -f1)
sed -n "${test_line},\$p" /tmp/original.rs > tests.rs
```

The tests file needs broader imports:
```rust
use super::*;
use super::registry::*;
use super::dispatcher::*;
use super::descriptors::*;
```

## Pitfalls encountered

### Multi-line `use` statement orphaning

`sed` extracts by line number, but multi-line `use` statements get split:
the first line (`use aidens_contracts::{`) is in one section, the continuation
lines (`    Foo, Bar,`) end up in another. Fix: extract from git-committed
original and don't re-process files that already have `use super::*;` —
the `pub(crate) use` in lib.rs handles all imports.

### Module name collisions with test variables

If you name a module `config` and the test code has a local variable `config`,
`use config::*;` in lib.rs creates an ambiguity. Fix: rename the module
(`config` → `cfg`, `scaffold` → `scaff`, `schemas` → `sch`).

### `use super::*;` doesn't bring sibling module items

`use super::*;` brings in items from the PARENT (lib.rs), not from SIBLING
modules. Items in `registry.rs` are not visible in `dispatcher.rs` via
`use super::*;` alone. You need explicit `use super::registry::ItemName;`.

### Platform-specific imports

Items like `#[cfg(unix)] use std::os::unix::fs::MetadataExt;` need to be
`pub(crate)` in lib.rs AND the `#[cfg(unix)]` guard must be preserved.
If a submodule uses `metadata.nlink()` (Unix-only), it needs:
```rust
// In the submodule:
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
```

### `pub(crate)` field visibility for tests

If tests access struct fields directly (`output.timed_out`), those fields
must be `pub` or `pub(crate)`:
```rust
pub(crate) struct TimedCommandOutput {
    pub timed_out: bool,  // was private, tests need it
    pub stdout: String,
    pub stderr: String,
}
```

## When codex can help vs when to do it directly

- **Codex can help** on the ANALYSIS step (reading the file, identifying
  boundaries, suggesting module groupings). Dispatch a codex agent with
  the file and ask for the module structure recommendation.
- **Codex CANNOT help** on the mechanical extraction step for large files.
  A 3,396-line file consumes the agent's context budget on reading, leaving
  nothing for writing. Do the sed extraction directly in the controller.
- **Codex CAN help** on small files (< 500 lines). The codex agent
  successfully split small modules by writing new files with correct
  imports. But for the initial extraction of a monolith, the controller
  should do the sed work.
