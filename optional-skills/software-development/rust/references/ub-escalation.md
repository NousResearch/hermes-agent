# Rust UB Escalation

A routing rule, not a judgment call. Run the trigger check on every Rust change before anything
else and state the verdict first.

## 1. Trigger check

The change is **escalated** if it adds, moves or modifies any of:

- an `unsafe` block, `unsafe fn`, or `unsafe trait`/`unsafe impl`
- a raw pointer (`*mut T`, `*const T`) or any pointer arithmetic
- `MaybeUninit`, `mem::transmute`, `mem::zeroed`, `mem::uninitialized`, `ptr::read`/`ptr::write`
  and friends, `slice::from_raw_parts`
- an FFI boundary: `extern "C"`, `#[no_mangle]` / `#[unsafe(no_mangle)]`, a `-sys` crate binding,
  a callback handed to foreign code
- `unsafe impl Send` or `unsafe impl Sync`
- a hand-written lock-free primitive, or `core::sync::atomic` with an ordering weaker than `SeqCst`
- `Pin` projection written by hand rather than through `pin-project`/`pin-project-lite`
- a `#[repr(...)]` change on a type that crosses an FFI or transmute boundary

If the change cannot be inspected well enough to answer, the verdict is **escalated** and labelled
conservative. An unmeasured "not escalated" is a false clean. Ordinary Rust touching none of the
above is not escalated, and the SKILL.md gate list is the whole bar.

## 2. What escalation adds (all four are blocking)

1. **The invariant.** Every `unsafe` block ships with a `// SAFETY:` comment stating what it
   asserts and why that holds. Once the compiler stops checking, the comment is the only
   specification left. Enabling `#![deny(unsafe_op_in_unsafe_fn)]` and
   `clippy::undocumented_unsafe_blocks` makes this checkable.
2. **Miri** on the tests that reach the `unsafe` path:
   `cargo +nightly miri test`. Add `MIRIFLAGS=-Zmiri-strict-provenance` for pointer-provenance
   work; `-Zmiri-tree-borrows` checks the newer aliasing model. Miri is the oracle for aliasing,
   use-after-free, uninitialised reads, invalid values, misalignment, out-of-bounds, provenance and
   double free.
3. **A sanitizer** where Miri can't reach (FFI, real I/O, inline asm). Name which and why:
   `-Zsanitizer=address` (UAF, OOB, double free), `=leak`, `=thread` (data races across FFI),
   `=memory` (uninitialised reads; needs `-Zbuild-std`). Run with
   `RUSTFLAGS=-Zsanitizer=<kind> cargo +nightly test --target <host-triple>`.
4. **Concurrency testing** for lock-free or ordering-sensitive code: a loom model
   (`RUSTFLAGS="--cfg loom" cargo test --release`), not a stress loop.

## 3. Category → tool that catches it

| Category | Caught by |
| --- | --- |
| aliasing violation (stacked/tree borrows) | Miri |
| data race | Miri; thread sanitizer across FFI |
| use after free / dangling pointer | Miri; address sanitizer |
| uninitialised memory read | Miri; memory sanitizer |
| invalid value for its type | Miri |
| misaligned access | Miri |
| out-of-bounds access | Miri; address sanitizer |
| provenance violation | Miri, strict-provenance mode |
| double free / invalid free | Miri; address sanitizer |
| incorrect `Send`/`Sync` | Miri, via the race it enables |
| `Pin` invariant violation | partially: reasoning plus Miri |
| FFI boundary UB | sanitizers; Miri cannot cross most FFI |
| unwinding across `extern "C"` | reasoning plus a targeted panic test (or `extern "C-unwind"`) |
| unsafe-contract violation in a dependency | reasoning: read the callee's documented safety contract |

The last three rows are why escalation is not "run Miri and move on". Name the category the change
risks, then the tool that actually reaches it.

## 4. What each observation proves

| Observation | Proves | Does not prove |
| --- | --- | --- |
| `cargo build` succeeds | the type checker accepted it | anything about `unsafe` invariants |
| `cargo test` passes | tested paths ran without a detected fault | untested `unsafe` paths |
| Miri passes on a test | that execution path has no Miri-detectable UB | paths the test doesn't reach |
| a sanitizer passes | that run had no detected fault | other inputs or interleavings |
| a loom check passes | the modelled interleavings are sound | interleavings outside the model |

Coverage limits all of them. If the `unsafe` path is untested, a green Miri run does not satisfy
escalation; a test that reaches the path, then Miri on that test, does.

## 5. When the toolchain can't run it

The change stays blocked. Name the smallest substitute proof: a narrower test Miri can run, a safe
wrapper that shrinks the `unsafe` surface, or a second reader reviewing the `SAFETY:` comment.
Keep the verdict escalated; downgrading it because a tool is unavailable is the failure this
reference exists to prevent.
