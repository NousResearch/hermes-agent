# AXLE verify-proof Security Considerations

## Known limitation

AXLE's `verify_proof` trusts the Lean environment to behave correctly. A sufficiently creative adversary can exploit Lean metaprogramming to make invalid proofs appear valid. This is a known limitation that AXLE does not plan to address.

## What AXLE blocks

Despite trusting the environment, AXLE does apply stricter validation than the Lean compiler:

- **open private** command: banned entirely
- **Unsafe/partial functions**: detected and blocked
- **Non-standard axioms**: must be in the allowed set of standard axioms
- **sorry in proofs**: blocked unless listed in `permitted_sorries`
- **native_decide**: blocked unless axioms `Lean.trustCompiler`, `Lean.ofReduceBool`, `Lean.ofReduceNat` are in `permitted_sorries`

## Verification error patterns

These appear in `tool_messages.errors`:
- "Missing required declaration '{name}'"
- "Kind mismatch for '{name}': candidate has {X} but expected {Y}"
- "Theorem '{name}' does not match expected signature: expected {X}, got {Y}"
- "Definition '{name}' does not match expected signature: expected {X}, got {Y}"
- "Unsafe/partial function '{name}' detected"
- "In '{name}': Axiom '{axiom}' is not in the allowed set of standard axioms"
- "Declaration '{name}' uses 'sorry' which is not allowed in a valid proof"
- "Candidate uses banned 'open private' command"

## For untrusted code

Use these alternatives instead of AXLE:

1. **lean4checker** — Lean FRO-developed .olean verifier. Runs proofs in isolated environments.
2. **Comparator** — Lean FRO-developed gold standard for proof judges.
3. **SafeVerify** — Battle-tested public proof checker.

These alternatives trade speed for security. They are less susceptible to known exploits.

## use_def_eq parameter

- `True` (default): types compared using definitional equality after kernel reduction. More accurate but slower.
- `False`: types compared at face value. Faster but may rarely reject valid proofs where types are definitionally equal but syntactically different.
