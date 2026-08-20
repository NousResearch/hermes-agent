# Rust Workspace Package Replay Debugging Notes

Session source: AiDENs P32 package/release execution, 2026-06-06.

## Durable lesson

For Rust source/context packages that include sibling path dependencies, separate three states:

1. Package validation passes: archive and sidecars are structurally coherent.
2. Extracted self-replay passes: the extracted archive can run the verifier in isolation.
3. Extracted self-replay is honestly blocked: a path dependency resolves outside the extracted tree.

Do not collapse (2) and (3), and do not misclassify (3) as `cargo` missing just because the stderr includes `cargo fmt` help text.

## Diagnostic pattern

When self-replay fails inside a temp extraction directory:

- Read the replay receipt first, especially stderr tail and blocker classification.
- Look for path resolution evidence:
  - `failed to read /tmp/.../Cargo.toml`
  - `failed to find a workspace root`
  - dependency chains like `semantic-memory -> poly-kv -> fib-quant`
  - relative paths such as `../../Libraries/fib-quant` that resolve outside the extracted tree.
- If the host has `cargo`, but the extracted manifest graph points outside the archive, the blocker is `external_path_dependency_unavailable_in_extracted_package`, not `cargo_or_toolchain_missing_in_replay`.

## Packaging pitfalls found

- If a package archive root is above the actual project root, stale-artifact classification must normalize both `scripts/foo.py` and `Project/scripts/foo.py` forms. Otherwise durable scripts can be falsely treated as stale run residue.
- Root package artifact archivers can delete important root audit docs if broad patterns like `*_AUDIT.md` run before protected-file checks. Protect release-critical docs explicitly.
- A synthetic root `Cargo.toml` for extracted replay must preserve `[workspace.dependencies]` while rewriting both `members` and `default-members` to packaged entries only. Rewriting only `members` can leave stale default members such as `agent-graph`.

## Fix patterns

- Add regression tests for path-prefix classification bugs before patching package logic.
- For synthetic root manifests, rewrite array sections surgically and leave dependency/lint tables intact.
- After package generation, inspect the generated zip/manifest for:
  - root `Cargo.toml` presence,
  - expected project-root docs,
  - stale workspace members not present in archive,
  - package validation status,
  - self-replay receipt status.

## Honest release wording

If package validation passes but self-replay is blocked by external path dependencies, report it as:

> Package sidecars validate, but extracted self-replay is blocked by external path dependency layout. This is not a certified isolated replay package yet.

Do not say "all release gates passed" until the full verifier is rerun after fixes and self-replay status is either passed or explicitly accepted by policy as a known limit.
