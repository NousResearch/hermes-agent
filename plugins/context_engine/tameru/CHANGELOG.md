# Tameru Hermes adapter changelog

## 1.4.1 source update — 2026-10-12

- Synchronise vendored engine modules and both Tameru/legacy-extractive manifests to the upstream 1.4.1 source at `e7e32866ded56071af13fd5fb91a2214c1b3c018`. Engine tag/Release publication remains pending.
- Forward Hermes compression keyword arguments, including `bypass_cooldown`, to the parent compressor. The previous signature raised TypeError during an overflow cooldown override.
- Build relevance queries from the text parts of multimodal user content, excluding image payloads, in both proactive pruning and full compression. An explicit focus topic still takes precedence.
- Add two regression tests through the real context-engine discovery path; both failed on the previous adapter and pass after the fixes.

Verification on Python 3.14: 322 Hermes tests passed across the adapter, context engine/compressor, focus, plugin init/discovery and manual compression. Ruff and compile checks pass; the upstream vendor drift check is clean. Upstream engine: 1,168 passed / 3 skipped, production battery 15/15.

This adapter retains the existing preliminary transcript-prune and summary-validation design. It does not include the standalone plugin's per-result ladder, `tameru_expand` store or artifact ledger, and it is not merged into Hermes main by this update.
