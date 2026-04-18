# Rasputin memory provider

Rasputin is a bundled Hermes memory provider that treats Rasputin as a derived retrieval sidecar rather than the canonical memory store.

Behavior in this V1 implementation:
- best-effort `/health` check during initialize
- best-effort `/search` recall for prompt injection
- best-effort background `/commit` mirroring for completed turns
- mirrored commits for built-in memory writes
- pre-compaction checkpoint commits before old turns are summarized away
- fail-open behavior throughout so Hermes continues normally when Rasputin is unavailable

Canonical memory remains Hermes built-ins plus Ryan Book/JSONL. No model-facing Rasputin tools are exposed in V1.

Environment variables:
- `RASPUTIN_ENABLED` (default: `true`)
- `RASPUTIN_BASE_URL` (default: `http://127.0.0.1:7777`)
- `RASPUTIN_TIMEOUT_SECONDS` (default: `8.0`)
- `RASPUTIN_COMMIT_TIMEOUT_SECONDS` (default: `20.0`)
- `RASPUTIN_SOURCE_NAMESPACE` (default: `hermes`)
- `RASPUTIN_PREFETCH_LIMIT` (default: `8`)
- `RASPUTIN_COMMIT_IMPORTANCE_DEFAULT` (default: `60`)
- `RASPUTIN_FAIL_OPEN` (default: `true`)
