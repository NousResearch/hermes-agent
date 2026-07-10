# mem0 resolution - 2026-07-10

Resolution shape: take-fork, per RESOLUTION-SPEC-2026-07-10.

Kept the fork's self-contained `plugins/memory/mem0/__init__.py` implementation wholesale. Removed upstream's refactor shards:

- `plugins/memory/mem0/_backend.py`
- `plugins/memory/mem0/_setup.py`
- `tests/plugins/memory/test_mem0_backend.py`
- `tests/plugins/memory/test_mem0_setup.py`
- `tests/plugins/memory/test_mem0_v3.py`

Reason: the fork implementation is the production fleet superset, including self-host direct REST, capture controls, QMD/gbrain recall, rerank gates, prefetch telemetry, destructive-tool safeguards, provider dead-letter handling, and fleet capture/routing behavior.

Upstream capabilities reviewed for follow-up: upstream split the backend/setup implementation into `Mem0Backend` variants and added/refined setup wizard paths for platform/self-hosted/OSS modes. Those are mostly architectural extractions around capabilities the fork already carries inline. If the orchestrator wants the upstream setup-wizard UX/refactor shape, it should be ported deliberately onto the fork client rather than reintroducing `_backend.py`/`_setup.py` as orphan modules.
