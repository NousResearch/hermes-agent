# mem0 conflict resolution decision (Task 6, INV-1/INV-6)

## The situation
**Two architecturally INCOMPATIBLE mem0 implementations** were built independently:

| | Fork (live fleet) | Upstream `929dd9c0d` |
|---|---|---|
| `__init__.py` | **1653 lines, self-contained** — `_DirectRestMem0Client`, hybrid retrieval, dedup ladder (`_dedup_*`), rerank (`_rerank_killed`), temporal boost, forget/restore, user-id pinning, CA-bundle, mint-token velocity | 574 lines — thin, **delegates** to `_backend.py` |
| Helper modules | `temporal_parse.py` only | `_backend.py`, `_oss_providers.py`, `_setup.py` (`Mem0Backend`/`PlatformBackend`/`OSSBackend`, `post_setup` hook) |
| Self-hosted OSS | YES — fork's own `_DirectRestMem0Client` (MEM0_HOST/admin key, direct REST) | YES — upstream's `OSSBackend` (different impl) |
| Commits since base | 10 (Wave-2 hybrid, CA-bundle, user-pinning, dedup) | 4 (v3 API, MEM0_HOST, update/delete, lazy-install) |

Both built self-hosted support → **upstream's `MEM0_HOST`/`OSSBackend` is redundant** with fork's
`_DirectRestMem0Client`. The fork version is a strict superset of capability and is what the live
fleet runs.

## Decision: take FORK's mem0 wholesale; drop upstream's re-architecture
- **`plugins/memory/mem0/__init__.py`** → resolve to **fork's version** (`git checkout --ours`).
  Preserves all fleet-critical logic (INV-1). Upstream's 6 net-new functions (`post_setup`,
  `_create_backend`, `_format_error`, `_is_client_error`, `_shutdown_backend`, `_write_metadata`)
  belong to a delegation architecture fork doesn't use — not portable without rewriting fork's
  self-contained client, which would be a regression.
- **`temporal_parse.py`** → keep (fork-only, fork `__init__` imports it).
- **`plugins/memory/mem0/_backend.py`, `_oss_providers.py`, `_setup.py`** → **`git rm`** (clean ADDs
  from upstream that nothing in the merged tree imports once we keep fork's `__init__`; they assume
  the thin delegating `__init__` we're not taking).
- **`tests/plugins/memory/test_mem0_backend.py`, `test_mem0_setup.py`, `test_mem0_oss_providers.py`**
  → **`git rm`** (clean ADDs that import `from plugins.memory.mem0._backend import Mem0Backend,
  PlatformBackend, OSSBackend` — symbols fork's `__init__` doesn't export; they'd import-fail).
- **`tests/plugins/memory/test_mem0_v2.py` (UD — we modified, upstream deleted)** → **accept
  upstream's delete** (`git rm`). Coverage of fork hybrid retrieval lives in fork's own retained
  suite: `test_mem0_remember.py`, `test_mem0_selfhost.py`, `test_mem0_destructive.py` (INV-6
  satisfied — coverage genuinely relocated/retained elsewhere).

## INV-6 / INV-1 compliance
- No fork-only behavior lost: fork's `__init__` (the superset) is kept intact.
- mem0 hybrid retrieval coverage retained in `test_mem0_remember.py` + `test_mem0_selfhost.py`.
- The dropped files are upstream's *alternative* architecture, not fork coverage — dropping them is
  the correct "supersede" call, recorded here per INV-6 ("deliberate recorded decision, never
  silently dropped").

## Net effect on conflict count
Resolving `__init__.py` = ours and `git rm`-ing the 3 upstream helper modules + 3 upstream tests
collapses 27 hunks (the dominant block) + 1 UD into a clean, fleet-faithful state. mem0 ends as
fork's proven implementation, unchanged.
