# MemoryMunch Hermes Plugin

This directory vendors Al's Hermes-side MemoryMunch plugin so it can be reviewed,
committed, tagged, and reused instead of living only in `~/.hermes/plugins`.

## Lame terms

MemoryMunch is the memory add-on. It gives Hermes a safer "second brain" lane:

- recent session notebook so Hermes does not lose the thread;
- source labels so old memories do not boss around the live user message;
- soft wall so graph-linked/outward memories can help but cannot become identity;
- bridge to the original MemoryMunch smart_search path: vault + DB keyword + vector + activation;
- DB/vault capture remains off unless explicit approval/env gates are set.

## What the source/test branch does

The source/test branch is not a random code blob. It is the proof harness around
this plugin:

- `agent/background_compaction.py` and `agent/conversation_compression.py` contain
  the Hermes-side compaction slice that lets MemoryMunch provide source-of-truth
  context instead of relying only on a blocking generic summary.
- `tests/run_agent/test_memorymunch_compression_context.py` proves MemoryMunch
  provider text is included in compression context.
- `tests/run_agent/test_background_compaction.py` proves stale-safe background
  compaction scheduling behavior.
- `tests/run_agent/test_memorymunch_softwall.py` proves soft-wall labels,
  query-aware cache, Curator-lite filtering, and `/new` cache cleanup.

Lame terms: the plugin is the engine; the source/test branch is the inspection
station that proves the engine stays safe when Hermes compresses context or
switches sessions.

## Update-safe senior protocol

Normal senior-engineer protocol for this type of local plugin is:

1. Vendor the plugin in git. This folder is that source-of-truth copy.
2. Keep runtime install outside Hermes core at `$HERMES_HOME/plugins/memorymunch`.
3. Add a watchdog that compares runtime files against this vendored copy.
4. Tag every known-good point.
5. After native `hermes update`, run the watchdog and focused tests.
6. If native update moved the source branch, reapply or merge from the golden tag.
7. Do not patch Hermes core unless there is no plugin/config/wrapper path.

## Install / refresh runtime plugin

From the Hermes repo root:

```bash
python contrib/plugins/memorymunch/watchdog.py --json
python contrib/plugins/memorymunch/watchdog.py --repair --json
```

`--json` only checks by default. `--repair` copies this vendored plugin into
`~/.hermes/plugins/memorymunch` after reporting drift.

## Required runtime gates

The provider only reports available when explicitly enabled and wrapper exists:

```bash
export HERMES_MEMORYMUNCH_ENABLE=1
```

Live DB/vault capture remains off unless both are set:

```bash
export HERMES_MEMORYMUNCH_LIVE_WRITE_ENABLE=1
export HERMES_MEMORYMUNCH_AUTO_CAPTURE_ENABLE=1
```

Do not enable live capture without a rollback pack and explicit approval.

## Files

- `__init__.py` — Hermes MemoryProvider implementation and tool schemas.
- `readonly_recall.py` — wrapper that calls original MemoryMunch smart_search.
- `original_bridge.py` — direct bridge into the original MemoryMunch repo tools.
- `plugin.yaml` — plugin metadata.
- `watchdog.py` — drift/status checker and optional runtime repair copier.

## Known proof command

```bash
/home/alcoo/.hermes/hermes-agent/venv/bin/python -m pytest \
  tests/run_agent/test_background_compaction.py \
  tests/run_agent/test_memorymunch_compression_context.py \
  tests/run_agent/test_memorymunch_softwall.py \
  tests/plugins/test_memorymunch_watchdog.py \
  -q -o 'addopts='
```
