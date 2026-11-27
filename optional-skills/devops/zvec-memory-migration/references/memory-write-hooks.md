# Memory Write Hooks — Hermes-side Orchestration Logic

## Write paths

```
run_conversation() completes
  → finalize_turn()                          # turn_finalizer.py:441
    → _sync_external_memory_for_turn(...)
      → if interrupted: return                # interrupted turns are skipped (run_agent.py:3112)
      → memory_manager.sync_all(user, response)  # background thread
        → provider.sync_turn(...)                 # role="turn", daemon thread
      → memory_manager.queue_prefetch_all(...)    # warm up next-turn retrieval

/new, /reset, CLI exit, session expiry
  → commit_memory_session(messages)             # session rollover (run_agent.py:3060)
    → memory_manager.on_session_end(messages)   # background thread
      → provider.on_session_end(...)             # role="session_end"

Context compression (conversation_compression.py:525)
  → agent.commit_memory_session(messages)        # triggered before compression, ensuring compressed turns are stored

Real shutdown (gateway shutdown / CLI exit)
  → shutdown_memory_provider(messages)
    → memory_manager.on_session_end(messages)   # same as above
    → memory_manager.shutdown_all()
      → _drain_sync_executor(timeout=5s)         # wait for background sync threads to finish
      → provider.shutdown()                       # del self._coll + gc.collect()
```

## sync_turn vs on_session_end

| Dimension | sync_turn | on_session_end |
|------|-----------|-----------------|
| Trigger | after each turn completes | /new /reset /exit / compression |
| Content | single turn user+assistant | all turns of the whole session |
| role | "turn" | "session_end" |
| Thread | daemon=True | MemoryManager background worker |
| Skipped when interrupted | yes (`if interrupted: return`) | N/A (fires only at session end) |
| On Ollama failure | whole-call try/except silent failure (that turn is lost) | per-item try/except continue (skips failed items) |
| Double-write | possible, overlapping with on_session_end | yes, the same conversation is stored twice |

## Interrupted-turn handling

`_mirror_to_memory()` source (run_agent.py ~3112):
```python
if interrupted:
    return  # skip the whole function; neither sync nor prefetch runs
```

Reason: an interrupted turn is not a "completed conversation"; writing it would pollute memory.
But note: if the user actively interrupts a **valuable** reply, that turn's memory is lost.
A later `/new`-triggered `on_session_end` writes it back (because it iterates the full messages list).

## on_session_end's per-item skip behavior

`on_session_end`'s batch write loop (`__init__.py` lines 643-646):
```python
try:
    vec = _ollama_embed_single(combined, self.base_url, self.model)
except Exception:
    continue  # silently skip; no retry, no logging
```

On a single embedding failure it `continue`s directly. No retry, no warning log.
Actual impact is low (sync_turn already covers it), but if both sync_turn and on_session_end fail, that turn is lost entirely.

## Lock release during shutdown

`shutdown_all()` calls each provider's `shutdown()`.
memory-zvec's shutdown does:
```python
def shutdown(self) -> None:
    if self._coll is not None:
        try:
            del self._coll
            import gc; gc.collect()  # release the Zvec LOCK file
        except Exception:
            pass
    self._coll = None
```

The Zvec Collection has no `close()`/`release()` method; the lock is released by Python GC.
`del` + `gc.collect()` is the only experimentally verified reliable release method.
Confirmed by experiment: after `del coll`, without `gc.collect()` the lock is not necessarily released immediately.

## Memory-safe path of the gateway's graceful shutdown

`gateway/run.py` `_stop_impl()`'s memory guarantee sequence:

```
① _drain_active_agents(timeout)
   — wait for all in-flight turns to finish (including finalize_turn → sync_turn)
   — after the drain timeout, interrupt unfinished turns (marked resume_pending)

② _finalize_shutdown_agents(active_agents)
   — call _cleanup_agent_resources for each agent active during the drain
   — → shutdown_memory_provider(messages) → on_session_end + shutdown_all

③ iterate _agent_cache (idle cached agents)  ← key! source lines 6740-6755
   for _entry in _idle_agents:
       _agent = _entry[0] if isinstance(_entry, tuple) else _entry
       self._cleanup_agent_resources(_agent)
   — ensure the memory providers of all idle agents also correctly trigger on_session_end

④ disconnect adapters, clean up resources, exit
```

**Step ③ is the core confirmation of this source audit**: the gateway's graceful shutdown does not miss idle cached agents.
But note that the idle cache's LRU cap / TTL sweep eviction (at runtime, not at shutdown) goes through
`_release_evicted_agent_soft` → `release_clients()`, which **does not call** `on_session_end`.
An evicted agent's memory provider instance remains in memory, awaiting cleanup during the next normal shutdown.

## prefetch mechanism

`queue_prefetch_all()` warms up the next turn's retrieval context after each turn completes.
It searches with the current turn's user_text and caches results for 30s.
When the next turn starts, `prefetch()` returns the cache directly, reducing latency.
