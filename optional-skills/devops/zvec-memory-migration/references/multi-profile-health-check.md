# Multi-Profile Zvec Memory System Health Check

## 4-Layer Diagnostic Methodology

To systematically check the memory system health of all Hermes profiles, inspect layer by layer along the following four layers:

```
Layer 1: Config layer   → memory.provider + plugins.memory-zvec configuration
Layer 2: Plugin layer   → symlink integrity + plugin file timestamps
Layer 3: Data layer     → zvec data directory, size, last write time
Layer 4: Runtime layer  → initialization/activation sequence in the logs
```

### Layer 1: Configuration Layer

Check whether each profile's `memory.provider` points to `memory-zvec`:

```bash
echo "=== default ===" && grep -A6 '^memory:' ~/.hermes/config.yaml 2>/dev/null
for p in chip_expert financial_expert health_manager zunhunfan; do
  echo "=== $p ==="
  grep -A6 '^memory:' ~/.hermes/profiles/$p/config.yaml 2>/dev/null || echo "NO CONFIG"
done
```

Expected output for each item contains `provider: memory-zvec`.

Also check that a `memory-zvec` config block exists under the `plugins:` section (zvec_dir / embedding_model / vector_dim, etc.):

```bash
for p in chip_expert financial_expert health_manager zunhunfan; do
  echo "=== $p plugins ==="
  grep -A10 '^plugins:' ~/.hermes/profiles/$p/config.yaml 2>/dev/null
done
```

### Layer 2: Plugin Layer

Check that the plugin itself exists and that the symlinks in each sub-profile are intact:

```bash
echo "=== memory-zvec plugin ===" && ls -la ~/.hermes/plugins/memory-zvec/ | head -5
echo ""
for p in chip_expert financial_expert health_manager zunhunfan; do
  link="$HOME/.hermes/profiles/$p/plugins/memory-zvec"
  if [ -L "$link" ] && [ -d "$(readlink -f "$link")" ]; then
    echo "✅ $p: symlink OK → $(readlink "$link")"
  else
    echo "❌ $p: BROKEN or missing"
  fi
done
```

**Key check**: the modification timestamp of the plugin's `__init__.py`, to confirm whether all profiles use the same version:

```bash
stat -c '%y %n' ~/.hermes/plugins/memory-zvec/__init__.py
```

If the plugin was just updated but a profile's gateway has not been restarted, the runtime still runs the old code.

### Layer 3: Data Layer

Check whether each profile's data directory exists, its size, and its last activity time:

```bash
echo "=== default ===" && du -sh ~/.hermes/记忆数据库/zvec_memory/ 2>/dev/null && \
  ls -lt ~/.hermes/记忆数据库/zvec_memory/memories/manifest.* 2>/dev/null | head -1

for p in chip_expert financial_expert health_manager zunhunfan; do
  dir="$HOME/.hermes/profiles/$p/记忆数据库/zvec_memory/"
  if [ -d "$dir" ]; then
    echo "=== $p ==="
    du -sh "$dir" 2>/dev/null
    manifest=$(ls -t "$dir"/memories/manifest.* 2>/dev/null | head -1)
    if [ -n "$manifest" ]; then
      echo "  last manifest: $(stat -c '%y' "$manifest")"
    fi
  else
    echo "❌ $p: NO DATA DIR"
  fi
done
```

The last modification time of the manifest is the most reliable indicator of whether a profile has had recent memory writes (more direct than agent.log).

### Layer 4: Runtime Layer — Log Activation Sequence

Read each profile's agent.log and check whether the memory-zvec initialization sequence is complete:

```bash
for p in chip_expert financial_expert health_manager zunhunfan; do
  if [ "$p" = "zunhunfan" ]; then
    logfile="$HOME/.hermes/profiles/$p/logs/agent.log"
  else
    logfile="$HOME/.hermes/profiles/$p/logs/agent.log"
  fi
  echo "=== $p ==="
  if [ -f "$logfile" ]; then
    # Check the full initialization sequence
    grep "Memory provider.*memory-zvec registered\|ZvecMemoryProvider.*opened existing\|ZvecMemoryProvider initialized\|Memory provider.*activated\|initialize failed" "$logfile" 2>/dev/null | tail -5
  else
    echo "  No agent.log"
  fi
  echo ""
done
```

#### Normal Activation Sequence (3 Key Log Lines)

```
① Memory provider 'memory-zvec' registered (5 tools)     ← plugin discovered and 5 tools registered
② ZvecMemoryProvider opened existing collection: ...      ← zvec collection opened successfully
③ ZvecMemoryProvider initialized — model=... dim=...     ← initialization complete (includes Ollama warmup)
④ run_agent: Memory provider 'memory-zvec' activated      ← hermes-agent confirms activation
```

**⚠️ Common misjudgment**: hermes-agent prints log line ④ even when `initialize()` throws an exception. So seeing `activated` does **not** mean initialization succeeded — you must also see lines ②/③.

#### Failure Pattern Reference Table

| Log signature | Root cause | Impact |
|---------|------|------|
| Only ① + ④, missing ②③ | `initialize()` exits with an exception (NameError / LOCK, etc.) | `_coll=None`, all tools report NoneType errors |
| `initialize failed: name '_open_read_only' is not defined` | Bare-name call to a static method inside the plugin code (pitfall #14) | `_coll` is always None |
| `still locked after retry, falling back to read-only` | The zvec LOCK is held by an old process or an old session (pitfall #12) | See the fallback chain in pitfall #12 |
| `session_end batch store failed: 'ZvecMemoryProvider' object has no attribute '_coll'` | `__init__` did not initialize `self._coll = None` | A warning at every session end, no actual writes |
| `session_end batch store failed: ...` (other errors) | Exception inside the `_batch_store` thread | The whole batch write fails; `sync_turn` already provides a turn-level fallback |

### Overall Assessment Matrix

| Profile | Config | Plugin | Data | Runtime | Verdict |
|---------|------|------|------|--------|------|
| default | ✅ | ✅ | ✅ 76MB/694 entries | ✅ Normal | ✅ |
| chip_expert | ✅ | ✅ (old code) | ✅ 20MB | ⚠️ NameError → read-only | Gateway restart needed |
| financial_expert | ✅ | ✅ (old code) | ✅ 7.8MB | ⚠️ NameError → read-only | Gateway restart needed |
| health_manager | ✅ | ✅ | ✅ 19MB | ✅ Normal (new code) | ✅ |
| zunhunfan | ✅ | ✅ | ✅ 6.0MB | ❓ No recent conversations → not activated | Auto-loads on the next message |

### Post-Fix Verification: Restart the Gateway

After a plugin update, the old gateway process is still running old code in memory. A restart is required:

```bash
sudo systemctl restart hermes-gateway-<profile>
```

After the restart, check the logs for the complete activation sequence (all three key log lines present):

```bash
grep -E "ZvecMemoryProvider opened|ZvecMemoryProvider initialized|Memory provider.*activated" \
  ~/.hermes/profiles/<profile>/logs/agent.log | tail -5
```

Then use `vec_memory_stats` to confirm it returns a normal count rather than a NoneType error.

### References

- Corresponding pitfall records: pitfall #13 (session_end write failure), #14 (_open_read_only NameError), #12 (LOCK conflict)
- Automated check script: [`scripts/check_all_profiles.py`](../scripts/check_all_profiles.py) (data-layer checks only, no runtime checks)
