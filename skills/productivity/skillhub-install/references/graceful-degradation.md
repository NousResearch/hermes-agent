# Graceful Degradation Pattern

When code depends on optional modules (e.g., Hermes core's `tools.skills_guard`), use graceful degradation to work in all environments.

## Pattern

```python
def _try_core_scanner():
    """Try to import Hermes core's scanner.
    
    Returns (scan_func, is_cached) or (None, False) if unavailable.
    Prefers scan_skill_cached (newer cores); falls back to scan_skill.
    """
    try:
        from tools.skills_guard import scan_skill_cached
        return scan_skill_cached, True
    except ImportError:
        pass
    try:
        from tools.skills_guard import scan_skill
        return scan_skill, False
    except ImportError:
        pass
    return None, False


def scan_quarantine(q_path, slug, files):
    """Scan with core scanner if available, else fallback to built-in."""
    core_scan, is_cached = _try_core_scanner()

    if core_scan is not None:
        label = "core (cached)" if is_cached else "core"
        print(f"  Scanner: {label}")
        try:
            result = core_scan(q_path, source=slug)
            verdict = getattr(result, "verdict", "safe")
            findings_raw = getattr(result, "findings", [])
            findings = [
                f if isinstance(f, str) else getattr(f, "message", str(f))
                for f in findings_raw
            ]
            return verdict, findings
        except Exception as e:
            print(f"  [Core scanner failed: {e}, falling back to built-in]")

    # Fallback: use our own scan_bundle
    print("  Scanner: built-in")
    return scan_bundle(files)
```

## Why This Matters

**Scenario 1: Running inside hermes-agent repo**
- `tools.skills_guard` is available
- Uses core scanner for parity with `do_install`
- Benefits from scan cache (faster repeated scans)

**Scenario 2: Running standalone (skill directory only)**
- `tools.skills_guard` is unavailable
- Falls back to built-in `scan_bundle()`
- Still provides security scanning, just without cache

**Scenario 3: Core scanner crashes**
- Exception caught and logged
- Automatically falls back to built-in
- Installation continues instead of failing

## Key Principles

1. **Try preferred option first** — newer/better API (e.g., `scan_skill_cached`)
2. **Fall back to older option** — legacy API (e.g., `scan_skill`)
3. **Fall back to built-in** — no external dependencies
4. **Catch and log failures** — don't crash on fallback
5. **Print which option was used** — helps debugging

## Common Use Cases

- **HTTP clients**: Try `httpx` → `requests` → `urllib`
- **JSON parsers**: Try `orjson` → `ujson` → `json`
- **Image libraries**: Try `PIL` → `cv2` → skip image features
- **Async runtimes**: Try `asyncio` → `trio` → sync fallback

## Anti-Patterns

❌ **Hard dependency**
```python
from tools.skills_guard import scan_skill_cached
# Crashes if tools module unavailable
```

❌ **Silent fallback**
```python
try:
    from tools.skills_guard import scan_skill_cached
except ImportError:
    pass  # No indication which scanner was used
```

❌ **No fallback**
```python
if not scan_skill_cached:
    raise RuntimeError("Scanner not available")
# Fails instead of degrading gracefully
```

✅ **Graceful degradation**
```python
core_scan, is_cached = _try_core_scanner()
if core_scan:
    print(f"  Scanner: core ({'cached' if is_cached else 'legacy'})")
    # ... use it
else:
    print("  Scanner: built-in")
    # ... use fallback
```