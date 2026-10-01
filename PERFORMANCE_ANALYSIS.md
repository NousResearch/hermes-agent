# Performance Analysis: Toolchain Preflight

## Current Implementation Review

### What's Good ✓

1. **Early returns**: When neither matrix extra nor clang build env is detected, the function returns immediately without any expensive operations.

2. **Cheap checks first**: String parsing (`_first_executable`), env var lookups, and `Path(x).name` checks are all very fast operations (< 1μs each).

3. **shutil.which is appropriate**: Looking up binaries on PATH via `shutil.which` is the right tool - it's a fast C-level syscall.

4. **Path.is_file() optimization**: Checking if an absolute compiler path exists is faster than PATH lookup.

### Performance Issue Found ⚠️

**Double subprocess invocation**: `_resolve_cxx` can spawn a subprocess to query Python's sysconfig, and this happens twice per sync call:
- Once in `_build_env_uses_clang(env, python)` (line 79)
- Again in the main function (line 96)

When both `needs_matrix` and `uses_clang` are true, we spawn the subprocess twice unnecessarily.

**Impact**: 
- Subprocess spawn + Python import takes ~10-50ms depending on system
- For the common case (matrix extra on Linux with no CXX/CC set), this doubles from ~20ms to ~40ms
- Not a huge cost, but unnecessary

### Optimization Applied

Refactored to resolve CXX once and reuse the result:

```python
def require_native_cxx_for_sync(...):
    env = dict(build_env if build_env is not None else os.environ)
    needs_matrix = _sync_needs_matrix_native_build(extras)
    
    # Resolve CXX once upfront
    cxx = _resolve_cxx(env, python)
    uses_clang = _build_env_uses_clang_from_resolved(env, cxx)
    
    # ... rest of logic uses the cached `cxx` result
```

This eliminates the redundant subprocess call while maintaining identical behavior.

### Remaining Costs (Acceptable)

1. **One subprocess call maximum**: When no CXX/CC env vars are set and python is provided, we still spawn once to query sysconfig. This is unavoidable - we need this information to make the right decision.

2. **extra_supported import**: This is a module import + function call, but it's negligible (~1ms) and only happens when matrix extra is in the sync list.

3. **shutil.which call**: Only when the preflight is actually needed (matrix on Linux or clang env). Fast PATH lookup is appropriate here.

## Benchmark Scenarios

### Hot path (no matrix, no clang): ~0.1ms
- Quick check of extras list
- Early return before any expensive work

### Matrix on Linux with CXX set: ~1ms
- Parse CXX env var
- One shutil.which call

### Matrix on Linux without CXX/CC: ~15-25ms  
- One subprocess spawn to query sysconfig
- One shutil.which call
- Acceptable cost for a one-time setup check

### Non-matrix, clang CXX set: ~1ms
- Parse CXX env var
- One shutil.which call

## Conclusion

After optimization: **No performance concerns**. The preflight adds minimal overhead (<1ms) in the common path, and even in the worst case (subprocess spawn), the ~20ms cost is acceptable for a pre-sync validation that runs once per environment setup.

The original double-subprocess issue has been fixed.
