# Windows Native Support: Clipboard ctypes + Sandbox TCP Fallback

## Summary

Two fixes that make Hermes work properly on native Windows (no WSL required):

1. **Clipboard: 50,000x faster** — Replace PowerShell subprocess calls with native win32 API via ctypes
2. **Sandbox: works everywhere** — TCP localhost fallback when AF_UNIX sockets aren't available

Also built CPython 3.13.12 from source with the AF_UNIX Windows patch ([python/cpython#137420](https://github.com/python/cpython/pull/137420)) — 10 lines, 3 C files, works.

## Clipboard: ctypes instead of PowerShell

**Problem:** Every Ctrl+V on Windows spawned a `powershell.exe` process to check the clipboard. First call took 2-15 seconds (.NET cold start). Image paste felt completely broken.

**Fix:** Native win32 API via `ctypes.windll.user32`:

| Function | Before (PowerShell) | After (ctypes) |
|----------|-------------------|----------------|
| `has_clipboard_image()` | 2,000-15,000ms | **0.03ms** |
| `has_clipboard_text()` | 2,000-15,000ms | **instant** |
| `get_clipboard_text()` | 2,000-15,000ms | **instant** |
| `save_clipboard_image()` | 3,000-20,000ms | **instant check** + PowerShell only for PNG conversion |

Uses `IsClipboardFormatAvailable` (no clipboard open needed), `OpenClipboard/GetClipboardData/GlobalLock/wstring_at` for text. PowerShell is only invoked when there's actually an image to extract (DIB→PNG needs .NET).

**82/82 clipboard tests pass.**

## Sandbox: Dual-Transport RPC

**Problem:** `execute_code` uses Unix domain sockets for parent↔child RPC. Stock CPython on Windows doesn't have `socket.AF_UNIX` (the upstream PR [python/cpython#137420](https://github.com/python/cpython/pull/137420) has been open since August 2025). The tool showed red/disabled in the banner and returned a hard error.

**Fix:** `_detect_transport()` picks the socket type at import time:
- **POSIX**: AF_UNIX socket at `/tmp/hermes_rpc_xxx.sock` (unchanged)
- **Windows**: AF_INET socket at `127.0.0.1:0` (OS picks random port)

The env var `HERMES_RPC_SOCKET` carries either a file path or `tcp:host:port`. The generated `hermes_tools.py` stub in the child process parses the prefix to decide which socket family to use.

`SANDBOX_AVAILABLE = True` always. No more red in the banner.

**28/28 tests pass** (20 unit + 8 end-to-end integration including actual subprocess sandbox execution with RPC tool calls).

## Bonus: CPython AF_UNIX Build

Built CPython 3.13.12 with the AF_UNIX patch applied — 10 lines across 3 files:
- `PC/pyconfig.h.in` — detect `<afunix.h>` via `__has_include`
- `Modules/socketmodule.h` — include `<afunix.h>`
- `Modules/socketmodule.c` — register `AF_UNIX` in `win_runtime_flags`

Verified: `socket.AF_UNIX = 1`, socket creation works on Windows 11 build 26200.

When Hermes detects this build (or whenever CPython merges the upstream PR), it automatically uses UDS instead of TCP via `_detect_transport()`.

## Files Changed

- `tools/code_execution_tool.py` — Dual-transport sandbox RPC
- `tests/tools/test_code_execution.py` — Updated tests, new transport detection tests
- `hermes_cli/clipboard.py` — Native win32 ctypes clipboard

## Testing

```bash
# Clipboard tests
python -m pytest tests/tools/test_clipboard.py -o "addopts=" -q
# 82 passed

# Sandbox tests (run outside test file due to pytestmark skip)
python /tmp/test_full_transport.py    # 20 passed
python /tmp/test_e2e_sandbox.py       # 8 passed
```

## Docs

GitHub Pages site with full writeup: [docs/windows/index.html](docs/windows/index.html)
