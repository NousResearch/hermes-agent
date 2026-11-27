#!/usr/bin/env python3
"""
ollama_status.py — Ollama model health status report

Hits the local Ollama API and prints a formatted summary of:
  - API reachability and Ollama version
  - All installed models with sizes
  - Models currently loaded in VRAM

Usage:
    python3 ollama_status.py

    # Custom host
    OLLAMA_HOST=http://192.168.1.10:11434 python3 ollama_status.py
"""

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

# ── ANSI colours (disabled when not a tty) ────────────────────────────────────
_USE_COLOR = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

GREEN  = lambda t: _c("32", t)
YELLOW = lambda t: _c("33", t)
RED    = lambda t: _c("31", t)
BOLD   = lambda t: _c("1",  t)
DIM    = lambda t: _c("2",  t)


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _get(path: str, timeout: int = 5) -> dict:
    url = f"{OLLAMA_HOST}{path}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


# ── Sections ──────────────────────────────────────────────────────────────────

def check_api() -> bool:
    """Return True if Ollama API is reachable."""
    print(BOLD("── Ollama API ──────────────────────────────────────────────"))
    try:
        data = _get("/api/version")
        version = data.get("version", "unknown")
        print(f"  Status  : {GREEN('reachable')}")
        print(f"  Version : {version}")
        print(f"  Host    : {OLLAMA_HOST}")
        return True
    except urllib.error.URLError as exc:
        print(f"  Status  : {RED('unreachable')}  ({exc.reason})")
        print(f"  Host    : {OLLAMA_HOST}")
        print()
        print(DIM("  Start Ollama with:  ollama serve"))
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"  Status  : {RED('error')}  ({exc})")
        return False


def show_installed_models() -> None:
    """Print all installed models with sizes."""
    print()
    print(BOLD("── Installed Models ────────────────────────────────────────"))
    try:
        data = _get("/api/tags")
        models = data.get("models", [])
        if not models:
            print(f"  {DIM('No models installed.')}")
            return

        col_name = 46
        header = f"  {'NAME':<{col_name}}  {'SIZE':>9}  {'MODIFIED'}"
        print(DIM(header))
        print(DIM("  " + "-" * (col_name + 22)))
        total = 0
        for m in sorted(models, key=lambda x: x.get("name", "")):
            name    = m.get("name", "?")
            size    = m.get("size", 0)
            total  += size
            mod_raw = m.get("modified_at", "")
            mod     = mod_raw[:10] if mod_raw else "-"
            print(f"  {name:<{col_name}}  {_fmt_bytes(size):>9}  {mod}")
        print(DIM("  " + "-" * (col_name + 22)))
        print(f"  {'Total':<{col_name}}  {_fmt_bytes(total):>9}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {RED('Error fetching installed models:')} {exc}")


def show_running_models() -> None:
    """Print models currently loaded in VRAM."""
    print()
    print(BOLD("── Running Models (VRAM) ───────────────────────────────────"))
    try:
        data = _get("/api/ps")
        models = data.get("models", [])
        if not models:
            print(f"  {DIM('No models currently loaded in VRAM.')}")
            return

        col_name = 46
        header = f"  {'NAME':<{col_name}}  {'SIZE':>9}  {'VRAM':>9}  EXPIRES"
        print(DIM(header))
        print(DIM("  " + "-" * (col_name + 34)))
        for m in models:
            name       = m.get("name", "?")
            size       = m.get("size", 0)
            size_vram  = m.get("size_vram", 0)
            expires_raw = m.get("expires_at", "")
            expires    = expires_raw[11:16] if len(expires_raw) >= 16 else "-"
            vram_str   = _fmt_bytes(size_vram) if size_vram else DIM("  n/a   ")
            print(
                f"  {name:<{col_name}}  {_fmt_bytes(size):>9}  "
                f"{vram_str:>9}  {expires}"
            )
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            # Older Ollama builds don't have /api/ps
            _show_running_fallback()
        else:
            print(f"  {RED('Error fetching running models:')} {exc}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {RED('Error fetching running models:')} {exc}")


def _show_running_fallback() -> None:
    """Fall back to `ollama ps` CLI when /api/ps is unavailable."""
    try:
        result = subprocess.run(
            ["ollama", "ps"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().splitlines()
            if len(lines) <= 1:
                print(f"  {DIM('No models currently loaded (ollama ps).')}")
            else:
                for line in lines:
                    print(f"  {line}")
        else:
            print(f"  {YELLOW('ollama ps not available or no models loaded.')}")
    except FileNotFoundError:
        print(f"  {YELLOW('ollama CLI not found; cannot show running models.')}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {RED('Error running ollama ps:')} {exc}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print(BOLD("╔══════════════════════════════════════════════════════════╗"))
    print(BOLD("║             Ollama Model Health Status                   ║"))
    print(BOLD("╚══════════════════════════════════════════════════════════╝"))
    print()

    ok = check_api()
    if not ok:
        print()
        sys.exit(1)

    show_installed_models()
    show_running_models()
    print()


if __name__ == "__main__":
    main()
