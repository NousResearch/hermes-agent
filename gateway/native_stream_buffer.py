"""Optional Rust-backed storage for high-frequency streamed text.

The gateway remains fully functional without the native library. A built
library is discovered from the checkout's ``native/rust_stream_buffer`` target
or ``<HERMES_HOME>/lib``. The Python backend is intentionally the default
fallback so source checkouts and lean installs do not require a Rust toolchain
at runtime.
"""

from __future__ import annotations

import ctypes
import platform
from functools import lru_cache
from pathlib import Path
from typing import Optional


class _RustApi:
    def __init__(self, library: ctypes.CDLL) -> None:
        self.library = library
        self.new = library.hermes_stream_buffer_new
        self.new.restype = ctypes.c_void_p
        self.free = library.hermes_stream_buffer_free
        self.free.argtypes = [ctypes.c_void_p]
        self.free.restype = None
        self.append = library.hermes_stream_buffer_append
        self.append.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.append.restype = ctypes.c_int
        self.set_active = library.hermes_stream_buffer_set_active
        self.set_active.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.set_active.restype = ctypes.c_int
        self.set_ledger = library.hermes_stream_buffer_set_ledger
        self.set_ledger.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.set_ledger.restype = ctypes.c_int
        self.clear = library.hermes_stream_buffer_clear
        self.clear.argtypes = [ctypes.c_void_p]
        self.clear.restype = None
        self.active_len = library.hermes_stream_buffer_active_len
        self.active_len.argtypes = [ctypes.c_void_p]
        self.active_len.restype = ctypes.c_size_t
        self.ledger_len = library.hermes_stream_buffer_ledger_len
        self.ledger_len.argtypes = [ctypes.c_void_p]
        self.ledger_len.restype = ctypes.c_size_t
        self.active_copy = library.hermes_stream_buffer_active_copy
        self.active_copy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.active_copy.restype = ctypes.c_size_t
        self.ledger_copy = library.hermes_stream_buffer_ledger_copy
        self.ledger_copy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.ledger_copy.restype = ctypes.c_size_t


def _library_names() -> tuple[str, ...]:
    system = platform.system().lower()
    if system == "windows":
        return ("rust_stream_buffer.dll",)
    if system == "darwin":
        return ("librust_stream_buffer.dylib", "rust_stream_buffer.dylib")
    return ("librust_stream_buffer.so", "rust_stream_buffer.so")


def _candidate_paths() -> list[Path]:
    names = _library_names()
    candidates: list[Path] = []
    root = Path(__file__).resolve().parents[1]
    native_root = root / "native" / "rust_stream_buffer" / "target"
    for profile in ("release", "debug"):
        candidates.extend(native_root.joinpath(profile, name) for name in names)
    return candidates


@lru_cache(maxsize=1)
def _load_rust_api() -> Optional[_RustApi]:
    for path in _candidate_paths():
        try:
            if not path.is_file():
                continue
            return _RustApi(ctypes.CDLL(str(path)))
        except (OSError, AttributeError):
            continue
    return None


def rust_stream_buffer_available() -> bool:
    """Return whether a compatible Rust stream buffer can be loaded."""
    return _load_rust_api() is not None


def _as_bytes(text: str) -> bytes:
    return str(text).encode("utf-8", errors="surrogatepass")


def _call_bytes(function, handle: ctypes.c_void_p, text: str) -> None:
    data = _as_bytes(text)
    pointer = ctypes.c_char_p(data) if data else ctypes.c_void_p()
    if not function(handle, pointer, len(data)):
        raise RuntimeError("Rust stream buffer rejected a write")


class StreamBuffer:
    """Two-view accumulator with a Python fallback and optional Rust storage."""

    __slots__ = ("_backend", "_api", "_handle", "_active_cache", "_ledger_cache")

    def __init__(self, *, backend: str = "auto") -> None:
        requested = str(backend or "auto").strip().lower()
        if requested not in {"auto", "python", "rust"}:
            raise ValueError(f"unknown stream buffer backend: {backend!r}")
        api = _load_rust_api() if requested in {"auto", "rust"} else None
        if requested == "rust" and api is None:
            raise RuntimeError(
                "Rust stream buffer is unavailable; build native/rust_stream_buffer or use backend='python'"
            )
        handle = api.new() if api is not None else None
        if api is not None and handle is None:
            if requested == "rust":
                raise RuntimeError("Rust stream buffer could not allocate a handle")
            api = None
        self._backend = "rust" if api is not None else "python"
        self._api = api
        self._handle = handle if api is not None else None
        self._active_cache: Optional[str] = ""
        self._ledger_cache: Optional[str] = ""

    @property
    def backend(self) -> str:
        return self._backend

    def append(self, text: str) -> None:
        if not text:
            return
        api = self._api
        if api is None:
            self._active_cache = (self._active_cache or "") + text
            self._ledger_cache = (self._ledger_cache or "") + text
            return
        handle = self._handle
        if handle is None:
            raise RuntimeError("Rust stream buffer handle is closed")
        _call_bytes(api.append, handle, text)
        self._active_cache = None
        self._ledger_cache = None

    def set_active(self, text: str) -> None:
        api = self._api
        if api is None:
            self._active_cache = text
            return
        handle = self._handle
        if handle is None:
            raise RuntimeError("Rust stream buffer handle is closed")
        _call_bytes(api.set_active, handle, text)
        self._active_cache = None

    def set_ledger(self, text: str) -> None:
        api = self._api
        if api is None:
            self._ledger_cache = text
            return
        handle = self._handle
        if handle is None:
            raise RuntimeError("Rust stream buffer handle is closed")
        _call_bytes(api.set_ledger, handle, text)
        self._ledger_cache = None

    def clear(self) -> None:
        if self._api is None:
            self._active_cache = ""
            self._ledger_cache = ""
            return
        self._api.clear(self._handle)
        self._active_cache = ""
        self._ledger_cache = ""

    def _read(self, length_function, copy_function) -> str:
        api = self._api
        handle = self._handle
        if api is None or handle is None:
            raise RuntimeError("Rust stream buffer handle is closed")
        length = int(length_function(handle))
        if length == 0:
            return ""
        data = ctypes.create_string_buffer(length)
        written = int(copy_function(handle, ctypes.cast(data, ctypes.c_void_p), length))
        return data.raw[: min(length, written)].decode("utf-8", errors="surrogatepass")

    def active(self) -> str:
        if self._active_cache is None:
            api = self._api
            if api is None:
                raise RuntimeError("Rust stream buffer backend is unavailable")
            self._active_cache = self._read(api.active_len, api.active_copy)
        return self._active_cache

    def ledger(self) -> str:
        if self._ledger_cache is None:
            api = self._api
            if api is None:
                raise RuntimeError("Rust stream buffer backend is unavailable")
            self._ledger_cache = self._read(api.ledger_len, api.ledger_copy)
        return self._ledger_cache

    def close(self) -> None:
        if self._api is not None and self._handle:
            self._api.free(self._handle)
            self._handle = None

    def __del__(self) -> None:  # pragma: no cover - interpreter teardown is nondeterministic
        try:
            self.close()
        except Exception:
            pass
