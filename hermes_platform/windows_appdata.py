"""The calling MSIX package's physical cache path."""

from functools import lru_cache
from pathlib import Path
import sys


@lru_cache(maxsize=1)
def local_cache_folder() -> Path | None:
    if sys.platform != "win32":
        return None
    try:
        # Read WinRT LocalCache; unpackaged installs need no WinRT dependencies.
        from winrt.windows.storage import ApplicationData

        return Path(ApplicationData.current.local_cache_folder.path)
    except ModuleNotFoundError as exc:
        if exc.name not in ("winrt", "winrt.windows", "winrt.windows.storage"):
            raise
    except OSError as exc:
        if getattr(exc, "winerror", None) != -2147009196:  # HRESULT: no package identity
            raise
    return None
