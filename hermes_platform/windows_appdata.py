"""The calling MSIX package's physical cache path."""

from functools import lru_cache
from pathlib import Path
import sys


@lru_cache(maxsize=1)
def local_cache_folder() -> Path | None:
    """Read WinRT LocalCache; unpackaged installs need no WinRT dependencies."""
    if sys.platform != "win32":
        return None
    import ctypes

    query = ctypes.WinDLL("kernel32").GetCurrentPackageFullName
    query.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_wchar_p]
    query.restype = ctypes.c_long
    result = query(ctypes.byref(ctypes.c_uint32()), None)
    if result == 15700:  # APPMODEL_ERROR_NO_PACKAGE
        return None
    if result != 122:  # ERROR_INSUFFICIENT_BUFFER: this process has package identity
        raise ctypes.WinError(result)

    from winrt.windows.storage import ApplicationData

    return Path(ApplicationData.current.local_cache_folder.path)
