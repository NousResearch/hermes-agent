"""The calling MSIX package's physical cache path, independent of AppData redirection."""

from functools import lru_cache
from pathlib import Path
import sys


@lru_cache(maxsize=1)
def local_cache_folder() -> Path | None:
    """Read ApplicationData.Current.LocalCacheFolder; an unpackaged process has none.

    Use WinRT's ABI through ctypes so this also works before optional dependencies
    load. Do not construct a Packages path from environment variables: Windows
    owns both the package identity and the data location.
    """
    if sys.platform != "win32":
        return None
    import ctypes as c
    from uuid import UUID

    kernel = c.WinDLL("kernel32")
    query = kernel.GetCurrentPackageFullName
    query.argtypes = [c.POINTER(c.c_uint32), c.c_wchar_p]
    query.restype = c.c_long
    result = query(c.byref(c.c_uint32()), None)
    if result == 15700:  # APPMODEL_ERROR_NO_PACKAGE
        return None
    if result != 122:  # ERROR_INSUFFICIENT_BUFFER: this process has package identity
        raise c.WinError(result)

    api = c.WinDLL("combase")
    pointer = c.c_void_p
    for name, args, returns in (
        ("RoInitialize", [c.c_uint32], c.c_long),
        ("RoUninitialize", [], None),
        ("WindowsCreateString", [c.c_wchar_p, c.c_uint32, c.POINTER(pointer)], c.c_long),
        ("WindowsDeleteString", [pointer], c.c_long),
        ("WindowsGetStringRawBuffer", [pointer, c.POINTER(c.c_uint32)], c.c_wchar_p),
        ("RoGetActivationFactory", [pointer, pointer, c.POINTER(pointer)], c.c_long),
    ):
        function = getattr(api, name)
        function.argtypes, function.restype = args, returns

    def check(hr):
        if hr < 0:
            raise c.WinError(hr & 0xFFFFFFFF)

    def iid(value):
        return (c.c_ubyte * 16).from_buffer_copy(UUID(value).bytes_le)

    def method(instance, slot, *args):
        table = c.cast(instance, c.POINTER(c.POINTER(pointer))).contents
        return c.WINFUNCTYPE(c.c_long, pointer, *args)(table[slot])

    initialized = api.RoInitialize(1)  # RO_INIT_MULTITHREADED
    if initialized != -2147417850:  # RPC_E_CHANGED_MODE: an existing STA is also fine
        check(initialized)
    strings, interfaces = [], []

    def out_interface():
        value = pointer()
        interfaces.append(value)
        return value

    try:
        name = "Windows.Storage.ApplicationData"
        class_name = pointer()
        strings.append(class_name)
        check(api.WindowsCreateString(name, len(name), c.byref(class_name)))
        factory = out_interface()
        check(api.RoGetActivationFactory(class_name, iid("5612147b-e843-45e3-94d8-06169e3c8e17"),
                                         c.byref(factory)))
        current = out_interface()
        check(method(factory, 6, c.POINTER(pointer))(factory, c.byref(current)))
        data2 = out_interface()
        check(method(current, 0, pointer, c.POINTER(pointer))(
            current, iid("9e65cd69-0ba3-4e32-be29-b02de6607638"), c.byref(data2)))
        folder = out_interface()
        check(method(data2, 6, c.POINTER(pointer))(data2, c.byref(folder)))
        item = out_interface()
        check(method(folder, 0, pointer, c.POINTER(pointer))(
            folder, iid("4207a996-ca2f-42f7-bde8-8b10457a7f30"), c.byref(item)))
        path = pointer()
        strings.append(path)
        check(method(item, 12, c.POINTER(pointer))(item, c.byref(path)))
        return Path(api.WindowsGetStringRawBuffer(path, None))
    finally:
        for value in reversed(interfaces):
            if value:
                method(value, 2)(value)  # IUnknown::Release
        for value in strings:
            api.WindowsDeleteString(value)
        if initialized >= 0:
            api.RoUninitialize()
