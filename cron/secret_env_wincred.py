"""``wincred://<target>``: a generic credential in the Windows Credential Manager.

Standard-library ctypes only. Register a value without echoing it (``cmdkey`` prompts for it):
``cmdkey /generic:<target> /user:hermes /pass``. ``cmdkey`` stores the blob as UTF-16LE; any other
encoding fails closed instead of being guessed. Importable everywhere; off Windows every
resolution is NOT_CONFIGURED.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

from agent.secret_sources.base import ErrorKind
from cron.secret_env import SecretBackendError

CRED_TYPE_GENERIC = 1
ERROR_NOT_FOUND = 1168


class CREDENTIALW(ctypes.Structure):
    _fields_ = [
        ("Flags", wintypes.DWORD),
        ("Type", wintypes.DWORD),
        ("TargetName", wintypes.LPWSTR),
        ("Comment", wintypes.LPWSTR),
        ("LastWritten", wintypes.FILETIME),
        ("CredentialBlobSize", wintypes.DWORD),
        ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
        ("Persist", wintypes.DWORD),
        ("AttributeCount", wintypes.DWORD),
        ("Attributes", ctypes.c_void_p),
        ("TargetAlias", wintypes.LPWSTR),
        ("UserName", wintypes.LPWSTR),
    ]


def _load_advapi32():
    if sys.platform != "win32":
        raise SecretBackendError(ErrorKind.NOT_CONFIGURED)
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    advapi32.CredReadW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
        ctypes.POINTER(ctypes.POINTER(CREDENTIALW))]
    advapi32.CredReadW.restype = wintypes.BOOL
    advapi32.CredFree.argtypes = [ctypes.c_void_p]
    advapi32.CredFree.restype = None
    return advapi32


def _last_error() -> int:
    return ctypes.get_last_error()


class WinCredBackend:
    def resolve(self, target: str) -> str:
        advapi32 = _load_advapi32()
        pcred = ctypes.POINTER(CREDENTIALW)()
        if not advapi32.CredReadW(target, CRED_TYPE_GENERIC, 0, ctypes.pointer(pcred)):
            raise SecretBackendError(
                ErrorKind.REF_INVALID if _last_error() == ERROR_NOT_FOUND else ErrorKind.INTERNAL)
        try:
            size = pcred.contents.CredentialBlobSize
            if size == 0:
                raise SecretBackendError(ErrorKind.EMPTY_VALUE)
            blob = ctypes.string_at(pcred.contents.CredentialBlob, size)
        finally:
            advapi32.CredFree(pcred)
        if size % 2:
            raise SecretBackendError(ErrorKind.INTERNAL)
        try:
            return blob.decode("utf-16-le")
        except UnicodeDecodeError:
            raise SecretBackendError(ErrorKind.INTERNAL) from None
