"""Reconstruct a process argv from flattened ``ps`` output — structurally or not at all.

``ps -o command=`` flattens a process argv into one string: elements are joined with
single spaces and the original quoting is neither preserved nor re-derivable. For plain
argvs (``hermes serve --port 9119``) whitespace tokenizing plus ``shlex.split``
round-trips. For inline-code launchers (``python -I -c <bootstrap>``) it does not: the
bootstrap element contains internal spaces, so re-tokenizing shatters it into many
tokens, and replaying that argv makes python parse ``-c import`` and die with
SyntaxError (regression: ``tests/hermes_cli/test_update_stale_dashboard.py ::
TestCmdlineCapture::test_flattened_ps_output_with_inline_c_payload_is_not_shattered``).

The only sound argv sources are (a) ``/proc/<pid>/cmdline`` — handled by the caller,
(b) the kernel's saved exec-time argument block — macOS ``sysctl KERN_PROCARGS2``, and
(c) refusal. This module implements (b) and (c). ``None`` means "cannot reconstruct":
callers must report a manual start instead of exec'ing a guessed argv.
"""

import shlex
import sys

_CTL_KERN = 1
# ``sys/sysctl.h`` on macOS: KERN_PROCARGS2 returns the saved exec-time argument block
# (argc, exec path, padding, argv strings) for one pid.
_KERN_PROCARGS2 = 49


def reconstruct_argv(pid: int, command: str) -> list[str] | None:
    """Exact argv of the process ``ps`` reported as *command*, or ``None`` when that
    cannot be done without guessing.

    Plain lines round-trip through ``shlex.split``. A line containing ``-c`` is only
    answered from the kernel's structural record: on macOS via ``KERN_PROCARGS2``,
    anywhere else by refusal. Even a kernel answer is refused when it disagrees with
    the ``ps`` line (first token or ``-c`` presence) — the signature of a recycled pid
    or a ps/kernel race. Refusal (``None``) lets the caller report a manual start
    instead of respawning an argv that dies on exec.
    """
    stripped = command.strip()
    if not stripped:
        return None
    head = stripped.split()
    if "-c" not in head:
        try:
            return shlex.split(stripped)
        except ValueError:
            # Historical fallback for unbalanced quotes: whitespace only. Such a line
            # never round-tripped quoted elements; plain argvs still do.
            return stripped.split() or None
    if sys.platform != "darwin":
        return None
    raw = _macos_procargs_bytes(pid)
    if raw is None:
        return None
    argv = _decode_procargs2(raw)
    if not argv or argv[0] != head[0] or "-c" not in argv:
        return None
    return argv


def _macos_procargs_bytes(pid: int) -> bytes | None:
    """Raw ``KERN_PROCARGS2`` buffer for *pid* via libc sysctl; ``None`` when the kernel
    refuses (no such pid, permission) or returns an empty buffer."""
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    mib = (ctypes.c_int * 3)(_CTL_KERN, _KERN_PROCARGS2, pid)
    size = ctypes.c_size_t(0)
    if libc.sysctl(mib, 3, None, ctypes.byref(size), None, 0) != 0 or size.value == 0:
        return None
    buf = ctypes.create_string_buffer(size.value)
    if libc.sysctl(mib, 3, buf, ctypes.byref(size), None, 0) != 0:
        return None
    return buf.raw[: size.value]


def _decode_procargs2(raw: bytes) -> list[str] | None:
    """Parse the ``KERN_PROCARGS2`` layout: ``int32 argc``, the NUL-terminated exec path,
    NUL padding, then the saved ``argv`` strings. ``None`` on any structural surprise —
    a malformed buffer must read as "unknown", never as a partial argv."""
    if len(raw) < 5:
        return None
    argc = int.from_bytes(raw[:4], sys.byteorder)
    if argc < 1:
        return None
    body = raw[4:]
    path_end = body.find(b"\x00")
    if path_end < 0:
        return None
    parts = body[path_end + 1 :].lstrip(b"\x00").split(b"\x00")
    if len(parts) < argc or any(not part for part in parts[:argc]):
        return None
    return [part.decode("utf-8", errors="replace") for part in parts[:argc]]
