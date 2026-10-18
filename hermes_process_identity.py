"""Process birth for worker scopes, without wall-clock identity on Linux.

Keep this module dependency-free: the worker sends hello before binding the
owner's frozen configuration. The PID remains a separate scope field. Numeric
legacy births retain exact timestamp matching, never reinterpretation as ticks.
"""
import math
from pathlib import Path
import re
import sys

_LINUX_BIRTH = re.compile(r'linux-v1:[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}:(?:0|[1-9][0-9]*)')


def _proc_start_ticks(pid: int):
    """Kernel start ticks, or None on failure; callers must fail closed.

    Shared with SessionDB's lock-holder checks. comm (field 2) can contain
    spaces/parens, so field 22 is counted after the LAST ')'.
    """
    try:
        with open(f'/proc/{pid}/stat', 'rb') as fh:
            return int(fh.read().rsplit(b')', 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def valid_process_birth(birth):
    """Accept the versioned wire identity or a finite legacy creation timestamp."""
    if type(birth) is int:
        return birth >= 0
    if type(birth) is float:
        return math.isfinite(birth) and birth >= 0
    return isinstance(birth, str) and len(birth) <= 128 and _LINUX_BIRTH.fullmatch(birth) is not None


def process_birth(process):
    """Read fresh kernel identity on Linux; keep other hosts' timestamp contract.

    Failure to read either kernel field is unknowable, never a reason to fall
    back to wall time or a cached identity.
    """
    if sys.platform != 'linux':
        return process.create_time()
    ticks = _proc_start_ticks(process.pid)
    boot = Path('/proc/sys/kernel/random/boot_id').read_text(encoding='ascii').strip()
    birth = f'linux-v1:{boot}:{ticks}'
    if type(ticks) is not int or not valid_process_birth(birth):
        raise OSError('worker process identity unavailable')
    return birth


def matches_process_birth(process, birth):
    """Compare like versions exactly; Linux read failure always rejects."""
    if not valid_process_birth(birth):
        return False
    if type(birth) in (int, float):
        return process.create_time() == birth
    if sys.platform != 'linux':
        return False
    try:
        return process_birth(process) == birth
    except (OSError, UnicodeError):
        return False
