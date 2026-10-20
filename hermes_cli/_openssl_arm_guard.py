"""Neutralize aarch64 hosts that advertise SVE/SME they cannot execute.

Some aarch64 VMs (and some emulated/sandboxed CPU models) list ``sve``/``sve2``/``sme`` in
``AT_HWCAP`` — so ``/proc/cpuinfo`` advertises them and every runtime CPU probe believes it —
while the CPU faults on the first SVE instruction. OpenSSL's aarch64 cpuprobe executes
``cntb`` to derive the vector length, so the fault lands *at library load*: every process that
imports OpenSSL-backed code (``cryptography``, ``ssl`` via a bundled OpenSSL, ``nemo_relay``)
dies with SIGILL before it can report anything, which reads as a random startup crash.

``OPENSSL_armcap=0`` makes OpenSSL skip the probe entirely and use its portable paths — the
cost is ARM hardware crypto acceleration, which is the right trade against not starting at all.

The probe cannot run inside this interpreter: the failure mode is a signal raised at import
time, and a ``SIGILL`` cannot be caught by the process that took it. So it runs one throwaway
child with the variable stripped and only acts when that child was killed by ``SIGILL`` — a
missing ``cryptography`` (or any other ordinary import error) is not evidence of a broken CPU.

Probing beats reading ``AT_HWCAP``: whether the crash happens also depends on the OpenSSL build
in the running interpreter, so two interpreters on the *same* faulty host can disagree — an
older ``cryptography`` may never execute the offending instruction and work fine. Only the
child tells the truth about the interpreter that is actually starting.

This module is stdlib-only and must be imported before anything that loads OpenSSL; see
``hermes_cli/main.py``, which calls :func:`ensure_openssl_armcap_safe` beside
``_early_recovery`` for exactly that reason.
"""

from __future__ import annotations

import os
import platform
import signal
import subprocess
import sys

OPENSSL_ARMCAP_ENV = "OPENSSL_armcap"
# The fully-broken mask: OpenSSL accepts 0 as "no capabilities" without validating it against
# what the probe found, so it never trips OpenSSL 4's subset check either.
SAFE_ARMCAP = "0"

# ``ssl`` covers a bundled/system OpenSSL that probes at load; ``cryptography``'s Rust binding
# is the one that crashes on the hosts seen in the wild. A missing module is an ordinary
# ImportError, not a SIGILL, so listing both cannot produce a false positive.
_PROBE_SOURCE = "import ssl\nimport cryptography.hazmat.bindings._rust\n"
_PROBE_TIMEOUT_SECS = 30


def _is_aarch64_linux() -> bool:
    """Only aarch64 Linux advertises SVE in HWCAP; nowhere else needs the probe."""
    return sys.platform.startswith("linux") and platform.machine().lower() in {"aarch64", "arm64"}


def openssl_probe_sigills() -> bool:
    """True when a child interpreter is killed by SIGILL while loading OpenSSL.

    The child inherits this process's environment minus ``OPENSSL_armcap``, so it probes the
    same way a fresh, unguarded Hermes process would.
    """
    env = {k: v for k, v in os.environ.items() if k != OPENSSL_ARMCAP_ENV}
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE_SOURCE],
            capture_output=True,
            env=env,
            stdin=subprocess.DEVNULL,
            timeout=_PROBE_TIMEOUT_SECS,
        )
    except (OSError, subprocess.SubprocessError):
        # No child, no evidence: never disable hardware crypto on a guess.
        return False
    return proc.returncode == -signal.SIGILL


def ensure_openssl_armcap_safe() -> str | None:
    """Set ``OPENSSL_armcap=0`` when this host cannot run the SVE code it advertises.

    Returns the reason when it acted (for the caller to log/print), else ``None``. Idempotent
    and a no-op off aarch64 Linux, when ``OPENSSL_armcap`` is already configured (an operator's
    choice, including a partial mask, is respected), and when the probe does not SIGILL.
    """
    if not _is_aarch64_linux():
        return None
    if os.environ.get(OPENSSL_ARMCAP_ENV):
        return None
    if not openssl_probe_sigills():
        return None
    os.environ[OPENSSL_ARMCAP_ENV] = SAFE_ARMCAP
    return (
        "this host advertises SVE/SME in AT_HWCAP but faults on the SVE instructions OpenSSL's "
        f"cpuprobe executes at load time; set {OPENSSL_ARMCAP_ENV}={SAFE_ARMCAP} to skip the probe"
    )
