"""
Capabilities startup assertion for session-orchestration adapters.

At watcher startup, call ``verify_adapters(adapters)`` to probe each adapter's
declared ``Capabilities`` against cheaply-observed behavior (binary-on-PATH,
``--help`` output).  A mismatch logs an error and marks that adapter
unavailable.  A missing binary also marks the adapter unavailable.  Neither
condition crashes the watcher.

Design
------
Each ``Capabilities`` field has a corresponding ``_probe_*`` function that
returns ``True`` if the observed environment supports that feature, or
``False`` if it does not.  The probe is intentionally **cheap**:

- Binary existence: ``shutil.which(binary)``
- Feature flags: run ``<binary> --help`` once and parse stdout+stderr for
  known flag patterns.  No full session is launched.

The ``ProbeRunner`` protocol is injectable so unit tests can supply fake
``--help`` output without invoking real binaries.

Return value
------------
``verify_adapters`` returns a dict mapping adapter name → adapter instance
for all adapters that pass verification.  Adapters that fail (mismatch or
missing binary) are absent from the dict; the watcher should treat missing
entries as unavailable and skip them.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Protocol

from session_orchestration.adapters.base import AgentAdapter
from session_orchestration.types import Capabilities

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ProbeRunner protocol (injectable for tests)
# ---------------------------------------------------------------------------


class ProbeRunner(Protocol):
    """Cheap probe interface over a CLI binary.

    Production code uses ``_SubprocessProbeRunner``; tests inject a fake that
    returns pre-canned stdout/stderr without spawning a process.
    """

    def which(self, binary: str) -> str | None:
        """Return the full path to ``binary`` if it is on PATH, else ``None``."""
        ...

    def help_text(self, binary: str) -> str:
        """Return the combined stdout+stderr of ``<binary> --help``.

        Must not raise even if the binary exits with a nonzero code (many CLIs
        exit 1 on ``--help``).  Returns an empty string if the binary cannot
        be run.
        """
        ...


class _SubprocessProbeRunner:
    """Production ``ProbeRunner`` — uses ``shutil.which`` + ``subprocess``."""

    def which(self, binary: str) -> str | None:
        return shutil.which(binary)

    def help_text(self, binary: str) -> str:
        try:
            result = subprocess.run(
                [binary, "--help"],
                capture_output=True,
                text=True,
                # omp's `--help` is slow (~8s: update-check + init) and slower
                # under load; a 10s budget timed out intermittently, returning
                # empty text that then read as "every flag missing" and disabled
                # the adapter. Give it generous headroom.
                timeout=30,
            )
            return result.stdout + result.stderr
        except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired):
            return ""


# ---------------------------------------------------------------------------
# AdapterProbeSpec — maps an adapter name to its binary + probe config
# ---------------------------------------------------------------------------


@dataclass
class AdapterProbeSpec:
    """Configuration for probing a specific adapter.

    Fields
    ------
    binary:
        The CLI binary name to locate on PATH (e.g. ``"claude"``, ``"omp"``).
    supports_print_mode_flag:
        String that must appear in ``--help`` output for ``supports_print_mode``
        to be considered supported.  ``None`` means: skip the probe (never
        mis-match).
    has_hooks_flag:
        String that must appear in ``--help`` output for ``has_hooks=True``
        to be considered supported.
    rpc_mode_flag:
        String that must appear in ``--help`` for ``rpc_mode=True``.
    json_mode_flag:
        String that must appear in ``--help`` for ``json_mode=True``.
    """

    binary: str
    supports_print_mode_flag: str | None = None
    has_hooks_flag: str | None = None
    rpc_mode_flag: str | None = None
    json_mode_flag: str | None = None


# Default probe specs for the two v1 adapters.
_CLAUDE_SPEC = AdapterProbeSpec(
    binary="claude",
    supports_print_mode_flag="--print",
    # Claude Code no longer exposes a bare `--hook` flag (hooks are configured
    # via settings.json); its help lists `--include-hook-events`. The old
    # `--hook` probe string never matched, so verify_adapters disabled the
    # entire claude adapter every tick even though has_hooks is unused at
    # runtime. Probe the flag that actually proves hook support.
    has_hooks_flag="--include-hook-events",
    rpc_mode_flag=None,       # claude does not declare rpc_mode
    json_mode_flag="--output-format",
)

_OMP_SPEC = AdapterProbeSpec(
    binary="omp",
    supports_print_mode_flag="--print",
    has_hooks_flag="--hook",
    rpc_mode_flag="--mode",   # omp --help shows "--mode=<value>" including rpc
    json_mode_flag="--mode",  # same flag covers json mode
)

# Registry mapping adapter class name → probe spec.
# Extend this dict when new adapters are added.
_ADAPTER_PROBE_SPECS: dict[str, AdapterProbeSpec] = {
    "ClaudeCodeAdapter": _CLAUDE_SPEC,
    "OmpAdapter": _OMP_SPEC,
}


# ---------------------------------------------------------------------------
# Core verification logic
# ---------------------------------------------------------------------------


def _check_capabilities(
    declared: Capabilities,
    spec: AdapterProbeSpec,
    runner: ProbeRunner,
) -> list[str]:
    """Return a list of mismatch descriptions (empty = all OK).

    Each mismatch is a human-readable string suitable for an error log.

    Parameters
    ----------
    declared:
        The ``Capabilities`` returned by the adapter's ``capabilities()``
        method.
    spec:
        The probe spec for this adapter (which binary to run, which flag
        strings to look for).
    runner:
        The ``ProbeRunner`` to use for binary checks and ``--help`` fetches.
    """
    mismatches: list[str] = []

    # --- Binary check ---
    binary_path = runner.which(spec.binary)
    if binary_path is None:
        mismatches.append(f"binary '{spec.binary}' not found on PATH")
        # No help text possible; short-circuit.
        return mismatches

    # Fetch help text once and reuse for all flag checks.
    help_text = runner.help_text(spec.binary)

    # The binary EXISTS (checked above) but `--help` returned nothing — a slow
    # or timed-out probe (omp's --help does an update-check + init and can be
    # slow, especially under load). We cannot verify flags against empty text;
    # treating every flag as "missing" here would spuriously DISABLE a healthy
    # adapter, which stops the watcher from processing its sessions entirely
    # (no state detection, no idle notification). Unverifiable ≠ broken: skip
    # the flag checks and keep the adapter available.
    if not help_text.strip():
        _logger.warning(
            "verify_adapters: '%s --help' returned no output (slow/timed-out "
            "probe); skipping capability checks and keeping adapter available.",
            spec.binary,
        )
        return mismatches

    def _flag_present(flag: str | None) -> bool:
        if flag is None:
            return True  # probe skipped; always considered OK
        return flag in help_text

    # --- supports_print_mode ---
    if declared.supports_print_mode and not _flag_present(spec.supports_print_mode_flag):
        mismatches.append(
            f"declared supports_print_mode=True but '{spec.supports_print_mode_flag}' "
            f"not found in '{spec.binary} --help'"
        )

    # --- has_hooks ---
    if declared.has_hooks and not _flag_present(spec.has_hooks_flag):
        mismatches.append(
            f"declared has_hooks=True but '{spec.has_hooks_flag}' "
            f"not found in '{spec.binary} --help'"
        )

    # --- rpc_mode ---
    if declared.rpc_mode and not _flag_present(spec.rpc_mode_flag):
        mismatches.append(
            f"declared rpc_mode=True but '{spec.rpc_mode_flag}' "
            f"not found in '{spec.binary} --help'"
        )

    # --- json_mode ---
    if declared.json_mode and not _flag_present(spec.json_mode_flag):
        mismatches.append(
            f"declared json_mode=True but '{spec.json_mode_flag}' "
            f"not found in '{spec.binary} --help'"
        )

    return mismatches


def verify_adapters(
    adapters: dict[str, AgentAdapter],
    *,
    probe_runner: ProbeRunner | None = None,
    probe_specs: dict[str, AdapterProbeSpec] | None = None,
) -> dict[str, AgentAdapter]:
    """Assert declared ``Capabilities`` against cheap observed probes for each adapter.

    Called once at watcher startup.  Adapters that pass verification are
    returned in the result dict; adapters that fail are logged as errors and
    omitted.  The watcher NEVER crashes due to a verification failure.

    Parameters
    ----------
    adapters:
        A dict mapping adapter name (arbitrary label used in the registry) →
        adapter instance.  Typically ``{"claude": ClaudeCodeAdapter(), "omp":
        OmpAdapter()}``.
    probe_runner:
        Override the probe back-end.  Defaults to the real subprocess runner.
        Inject a fake in tests.
    probe_specs:
        Override the mapping of adapter class name → ``AdapterProbeSpec``.
        Defaults to ``_ADAPTER_PROBE_SPECS``.  Override in tests to supply
        specs for stub adapters.

    Returns
    -------
    dict[str, AgentAdapter]
        All adapters whose probes passed.  Adapters absent from the return
        value are unavailable for this run.
    """
    runner = probe_runner or _SubprocessProbeRunner()
    specs = probe_specs if probe_specs is not None else _ADAPTER_PROBE_SPECS

    available: dict[str, AgentAdapter] = {}

    for name, adapter in adapters.items():
        class_name = type(adapter).__name__
        spec = specs.get(class_name)

        if spec is None:
            # No probe spec registered for this adapter class.  Log a warning
            # and accept it (we cannot verify what we don't know how to probe).
            _logger.warning(
                "verify_adapters: no probe spec for adapter '%s' (class %s); "
                "marking available without verification.",
                name,
                class_name,
            )
            available[name] = adapter
            continue

        try:
            declared = adapter.capabilities()
        except Exception as exc:
            _logger.error(
                "verify_adapters: adapter '%s' raised %r from capabilities(); "
                "marking unavailable.",
                name,
                exc,
            )
            continue

        mismatches = _check_capabilities(declared, spec, runner)

        if mismatches:
            for mismatch in mismatches:
                _logger.error(
                    "verify_adapters: adapter '%s' capability mismatch — %s",
                    name,
                    mismatch,
                )
            _logger.error(
                "verify_adapters: adapter '%s' disabled due to %d capability mismatch(es).",
                name,
                len(mismatches),
            )
        else:
            _logger.info(
                "verify_adapters: adapter '%s' OK (binary '%s' found; all declared "
                "capabilities confirmed).",
                name,
                spec.binary,
            )
            available[name] = adapter

    return available
