"""Managed local llama-server runtime diagnostics (#117326, slice 3).

Two facts the product never surfaces anywhere else:

1. **server.json record truth** — the recorded PID vs the actual listener on the
   recorded port. A reused PID (macOS `distnoted`, etc.) makes the Local Models
   page say "running" while every request fails with a connection error, and the
   only trace is a stale `runtimes/llamacpp/server.json`.
2. **The downloaded-but-excluded GGUF census with reasons** — the preset policy
   already knows why a model was refused (physics refusal, unreadable header);
   doctor says it, so support stops asking for a diagnostics bundle for this class.
"""

from __future__ import annotations

import socket
import urllib.parse

from hermes_cli.doctor_report import Finding, check_info, check_ok, check_warn, doctor_check

# Cap the per-model reason list so a 40-model staging dir does not flood the report.
_MAX_REASONS = 6


def _listener_on(port: int, timeout_s: float = 1.0) -> bool:
    """True when anything accepts TCP on 127.0.0.1:port."""
    with socket.socket() as s:
        s.settimeout(timeout_s)
        return s.connect_ex(("127.0.0.1", port)) == 0


@doctor_check(on_error="local models runtime diagnostics unavailable ({e})")
def _check_local_runtime(should_fix: bool, f: Finding) -> None:
    from hermes_cli.local_runtime import recovery
    from hermes_cli.local_runtime.bootstrap import models_dir, staged_in
    from hermes_cli.local_runtime.hardware import probe_budget
    from hermes_cli.local_runtime.presets import preset_for_model
    from hermes_cli.local_runtime.supervisor import state_path

    # ── server.json record truth ────────────────────────────────────────────
    state = recovery.read_state()
    if not state:
        check_info("no managed llama-server record (Local Models not started yet)")
    else:
        url = urllib.parse.urlsplit(state.get("base_url", ""))
        port = url.port
        recorded_pid = state.get("pid")
        proc = recovery.recorded_process(state)
        if proc is not None:
            check_ok(f"managed llama-server pid {proc.pid} on port {port} — record matches the live process")
        elif port is None:
            check_warn(
                f"stale llama-server record: recorded pid {recorded_pid} has no parseable base_url",
                "(move server.json aside or restart Hermes to reset Local Models)")
            f.issues.append(
                f"reset the stale local-models record by quitting this profile's gateway/Desktop and moving "
                f"{state_path()} aside (or restarting Hermes), then re-add the model")
        elif not _listener_on(port):
            check_warn(
                f"stale llama-server record: recorded pid {recorded_pid} is not a live llama-server and "
                f"nothing listens on port {port}",
                "(Local Models can show 'running' while every request fails)")
            f.issues.append(
                f"reset the stale local-models record by quitting this profile's gateway/Desktop and moving "
                f"{state_path()} aside (or restarting Hermes), then re-add the model")
        else:
            check_warn(
                f"stale llama-server record: recorded pid {recorded_pid} is not the live llama-server, "
                f"but something else listens on port {port}",
                "(a second Hermes backend may be running — quit all Hermes processes and restart)")
            f.issues.append(
                f"another process is listening on port {port} while the recorded llama-server pid "
                f"{recorded_pid} is gone — quit all Hermes processes and restart so the managed server "
                f"rebinds the port")

    # ── downloaded-but-excluded GGUF census ─────────────────────────────────
    staged = staged_in(models_dir())
    if not staged:
        check_info("no staged GGUF models in the local models directory")
        return
    try:
        budget = probe_budget(planning=True)
    except Exception as exc:  # noqa: BLE001
        check_warn("could not probe the hardware budget for the model census", f"({exc})")
        return
    excluded: list[tuple[str, str]] = []
    for gguf in staged:
        entry = preset_for_model(gguf, budget, set())
        if entry is None:
            # preset_for_model only logs the header failure; re-read it for the reason.
            from hermes_cli.local_runtime.gguf import read_gguf_header

            try:
                read_gguf_header(gguf)
                reason = "unreadable model header"
            except (ValueError, OSError) as exc:
                reason = str(exc)
            excluded.append((gguf.name, reason))
        elif entry.refusal:
            excluded.append((gguf.name, entry.refusal))
    if excluded:
        check_warn(f"{len(excluded)} of {len(staged)} staged GGUF model(s) excluded from the Local Models listing")
        lines = "; ".join(f"{name}: {reason}" for name, reason in excluded[:_MAX_REASONS])
        if len(excluded) > _MAX_REASONS:
            lines += f"; …and {len(excluded) - _MAX_REASONS} more"
        f.issues.append(f"excluded local models: {lines}")
    else:
        check_ok(f"all {len(staged)} staged GGUF model(s) admitted to the Local Models listing")