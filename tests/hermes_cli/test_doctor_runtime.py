"""`hermes doctor` local-models runtime check (#117326, slice 3).

The check reports two facts the product never surfaces:
- the recorded `server.json` PID vs the actual listener on the recorded port
  (a reused PID makes Local Models say "running" while every request fails);
- the downloaded-but-excluded GGUF census with reasons, so support stops asking
  for a diagnostics bundle for this class.
"""

from __future__ import annotations

from pathlib import Path

from hermes_cli.doctor_runtime import _check_local_runtime
from hermes_cli.local_runtime.presets import PresetEntry

STATE = {"base_url": "http://127.0.0.1:18434/v1", "pid": 4242}


class _Proc:
    pid = 1234


def _install(monkeypatch, *, state=None, proc=None, listener=True,
             staged=(), budget=None, preset=None, header_error=None):
    """Point every external dependency the check touches at fakes."""
    import hermes_cli.doctor_runtime as doctor_runtime
    import hermes_cli.local_runtime.bootstrap as bootstrap
    import hermes_cli.local_runtime.gguf as gguf
    import hermes_cli.local_runtime.hardware as hardware
    import hermes_cli.local_runtime.presets as presets
    import hermes_cli.local_runtime.recovery as recovery

    monkeypatch.setattr(recovery, "read_state", lambda: state if state is not None else {})
    monkeypatch.setattr(recovery, "recorded_process", lambda s: proc)
    monkeypatch.setattr(doctor_runtime, "_listener_on", lambda port, timeout_s=1.0: listener)
    monkeypatch.setattr(bootstrap, "staged_in", lambda mdir, require_complete=True: list(staged))
    monkeypatch.setattr(hardware, "probe_budget", lambda planning=False: budget)

    def fake_preset(gguf_path, _budget, _mtp):
        # A model absent from the map is treated as header-unreadable (preset_for_model -> None).
        return preset.get(gguf_path) if preset else None

    monkeypatch.setattr(presets, "preset_for_model", fake_preset)
    if header_error is not None:
        def raise_header(_path):
            raise ValueError(header_error)

        monkeypatch.setattr(gguf, "read_gguf_header", raise_header)


def test_no_record_is_informational_only(monkeypatch, capsys):
    """No server.json yet: an info line, never an issue."""
    _install(monkeypatch)
    finding = _check_local_runtime(False)
    out = capsys.readouterr().out
    assert "no managed llama-server record" in out
    assert finding.issues == []


def test_live_record_reports_ok(monkeypatch, capsys):
    """A modern record whose pid matches the live process is green."""
    _install(monkeypatch, state=STATE, proc=_Proc())
    finding = _check_local_runtime(True)
    out = capsys.readouterr().out
    assert "record matches the live process" in out
    assert finding.issues == []


def test_stale_record_with_nothing_listening_warns(monkeypatch, capsys):
    """The Local Models 'running' lie: recorded pid dead, port silent."""
    _install(monkeypatch, state=STATE, proc=None, listener=False)
    finding = _check_local_runtime(True)
    out = capsys.readouterr().out
    assert "stale llama-server record" in out
    assert "nothing listens on port 18434" in out
    assert any("server.json" in issue and "aside" in issue for issue in finding.issues)


def test_stale_record_but_port_busy_warns_second_backend(monkeypatch, capsys):
    """The recorded pid is gone but something else answers — a second backend."""
    _install(monkeypatch, state=STATE, proc=None, listener=True)
    finding = _check_local_runtime(True)
    out = capsys.readouterr().out
    assert "stale llama-server record" in out
    assert "something else listens" in out
    assert any("another process is listening" in issue for issue in finding.issues)


def test_excluded_census_reports_refusals_and_unreadable(tmp_path, monkeypatch, capsys):
    """Both exclusion classes surface with their reasons, capped at _MAX_REASONS."""
    refused = Path(tmp_path / "gemma-4-12b-it-qat-q4_0.gguf")
    unreadable = Path(tmp_path / "qwen3.6-14b-mxfp4_moe.gguf")
    _install(
        monkeypatch, state=STATE, proc=_Proc(),
        staged=[unreadable, refused],
        budget=object(),
        preset={
            refused: PresetEntry(model_id="gemma-4-12b-it-qat-q4_0", window=0, spilled=False,
                                 refusal="estimated 31.8 GiB > 25.2 GiB available"),
        },
        header_error="unknown ggml tensor type 39")
    finding = _check_local_runtime(True)
    out = capsys.readouterr().out
    assert "2 of 2 staged GGUF model(s) excluded" in out
    assert any("excluded local models" in issue and "31.8 GiB" in issue for issue in finding.issues)
    assert any("unknown ggml tensor type 39" in issue for issue in finding.issues)


def test_census_all_admitted_is_green(tmp_path, monkeypatch, capsys):
    """Every staged model priced into the listing: no issue, explicit ok line."""
    admitted = Path(tmp_path / "gemma-4-12b-it-qat-q4_0.gguf")
    _install(
        monkeypatch, state=STATE, proc=_Proc(),
        staged=[admitted],
        budget=object(),
        preset={admitted: PresetEntry(model_id="gemma-4-12b-it-qat-q4_0", window=2048, spilled=False,
                                      keys={"model": str(admitted)})})
    finding = _check_local_runtime(True)
    out = capsys.readouterr().out
    assert "all 1 staged GGUF model(s) admitted" in out
    assert finding.issues == []