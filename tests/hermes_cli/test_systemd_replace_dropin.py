"""A Hermes-authored systemd ``--replace`` drop-in must not survive unit refresh.

An older Hermes wrote ``<unit>.d/20-replace.conf`` to end a respawn storm; it overrides ExecStart
with ``gateway run --replace``. Combined with the cross-profile ownership guard, that override turns
a per-profile fleet's unit into one that can never start (#119467). Refresh retires the Hermes file,
and only that file: a drop-in the operator wrote is theirs.
"""
from gateway import systemd_identity
from gateway import systemd_runtime
from gateway import systemd_unit_render
from gateway import systemd_unit_state

import hermes_cli.gateway as gateway_cli

HERMES_DROPIN = (
    "# Added to end the gateway respawn storm: a stray lock-holder used to make the\n"
    "# plain `gateway run` exit 1, and Restart=always turned that into thousands of\n"
    "# restarts. `--replace` makes systemd's start reclaim the lock instead of\n"
    "# crash-looping. Remove this file (and daemon-reload) to revert.\n"
    "[Service]\n"
    "ExecStart=\n"
    "ExecStart=/usr/bin/python -m hermes_cli.main gateway run --replace\n"
)


def _current_unit_with_dropin(tmp_path, monkeypatch, dropin_text: str):
    unit = tmp_path / "hermes-gateway.service"
    unit.write_text("[Unit]\nDescription=current\n", encoding="utf-8")
    monkeypatch.setattr(systemd_identity, "unit_path", lambda system=False: unit)
    monkeypatch.setattr(systemd_unit_render, "generate_systemd_unit", lambda **_: "[Unit]\nDescription=current\n")
    monkeypatch.setattr(systemd_runtime, "sync_home_from_unit", lambda *_, **__: None)
    calls = []
    monkeypatch.setattr(systemd_runtime, "run_systemctl", lambda args, **kwargs: calls.append((args, kwargs)))
    dropin = unit.parent / f"{unit.name}.d" / "20-replace.conf"
    dropin.parent.mkdir()
    dropin.write_text(dropin_text, encoding="utf-8")
    return unit, dropin, calls


def test_refresh_retires_the_hermes_replace_dropin_even_when_the_unit_text_is_current(tmp_path, monkeypatch, capsys):
    """The base unit already matches the generator, so the old gate said "nothing to do" and the
    drop-in kept re-arming ``--replace`` on every restart."""
    unit, dropin, calls = _current_unit_with_dropin(tmp_path, monkeypatch, HERMES_DROPIN)
    before = unit.read_bytes()

    assert systemd_unit_state.refresh_if_needed(system=True) is True

    assert not dropin.exists()
    assert unit.read_bytes() == before, "the current unit itself is left alone"
    assert [args for args, _ in calls] == [["daemon-reload"]] and calls[0][1]["system"] is True
    assert "--replace drop-in" in capsys.readouterr().out


def test_refresh_keeps_an_operator_written_dropin_of_the_same_name(tmp_path, monkeypatch):
    """Only the file Hermes authored (recognised by its own comment) is retired."""
    unit, dropin, calls = _current_unit_with_dropin(
        tmp_path, monkeypatch, "[Service]\nExecStart=\nExecStart=/opt/hermes gateway run --replace\n")

    assert systemd_unit_state.refresh_if_needed() is False

    assert dropin.exists() and calls == []
