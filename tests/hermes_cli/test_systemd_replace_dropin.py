"""Hermes-authored systemd --replace drop-ins must not survive unit refresh.

The generator emits ``gateway run`` without ``--replace``. A leftover
``20-replace.conf`` overrides that ExecStart, so refresh has to retire the
Hermes file even when the base unit text already matches.
"""

import pytest

import hermes_cli.gateway as gateway_cli


class TestSystemdReplaceDropinRetirement:
    POSITIVE = (
        "# Added to end the gateway respawn storm: a stray lock-holder used to make the\n"
        "# plain `gateway run` exit 1, and Restart=always turned that into thousands of\n"
        "# restarts. `--replace` makes systemd's start reclaim the lock instead of\n"
        "# crash-looping. Remove this file (and daemon-reload) to revert.\n"
        "[Service]\n"
        "ExecStart=\n"
        "ExecStart=/usr/bin/python -m hermes_cli.main gateway run --replace\n"
    )

    def _setup(self, tmp_path, monkeypatch, *, dropin=None):
        unit = tmp_path / "hermes-gateway.service"
        unit.write_bytes(b"[Unit]\nDescription=current\n")
        monkeypatch.setattr(gateway_cli, "get_systemd_unit_path", lambda system=False: unit)
        monkeypatch.setattr(
            gateway_cli, "generate_systemd_unit", lambda **_: "[Unit]\nDescription=current\n"
        )
        monkeypatch.setattr(gateway_cli, "_sync_hermes_home_from_systemd_unit", lambda **_: None)
        calls = []
        monkeypatch.setattr(
            gateway_cli, "_run_systemctl", lambda args, **kwargs: calls.append((args, kwargs))
        )
        if dropin is not None:
            dropin_path = unit.parent / f"{unit.name}.d" / "20-replace.conf"
            dropin_path.parent.mkdir()
            dropin_path.write_text(dropin, encoding="utf-8")
        return unit, calls

    def test_current_unit_retire_dropin_and_reload_once(self, tmp_path, monkeypatch, capsys):
        unit, calls = self._setup(tmp_path, monkeypatch, dropin=self.POSITIVE)
        before = unit.read_bytes()
        assert gateway_cli.systemd_unit_is_current() is False
        assert gateway_cli.refresh_systemd_unit_if_needed() is True
        assert unit.read_bytes() == before
        assert not (unit.parent / f"{unit.name}.d" / "20-replace.conf").exists()
        assert [args for args, _kwargs in calls] == [["daemon-reload"]]
        assert calls[0][1]["system"] is False
        assert "Removed stale Hermes --replace drop-in" in capsys.readouterr().out

    def test_retired_dropin_is_reloaded_when_unit_write_is_refused(
        self, tmp_path, monkeypatch, capsys
    ):
        unit, calls = self._setup(tmp_path, monkeypatch, dropin=self.POSITIVE)
        before = unit.read_bytes()
        monkeypatch.setattr(
            gateway_cli,
            "generate_systemd_unit",
            lambda **_: "[Unit]\nHERMES_HOME=/tmp/pytest-of-example/hermes_test\n",
        )

        assert gateway_cli.refresh_systemd_unit_if_needed() is False
        assert not (unit.parent / f"{unit.name}.d" / "20-replace.conf").exists()
        assert unit.read_bytes() == before
        assert [args for args, _kwargs in calls] == [["daemon-reload"]]
        assert "Removed stale Hermes --replace drop-in" in capsys.readouterr().out

    def test_stale_unit_rewrites_and_retires_dropin_once(self, tmp_path, monkeypatch):
        unit, calls = self._setup(tmp_path, monkeypatch, dropin=self.POSITIVE)
        replacement = "[Unit]\nDescription=rewritten\n"
        monkeypatch.setattr(gateway_cli, "generate_systemd_unit", lambda **_: replacement)

        assert gateway_cli.refresh_systemd_unit_if_needed() is True
        assert unit.read_text(encoding="utf-8") == replacement
        assert not (unit.parent / f"{unit.name}.d" / "20-replace.conf").exists()
        assert [args for args, _kwargs in calls] == [["daemon-reload"]]

    def test_same_filename_without_marker_is_kept(self, tmp_path, monkeypatch):
        unit, calls = self._setup(
            tmp_path, monkeypatch, dropin="[Service]\nExecStart=foo --replace\n"
        )
        assert gateway_cli.systemd_unit_is_current() is True
        assert gateway_cli.refresh_systemd_unit_if_needed() is False
        assert (unit.parent / f"{unit.name}.d" / "20-replace.conf").exists()
        assert calls == []

    def test_other_dropin_name_is_kept(self, tmp_path, monkeypatch):
        unit, calls = self._setup(tmp_path, monkeypatch)
        other = unit.parent / f"{unit.name}.d" / "10-custom.conf"
        other.parent.mkdir(parents=True)
        other.write_text(self.POSITIVE, encoding="utf-8")
        assert gateway_cli.refresh_systemd_unit_if_needed() is False
        assert other.exists() and calls == []

    @pytest.mark.parametrize(
        "text",
        [
            "# Added to end the gateway respawn storm\n[Service]\n",
            "# Added to end the gateway respawn storm\n[Service]\nExecStart=\n",
            "# Added to end the gateway respawn storm\n[Service]\nExecStart=foo\n",
        ],
    )
    def test_incomplete_marker_is_kept(self, tmp_path, monkeypatch, text):
        unit, calls = self._setup(tmp_path, monkeypatch, dropin=text)
        assert gateway_cli.refresh_systemd_unit_if_needed() is False
        assert (unit.parent / f"{unit.name}.d" / "20-replace.conf").exists()
        assert calls == []

    def test_no_dropin_current_unit_is_unchanged(self, tmp_path, monkeypatch):
        unit, calls = self._setup(tmp_path, monkeypatch)
        assert gateway_cli.refresh_systemd_unit_if_needed() is False
        assert unit.read_bytes() == b"[Unit]\nDescription=current\n"
        assert calls == []

    def test_system_scope_uses_patched_unit_path(self, tmp_path, monkeypatch):
        unit, calls = self._setup(tmp_path, monkeypatch, dropin=self.POSITIVE)
        assert gateway_cli.refresh_systemd_unit_if_needed(system=True) is True
        assert not (unit.parent / f"{unit.name}.d" / "20-replace.conf").exists()
        assert calls == [(["daemon-reload"], {"system": True, "check": True, "timeout": 30})]
