"""Packaged-install resource pointers (``HERMES_BUNDLED_*``) survive into generated service units."""

import plistlib

import pytest

pytest.importorskip("pwd")

import hermes_cli.gateway as gateway_cli
from hermes_cli.gateway_bundled_env import BUNDLED_RESOURCE_ENV_VARS
from hermes_cli.gateway_launchd import generate_launchd_plist


class TestBundledResourceEnvPropagation:
    """#85357: HERMES_BUNDLED_* pointers the ``hermes`` wrapper exports must be baked into the
    generated launchd/systemd units, since the supervisor starts the venv python directly (no
    wrapper) — otherwise a Homebrew/Nix gateway can't find bundled adapters ("No adapter available
    for telegram/discord/slack"). Regeneration must also carry pointers already on disk forward, so
    an ordinary start from a shell without the wrapper env doesn't silently strip them."""

    def _clear_bundled_env(self, monkeypatch):
        for name in BUNDLED_RESOURCE_ENV_VARS:
            monkeypatch.delenv(name, raising=False)

    def test_systemd_unit_omits_bundled_env_when_absent(self, monkeypatch, tmp_path):
        self._clear_bundled_env(monkeypatch)
        monkeypatch.setattr(gateway_cli, "get_systemd_unit_path", lambda system=False: tmp_path / "hermes-gateway.service")
        unit = gateway_cli.generate_systemd_unit(system=False)
        for name in BUNDLED_RESOURCE_ENV_VARS:
            assert name not in unit

    def test_systemd_unit_bakes_wrapper_pointers_escaped(self, monkeypatch, tmp_path):
        self._clear_bundled_env(monkeypatch)
        monkeypatch.setattr(gateway_cli, "get_systemd_unit_path", lambda system=False: tmp_path / "hermes-gateway.service")
        monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", '/opt/he"rmes/plu%gins')
        monkeypatch.setenv("HERMES_TUI_DIR", "/opt/tui")
        unit = gateway_cli.generate_systemd_unit(system=False)
        # systemd quoting: \\ and " escaped, % doubled to defeat specifier expansion.
        assert 'Environment="HERMES_BUNDLED_PLUGINS=/opt/he\\"rmes/plu%%gins"' in unit
        assert 'Environment="HERMES_TUI_DIR=/opt/tui"' in unit
        # Injected between HERMES_SUPERVISED_CHILD and Restart=always, not elsewhere.
        body = unit.split('Environment="HERMES_SUPERVISED_CHILD=1"', 1)[1]
        assert body.index("HERMES_BUNDLED_PLUGINS") < body.index("Restart=always")

    def test_systemd_unit_carries_on_disk_pointers_forward(self, monkeypatch, tmp_path):
        self._clear_bundled_env(monkeypatch)
        unit_path = tmp_path / "hermes-gateway.service"
        monkeypatch.setattr(gateway_cli, "get_systemd_unit_path", lambda system=False: unit_path)
        monkeypatch.setenv("HERMES_BUNDLED_SKILLS", "/opt/skills")
        unit_path.write_text(gateway_cli.generate_systemd_unit(system=False), encoding="utf-8")
        # Wrapper env gone (a plain shell start): the on-disk pointer must survive regeneration.
        monkeypatch.delenv("HERMES_BUNDLED_SKILLS")
        regenerated = gateway_cli.generate_systemd_unit(system=False)
        assert 'Environment="HERMES_BUNDLED_SKILLS=/opt/skills"' in regenerated

    def test_systemd_wrapper_env_overrides_on_disk_pointer(self, monkeypatch, tmp_path):
        self._clear_bundled_env(monkeypatch)
        unit_path = tmp_path / "hermes-gateway.service"
        monkeypatch.setattr(gateway_cli, "get_systemd_unit_path", lambda system=False: unit_path)
        monkeypatch.setenv("HERMES_BUNDLED_SKILLS", "/opt/old")
        unit_path.write_text(gateway_cli.generate_systemd_unit(system=False), encoding="utf-8")
        monkeypatch.setenv("HERMES_BUNDLED_SKILLS", "/opt/new")
        regenerated = gateway_cli.generate_systemd_unit(system=False)
        assert 'Environment="HERMES_BUNDLED_SKILLS=/opt/new"' in regenerated
        assert "/opt/old" not in regenerated

    def test_systemd_unit_carries_hostile_chars_forward_round_trip(self, monkeypatch, tmp_path):
        """A carried-forward pointer holding systemd's quoting metacharacters (``%``/``"``/``\\``)
        must survive the write→read-back→rewrite round-trip byte-for-byte. The first unit escapes it
        via ``_systemd_env_line``; regeneration with the wrapper env gone reads it back through
        ``_unit_environment_value`` (undoing that quoting) and re-escapes it, so the on-disk line
        must be identical and ``_unit_environment_value`` must recover the exact original value."""
        self._clear_bundled_env(monkeypatch)
        unit_path = tmp_path / "hermes-gateway.service"
        monkeypatch.setattr(gateway_cli, "get_systemd_unit_path", lambda system=False: unit_path)
        hostile = '/opt/he"rmes/pl%ug\\ins'
        monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", hostile)
        first = gateway_cli.generate_systemd_unit(system=False)
        expected_line = 'Environment="HERMES_BUNDLED_PLUGINS=/opt/he\\"rmes/pl%%ug\\\\ins"'
        assert expected_line in first
        unit_path.write_text(first, encoding="utf-8")
        # Wrapper env gone (a plain shell start): the on-disk pointer is read back and re-baked.
        monkeypatch.delenv("HERMES_BUNDLED_PLUGINS")
        regenerated = gateway_cli.generate_systemd_unit(system=False)
        assert expected_line in regenerated
        # The read-back parser recovers the original, unescaped value from the regenerated unit.
        unit_path.write_text(regenerated, encoding="utf-8")
        assert gateway_cli._unit_environment_value(unit_path, "HERMES_BUNDLED_PLUGINS") == hostile

    def test_launchd_plist_omits_bundled_env_when_absent(self, monkeypatch, tmp_path):
        self._clear_bundled_env(monkeypatch)
        monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: tmp_path / "gw.plist")
        plist = generate_launchd_plist()
        env = plistlib.loads(plist.encode("utf-8"))["EnvironmentVariables"]
        for name in BUNDLED_RESOURCE_ENV_VARS:
            assert name not in env

    def test_launchd_plist_bakes_wrapper_pointers_xml_escaped(self, monkeypatch, tmp_path):
        self._clear_bundled_env(monkeypatch)
        monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: tmp_path / "gw.plist")
        monkeypatch.setenv("HERMES_BUNDLED_SKILLS", "/opt/sk&ills")
        plist = generate_launchd_plist()
        assert "<string>/opt/sk&amp;ills</string>" in plist  # XML-escaped on the wire
        env = plistlib.loads(plist.encode("utf-8"))["EnvironmentVariables"]
        assert env["HERMES_BUNDLED_SKILLS"] == "/opt/sk&ills"  # unescapes back to the real value

    def test_launchd_plist_carries_on_disk_pointers_forward(self, monkeypatch, tmp_path):
        self._clear_bundled_env(monkeypatch)
        plist_path = tmp_path / "gw.plist"
        monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)
        monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", "/opt/plugins")
        plist_path.write_text(generate_launchd_plist(), encoding="utf-8")
        monkeypatch.delenv("HERMES_BUNDLED_PLUGINS")
        env = plistlib.loads(generate_launchd_plist().encode("utf-8"))["EnvironmentVariables"]
        assert env["HERMES_BUNDLED_PLUGINS"] == "/opt/plugins"
