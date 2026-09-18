"""Regression tests for #110069: ``hermes config set dashboard.basic_auth.*`` must
reject writes that leave the dashboard with no usable credential.

A password_hash passed through a double-quoted shell argument has its ``$``
segments expanded away, so a mangled hash lands in config.yaml; combined with
an accepted empty ``password``, the basic-auth provider then holds a credential
nobody can authenticate with — a silent lockout of the admin surface. Both
writes are now hard-rejected at config-set time (fail-closed).
"""

import base64
from pathlib import Path

import pytest
import yaml

VALID_HASH = (
    "scrypt$16384$8$1$"
    + base64.b64encode(b"s" * 16).decode()
    + "$"
    + base64.b64encode(b"d" * 32).decode()
)


def _write_config(hermes_home: Path, data: dict) -> None:
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "config.yaml").write_text(yaml.dump(data))


def _set(monkeypatch, hermes_home, key, value, force=False):
    """Isolated call to set_config_value against a temp HERMES_HOME."""
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    from hermes_cli.config import set_config_value
    set_config_value(key, value, force=force)


def _read(hermes_home: Path) -> dict:
    return yaml.safe_load((hermes_home / "config.yaml").read_text())


@pytest.fixture
def hermes_home(tmp_path):
    home = tmp_path / ".hermes"
    _write_config(home, {"model": {"default": "test-model"}})
    return home


class TestPasswordHashWriteValidation:
    def test_shell_expanded_hash_is_rejected_fail_closed(self, hermes_home, monkeypatch, capsys):
        # `config set dashboard.basic_auth.password_hash "scrypt$16384$8$1$..."` in a
        # double-quoted shell argument expands every $segment away, leaving "scrypt".
        with pytest.raises(SystemExit) as exc:
            _set(monkeypatch, hermes_home, "dashboard.basic_auth.password_hash", "scrypt")
        assert exc.value.code == 1
        assert "does not parse" in capsys.readouterr().err
        # fail-closed: nothing was written
        assert "basic_auth" not in _read(hermes_home).get("dashboard", {})

    def test_wrong_segment_count_rejected(self, hermes_home, monkeypatch):
        with pytest.raises(SystemExit):
            _set(monkeypatch, hermes_home, "dashboard.basic_auth.password_hash", "scrypt$16384$8$1")

    def test_non_base64_payload_rejected(self, hermes_home, monkeypatch):
        bad = "scrypt$16384$8$1$not*base64*$AAAA"
        with pytest.raises(SystemExit):
            _set(monkeypatch, hermes_home, "dashboard.basic_auth.password_hash", bad)

    def test_valid_hash_is_written(self, hermes_home, monkeypatch):
        _set(monkeypatch, hermes_home, "dashboard.basic_auth.password_hash", VALID_HASH)
        assert _read(hermes_home)["dashboard"]["basic_auth"]["password_hash"] == VALID_HASH

    def test_empty_hash_is_a_legal_clear(self, hermes_home, monkeypatch):
        # Clearing password_hash (falls back to the plaintext password / provider
        # skip) must stay a legal write — it disables, it does not lock out.
        _set(monkeypatch, hermes_home, "dashboard.basic_auth.password_hash", "")
        assert _read(hermes_home)["dashboard"]["basic_auth"]["password_hash"] == ""


class TestEmptyPasswordWriteValidation:
    def test_empty_password_without_hash_rejected(self, hermes_home, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _set(monkeypatch, hermes_home, "dashboard.basic_auth.password", "")
        assert exc.value.code == 1
        assert "lockout" in capsys.readouterr().err
        assert "basic_auth" not in _read(hermes_home).get("dashboard", {})

    def test_empty_password_with_stored_mangled_hash_rejected(self, hermes_home, monkeypatch):
        # The exact incident shape from #110069: mangled hash already stored, then
        # the plaintext fallback is cleared -> no usable credential left.
        _write_config(
            hermes_home,
            {"dashboard": {"basic_auth": {"username": "op", "password_hash": "scrypt"}}},
        )
        with pytest.raises(SystemExit):
            _set(monkeypatch, hermes_home, "dashboard.basic_auth.password", "")

    def test_empty_password_with_valid_hash_allowed(self, hermes_home, monkeypatch):
        # Migrating off the plaintext password once a real hash exists is the
        # documented production setup; clearing must be allowed then.
        _write_config(
            hermes_home,
            {"dashboard": {"basic_auth": {"username": "op", "password_hash": VALID_HASH}}},
        )
        _set(monkeypatch, hermes_home, "dashboard.basic_auth.password", "")
        assert _read(hermes_home)["dashboard"]["basic_auth"]["password"] == ""


class TestProviderEncodingAgreement:
    def test_provider_generated_hash_passes_cli_validation(self, hermes_home, monkeypatch):
        # The CLI guard must accept exactly what the provider's own hash_password()
        # emits, or operators following the documented precompute flow get rejected.
        from plugins.dashboard_auth.basic import hash_password

        _set(monkeypatch, hermes_home, "dashboard.basic_auth.password_hash", hash_password("probe"))

        from hermes_cli.config import _is_valid_scrypt_hash
        stored = _read(hermes_home)["dashboard"]["basic_auth"]["password_hash"]
        assert _is_valid_scrypt_hash(stored)
