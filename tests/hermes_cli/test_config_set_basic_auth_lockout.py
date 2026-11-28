"""Regression tests for #110069: ``hermes config set dashboard.basic_auth.*`` must
reject writes that leave the dashboard with no usable credential.

The write is the entry point for the reported lockout: a ``password_hash`` passed
through a double-quoted shell argument has its ``$`` segments expanded away, so a
mangled value lands in config.yaml, and the basic-auth provider treats any non-empty
``password_hash`` as authoritative while ``_verify_password`` returns ``False`` for
every password. The operator then cannot sign in to the surface used to administer the
agent. Both halves of that write are now hard-rejected at config-set time
(fail-closed: nothing is written), instead of failing silently until login.
"""

import base64
import os
from unittest.mock import patch

import pytest
import hermes_yaml as yaml

from hermes_cli.config import set_config_value

VALID_HASH = (
    "scrypt$16384$8$1$"
    + base64.b64encode(b"s" * 16).decode()
    + "$"
    + base64.b64encode(b"d" * 32).decode()
)

# Shapes a mangled or hand-typed hash actually takes.
MALFORMED_HASHES = [
    pytest.param("scrypt", id="shell-expanded-to-the-scheme"),
    pytest.param("scrypt$", id="trailing-separator"),
    pytest.param("scrypt$16384$8$1", id="too-few-segments"),
    pytest.param("scrypt$16384$8$1$AAAA$BBBB$CCCC", id="too-many-segments"),
    pytest.param("Scrypt$16384$8$1$AAAA$BBBB", id="wrong-scheme-case"),
    pytest.param("scrypt$0$8$1$AAAA$BBBB", id="zero-n"),
    pytest.param("scrypt$16384$0$1$AAAA$BBBB", id="zero-r"),
    pytest.param("scrypt$16384$8$0$AAAA$BBBB", id="zero-p"),
    pytest.param("scrypt$nope$8$1$AAAA$BBBB", id="non-integer-n"),
    pytest.param("scrypt$16384$8$1$not*base64*$BBBB", id="non-base64-salt"),
    pytest.param("scrypt$16384$8$1$AAAA$not*base64*", id="non-base64-derived-key"),
]


@pytest.fixture(autouse=True)
def _isolated_hermes_home(tmp_path):
    """Point HERMES_HOME at a temp dir so tests never touch real config."""
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        yield tmp_path


def _write_config(home, data):
    (home / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


def _read_config(home):
    """Merged-from-disk config; ``{}`` when the guard refused the write and wrote nothing."""
    config_path = home / "config.yaml"
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def _stored_basic_auth(home):
    return (_read_config(home).get("dashboard") or {}).get("basic_auth") or {}


class TestPasswordHashValidation:
    @pytest.mark.parametrize("value", MALFORMED_HASHES)
    def test_malformed_hash_rejected_and_nothing_written(self, _isolated_hermes_home, value, capsys):
        with pytest.raises(SystemExit) as exc:
            set_config_value("dashboard.basic_auth.password_hash", value)
        assert exc.value.code == 1
        assert "does not parse" in capsys.readouterr().err
        # Fail-closed: the broken credential never reaches disk.
        assert "basic_auth" not in _read_config(_isolated_hermes_home).get("dashboard", {})

    def test_shell_expansion_hint_is_named(self, _isolated_hermes_home, capsys):
        """The operator typo that caused the report must be recognisable in the error."""
        with pytest.raises(SystemExit):
            set_config_value("dashboard.basic_auth.password_hash", "scrypt")
        err = capsys.readouterr().err
        assert "single" in err and "quote" in err

    @pytest.mark.parametrize("value", ["", "   "])
    def test_empty_hash_is_a_legal_clear(self, _isolated_hermes_home, value):
        """Empty disables the provider (falls back to password / provider skip) — not a lockout."""
        set_config_value("dashboard.basic_auth.password_hash", value)
        assert _stored_basic_auth(_isolated_hermes_home)["password_hash"] == value

    def test_valid_hash_is_written(self, _isolated_hermes_home):
        set_config_value("dashboard.basic_auth.password_hash", VALID_HASH)
        assert _stored_basic_auth(_isolated_hermes_home)["password_hash"] == VALID_HASH


class TestEmptyPasswordValidation:
    def test_empty_password_without_stored_hash_rejected(self, _isolated_hermes_home, capsys):
        with pytest.raises(SystemExit) as exc:
            set_config_value("dashboard.basic_auth.password", "")
        assert exc.value.code == 1
        assert "no usable credential" in capsys.readouterr().err
        assert "basic_auth" not in _read_config(_isolated_hermes_home).get("dashboard", {})

    def test_empty_password_over_mangled_stored_hash_rejected(self, _isolated_hermes_home):
        """The exact incident shape: mangled hash already stored, then clear the fallback."""
        _write_config(
            _isolated_hermes_home,
            {"dashboard": {"basic_auth": {"username": "op", "password_hash": "scrypt"}}},
        )
        before = (_isolated_hermes_home / "config.yaml").read_text(encoding="utf-8")
        with pytest.raises(SystemExit):
            set_config_value("dashboard.basic_auth.password", "")
        assert (_isolated_hermes_home / "config.yaml").read_text(encoding="utf-8") == before

    def test_empty_password_with_valid_stored_hash_allowed(self, _isolated_hermes_home):
        """Dropping the plaintext once a real hash exists is the documented production setup."""
        _write_config(
            _isolated_hermes_home,
            {"dashboard": {"basic_auth": {"username": "op", "password_hash": VALID_HASH}}},
        )
        set_config_value("dashboard.basic_auth.password", "")
        assert _stored_basic_auth(_isolated_hermes_home)["password"] == ""
        assert _stored_basic_auth(_isolated_hermes_home)["password_hash"] == VALID_HASH

    def test_non_empty_password_is_untouched(self, _isolated_hermes_home):
        """A real plaintext password is a usable credential; the guard must not interfere."""
        set_config_value("dashboard.basic_auth.password", "hunter2")
        assert _stored_basic_auth(_isolated_hermes_home)["password"] == "hunter2"


class TestProviderEncodingAgreement:
    def test_provider_generated_hash_passes_cli_validation(self, _isolated_hermes_home):
        """Guard accepts exactly what the provider's own hash_password() emits."""
        from plugins.dashboard_auth.basic import hash_password

        generated = hash_password("probe-password")
        set_config_value("dashboard.basic_auth.password_hash", generated)
        assert _stored_basic_auth(_isolated_hermes_home)["password_hash"] == generated

    def test_guard_agrees_with_provider_on_a_forced_password_login(self, _isolated_hermes_home):
        """End the loop: a hash that survives the guard must actually authenticate."""
        from plugins.dashboard_auth.basic import _verify_password, hash_password

        generated = hash_password("probe-password")
        set_config_value("dashboard.basic_auth.password_hash", generated)
        stored = _stored_basic_auth(_isolated_hermes_home)["password_hash"]
        assert _verify_password("probe-password", stored) is True
        assert _verify_password("wrong", stored) is False


class TestOtherKeysUnaffected:
    @pytest.mark.parametrize(
        "key,value",
        [
            ("dashboard.basic_auth.username", "operator"),
            ("dashboard.basic_auth.session_ttl_seconds", "3600"),
            ("dashboard.basic_auth.secret", "x" * 32),
            ("dashboard.enabled", "true"),
            ("model.default", "some-model"),
        ],
    )
    def test_unrelated_writes_still_work(self, _isolated_hermes_home, key, value):
        set_config_value(key, value)
        assert _read_config(_isolated_hermes_home), f"{key} did not write"

    def test_shorter_dotted_path_is_not_guarded(self, _isolated_hermes_home):
        """Only the three-segment leaf is ours — a whole-section write stays the section guard's."""
        set_config_value("dashboard.basic_auth.password_hash", VALID_HASH)
        set_config_value("dashboard.basic_auth.username", "operator")
        assert _stored_basic_auth(_isolated_hermes_home)["username"] == "operator"
