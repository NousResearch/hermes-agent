"""#139398: an E2EE-off Matrix gateway must not require ``mautrix[encryption]``.

The ``matrix`` extra pins ``mautrix[encryption]`` (python-olm → deprecated libolm, which needs cmake
and a C++ toolchain to build), and its anchors include asyncpg/aiosqlite (the E2EE crypto store) and
aiohttp_socks (SOCKS proxies only). With ``MATRIX_E2EE_MODE=off`` none of those are used, so a plain
``mautrix`` install must pass the passive ``check_fn`` and must not trigger the lazy installer.
"""
from unittest.mock import patch

import pytest

import plugins.platforms.matrix.adapter as matrix_mod

_E2EE_ONLY = ("asyncpg", "aiosqlite", "aiohttp_socks")


@pytest.fixture
def matrix_env(monkeypatch):
    monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_test")
    monkeypatch.setenv("MATRIX_HOMESERVER", "https://matrix.example.org")
    monkeypatch.delenv("MATRIX_ENCRYPTION", raising=False)
    monkeypatch.setenv("MATRIX_E2EE_MODE", "off")
    return monkeypatch


def test_e2ee_off_plain_mautrix_skips_encryption_install(matrix_env):
    def _no_install(*_args, **_kwargs):
        pytest.fail("E2EE-off gateway must not install mautrix[encryption]")

    with patch("pm.extras.missing", return_value=_E2EE_ONLY), \
         patch("pm.extras.ensure_and_bind", side_effect=_no_install):
        assert matrix_mod.check_matrix_requirements() is True


def test_e2ee_off_still_installs_when_mautrix_missing(matrix_env):
    calls = []
    with patch("pm.extras.missing", return_value=("mautrix", "asyncpg")), \
         patch("pm.extras.ensure_and_bind", side_effect=lambda extra, *_a: calls.append(extra) or True):
        assert matrix_mod.ensure_matrix_deps() is True
    assert calls == ["matrix"]


@pytest.mark.parametrize("mode", ["optional", "required"])
def test_e2ee_on_still_installs_crypto_deps(matrix_env, mode):
    matrix_env.setenv("MATRIX_E2EE_MODE", mode)
    calls = []
    with patch("pm.extras.missing", return_value=("asyncpg",)), \
         patch("pm.extras.ensure_and_bind", side_effect=lambda extra, *_a: calls.append(extra) or True), \
         patch.object(matrix_mod, "_check_e2ee_deps", return_value=True):
        assert matrix_mod.ensure_matrix_deps() is True
    assert calls == ["matrix"]


@pytest.mark.parametrize("mode, expected", [("off", True), ("optional", False), ("required", False)])
def test_passive_check_accepts_plain_mautrix_only_when_e2ee_off(matrix_env, mode, expected):
    matrix_env.setenv("MATRIX_E2EE_MODE", mode)
    with patch("pm.available", return_value=False), patch("pm.extras.missing", return_value=_E2EE_ONLY):
        assert matrix_mod.matrix_deps_present() is expected


def test_passive_check_e2ee_off_still_requires_mautrix(matrix_env):
    with patch("pm.available", return_value=False), patch("pm.extras.missing", return_value=("mautrix",)):
        assert matrix_mod.matrix_deps_present() is False


def test_legacy_encryption_flag_still_requires_crypto_deps(matrix_env):
    matrix_env.delenv("MATRIX_E2EE_MODE")
    matrix_env.setenv("MATRIX_ENCRYPTION", "true")
    with patch("pm.available", return_value=False), patch("pm.extras.missing", return_value=_E2EE_ONLY):
        assert matrix_mod.matrix_deps_present() is False
