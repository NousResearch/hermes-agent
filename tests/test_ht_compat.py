"""Tests for ht_compat — backward-compatible HT_*/HERMES_* identity aliases."""

import os

import pytest

from ht_compat import (
    mirror_brand_env,
    mirror_brand_headers,
    new_env_name,
    old_env_name,
    read_brand_header,
    resolve_env,
)


class TestEnvNameMapping:
    def test_new_from_old(self):
        assert new_env_name("HERMES_HOME") == "HT_HOME"
        assert new_env_name("HERMES_BACKEND_READY") == "HT_BACKEND_READY"

    def test_old_from_new(self):
        assert old_env_name("HT_HOME") == "HERMES_HOME"
        assert old_env_name("HT_BACKEND_READY") == "HERMES_BACKEND_READY"

    def test_idempotent_on_wrong_prefix(self):
        # Already-new name passed to new_env_name is unchanged.
        assert new_env_name("HT_HOME") == "HT_HOME"
        # Already-old name passed to old_env_name is unchanged.
        assert old_env_name("HERMES_HOME") == "HERMES_HOME"
        # Unrelated names are untouched by either.
        assert new_env_name("PATH") == "PATH"
        assert old_env_name("PATH") == "PATH"


class TestResolveEnv:
    def test_legacy_wins_over_new(self):
        # Legacy-authoritative: HERMES_* wins when both are set, so an
        # explicitly-propagated legacy value beats a stale inherited HT_*.
        env = {"HT_HOME": "/new", "HERMES_HOME": "/old"}
        assert resolve_env("HERMES_HOME", env=env) == "/old"
        assert resolve_env("HT_HOME", env=env) == "/old"

    def test_stale_inherited_ht_does_not_shadow_explicit_hermes(self):
        # The #18594-class hazard in miniature: a subprocess sets HERMES_HOME to
        # its profile dir but inherits a stale HT_HOME from the parent. The
        # explicit legacy value must win.
        child_env = {"HT_HOME": "/root", "HERMES_HOME": "/root/profiles/coder"}
        assert resolve_env("HERMES_HOME", env=child_env) == "/root/profiles/coder"

    def test_falls_back_to_old(self):
        env = {"HERMES_HOME": "/old"}
        assert resolve_env("HT_HOME", env=env) == "/old"
        assert resolve_env("HERMES_HOME", env=env) == "/old"

    def test_default_when_absent(self):
        assert resolve_env("HERMES_HOME", default="/fallback", env={}) == "/fallback"
        assert resolve_env("HERMES_HOME", env={}) is None

    def test_reads_process_environ_by_default(self, monkeypatch):
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setenv("HT_HOME", "/proc-new")
        assert resolve_env("HERMES_HOME", "") == "/proc-new"

    def test_empty_new_falls_through_to_nonempty_old(self):
        env = {"HT_HOME": "", "HERMES_HOME": "/old"}
        assert resolve_env("HERMES_HOME", env=env) == "/old"


class TestMirrorBrandEnv:
    def test_fills_new_from_old(self):
        env = {"HERMES_HOME": "/h", "HERMES_BACKEND_READY": "1"}
        mirror_brand_env(env)
        assert env["HT_HOME"] == "/h"
        assert env["HT_BACKEND_READY"] == "1"
        # Legacy names preserved.
        assert env["HERMES_HOME"] == "/h"

    def test_fills_old_from_new(self):
        env = {"HT_HOME": "/h"}
        mirror_brand_env(env)
        assert env["HERMES_HOME"] == "/h"
        assert env["HT_HOME"] == "/h"

    def test_non_clobbering_when_both_present(self):
        env = {"HT_HOME": "/new", "HERMES_HOME": "/old"}
        mirror_brand_env(env)
        assert env["HT_HOME"] == "/new"
        assert env["HERMES_HOME"] == "/old"

    def test_idempotent(self):
        env = {"HERMES_HOME": "/h"}
        mirror_brand_env(env)
        snapshot = dict(env)
        mirror_brand_env(env)
        assert env == snapshot

    def test_ignores_unrelated_vars(self):
        env = {"PATH": "/usr/bin", "HERMES_HOME": "/h"}
        mirror_brand_env(env)
        assert "HT_PATH" not in env
        assert env["PATH"] == "/usr/bin"

    def test_defaults_to_os_environ(self, monkeypatch):
        monkeypatch.delenv("HT_HOME", raising=False)
        monkeypatch.setenv("HERMES_HOME", "/proc")
        try:
            mirror_brand_env()
            assert os.environ["HT_HOME"] == "/proc"
        finally:
            # mirror_brand_env mutates os.environ directly, outside
            # monkeypatch's tracking — remove the mirror we created so it
            # can't leak into and pollute later tests.
            os.environ.pop("HT_HOME", None)


class TestMirrorBrandHeaders:
    def test_adds_ht_mirror(self):
        headers = {"X-Hermes-Session-Id": "abc"}
        mirror_brand_headers(headers)
        assert headers["X-HT-Session-Id"] == "abc"
        # Legacy header preserved for existing clients.
        assert headers["X-Hermes-Session-Id"] == "abc"

    def test_non_clobbering(self):
        headers = {"X-Hermes-Session-Id": "old", "X-HT-Session-Id": "new"}
        mirror_brand_headers(headers)
        assert headers["X-HT-Session-Id"] == "new"

    def test_case_insensitive_existing(self):
        # An existing HT header in different casing must not be duplicated.
        headers = {"X-Hermes-Session-Id": "v", "x-ht-session-id": "kept"}
        mirror_brand_headers(headers)
        assert headers["x-ht-session-id"] == "kept"
        assert "X-HT-Session-Id" not in headers

    def test_ignores_unrelated_headers(self):
        headers = {"Content-Type": "application/json"}
        mirror_brand_headers(headers)
        assert headers == {"Content-Type": "application/json"}

    def test_returns_headers_for_chaining(self):
        headers = {"X-Hermes-Completed": "false"}
        assert mirror_brand_headers(headers) is headers


class TestReadBrandHeader:
    def test_new_wins(self):
        headers = {"X-HT-Session-Id": "new", "X-Hermes-Session-Id": "old"}
        assert read_brand_header(headers, "Session-Id") == "new"

    def test_falls_back_to_legacy(self):
        headers = {"X-Hermes-Session-Id": "old"}
        assert read_brand_header(headers, "Session-Id") == "old"

    def test_default_when_absent(self):
        assert read_brand_header({}, "Session-Id") == ""
        assert read_brand_header({}, "Session-Id", "d") == "d"

    def test_case_insensitive(self):
        headers = {"x-hermes-session-key": "k"}
        assert read_brand_header(headers, "Session-Key") == "k"


class TestCaseInsensitiveMultiDict:
    """The real request.headers is a case-insensitive multidict; emulate it."""

    class _CIDict(dict):
        def get(self, key, default=None):
            for k, v in self.items():
                if k.lower() == key.lower():
                    return v
            return default

    def test_read_via_get_case_insensitive(self):
        headers = self._CIDict({"X-Hermes-Session-Id": "sid"})
        assert read_brand_header(headers, "Session-Id") == "sid"
        # Also resolvable via the new spelling against the same source.
        headers2 = self._CIDict({"x-ht-session-id": "sid2"})
        assert read_brand_header(headers2, "Session-Id") == "sid2"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
