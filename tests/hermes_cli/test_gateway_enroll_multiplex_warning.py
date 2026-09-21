"""Gateway enroll: secondary-multiplex-profile warning for relay routing stamps.

``hermes gateway enroll --connector-url/--wake-url`` persists GATEWAY_RELAY_URL /
GATEWAY_RELAY_WAKE_URL into the active profile's .env. Those are process-global
deployment stamps (agent/secret_scope.py): a multiplexed gateway reads them from
the process environment only, never from a secondary profile's isolated secret
scope — so an enroll run from a secondary profile must WARN that the written
URLs will not activate from there (sol-reviewer round-4: the first cut read
``multiplex_profiles`` from the SECONDARY profile's config.yaml, where it never
lives, and the warning silently never fired in the real topology).

The topology decision is owned by the DEFAULT root: path relationship to
``<default_root>/profiles/`` plus the default root's config.yaml (or the
GATEWAY_MULTIPLEX_PROFILES env override).
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

import hermes_constants
from hermes_cli.gateway_enroll import _warn_if_secondary_multiplex_profile


@pytest.fixture()
def topology(tmp_path, monkeypatch):
    """Standard multiplex layout: default root + one secondary profile."""
    root = tmp_path / "root"
    (root / "profiles" / "alice").mkdir(parents=True)
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: root)
    monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
    return root


def _run() -> tuple[bool, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        fired = _warn_if_secondary_multiplex_profile()
    return fired, buf.getvalue()


def test_fires_for_secondary_when_default_root_has_multiplex_on(topology, monkeypatch):
    """The regression case: flag in the DEFAULT root's config.yaml, absent from
    the secondary profile's own config — the warning must still fire."""
    (topology / "config.yaml").write_text(
        "gateway:\n  multiplex_profiles: true\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(topology / "profiles" / "alice"))

    fired, out = _run()

    assert fired is True
    assert "SECONDARY profile" in out


def test_silent_when_multiplex_off_in_default_root(topology, monkeypatch):
    (topology / "config.yaml").write_text(
        "gateway:\n  multiplex_profiles: false\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(topology / "profiles" / "alice"))

    fired, out = _run()

    assert fired is False
    assert out == ""


def test_silent_for_default_profile_even_with_multiplex_on(topology, monkeypatch):
    (topology / "config.yaml").write_text(
        "gateway:\n  multiplex_profiles: true\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(topology))

    fired, _ = _run()

    assert fired is False


def test_env_override_forces_multiplex_on_without_config_flag(topology, monkeypatch):
    (topology / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(topology / "profiles" / "alice"))
    monkeypatch.setenv("GATEWAY_MULTIPLEX_PROFILES", "true")

    fired, _ = _run()

    assert fired is True


def test_env_override_off_wins_over_config_flag(topology, monkeypatch):
    (topology / "config.yaml").write_text(
        "gateway:\n  multiplex_profiles: true\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(topology / "profiles" / "alice"))
    monkeypatch.setenv("GATEWAY_MULTIPLEX_PROFILES", "false")

    fired, _ = _run()

    assert fired is False


def test_silent_for_unrelated_dir_named_profiles(topology, tmp_path, monkeypatch):
    """A directory whose parent happens to be named 'profiles' but is OUTSIDE
    the default root is not a secondary profile (sol-reviewer round-4 MINOR:
    name-based heuristics false-positive on unrelated layouts)."""
    (topology / "config.yaml").write_text(
        "gateway:\n  multiplex_profiles: true\n", encoding="utf-8"
    )
    other = tmp_path / "elsewhere" / "profiles" / "x"
    other.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(other))

    fired, _ = _run()

    assert fired is False


class _Resp:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> bool:
        return False


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(b"not json", id="non-json"),
        pytest.param(b"\xff\xfe\x00corrupt", id="non-utf8"),
    ],
)
def test_post_enroll_non_json_200_raises_runtime_error(monkeypatch, body):
    """A 200 with a non-JSON or non-UTF-8 body must surface as the documented
    RuntimeError, not a raw JSONDecodeError/UnicodeDecodeError."""
    import urllib.request
    from hermes_cli.gateway_enroll import _post_enroll

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp(body))

    with pytest.raises(RuntimeError, match="non-JSON"):
        _post_enroll(
            connector_base_url="https://connector.example",
            access_token="tok",
            enrollment_token="et",
            gateway_id="gw-1",
        )


def test_post_enroll_read_failure_raises_runtime_error(monkeypatch):
    """A mid-body socket reset on the enroll POST is a transport failure ->
    RuntimeError, not a bare OSError traceback."""
    import urllib.request
    from hermes_cli.gateway_enroll import _post_enroll

    class _ResetResp:
        def read(self):
            raise ConnectionResetError("reset")

        def __enter__(self):
            return self

        def __exit__(self, *_exc) -> bool:
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _ResetResp())

    with pytest.raises(RuntimeError, match="transport failure"):
        _post_enroll(
            connector_base_url="https://connector.example",
            access_token="tok",
            enrollment_token="et",
            gateway_id="gw-1",
        )
