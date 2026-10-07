"""Plugin data paths follow the active profile's RABBIT_HOME, including the ContextVar override.

Several plugins carried a ``~/.rabbit`` fallback (guarding an ImportError of ``rabbit_constants``
that cannot happen for a bundled plugin) or resolved the home at import time. Both are wrong on
Windows and under multiplex profile overrides. Every resolver below must land inside the override.
"""
from __future__ import annotations

import pytest

from rabbit_constants import reset_rabbit_home_override, set_rabbit_home_override


def _a2a_conversation(home):
    from plugins.platforms.a2a import protocol
    return protocol._conv_path("peer-x")


def _photon_auth(home):
    from plugins.platforms.photon import auth
    return auth._auth_json_path()


def _mem0_qdrant(home):
    from plugins.memory.mem0._oss_providers import vector_default_config
    return vector_default_config("qdrant")["path"]


def _openviking_log(home):
    import plugins.memory.openviking as ov
    return ov.get_rabbit_home() / ov._OPENVIKING_SERVER_LOG_RELATIVE_PATH


_RESOLVERS = {"a2a": _a2a_conversation, "photon": _photon_auth, "mem0-qdrant": _mem0_qdrant,
              "openviking": _openviking_log}


@pytest.mark.parametrize("name", sorted(_RESOLVERS))
def test_plugin_path_follows_profile_override(name, tmp_path, monkeypatch):
    monkeypatch.setenv("RABBIT_HOME", str(tmp_path / "default"))
    monkeypatch.setenv("HOME", str(tmp_path / "user-home"))
    profile = tmp_path / "profiles" / "b"
    profile.mkdir(parents=True)
    token = set_rabbit_home_override(profile)
    try:
        resolved = str(_RESOLVERS[name](profile))
    finally:
        reset_rabbit_home_override(token)
    assert resolved.startswith(str(profile)), f"{name} resolved {resolved!r} outside the active profile"
