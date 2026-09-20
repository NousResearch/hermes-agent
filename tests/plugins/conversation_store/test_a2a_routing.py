from types import SimpleNamespace

from gateway.config import PlatformConfig
from plugins.platforms.a2a import adapter as a2a
from plugins.platforms.a2a import protocol


def test_forwarded_profile_title_uses_hermes_session_seam(monkeypatch):
    reads = iter(["", "sess-1"])
    monkeypatch.setattr(a2a, "_state_db", lambda *_args, **_kwargs: next(reads))
    monkeypatch.setattr(a2a, "_profile_home", lambda _profile: "/profile")
    monkeypatch.setattr(
        "tools.environments.local.served_profile_child_env",
        lambda **_kwargs: {},
    )

    calls = []

    def fake_run(command, **_kwargs):
        calls.append(list(command))
        stdout = "reply" if command[1] == "chat" else "renamed"
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(a2a.subprocess, "run", fake_run)
    adapter = a2a.A2AAdapter(PlatformConfig(enabled=True, extra={
        "agents": {"dev": {"profile": "dev", "tenant": "dev", "timeout": 5}},
    }))

    reply, state = adapter._forward_to_profile(
        adapter._agents["dev"], "peer", "ctx/unsafe value", "hello")

    assert (reply, state) == ("reply", protocol.STATE_COMPLETED)
    assert calls[0][:2] == ["hermes", "chat"]
    assert calls[1] == [
        "hermes", "sessions", "rename", "sess-1", "a2a-dev-ctx-unsafe-value",
    ]
