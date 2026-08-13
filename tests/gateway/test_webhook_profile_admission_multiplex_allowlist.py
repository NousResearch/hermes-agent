"""Regression coverage for webhook multiplex profile allowlist propagation."""

from types import SimpleNamespace

from gateway.platforms.webhook import WebhookAdapter


class _Request:
    match_info = {"profile": "worker"}


def test_profile_admission_passes_configured_multiplex_allowlist(monkeypatch):
    """Profile admission must use the same selective set as gateway startup."""
    calls = []

    def fake_profiles_to_serve(*, multiplex, profile_allowlist=None):
        calls.append((multiplex, profile_allowlist))
        return [("default", "/profiles/default"), ("worker", "/profiles/worker")]

    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        fake_profiles_to_serve,
    )

    adapter = WebhookAdapter.__new__(WebhookAdapter)
    adapter.gateway_runner = SimpleNamespace(
        config=SimpleNamespace(
            multiplex_profiles=True,
            multiplex_profile_allowlist=["worker"],
        )
    )

    assert adapter._resolve_request_profile(_Request()) == "worker"
    assert calls == [(True, ["worker"])]
