"""Phase 1: HTTP-inbound /p/<profile>/ routing for the webhook adapter."""
import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, build_session_key


class TestSessionSourceProfileField:
    def test_profile_roundtrips(self):
        s = SessionSource(
            platform=Platform.WEBHOOK if hasattr(Platform, "WEBHOOK") else Platform.TELEGRAM,
            chat_id="c1",
            chat_type="webhook",
            profile="coder",
        )
        restored = SessionSource.from_dict(s.to_dict())
        assert restored.profile == "coder"


class TestWebhookProfileResolution:
    """_resolve_request_profile validates the /p/<profile>/ prefix."""

    def _adapter(
        self,
        multiplex: bool,
        served=("default", "coder"),
        allowlist=None,
    ):
        from gateway.platforms.webhook import WebhookAdapter, _PROFILE_REJECTED

        class _FakeReq:
            def __init__(self, profile):
                self.match_info = {"profile": profile} if profile is not None else {}

        cfg = GatewayConfig(
            multiplex_profiles=multiplex,
            multiplex_profile_allowlist=allowlist,
        )

        class _Runner:
            config = cfg

        # Construct minimally; we only call _resolve_request_profile.
        adapter = WebhookAdapter.__new__(WebhookAdapter)
        adapter.gateway_runner = _Runner()
        return adapter, _FakeReq, _PROFILE_REJECTED, served

    def test_no_prefix_returns_none(self):
        adapter, Req, _REJ, _ = self._adapter(multiplex=True)
        assert adapter._resolve_request_profile(Req(None)) is None

    def test_unserved_prefix_is_rejected(self, monkeypatch):
        adapter, Req, rejected, served = self._adapter(
            multiplex=True,
            served=("default", "worker"),
            allowlist=["worker"],
        )
        monkeypatch.setattr(
            "hermes_cli.profiles.profiles_to_serve",
            lambda multiplex, profile_allowlist=None: [
                (name, f"/profiles/{name}") for name in served
            ],
        )

        assert adapter._resolve_request_profile(Req("worker")) == "worker"
        assert adapter._resolve_request_profile(Req("restricted")) is rejected


