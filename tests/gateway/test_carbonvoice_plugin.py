"""Tests for the Carbon Voice platform plugin registration and wiring.

Covers the three review items on the original (native) submission:

1. Registration — ``register(ctx)`` produces a valid ``PlatformEntry`` with
   the hooks the gateway relies on (env enablement, cron delivery,
   standalone sending, auth env vars).
2. Voice-out configuration wiring — ``CARBONVOICE_VOICE_OUT=true`` flows
   env → ``_env_enablement()`` seed → ``PlatformConfig.extra`` →
   ``adapter._voice_out``, and the adapter declares
   ``voice_out_carries_text`` so core suppresses the duplicate text send
   (see ``test_voice_out_carries_text.py`` for the core contract itself).
3. Credential lock — ``connect()`` refuses to start when another gateway
   already holds the PAT lock, and ``disconnect()`` releases it.
"""

import asyncio

import pytest

from gateway.config import PlatformConfig
from gateway.platform_registry import PlatformEntry
from plugins.platforms.carbonvoice import setup as cv_setup
from plugins.platforms.carbonvoice.adapter import CarbonVoiceAdapter


class _FakeCtx:
    """Captures ``register_platform`` kwargs like the plugin manager would."""

    def __init__(self):
        self.kwargs: dict = {}

    def register_platform(self, **kwargs):
        self.kwargs.update(kwargs)


_CV_ENV_VARS = (
    "CARBONVOICE_PAT",
    "CARBONVOICE_VOICE_OUT",
    "CARBONVOICE_BASE_URL",
    "CARBONVOICE_HOME_CHANNEL",
    "CARBONVOICE_HOME_CHANNEL_NAME",
)


@pytest.fixture
def clean_cv_env(monkeypatch):
    for var in _CV_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


# ---------------------------------------------------------------------------
# 1. Registration
# ---------------------------------------------------------------------------

class TestRegistration:

    def _entry(self) -> PlatformEntry:
        ctx = _FakeCtx()
        cv_setup.register(ctx)
        return PlatformEntry(source="plugin", **ctx.kwargs)

    def test_register_builds_valid_platform_entry(self):
        entry = self._entry()
        assert entry.name == "carbonvoice"
        assert entry.label == "Carbon Voice"
        assert "CARBONVOICE_PAT" in entry.required_env

    def test_register_provides_gateway_hooks(self):
        entry = self._entry()
        assert callable(entry.env_enablement_fn)
        assert callable(entry.standalone_sender_fn)
        assert callable(entry.setup_fn)
        assert entry.cron_deliver_env_var == "CARBONVOICE_HOME_CHANNEL"

    def test_register_provides_auth_env_vars(self):
        entry = self._entry()
        assert entry.allowed_users_env == "CARBONVOICE_ALLOWED_USERS"
        assert entry.allow_all_env == "CARBONVOICE_ALLOW_ALL_USERS"

    def test_adapter_factory_builds_adapter(self):
        entry = self._entry()
        adapter = entry.adapter_factory(
            PlatformConfig(enabled=True, token="cv_pat_test")
        )
        assert isinstance(adapter, CarbonVoiceAdapter)


# ---------------------------------------------------------------------------
# 2. Voice-out configuration wiring (env → seed → adapter)
# ---------------------------------------------------------------------------

class TestVoiceOutWiring:

    def test_no_pat_returns_none(self, clean_cv_env):
        assert cv_setup._env_enablement() is None

    def test_voice_out_defaults_false(self, clean_cv_env):
        clean_cv_env.setenv("CARBONVOICE_PAT", "cv_pat_test")
        seed = cv_setup._env_enablement()
        assert seed["voice_out"] is False

    @pytest.mark.parametrize("value", ["true", "1", "yes", "TRUE"])
    def test_voice_out_env_seeds_extra(self, clean_cv_env, value):
        clean_cv_env.setenv("CARBONVOICE_PAT", "cv_pat_test")
        clean_cv_env.setenv("CARBONVOICE_VOICE_OUT", value)
        seed = cv_setup._env_enablement()
        assert seed["voice_out"] is True

    def test_seed_reaches_adapter_flag(self, clean_cv_env):
        """The full chain: env → seed → PlatformConfig.extra → adapter."""
        clean_cv_env.setenv("CARBONVOICE_PAT", "cv_pat_test")
        clean_cv_env.setenv("CARBONVOICE_VOICE_OUT", "true")
        seed = cv_setup._env_enablement()
        adapter = CarbonVoiceAdapter(
            PlatformConfig(enabled=True, token="cv_pat_test", extra=seed)
        )
        assert adapter._voice_out is True

    def test_voice_out_off_by_default_on_adapter(self, clean_cv_env):
        adapter = CarbonVoiceAdapter(
            PlatformConfig(enabled=True, token="cv_pat_test")
        )
        assert adapter._voice_out is False

    def test_adapter_declares_carries_text_contract(self):
        # Core suppresses the follow-up text send only for adapters that
        # opt in — Carbon Voice does (server-side transcript IS the text).
        assert CarbonVoiceAdapter.voice_out_carries_text is True


# ---------------------------------------------------------------------------
# 3. Credential-scoped lock
# ---------------------------------------------------------------------------

class _StubAPI:
    def __init__(self):
        self.opened = False
        self.closed = False

    async def open(self):
        self.opened = True

    async def close(self):
        self.closed = True


class TestCredentialLock:

    def _adapter(self) -> CarbonVoiceAdapter:
        return CarbonVoiceAdapter(
            PlatformConfig(enabled=True, token="cv_pat_locktest")
        )

    def test_connect_bails_when_lock_denied(self, monkeypatch):
        adapter = self._adapter()
        stub = _StubAPI()
        adapter._api = stub
        calls = []
        monkeypatch.setattr(
            adapter,
            "_acquire_platform_lock",
            lambda scope, identity, desc: calls.append((scope, identity)) or False,
        )
        ok = asyncio.run(adapter.connect())
        assert ok is False
        assert calls == [("carbonvoice-pat", "cv_pat_locktest")]
        # Bailed BEFORE opening the API client.
        assert stub.opened is False

    def test_disconnect_releases_lock(self, monkeypatch):
        adapter = self._adapter()
        adapter._api = _StubAPI()

        released = []
        monkeypatch.setattr(
            adapter, "_release_platform_lock", lambda: released.append(True)
        )

        async def _noop():
            return None

        monkeypatch.setattr(adapter._transport, "stop", _noop)
        monkeypatch.setattr(adapter._cursor, "stop", _noop)

        asyncio.run(adapter.disconnect())
        assert released == [True]
