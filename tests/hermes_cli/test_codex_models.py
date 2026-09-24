import base64
import json
from unittest.mock import patch

from hermes_cli.codex_models import (
    _FORWARD_COMPAT_TEMPLATE_MODELS,
    DEFAULT_CODEX_MODELS,
    get_codex_model_ids,
)


def _codex_jwt(subject: str) -> str:
    """JWT-shaped test stand-in: the live catalog path gates on the token parsing as a JWT
    (a custom base's gateway key must stay off chatgpt.com, #121486)."""
    enc = lambda raw: base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
    header = enc(b'{"alg":"RS256"}')
    payload = enc(json.dumps({"sub": subject}).encode())
    return f"{header}.{payload}.test-signature"


# dummy fixture value (not a real credential): a gateway pool key is opaque, never a JWT
_GATEWAY_POOL_KEY = "dummy-gateway-pool-key"


def _pro_slugs(model_ids):
    return [m for m in model_ids if m.removesuffix("-900k").endswith("-pro")]


def test_codex_catalog_never_offers_chatgpt_rejected_pro_slugs(monkeypatch, tmp_path):
    """The ChatGPT Codex OAuth backend 400s every ``-pro`` slug (#52492), so
    neither the offline fallback nor forward-compat synthesis over a live
    catalog may offer one, while the fallback still keeps every curated model."""
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))  # no config.toml default, no cache
    offline = get_codex_model_ids()
    assert set(DEFAULT_CODEX_MODELS) <= set(offline)
    assert _pro_slugs(offline) == []

    # Live discovery returning only template slugs fires every forward-compat
    # synthesis rule; none of what it adds may be -pro.
    templates = list(dict.fromkeys(t for _, ts in _FORWARD_COMPAT_TEMPLATE_MODELS for t in ts))
    monkeypatch.setattr(
        "hermes_cli.codex_models._fetch_models_from_api", lambda access_token, base_url="": templates
    )
    live = get_codex_model_ids(access_token="codex-access-token")
    assert {synthetic for synthetic, _ in _FORWARD_COMPAT_TEMPLATE_MODELS} <= set(live)
    assert _pro_slugs(live) == []



def test_picker_synthesizes_900k_variants_for_verified_slugs():
    """Every live-verified large-context slug gets an explicit ``-900k``
    picker variant directly after its base entry; slugs that genuinely
    enforce 272K (gpt-5.5, gpt-5.4-mini) never get one. Base slugs stay
    in the list as the cheaper 272K default."""
    model_ids = get_codex_model_ids()  # offline curated path

    for base in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.4"):
        assert base in model_ids
        assert f"{base}-900k" in model_ids
        assert model_ids.index(f"{base}-900k") == model_ids.index(base) + 1

    assert "gpt-5.5-900k" not in model_ids
    assert "gpt-5.4-mini-900k" not in model_ids
    assert "gpt-5.3-codex-900k" not in model_ids


def test_picker_never_synthesizes_900k_for_pro_or_unknown_slugs():
    """Eligibility is an exact predicate, not a family-prefix match:
    ``-pro`` slugs are not routable on Codex OAuth (backend 400s them) and
    unknown future descendants were never probed — neither may gain a
    synthetic ``-900k`` entry (#92797 review)."""
    from hermes_cli.codex_models import _finalize_codex_models

    out = _finalize_codex_models(["gpt-5.6-sol-pro", "gpt-5.6-nova"])
    assert "gpt-5.6-sol-pro-900k" not in out
    assert "gpt-5.6-nova-900k" not in out










def test_fetch_from_api_keeps_supported_in_api_false_models(monkeypatch):
    """Regression: gpt-5.3-codex-spark is returned by the live Codex backend
    with ``supported_in_api: false`` because it isn't in the public OpenAI
    API. The Codex CLI / OAuth route still serves it for ChatGPT Pro
    accounts, so we must not drop it on that flag. visibility=hidden is
    the separate signal that *should* still filter entries out.
    """
    import sys
    from hermes_cli import codex_models

    class _FakeResp:
        status_code = 200

        def json(self):
            return {
                "models": [
                    {"slug": "gpt-5.5", "priority": 0, "supported_in_api": True},
                    {"slug": "gpt-5.3-codex-spark", "priority": 7, "supported_in_api": False},
                    {"slug": "gpt-5-internal", "priority": 99, "visibility": "hidden"},
                ]
            }

    class _FakeHttpx:
        @staticmethod
        def get(url, headers=None, timeout=None):
            return _FakeResp()

    monkeypatch.setitem(sys.modules, "httpx", _FakeHttpx)

    models = codex_models._fetch_models_from_api(access_token=_codex_jwt("acct"))

    assert "gpt-5.5" in models
    assert "gpt-5.3-codex-spark" in models
    assert "gpt-5-internal" not in models


def test_astra_requires_live_codex_account_discovery(monkeypatch, tmp_path):
    """Cached/configured Astra names must not manufacture current OAuth entitlement."""
    from hermes_cli import codex_models

    (tmp_path / "config.toml").write_text('model = "gpt-6-astra"\n', encoding="utf-8")
    (tmp_path / "models_cache.json").write_text(
        json.dumps({"models": [
            {"slug": "gpt-6-astra", "priority": 0},
            {"slug": "openai/gpt-6-astra", "priority": 1},
            {"slug": "gpt-6-astra-900k", "priority": 2},
            {"slug": "openai/gpt-6-astra-900k", "priority": 3},
        ]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    monkeypatch.setattr(codex_models, "_fetch_models_from_api", lambda _token, base_url="": [])

    assert "gpt-6-astra" not in get_codex_model_ids(access_token="stale-token")
    assert "openai/gpt-6-astra" not in get_codex_model_ids(access_token="stale-token")
    assert "gpt-6-astra-900k" not in get_codex_model_ids(access_token="stale-token")
    assert "openai/gpt-6-astra-900k" not in get_codex_model_ids(access_token="stale-token")

    monkeypatch.setattr(
        codex_models,
        "_fetch_models_from_api",
        lambda _token, base_url="": codex_models._finalize_codex_models(["gpt-6-astra"]),
    )
    entitled = get_codex_model_ids(access_token="entitled-token")
    assert entitled[entitled.index("gpt-6-astra") + 1] == "gpt-6-astra-900k"






def test_model_command_prompts_to_reuse_or_reauthenticate_codex_session(monkeypatch, capsys):
    from hermes_cli.model_setup_flows import _model_flow_openai_codex

    captured = {"login_calls": 0}
    choices = iter(["2"])

    monkeypatch.setattr("builtins.input", lambda prompt="": next(choices))
    monkeypatch.setattr(
        "hermes_cli.auth.get_codex_auth_status",
        lambda: {"logged_in": True, "source": "hermes-auth-store"},
    )
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_codex_runtime_credentials",
        lambda *args, **kwargs: {"api_key": "fresh-codex-token"},
    )

    def _fake_login(*args, force_new_login=False, **kwargs):
        captured["login_calls"] += 1
        captured["force_new_login"] = force_new_login

    monkeypatch.setattr("hermes_cli.auth._login_openai_codex", _fake_login)
    monkeypatch.setattr(
        "hermes_cli.codex_models.get_codex_model_ids",
        lambda access_token=None, base_url="": ["gpt-5.4", "gpt-5.5"],
    )
    monkeypatch.setattr(
        "hermes_cli.auth._prompt_model_selection",
        lambda model_ids, current_model="", **_kwargs: None,
    )

    _model_flow_openai_codex({}, current_model="gpt-5.4")

    assert captured["login_calls"] == 1
    assert captured["force_new_login"] is True


# ── Tests for _normalize_model_for_provider ──────────────────────────


def _make_cli(model="anthropic/claude-opus-4.6", **kwargs):
    """Create a HermesCLI with minimal mocking."""
    import cli as _cli_mod
    from cli import HermesCLI

    _clean_config = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "auto",
        },
        "display": {"compact": False, "tool_progress": "all", "resume_display": "full"},
        "agent": {},
        "terminal": {"env_type": "local"},
    }
    clean_env = {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}
    with (
        patch("cli.get_tool_definitions", return_value=[]),
        patch.dict("os.environ", clean_env, clear=False),
        patch.dict(_cli_mod.__dict__, {"CLI_CONFIG": _clean_config}),
    ):
        cli = HermesCLI(model=model, **kwargs)
    return cli


class TestNormalizeModelForProvider:
    """_normalize_model_for_provider() trusts user-selected models.

    Only two things happen:
    1. Provider prefixes are stripped (API needs bare slugs)
    2. The *untouched default* model is swapped for a Codex model
    Everything else passes through — the API is the judge.
    """

    def test_non_codex_provider_is_noop(self):
        cli = _make_cli(model="gpt-5.4")
        changed = cli._normalize_model_for_provider("openrouter")
        assert changed is False
        assert cli.model == "gpt-5.4"


    def test_opencode_zen_claude_sets_messages_mode(self):
        cli = _make_cli(model="opencode-zen/claude-sonnet-4-6")
        cli.api_mode = "chat_completions"
        changed = cli._normalize_model_for_provider("opencode-zen")
        assert changed is True
        assert cli.model == "claude-sonnet-4-6"
        assert cli.api_mode == "anthropic_messages"

    def test_default_model_replaced(self):
        """No model configured (empty default) gets swapped for codex."""
        import cli as _cli_mod
        _clean_config = {
            "model": {
                "default": "",
                "base_url": "",
                "provider": "auto",
            },
            "display": {"compact": False, "tool_progress": "all", "resume_display": "full"},
            "agent": {},
            "terminal": {"env_type": "local"},
        }
        # Don't pass model= so _model_is_default is True
        with (
            patch("cli.get_tool_definitions", return_value=[]),
            patch.dict("os.environ", {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}, clear=False),
            patch.dict(_cli_mod.__dict__, {"CLI_CONFIG": _clean_config}),
        ):
            from cli import HermesCLI
            cli = HermesCLI()

        assert cli._model_is_default is True
        with patch(
            "hermes_cli.codex_models.get_codex_model_ids",
            return_value=["gpt-5.5", "gpt-5.4"],
        ):
            changed = cli._normalize_model_for_provider("openai-codex")
        assert changed is True
        # Uses first from available list
        assert cli.model == "gpt-5.5"


def _gated_codex_catalog(seen_urls):
    """Backend shape since the GPT-6 Sol/Luna rollout (#119412): the ``0.0.0`` sentinel answers a
    frozen legacy list, any newer client version answers the full account catalog."""
    from urllib.parse import parse_qs, urlparse

    class _FakeResp:
        status_code = 200

        def __init__(self, url):
            self.version = parse_qs(urlparse(url).query)["client_version"][0]

        def json(self):
            models = [{"slug": "gpt-5.6-sol", "visibility": "list", "context_window": 272000}]
            if self.version != "0.0.0":
                models.append({"slug": "gpt-6-sol", "visibility": "list", "context_window": 272000})
            return {"models": models}

    def get(url, headers=None, timeout=None, verify=None):
        seen_urls.append(url)
        return _FakeResp(url)

    return get


def test_catalog_requests_ask_as_the_newest_client(monkeypatch):
    """Both catalog request sites (picker + context probe) send a client version newer than every
    ``minimal_client_version`` on the first try, so account-visible GPT-6 Sol/Luna are not hidden
    behind the frozen ``0.0.0`` legacy list (#119412)."""
    import sys
    from urllib.parse import parse_qs, urlparse

    from agent import model_metadata
    from hermes_cli import codex_models

    seen_urls = []
    get = _gated_codex_catalog(seen_urls)
    monkeypatch.setitem(sys.modules, "httpx", type("_FakeHttpx", (), {"get": staticmethod(get)}))
    assert "gpt-6-sol" in codex_models._fetch_models_from_api(access_token=_codex_jwt("acct"))
    monkeypatch.setattr(model_metadata, "requests", type("_FakeRequests", (), {"get": staticmethod(get)}))
    monkeypatch.setattr(model_metadata, "_ensure_requests", lambda: None)
    monkeypatch.setattr(model_metadata, "_codex_oauth_context_cache", {})
    live, fresh = model_metadata._fetch_codex_oauth_context_lengths_with_source(_codex_jwt("acct"))
    assert fresh and "gpt-6-sol" in live

    assert len(seen_urls) == 2  # one request per site: the newest-client answer was non-empty
    for url in seen_urls:
        parsed = urlparse(url)
        assert parsed.netloc == "chatgpt.com" and parsed.path == "/backend-api/codex/models"
        assert parse_qs(parsed.query)["client_version"] != ["0.0.0"]


def test_gateway_key_is_never_sent_to_the_direct_catalog(monkeypatch):
    """A ``HERMES_CODEX_BASE_URL`` gateway's key is not a ChatGPT token; the picker must not
    send it to chatgpt.com (a service it does not belong to, #121486). Discovery declines
    and falls back to the offline sources instead of firing a doomed request."""
    import sys
    from hermes_cli import codex_models

    calls = {"n": 0}

    def _count(_url, **_kwargs):
        calls["n"] += 1
        raise RuntimeError("a gateway key must not reach any catalog endpoint")

    monkeypatch.setitem(sys.modules, "httpx", type("_FakeHttpx", (), {"get": staticmethod(_count)}))
    assert codex_models._fetch_models_from_api(access_token=_GATEWAY_POOL_KEY) == []
    assert calls["n"] == 0


def test_picker_catalog_honours_the_custom_codex_base(monkeypatch):
    """When the routing decision resolves a custom Codex base (``codex_route_base_url``), the
    picker's live discovery probes the gateway's own ``/models``, not the hard-coded chatgpt.com
    host — the credential and its destination travel as one pair (#121486)."""
    import sys
    from urllib.parse import urlparse

    from hermes_cli import codex_models

    seen_urls = []
    get = _gated_codex_catalog(seen_urls)
    monkeypatch.setitem(sys.modules, "httpx", type("_FakeHttpx", (), {"get": staticmethod(get)}))

    assert "gpt-6-sol" in codex_models._fetch_models_from_api(
        access_token=_codex_jwt("acct"), base_url="https://codex-gw.example/backend-api/codex")
    assert seen_urls, "discovery must still run for a JWT credential"
    for url in seen_urls:
        parsed = urlparse(url)
        assert parsed.netloc == "codex-gw.example" and parsed.path == "/backend-api/codex/models"


def test_catalog_falls_back_to_the_ungated_sentinel_when_newest_client_is_rejected():
    """If the backend goes back to rejecting out-of-sequence versions (empty list or non-200), the
    ``0.0.0`` sentinel is tried next; a sentinel that is itself empty yields no entries."""
    from agent.model_metadata import CODEX_UNGATED_CLIENT_VERSION, fetch_codex_catalog_entries

    class _Resp:
        def __init__(self, status, models):
            self.status_code, self._models = status, models

        def json(self):
            return {"models": self._models}

    def rejecting(url):
        return _Resp(200, [{"slug": "gpt-5.5"}]) if url.endswith(CODEX_UNGATED_CLIENT_VERSION) else _Resp(400, [])

    entries, status = fetch_codex_catalog_entries(rejecting)
    assert [e["slug"] for e in entries] == ["gpt-5.5"] and status == 200
    assert fetch_codex_catalog_entries(lambda url: _Resp(200, [])) == ([], 200)


def test_route_authority_prefers_model_base_url_when_env_unset(monkeypatch):
    """#121486 adversarial case (a): the effective Codex route is expressed through
    model.base_url with the env unset — the resolved authority is that base, so a credential
    carried with it can never be sent to chatgpt.com."""
    from hermes_cli.auth_codex import codex_route_base_url

    monkeypatch.setattr("agent.secret_scope.get_secret_str", lambda name, default="": "")
    resolved = codex_route_base_url({"provider": "openai-codex", "base_url": "https://gw.example/api"})
    assert resolved == "https://gw.example/api"


def test_route_authority_env_override_wins_over_model_base_url(monkeypatch):
    """#121486 adversarial case (b): env/base disagreement resolves exactly like the runtime
    (``_pool_entry_mode_and_url``) — the profile-scoped override wins for every reader, so the
    catalog probes what the runtime actually talks to."""
    from hermes_cli.auth_codex import codex_route_base_url

    monkeypatch.setattr(
        "agent.secret_scope.get_secret_str",
        lambda name, default="": "https://override.example/api" if name == "HERMES_CODEX_BASE_URL" else default,
    )
    assert codex_route_base_url({"provider": "openai-codex", "base_url": "https://gw.example/api"}) == \
        "https://override.example/api"
    assert codex_route_base_url() == "https://override.example/api"


def test_route_authority_ignores_a_foreign_providers_base_url(monkeypatch):
    """A stale model.base_url left over from another provider must not leak into the Codex route
    (``_config_base_url_for_provider`` only honours the matching provider)."""
    from hermes_cli.auth_codex import codex_route_base_url

    monkeypatch.setattr("agent.secret_scope.get_secret_str", lambda name, default="": "")
    resolved = codex_route_base_url({"provider": "openai", "base_url": "https://api.openai.com/v1"})
    assert resolved == "https://chatgpt.com/backend-api/codex"
