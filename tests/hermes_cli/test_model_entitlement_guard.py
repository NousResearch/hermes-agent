"""Entitlement gate for a model pick that would become the profile default (t_27cf7a7b).

The 2026-09-28 incident: ``custom:zai/glm-5.3-flashx`` (base_url = the Z.AI coding
endpoint) was persisted as ``profiles/hephaestus/config.yaml`` ``model.default``
although the plan refuses it with ``429 code=1311`` — every session afterwards paid
a 429 and then a fallback hop. Nothing on the persist path probed the pick.

These tests pin the gate:
  * only a PROVEN refusal blocks (a throttle, a 5xx, a transport error, a local
    endpoint or a missing credential/endpoint never do);
  * the refusal is printed and the config write does not happen — on both persist
    paths (``_persist_model`` for the setup flows, ``persist_model_selection`` for
    ``/model --global``);
  * ``HERMES_ALLOW_UNENTITLED_PICK=1`` restores warn-and-persist.

All file I/O is against ``tmp_path`` (tests/home_io_guard.py forbids the real home).
"""

from __future__ import annotations

import urllib.error

import pytest

from hermes_cli import model_entitlement_guard as guard

# Live bodies, recorded from the real endpoints on 2026-09-28.
PLAN_REFUSAL_BODY = {"error": {"code": "1311", "message": (
    "Your current subscription plan does not yet include access to GLM-5.3-FlashX")}}
THROTTLE_BODY = {"error": {"message": "Rate limit exceeded. Please retry after 20s.",
                           "type": "rate_limit_error"}}
RETIRED_FREE_BODY = {"error": {"message": (
    "This model is unavailable for free. The paid version is available now - use this "
    "slug instead: upstage/solar-pro4"), "code": 404}}


def _probe_returning(monkeypatch, status: int, body: dict):
    monkeypatch.setattr(guard, "_post_chat_completion",
                        lambda base_url, api_key, model, timeout, **_kw: (status, body))


def _probe_raising(monkeypatch, exc):
    def boom(base_url, api_key, model, timeout, **_kw):
        raise exc
    monkeypatch.setattr(guard, "_post_chat_completion", boom)


class TestRefusalProbe:
    def test_entitled_pick_is_allowed(self, monkeypatch):
        _probe_returning(monkeypatch, 200, {"choices": [{"message": {"content": "pong"}}]})
        assert guard.refusal_for_pick(
            "glm-5.3", provider="zai", base_url="https://api.z.ai/api/coding/paas/v4",
            api_key="k") is None

    def test_plan_refusal_is_reported(self, monkeypatch):
        _probe_returning(monkeypatch, 429, PLAN_REFUSAL_BODY)
        reason = guard.refusal_for_pick(
            "glm-5.3-flashx", provider="zai",
            base_url="https://api.z.ai/api/coding/paas/v4", api_key="k")
        assert reason and "include access" in reason

    def test_retired_free_slug_is_reported(self, monkeypatch):
        _probe_returning(monkeypatch, 404, RETIRED_FREE_BODY)
        assert guard.refusal_for_pick(
            "upstage/solar-pro4:free", provider="openrouter",
            base_url="https://openrouter.ai/api/v1", api_key="k")

    def test_throttle_is_not_a_refusal(self, monkeypatch):
        """A 429 window means "busy", not "not entitled": never block on it."""
        _probe_returning(monkeypatch, 429, THROTTLE_BODY)
        assert guard.refusal_for_pick(
            "glm-5.3", provider="zai", base_url="https://api.z.ai/api/coding/paas/v4",
            api_key="k") is None

    def test_server_error_is_not_a_refusal(self, monkeypatch):
        _probe_returning(monkeypatch, 503, {"error": {"message": "Service temporarily overloaded"}})
        assert guard.refusal_for_pick(
            "glm-5.3", provider="zai", base_url="https://api.z.ai/api/coding/paas/v4",
            api_key="k") is None

    def test_transport_error_is_not_a_refusal(self, monkeypatch):
        """A flaky network must never block a pick."""
        _probe_raising(monkeypatch, urllib.error.URLError("no route to host"))
        assert guard.refusal_for_pick(
            "glm-5.3", provider="zai", base_url="https://api.z.ai/api/coding/paas/v4",
            api_key="k") is None

    def test_local_endpoint_is_not_probed(self, monkeypatch):
        def forbidden(*a, **kw):  # pragma: no cover - must not be reached
            raise AssertionError("a local endpoint must not be probed")
        monkeypatch.setattr(guard, "_post_chat_completion", forbidden)
        assert guard.refusal_for_pick(
            "qwen2.5vl:7b", provider="ollama-launch",
            base_url="http://127.0.0.1:11434/v1", api_key="k") is None

    def test_missing_endpoint_or_key_is_conclusive_nothing(self, monkeypatch):
        def forbidden(*a, **kw):  # pragma: no cover - must not be reached
            raise AssertionError("no endpoint/key means nothing to probe")
        monkeypatch.setattr(guard, "_post_chat_completion", forbidden)
        assert guard.refusal_for_pick("m", provider="p", base_url="", api_key="k") is None
        assert guard.refusal_for_pick("m", provider="p", base_url="https://x/v1", api_key="") is None


class TestPolicy:
    def test_default_policy_refuses_and_names_the_override(self, monkeypatch, capsys):
        _probe_returning(monkeypatch, 429, PLAN_REFUSAL_BODY)
        monkeypatch.delenv(guard.POLICY_ENV, raising=False)
        allowed = guard.ensure_pick_entitled(
            "glm-5.3-flashx", provider="zai",
            base_url="https://api.z.ai/api/coding/paas/v4", api_key="k")
        out = capsys.readouterr().out
        assert allowed is False
        assert "was NOT saved as the default" in out
        assert "hermes -m glm-5.3-flashx" in out
        assert guard.POLICY_ENV in out

    def test_escape_hatch_warns_and_persists(self, monkeypatch, capsys):
        _probe_returning(monkeypatch, 429, PLAN_REFUSAL_BODY)
        monkeypatch.setenv(guard.POLICY_ENV, "1")
        allowed = guard.ensure_pick_entitled(
            "glm-5.3-flashx", provider="zai",
            base_url="https://api.z.ai/api/coding/paas/v4", api_key="k")
        out = capsys.readouterr().out
        assert allowed is True
        assert "Saving it anyway" in out

    def test_refusal_copy_never_echoes_a_credential(self, monkeypatch, capsys):
        _probe_returning(monkeypatch, 429, PLAN_REFUSAL_BODY)
        monkeypatch.delenv(guard.POLICY_ENV, raising=False)
        guard.ensure_pick_entitled(
            "glm-5.3-flashx", provider="zai",
            base_url="https://api.z.ai/api/coding/paas/v4", api_key="sk-super-secret")
        assert "sk-super-secret" not in capsys.readouterr().out


@pytest.fixture
def seeded_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a previous default to fall back to.

    ``HERMES_HOME`` is set BEFORE any Hermes module resolves a path, and
    ``tests/home_io_guard.py`` refuses any I/O that escapes to the real home —
    a stray write here fails loudly instead of touching the live config.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "model:\n  default: deepseek-v4.1-flash\n  provider: cheaper-inference\n"
        "  base_url: https://api.cheaperinference.com/v1\n", encoding="utf-8")
    return tmp_path


def _default(home) -> str:
    import hermes_yaml as yaml
    return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))["model"]["default"]


class TestSetupFlowPersistGate:
    def test_refused_pick_leaves_the_default_untouched(self, seeded_home, monkeypatch, capsys):
        from hermes_cli.model_setup_flows_common import _finish_model
        monkeypatch.setattr(guard, "ensure_pick_entitled", lambda *a, **kw: False)
        result = _finish_model(
            "glm-5.3-flashx", "custom:zai",
            "Default model set to: glm-5.3-flashx",
            base_url="https://api.z.ai/api/coding/paas/v4")
        out = capsys.readouterr().out
        assert result is None
        assert "deepseek-v4.1-flash" == _default(seeded_home)
        assert "Default model set to" not in out
        assert "No change." in out

    def test_allowed_pick_is_persisted(self, seeded_home, monkeypatch):
        from hermes_cli.model_setup_flows_common import _finish_model
        monkeypatch.setattr(guard, "ensure_pick_entitled", lambda *a, **kw: True)
        _finish_model(
            "glm-5.3", "zai", "Default model set to: glm-5.3",
            base_url="https://api.z.ai/api/coding/paas/v4")
        assert "glm-5.3" == _default(seeded_home)


class TestModelSwitchPersistGate:
    def _run(self, home, monkeypatch, allowed: bool):
        import hermes_yaml as yaml
        from hermes_cli.model_switch import ModelSwitchResult, persist_model_selection
        monkeypatch.setattr(guard, "ensure_pick_entitled", lambda *a, **kw: allowed)
        result = ModelSwitchResult(
            success=True, new_model="glm-5.3-flashx", target_provider="custom:zai",
            provider_changed=True, api_key="k",
            base_url="https://api.z.ai/api/coding/paas/v4", api_mode="", is_global=True)
        persist_model_selection(result, config_path=home / "config.yaml")
        return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))["model"]

    def test_refused_global_persist_keeps_the_previous_default(self, seeded_home, monkeypatch, capsys):
        model = self._run(seeded_home, monkeypatch, allowed=False)
        assert model["default"] == "deepseek-v4.1-flash"
        assert "was NOT changed" in capsys.readouterr().out

    def test_allowed_global_persist_writes_the_pick(self, seeded_home, monkeypatch):
        model = self._run(seeded_home, monkeypatch, allowed=True)
        assert model["default"] == "glm-5.3-flashx"


class TestNousPersistGate:
    """The Nous surfaces: ``_nous_persist_selection`` and the login that calls it.

    Review round 1 (t_27cf7a7b) found these reaching ``_save_model_choice`` with no
    gate — reproduced live: the gated ``_persist_model`` refused ``glm-5.3-flashx``
    while this one wrote it. They persist through their own path, so they carry
    their own gate; these tests are what would have caught the gap.
    """

    def _refuse(self, monkeypatch, capsys_rationale=PLAN_REFUSAL_BODY):
        _probe_returning(monkeypatch, 429, capsys_rationale)
        monkeypatch.delenv(guard.POLICY_ENV, raising=False)

    def test_refused_nous_pick_writes_nothing_and_returns_none(self, seeded_home, monkeypatch, capsys):
        from hermes_cli.model_setup_flows import _nous_persist_selection
        before = (seeded_home / "config.yaml").read_text(encoding="utf-8")
        self._refuse(monkeypatch)
        result = _nous_persist_selection(
            "glm-5.3-flashx", {"base_url": "https://api.z.ai/api/coding/paas/v4", "api_key": "k"})
        out = capsys.readouterr().out
        assert result is None
        assert (seeded_home / "config.yaml").read_text(encoding="utf-8") == before
        assert "deepseek-v4.1-flash" == _default(seeded_home)
        assert "was NOT saved as the default" in out

    def test_entitled_nous_pick_is_persisted(self, seeded_home, monkeypatch):
        from hermes_cli.model_setup_flows import _nous_persist_selection
        _probe_returning(monkeypatch, 200, {"choices": []})
        result = _nous_persist_selection(
            "upstage/solar-pro4", {"base_url": "https://inference-api.nousresearch.com/v1",
                                   "api_key": "k"})
        assert isinstance(result, dict)
        assert result["model"]["default"] == "upstage/solar-pro4"

    def test_nous_flow_prints_no_change_when_refused(self, seeded_home, monkeypatch, capsys):
        """The caller must not claim "Default model set to" after a refusal.

        ``_nous_persist_selection`` is stubbed to the refusal result (``None``) so
        the CALLER's branch is what is under test — the persist surface itself is
        covered above, and the whole flow would otherwise reach the network.
        """
        import hermes_cli.models as models_mod
        import hermes_cli.models_pricing as pricing_mod
        import hermes_cli.auth as auth_mod
        import hermes_cli.nous_account as account_mod
        import hermes_cli.model_setup_flows as flows
        monkeypatch.setattr(models_mod, "check_nous_free_tier", lambda **kw: False)
        monkeypatch.setattr(models_mod, "get_curated_nous_model_ids",
                            lambda: ["upstage/solar-pro4"])
        # Live pricing is a network call — the flow would hang without this stub.
        monkeypatch.setattr(pricing_mod, "get_pricing_for_provider",
                            lambda *a, **kw: {})
        monkeypatch.setattr(auth_mod, "get_provider_auth_state",
                            lambda p: {"access_token": "test-token"})
        monkeypatch.setattr(auth_mod, "_prompt_model_selection",
                            lambda *a, **kw: "glm-5.3-flashx")
        monkeypatch.setattr(flows, "_nous_verified_credentials",
                            lambda: {"base_url": "https://api.z.ai/api/coding/paas/v4",
                                     "api_key": "k"})
        monkeypatch.setattr(flows, "_nous_model_catalog",
                            lambda *a, **kw: (["glm-5.3-flashx"], {}, [], "", False))
        monkeypatch.setattr(account_mod, "nous_policy_notice", lambda **kw: "")
        monkeypatch.setattr(flows, "_nous_persist_selection", lambda *a, **kw: None)
        flows._model_flow_nous({}, current_model="deepseek-v4.1-flash")
        out = capsys.readouterr().out
        assert "Default model set to" not in out
        assert "No change." in out
        assert "deepseek-v4.1-flash" == _default(seeded_home)


class TestOAuthActivateGate:
    """``_activate_provider_model`` — the four OAuth flows' persist step."""

    def test_refused_oauth_pick_writes_nothing(self, seeded_home, monkeypatch, capsys):
        from hermes_cli.model_setup_flows_common import _activate_provider_model
        self._refuse_oauth(monkeypatch)
        before = (seeded_home / "config.yaml").read_text(encoding="utf-8")
        _activate_provider_model("grok-4.6", "xai-oauth", "https://api.x.ai/v1",
                                 "Default model set to: grok-4.6")
        out = capsys.readouterr().out
        assert (seeded_home / "config.yaml").read_text(encoding="utf-8") == before
        assert "Default model set to" not in out
        assert "No change." in out
        assert "was NOT saved as the default" in out

    def test_allowed_oauth_pick_is_persisted(self, seeded_home, monkeypatch):
        from hermes_cli.model_setup_flows_common import _activate_provider_model
        monkeypatch.setattr(guard, "ensure_pick_entitled", lambda *a, **kw: True)
        monkeypatch.setattr(guard, "resolve_endpoint", lambda *a, **kw: ("", ""))
        _activate_provider_model("grok-4.6", "xai-oauth", "https://api.x.ai/v1",
                                 "Default model set to: grok-4.6")
        assert "grok-4.6" == _default(seeded_home)

    def test_silent_no_change_flow_stays_silent_on_refusal(self, seeded_home, monkeypatch, capsys):
        """MiniMax passes ``no_change=None``; a refusal must not add output."""
        from hermes_cli.model_setup_flows_common import _activate_provider_model
        monkeypatch.setattr(guard, "ensure_pick_entitled", lambda *a, **kw: False)
        monkeypatch.setattr(guard, "resolve_endpoint", lambda *a, **kw: ("", ""))
        _activate_provider_model("MiniMax-M2", "minimax-oauth", "https://api.minimax.io/anthropic",
                                 "✓ Using MiniMax model", no_change=None)
        assert capsys.readouterr().out == ""

    def _refuse_oauth(self, monkeypatch):
        # The refusal must come through the gate's OWN path: patch the transport,
        # not ensure_pick_entitled, so the wiring is exercised end to end.
        _probe_returning(monkeypatch, 429, PLAN_REFUSAL_BODY)
        monkeypatch.delenv(guard.POLICY_ENV, raising=False)
        monkeypatch.setattr(guard, "resolve_endpoint",
                            lambda *a, **kw: ("https://api.x.ai/v1", "k"))


class TestMoAExemption:
    def test_moa_is_exempt_and_documented(self, monkeypatch, capsys):
        """A MoA pick is a preset NAME, not a route: no endpoint to probe."""
        def forbidden(*a, **kw):  # pragma: no cover - must not be reached
            raise AssertionError("a virtual provider must not be probed")
        monkeypatch.setattr(guard, "_post_chat_completion", forbidden)
        assert guard.is_exempt_provider("moa") is True
        assert guard.refusal_for_pick(
            "my-preset", provider="moa", base_url="moa://local", api_key="k") is None

    def test_only_moa_is_exempt(self):
        for provider in ("zai", "custom:zai", "nous", "xai-oauth", "openrouter", "openai-codex"):
            assert guard.is_exempt_provider(provider) is False, provider


class TestWireAwareProbe:
    """The probe must speak the wire the pick would run on.

    A wrong-wire request 404s, and that is a fact about the probe, not about
    entitlement — the reason ``404`` left ``REFUSAL_STATUSES``.
    """

    def test_wire_is_the_pick_route(self):
        assert guard._wire_for("zai", "https://api.z.ai/api/coding/paas/v4", "glm-5.3") == "chat_completions"
        assert guard._wire_for("openai-codex", "https://chatgpt.com/backend-api/codex", "gpt-5.2") == "codex_responses"
        assert guard._wire_for("xai-oauth", "https://api.x.ai/v1", "grok-4.6") == "codex_responses"
        assert guard._wire_for("minimax-oauth", "https://api.minimax.io/anthropic", "MiniMax-M2") == "anthropic_messages"
        assert guard._wire_for("qwen-oauth", "https://portal.qwen.ai/v1", "qwen3-coder-plus") == "chat_completions"

    def test_responses_wire_gets_the_responses_shape(self):
        url, body, auth = guard._wire_request("https://api.x.ai/v1", "k", "grok-4.6", "codex_responses")
        assert url == "https://api.x.ai/v1/responses"
        assert "input" in body and "messages" not in body
        assert auth is None  # Bearer

    def test_anthropic_wire_gets_v1_messages(self):
        url, body, auth = guard._wire_request(
            "https://api.minimax.io/anthropic", "k", "MiniMax-M2", "anthropic_messages")
        assert url == "https://api.minimax.io/anthropic/v1/messages"
        assert body["messages"] == [{"role": "user", "content": "ping"}]

    def test_bare_404_from_a_wrong_wire_does_not_block(self, monkeypatch):
        """The decisive regression: a 404 with no entitlement body is inconclusive."""
        _probe_returning(monkeypatch, 404, {"detail": "Not Found"})
        assert guard.refusal_for_pick(
            "gpt-5.2", provider="openai-codex",
            base_url="https://chatgpt.com/backend-api/codex", api_key="k") is None

    def test_404_with_a_retired_free_body_still_blocks(self, monkeypatch):
        """The real 404 wall carries its proof in the body — that path is unchanged."""
        _probe_returning(monkeypatch, 404, RETIRED_FREE_BODY)
        assert guard.refusal_for_pick(
            "upstage/solar-pro4:free", provider="openrouter",
            base_url="https://openrouter.ai/api/v1", api_key="k")


#: Functions that reach ``_save_model_choice`` with NO gate, on purpose. Each needs
#: a written reason here or the invariant test below fails — the point is that the
#: exemption is visible at review time, not discovered by a reviewer's probe.
_GATE_EXEMPT = {
    "_model_flow_moa": "persists a MoA PRESET NAME for a virtual provider (moa://local): "
                       "no endpoint to probe, no plan to be entitled by",
    "_begin_model_config": "internal half-step of the gated _persist_model — every "
                           "external caller goes through the gate first",
}


class TestPersistSurfaceInvariant:
    """Every surface that can write ``model.default`` is gated or exempt-with-reason.

    Review round 1 (t_27cf7a7b) requested this: the original claim "every persist
    surface refuses to write a proven-unusable pick" was false — four functions
    still reached ``_save_model_choice`` ungated and one of them
    (``_nous_persist_selection``) was reproduced writing a refused slug live. A
    hand-maintained list is what went stale; this derives the list from the AST, so
    a NEW persist surface fails the suite until it is gated or exempted on purpose.
    """

    def test_every_save_model_choice_caller_is_gated_or_exempt(self):
        import ast
        from pathlib import Path

        pkg = Path(guard.__file__).resolve().parent
        ungated = []
        for path in sorted(pkg.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                called = {getattr(c.func, "id", None) or getattr(c.func, "attr", None)
                          for c in ast.walk(node) if isinstance(c, ast.Call)}
                if "_save_model_choice" not in called:
                    continue
                if "ensure_pick_entitled" in called or node.name in _GATE_EXEMPT:
                    continue
                # A function that only calls the gated helper is covered by it.
                if "_persist_model" in called or "_activate_provider_model" in called:
                    continue
                ungated.append(f"{path.name}:{node.lineno} {node.name}")
        assert not ungated, (
            "ungated persist surface(s) — gate them or add a written exemption to "
            f"_GATE_EXEMPT: {ungated}")

    def test_every_activate_provider_model_caller_passes_its_credential(self):
        """The OAuth flows must hand the gate the key they resolved.

        Without it the gate falls back to config/env resolution and can be
        inconclusive where the flow itself held a conclusive credential.
        """
        import ast
        from pathlib import Path

        pkg = Path(guard.__file__).resolve().parent
        missing = []
        for path in sorted(pkg.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for call in (c for c in ast.walk(node) if isinstance(c, ast.Call)):
                    if getattr(call.func, "id", None) != "_activate_provider_model":
                        continue
                    if not any(k.arg == "api_key" for k in call.keywords):
                        missing.append(f"{path.name}:{call.lineno} in {node.name}")
        assert not missing, f"_activate_provider_model caller(s) without api_key=: {missing}"

    def test_the_exemptions_are_exactly_what_is_expected(self):
        """An exemption is a decision — a new one should be a deliberate edit."""
        assert set(_GATE_EXEMPT) == {"_model_flow_moa", "_begin_model_config"}
