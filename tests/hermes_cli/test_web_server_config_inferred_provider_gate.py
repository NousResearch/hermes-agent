"""#115079 (dashboard surface): the Settings flat Model field must not persist an INFERRED provider.

The Dashboard/Desktop Settings Model field carries no provider, so ``PUT /api/config``
denormalizes a model-name change by INFERRING the serving provider
(``_infer_provider_on_model_change``) — whose only signal gate is credential possession.
An ambient ``*_API_KEY`` therefore let a bare model edit write ``model.provider`` into
config.yaml: the incident's write class. Standing ruling (PR #107366): "Possessing a
credential is not selecting a provider." The save now fails closed with HTTP 400 through
the shared gate (``hermes_cli.model_switch.inferred_provider_persist_refusal``) unless the
user NAMED the provider (auth-store active provider).

The env-key rows are E2E controls in the spirit of ``test_inferred_provider_persist_gate.py``:
``DASHSCOPE_API_KEY`` present/absent must MOVE the outcome — an injection that changed nothing
would invalidate the guard. The catalog/validation patch set mirrors that file's ``_offline``;
the credential ladder, the persist gate, and the config write stay real, against an isolated
HERMES_HOME.
"""

from __future__ import annotations

import json
import unittest.mock as mock

import pytest
import yaml
from fastapi import HTTPException

from hermes_cli.web_server_config import _denormalize_config_from_web

_ACCEPTED = {"accepted": True, "persist": True, "recognized": True, "message": None}

# A configured (NOT fresh) install: deepseek is the standing route, and an ambient
# DASHSCOPE_API_KEY seeds the credential pool for the unconfigured alibaba provider.
_SEED = (
    "model:\n"
    "  default: deepseek-chat\n"
    "  provider: deepseek\n"
    "agent:\n"
    "  system_prompt: keepme\n"
)


def _seed_home(tmp_path, monkeypatch, *, dashscope: bool, config_text: str = _SEED):
    """Isolated HERMES_HOME with a seeded config.yaml and a live deepseek session key;
    the ambient alibaba (DashScope) key is the injected variable under test."""
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(config_text, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-dee...sion")
    if dashscope:
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-das...ient")
    return home


def _offline_stack():
    """``_offline`` plus alias/validation/metadata probes (needed once the save proceeds
    through ``_validated_main_model_selection`` → ``switch_model``)."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        with mock.patch.multiple(
                "hermes_cli.models",
                cached_provider_model_ids=lambda *a, **k: [],
                model_ids=lambda *a, **k: []), \
             mock.patch("hermes_cli.model_switch.resolve_alias", return_value=None), \
             mock.patch("hermes_cli.models_validate.validate_requested_model", return_value=_ACCEPTED), \
             mock.patch("hermes_cli.model_switch.get_model_info", return_value=None), \
             mock.patch("hermes_cli.model_switch.get_model_capabilities", return_value=None):
            yield

    return _ctx()


def _model_block(home) -> dict:
    return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))["model"]


class TestSettingsFieldInferredProviderGate:
    """Unit rows drive the real ``_denormalize_config_from_web`` entry that PUT /api/config uses."""

    def test_inferred_cross_provider_refused_400_names_provider_disk_unchanged(
            self, tmp_path, monkeypatch):
        """Flat model edit → the credential-gated detector names alibaba → the user never
        named it → 400 naming the provider, BEFORE any assignment/normalisation; the
        on-disk route is untouched."""
        home = _seed_home(tmp_path, monkeypatch, dashscope=True)
        before = (home / "config.yaml").read_bytes()

        with _offline_stack():
            with pytest.raises(HTTPException) as exc_info:
                _denormalize_config_from_web({"model": "qwen3.6-plus"})

        assert exc_info.value.status_code == 400
        detail = str(exc_info.value.detail).lower()
        assert "alibaba" in detail
        # The refusal must say how to confirm ON THIS SURFACE: the Models page is the
        # dashboard picker that submits provider+model together over POST /api/model/set.
        assert "models page" in detail
        assert (home / "config.yaml").read_bytes() == before

    def test_env_key_injection_moves_the_outcome_both_ways(self, tmp_path, monkeypatch):
        """The injection itself must MOVE the outcome (otherwise the rows prove nothing):
        WITH DASHSCOPE_API_KEY the same edit is refused; WITHOUT it the credential gate
        skips the guess (detect → None), no provider change is attempted, and the save
        proceeds with the standing provider."""
        home_with = _seed_home(tmp_path / "with", monkeypatch, dashscope=True)
        with _offline_stack():
            with pytest.raises(HTTPException) as exc_info:
                _denormalize_config_from_web({"model": "qwen3.6-plus"})
        assert exc_info.value.status_code == 400
        assert "alibaba" in str(exc_info.value.detail).lower()
        assert _model_block(home_with)["provider"] == "deepseek"

        home_no = _seed_home(tmp_path / "without", monkeypatch, dashscope=False)
        with _offline_stack():
            result = _denormalize_config_from_web({"model": "qwen3.6-plus"})
        assert result["model"]["default"] == "qwen3.6-plus"
        assert result["model"]["provider"] == "deepseek"  # model moved, provider did not

    def test_target_is_auth_store_active_provider_saves(self, tmp_path, monkeypatch):
        """The user NAMED alibaba (a login/selection wrote active_provider) → authorized →
        the save routes through the assignment chokepoint exactly as before."""
        home = _seed_home(tmp_path, monkeypatch, dashscope=True)
        # auth store lives in HERMES_HOME (the seeded home).
        (home / "auth.json").write_text(
            json.dumps({"version": 3, "providers": {}, "active_provider": "alibaba"}),
            encoding="utf-8")

        with _offline_stack():
            result = _denormalize_config_from_web({"model": "qwen3.6-plus"})

        assert result["model"]["provider"] == "alibaba"
        assert result["model"]["default"] == "qwen3.6-plus"

    def test_provider_named_flat_input_saves_no_false_400(self, tmp_path, monkeypatch):
        """F1 (round-1 review BLOCKER): the flat Model field accepts the documented
        ``provider/model`` form. ``alibaba/qwen3.6-plus`` on an openrouter config resolves
        through detect's naming branch — the user NAMED the provider, so the CLI's
        input-naming semantics must exempt it here too. Pre-fix the web gate fired on any
        detected provider change and answered with the factually FALSE refusal copy
        ('never selected by you') → HTTP 400 on an explicit input."""
        _seed_home(tmp_path, monkeypatch, dashscope=True,
                   config_text="model:\n  default: some-other-model\n  provider: openrouter\n"
                               "agent:\n  system_prompt: keepme\n")

        with _offline_stack(), mock.patch(
                "hermes_cli.models.detect_provider_for_model",
                return_value=("alibaba", "qwen3.6-plus")):
            result = _denormalize_config_from_web({"model": "alibaba/qwen3.6-plus"})

        assert result["model"]["provider"] == "alibaba"
        assert result["model"]["default"]  # routed through the assignment chokepoint

    def test_bare_provider_alias_input_saves_no_false_400(self, tmp_path, monkeypatch):
        """Same exemption via detect's step-0 branch: the flat field value ``alibaba`` IS
        the provider name (bare alias-form model value) — naming it IS selecting it."""
        _seed_home(tmp_path, monkeypatch, dashscope=True,
                   config_text="model:\n  default: some-other-model\n  provider: openrouter\n"
                               "agent:\n  system_prompt: keepme\n")

        with _offline_stack(), mock.patch(
                "hermes_cli.models.detect_provider_for_model",
                return_value=("alibaba", "qwen3.8-max")):
            result = _denormalize_config_from_web({"model": "alibaba"})

        assert result["model"]["provider"] == "alibaba"

    def test_unnamed_vendor_slug_openrouter_sentinel_still_gated(self, tmp_path, monkeypatch):
        """Guard row for the exemption's boundary: ``acme/some-model`` names the model's
        VENDOR, not the aggregator — ``_infer_provider_on_model_change``'s ``openrouter``
        answer is a credential-gated GUESS, and a guess must still 400 (with openrouter
        creds ambient, exactly the state where the sentinel fires)."""
        home = _seed_home(tmp_path, monkeypatch, dashscope=True,
                          config_text="model:\n  default: some-other-model\n"
                                      "  provider: deepseek\n"
                                      "agent:\n  system_prompt: keepme\n")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-openrouter-ambient")
        before = (home / "config.yaml").read_bytes()

        with _offline_stack():
            with pytest.raises(HTTPException) as exc_info:
                _denormalize_config_from_web({"model": "acme/some-model"})

        assert exc_info.value.status_code == 400
        assert "openrouter" in str(exc_info.value.detail).lower()
        assert (home / "config.yaml").read_bytes() == before

    def test_fresh_config_without_model_block_passes_through(self, tmp_path, monkeypatch):
        """A config with no ``model:`` block has no provider to displace: the flat string
        passes through unchanged (on THIS surface inference requires a dict model with a
        provider on disk; the gate's own fresh-install clause is covered in
        ``test_inferred_provider_persist_gate.py``)."""
        home = _seed_home(
            tmp_path, monkeypatch, dashscope=True,
            config_text="agent:\n  system_prompt: keepme\n")

        with _offline_stack():
            result = _denormalize_config_from_web({"model": "qwen3.6-plus"})

        assert result["model"] == "qwen3.6-plus"
        assert "model" not in yaml.safe_load(
            (home / "config.yaml").read_text(encoding="utf-8"))

    def test_model_change_within_current_provider_saves(self, tmp_path, monkeypatch):
        """Detection handing back the CURRENT provider is a model rename, not an inferred
        route — the gate must not fire even with an ambient key present."""
        _seed_home(tmp_path, monkeypatch, dashscope=True)

        with mock.patch("hermes_cli.models.detect_provider_for_model",
                        return_value=("deepseek", "deepseek-chat-v2")):
            result = _denormalize_config_from_web({"model": "deepseek-chat-v2"})

        assert result["model"]["default"] == "deepseek-chat-v2"
        assert result["model"]["provider"] == "deepseek"

    def test_alias_form_provider_same_provider_edit_saves_no_false_400(
            self, tmp_path, monkeypatch):
        """Alias-form provider on disk must not read as a provider change (review polish).
        ``normalize_provider('qwen') == 'alibaba'`` (providers._ALIAS_GROUPS); with the disk
        holding the ALIAS form (``provider: qwen``), a model edit the ladder resolves back to
        the same provider (typed model alias ``qwen`` → MODEL_ALIASES → alibaba's
        ``qwen3.8-max``) is the false-positive: the raw compare read ``'alibaba' != 'qwen'``
        as a change and the gate refused a save that moves no provider. Comparing canonical
        ids, this edit takes the SAME path a canonical-form ``provider: alibaba`` disk takes:
        fire condition false → default updated verbatim, provider untouched, no 400."""
        _seed_home(
            tmp_path, monkeypatch, dashscope=True,
            config_text="model:\n  default: qwen3.6-plus\n  provider: qwen\n"
                        "agent:\n  system_prompt: keepme\n")

        with _offline_stack():
            result = _denormalize_config_from_web({"model": "qwen"})

        assert result["model"]["provider"] == "qwen"  # same provider after normalize — no move
        assert result["model"]["default"] == "qwen"   # typed value preserved, verbatim,
        # exactly as the canonical-form ('alibaba' on disk) row of the same edit writes.

    def test_unchanged_model_save_is_unaffected(self, tmp_path, monkeypatch):
        """The Settings autosave PUTs the whole draft: saving an unrelated field with the
        model echoed back must not trip inference at all."""
        _seed_home(tmp_path, monkeypatch, dashscope=True)

        result = _denormalize_config_from_web({"model": "deepseek-chat", "approvals": {"mode": "read"}})

        assert result["model"]["provider"] == "deepseek"
        assert result["model"]["default"] == "deepseek-chat"


class TestPutConfigEndpointInferredProviderRefusal:
    """Endpoint guarantee: the 400 survives ``asyncio.to_thread`` + ``http_failure``
    (which re-raises HTTPException verbatim) and the config file is byte-identical."""

    def test_put_config_400_and_config_byte_identical(self, tmp_path, monkeypatch):
        from starlette.testclient import TestClient
        from hermes_constants import get_hermes_home
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

        home = _seed_home(tmp_path, monkeypatch, dashscope=True)
        cfg_path = get_hermes_home() / "config.yaml"
        assert cfg_path.read_bytes() == (home / "config.yaml").read_bytes()
        before = cfg_path.read_bytes()

        with _offline_stack():
            client = TestClient(app)
            client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
            resp = client.put("/api/config", json={"config": {"model": "qwen3.6-plus"}})

        assert resp.status_code == 400
        assert "alibaba" in resp.json()["detail"].lower()
        assert cfg_path.read_bytes() == before
