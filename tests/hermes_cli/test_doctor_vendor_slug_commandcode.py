"""Regression: doctor must not flag CommandCode's native vendor-prefixed model IDs.

CommandCode (https://api.commandcode.ai/provider/v1) publishes a natively
MIXED model namespace: 64 of its 85 catalog IDs are vendor-prefixed
(``deepseek/deepseek-v4-pro``, ``Qwen/Qwen3.7-Max``, ``moonshotai/Kimi-K2.6``,
``zai-org/GLM-5.1``, ...) while the rest are bare (``gpt-5.5``,
``claude-sonnet-4-6``).  Its provider plugin declares the same form in
``fallback_models`` and gates DeepSeek's native reasoning controls on the
``deepseek/`` prefix, so dropping the prefix (the doctor's suggested remedy)
would silently break ``/reasoning`` for those models.

``commandcode`` therefore belongs in ``_VENDOR_SLUG_PROVIDERS`` alongside
``fireworks``/``deepinfra``.  See NousResearch/hermes-agent doctor issue #1.
"""

from __future__ import annotations

import contextlib
import io
import sys
import types
from argparse import Namespace

import pytest

import hermes_cli.doctor as doctor_mod
from hermes_cli.doctor_config import _validate_model_config

# Real IDs from the live catalog (public /models endpoint). Slash-form and bare
# both occur, which is exactly the mixed namespace the allowlist must accept.
COMMANDCODE_MODEL_IDS = [
    "deepseek/deepseek-v4.1-flash-fast",
    "deepseek/deepseek-v4-pro",
    "Qwen/Qwen3.7-Max",
    "moonshotai/Kimi-K2.6",
    "zai-org/GLM-5.1",
    "gpt-5.5",
]

_VENDOR_WARNING = "uses a vendor/model slug but provider is"
_VENDOR_ISSUE = "is vendor-prefixed but model.provider is"


def _write_config(tmp_path, provider, default_model, base_url=None):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    body = f"model:\n  provider: {provider}\n  default: {default_model}\n"
    if base_url:
        body += f"  base_url: {base_url}\n"
    (home / "config.yaml").write_text(body, encoding="utf-8")
    return home


def _run_doctor(monkeypatch, tmp_path, home):
    """Drive the real doctor entry point offline, mirroring the sibling doctor tests."""
    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", tmp_path / "project")
    monkeypatch.setattr(doctor_mod, "_DHH", str(home))
    (tmp_path / "project").mkdir(exist_ok=True)

    monkeypatch.setitem(
        sys.modules,
        "model_tools",
        types.SimpleNamespace(
            check_tool_availability=lambda *a, **kw: ([], []),
            TOOLSET_REQUIREMENTS={},
        ),
    )

    # Keep the run offline: the CommandCode profile would otherwise hit its
    # live /models endpoint.
    import httpx

    monkeypatch.setattr(httpx, "get", lambda *a, **k: types.SimpleNamespace(status_code=200))

    try:
        from hermes_cli import auth as _auth_mod

        monkeypatch.setattr(_auth_mod, "get_nous_auth_status_local", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", lambda: {})
    except Exception:
        pass

    buf = io.StringIO()
    with contextlib.suppress(SystemExit), contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False))
    return buf.getvalue()


def test_run_doctor_accepts_commandcode_vendor_slug(monkeypatch, tmp_path):
    """End-to-end: the live config (provider=commandcode,
    default=deepseek/deepseek-v4.1-flash-fast) must produce NO vendor-slug
    warning and NO issue line."""
    home = _write_config(tmp_path, "commandcode", "deepseek/deepseek-v4.1-flash-fast")
    (home / ".env").write_text("COMMANDCODE_API_KEY=cc_test_key\n", encoding="utf-8")
    monkeypatch.setenv("COMMANDCODE_API_KEY", "cc_test_key")

    out = _run_doctor(monkeypatch, tmp_path, home)

    assert f"model.default 'deepseek/deepseek-v4.1-flash-fast' {_VENDOR_WARNING} 'commandcode'" not in out
    assert f"model.default 'deepseek/deepseek-v4.1-flash-fast' {_VENDOR_ISSUE} 'commandcode'" not in out
    assert "Either set model.provider to 'openrouter', or drop the vendor prefix." not in out


@pytest.mark.parametrize("model_id", COMMANDCODE_MODEL_IDS)
def test_validate_model_config_accepts_commandcode_catalog_ids(monkeypatch, tmp_path, model_id):
    """Both halves of the mixed catalog (slash-form and bare) resolve cleanly."""
    monkeypatch.setenv("COMMANDCODE_API_KEY", "cc_test_key")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"model:\n  provider: commandcode\n  default: {model_id}\n", encoding="utf-8"
    )

    issues: list[str] = []
    _validate_model_config(config_path, issues)

    assert not [i for i in issues if _VENDOR_ISSUE in i], (
        f"CommandCode model '{model_id}' falsely flagged as vendor-prefixed: {issues}"
    )


def test_validate_model_config_still_flags_unlisted_provider(monkeypatch, tmp_path):
    """Control: the heuristic is not neutered — a provider outside the allowlist
    with a real OpenAI endpoint still gets the vendor-slug issue."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model:\n"
        "  provider: openai-api\n"
        "  default: nvidia/z-ai/glm-5.2\n"
        "  base_url: https://api.openai.com/v1\n",
        encoding="utf-8",
    )

    issues: list[str] = []
    _validate_model_config(config_path, issues)

    assert [i for i in issues if _VENDOR_ISSUE in i], (
        f"unlisted provider should still be flagged, got: {issues}"
    )
