"""Azure Foundry doctor connectivity probe (issue #66756).

Anthropic-style Foundry bases (`…/anthropic`) return HTTP 404 for the
generic Bearer `/models` health check even when chat works. Doctor must
use Anthropic headers / a messages probe instead.
"""

from __future__ import annotations

import types
from argparse import Namespace

import pytest


def test_azure_foundry_skipped_by_generic_apikey_loop():
    from hermes_cli import doctor

    doctor._APIKEY_PROVIDERS_CACHE = None
    entries = doctor._build_apikey_providers_list()
    names = {entry[0].lower() for entry in entries}
    assert not any("azure foundry" in name or "azure-foundry" in name for name in names), (
        f"Azure Foundry must use the dedicated probe, not the generic Bearer "
        f"loop. Got: {sorted(names)}"
    )


def test_run_doctor_azure_foundry_anthropic_endpoint_reports_healthy(
    monkeypatch, tmp_path
):
    from hermes_cli import doctor as doctor_mod

    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    base = "https://myresource.services.ai.azure.com/anthropic"
    (home / "config.yaml").write_text(
        "model:\n"
        "  provider: azure-foundry\n"
        "  default: claude-sonnet-5\n"
        f"  base_url: {base}\n",
        encoding="utf-8",
    )
    (home / ".env").write_text("AZURE_FOUNDRY_API_KEY=sk-foundry-test\n", encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)

    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)
    monkeypatch.setattr(doctor_mod, "_DHH", str(home))
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "sk-foundry-test")
    monkeypatch.setenv("AZURE_FOUNDRY_BASE_URL", base)

    fake_model_tools = types.SimpleNamespace(
        check_tool_availability=lambda *a, **kw: ([], []),
        TOOLSET_REQUIREMENTS={},
    )
    monkeypatch.setitem(__import__("sys").modules, "model_tools", fake_model_tools)

    try:
        from hermes_cli import auth as _auth_mod

        monkeypatch.setattr(_auth_mod, "get_nous_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", lambda: {})
    except Exception:
        pass

    calls: list[tuple] = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(("GET", url, headers))
        # Anthropic-style Foundry: /models is often missing
        return types.SimpleNamespace(status_code=404, text="not found")

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(("POST", url, headers, json))
        return types.SimpleNamespace(status_code=200, text="{}")

    import httpx
    import io
    import contextlib

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(httpx, "post", fake_post)

    # Clear cached provider list so azure-foundry skip is re-evaluated.
    doctor_mod._APIKEY_PROVIDERS_CACHE = None

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False))
    out = buf.getvalue()

    assert "Azure Foundry" in out
    assert "HTTP 404" not in out
    # Must have used x-api-key on the anthropic probe path
    anth_calls = [c for c in calls if isinstance(c[1], str) and "/anthropic" in c[1]]
    assert anth_calls, f"expected anthropic endpoint probe, got {calls}"
    headers = anth_calls[0][2] or {}
    assert headers.get("x-api-key") == "sk-foundry-test"
    assert "Authorization" not in headers or not str(headers.get("Authorization", "")).startswith(
        "Bearer sk-foundry"
    )
