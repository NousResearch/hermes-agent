"""The chat-platform /model picker (``list_picker_providers``) lists the managed local runtime.

Contract (#152473): staged GGUFs that make the desktop/CLI payload show a selectable ``llamacpp``
row make the gateway picker (Telegram, Discord, Slack) show the same row, and that row's slug
resolves as a provider so selecting it can switch."""

from __future__ import annotations

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "models").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _stage(home, *names):
    for name in names:
        (home / "models" / f"{name}.gguf").write_bytes(b"GGUF" + b"\x00" * 64)


def _picker(**kw):
    from hermes_cli.model_switch_providers import list_picker_providers

    return list_picker_providers(include_moa=True, non_blocking_catalogs=True,
                                 probe_custom_providers=False, **kw)


def test_chat_picker_lists_the_same_local_row_as_the_desktop_payload(hermes_home):
    from hermes_cli.inventory import build_models_payload, load_picker_context
    from hermes_cli.providers import resolve_provider_full

    _stage(hermes_home, "Qwen3-4B-Instruct-2507-UD-Q8_K_XL", "Local-Model-X")
    rows = {r["slug"]: r for r in _picker()}
    assert "llamacpp" in rows, f"chat picker omitted the local runtime row: {sorted(rows)}"
    desktop = {r["slug"]: r for r in build_models_payload(
        load_picker_context(), probe_custom_providers=False)["providers"]}
    assert sorted(rows["llamacpp"]["models"]) == sorted(desktop["llamacpp"]["models"])
    assert rows["llamacpp"]["total_models"] == 2
    assert resolve_provider_full("llamacpp") is not None


def test_current_local_row_leads_and_nothing_staged_means_no_row(hermes_home):
    assert "llamacpp" not in {r["slug"] for r in _picker()}
    _stage(hermes_home, "M")
    rows = _picker(current_provider="llamacpp", current_model="M")
    assert rows[0]["slug"] == "llamacpp" and rows[0]["is_current"] is True
