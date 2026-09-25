"""Copilot picker honors per-account policy state instead of listing every catalog row.

GitHub's ``/models`` catalog is platform-wide, not account-scoped: Free/Student quota
accounts observe rows they cannot pin (``policy.state: disabled`` → HTTP 400
``model_not_supported``) plus internal routing slugs no client offers. The catalog
fetcher must reflect what the signed-in account can actually select.
"""

from hermes_cli import models


def _row(mid, policy=None, type_="chat"):
    item = {"id": mid, "capabilities": {"type": type_}}
    if policy is not None:
        item["policy"] = {"state": policy}
    return item


CATALOG = [
    _row("claude-sonnet-5", "disabled"),
    _row("copilot-search-a"),
    _row("exec-agent-b"),
    _row("trajectory-compaction"),
    _row("gpt-5.6-luna", "enabled"),
    _row("gpt-4.1"),
    _row("gpt-4o"),
    _row("embedding-thing", type_="embedding"),
]


def test_policy_disabled_and_internal_slugs_are_hidden():
    assert models._policy_available_copilot_models(CATALOG) == ["gpt-5.6-luna", "gpt-4.1", "gpt-4o"]


def test_missing_policy_means_allowed_not_disabled():
    # Legacy/utility rows (gpt-4o, gpt-4.1) carry no policy block and remain servable.
    assert models._policy_available_copilot_models([_row("gpt-4.1")]) == ["gpt-4.1"]


def test_picker_flag_ignored_but_chat_type_respected():
    # Accounts exist where every row reports model_picker_enabled: false — display hint only.
    rows = [
        {"id": "x", "model_picker_enabled": False, "capabilities": {"type": "chat"}},
        {"id": "y", "model_picker_enabled": True, "capabilities": {"type": "embedding"}},
    ]
    assert models._policy_available_copilot_models(rows) == ["x"]


def test_copilot_catalog_serves_filtered_rows_not_curated_fallback(monkeypatch):
    monkeypatch.setattr(models, "_resolve_copilot_catalog_api_key", lambda: "k")
    monkeypatch.setattr(models, "fetch_github_model_catalog", lambda api_key=None, timeout=5.0: CATALOG)
    out = models._copilot_catalog("copilot", False)
    assert out == ["gpt-5.6-luna", "gpt-4.1", "gpt-4o"]
    assert not isinstance(out, models.CuratedFallbackModels)


def test_copilot_catalog_failure_still_falls_back_to_curated(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("network down")
    monkeypatch.setattr(models, "_resolve_copilot_catalog_api_key", lambda: "k")
    monkeypatch.setattr(models, "fetch_github_model_catalog", boom)
    assert isinstance(models._copilot_catalog("copilot", False), models.CuratedFallbackModels)
