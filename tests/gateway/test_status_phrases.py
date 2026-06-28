import random

from gateway.status_phrases import (
    classify_status_context,
    choose_status_phrase,
    resolve_status_phrase_catalog,
)


def test_terminal_tool_uses_command_surface_bucket():
    assert classify_status_context("tool", tool_name="terminal") == "command"
    assert classify_status_context("command") == "command"


def test_regular_tool_uses_tool_surface_bucket_not_domain_bucket():
    assert classify_status_context("tool", tool_name="web_search", preview="weather") == "tool"
    assert classify_status_context("tool", tool_name="mcp_atlassian_infohealth_getJiraIssue") == "tool"


def test_interim_uses_interim_surface_bucket():
    assert classify_status_context("interim_assistant") == "interim"


def test_generic_phrase_does_not_leak_raw_thinking_text():
    msg = choose_status_phrase(
        "thinking",
        preview="actual private scratch text should not be sent",
        rng=random.Random(4),
    )

    assert "actual private scratch" not in msg
    assert msg


def test_generic_tool_phrase_does_not_leak_vendor_or_args():
    msg = choose_status_phrase(
        "tool",
        tool_name="mcp_atlassian_infohealth_getJiraIssue",
        args={"issueIdOrKey": "SECRET-123"},
        rng=random.Random(1),
    )

    assert "jira" not in msg.lower()
    assert "confluence" not in msg.lower()
    assert "atlassian" not in msg.lower()
    assert "SECRET-123" not in msg


def test_generic_phrase_avoids_recent_repetition():
    recent: list[str] = []
    first = choose_status_phrase("tool", tool_name="web_search", rng=random.Random(2), recent=recent)
    second = choose_status_phrase("tool", tool_name="web_search", rng=random.Random(2), recent=recent)

    assert first != second
    assert recent[-2:] == [first, second]


def test_builtin_catalog_is_loaded_from_external_asset_and_is_not_tiny():
    catalog = resolve_status_phrase_catalog({}, "whatsapp")

    for surface in ("thinking", "tool", "command", "interim", "status"):
        assert len(catalog[surface]) >= 25, surface


def test_relative_status_phrase_path_loads_from_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    phrase_file = tmp_path / "phrases.yaml"
    phrase_file.write_text("mode: replace\ntool:\n  - relative safe tool text\n", encoding="utf-8")

    catalog = resolve_status_phrase_catalog(
        {"display": {"status_phrases": {"path": "phrases.yaml"}}},
        "whatsapp",
    )

    assert catalog["tool"] == ["relative safe tool text"]


def test_status_phrase_path_can_load_relative_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    phrase_dir = tmp_path / "phrase-catalog"
    phrase_dir.mkdir()
    (phrase_dir / "01-status.yaml").write_text("status:\n  - relative dir status text\n", encoding="utf-8")

    catalog = resolve_status_phrase_catalog(
        {"display": {"status_phrases": {"path": "phrase-catalog"}}},
        "whatsapp",
    )

    assert "relative dir status text" in catalog["status"]


def test_absolute_or_parent_phrase_paths_are_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    outside = tmp_path.parent / "outside-phrases.yaml"
    outside.write_text("mode: replace\ntool:\n  - should not load\n", encoding="utf-8")

    catalog = resolve_status_phrase_catalog(
        {"display": {"status_phrases": {"path": str(outside)}}},
        "whatsapp",
    )
    escaped = resolve_status_phrase_catalog(
        {"display": {"status_phrases": {"path": "../outside-phrases.yaml"}}},
        "whatsapp",
    )

    assert catalog["tool"] != ["should not load"]
    assert escaped["tool"] != ["should not load"]


def test_conventional_relative_status_phrase_file_is_loaded(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "status_phrases.yaml").write_text(
        "mode: replace\nstatus:\n  - conventional status text\n",
        encoding="utf-8",
    )

    catalog = resolve_status_phrase_catalog({}, "whatsapp")

    assert catalog["status"] == ["conventional status text"]


def test_global_custom_phrase_catalog_appends_to_builtin():
    catalog = resolve_status_phrase_catalog(
        {
            "display": {
                "status_phrases": {
                    "thinking": ["custom thinking placeholder"],
                }
            }
        },
        "whatsapp",
    )

    assert "custom thinking placeholder" in catalog["thinking"]
    assert len(catalog["thinking"]) > 1


def test_platform_custom_phrase_catalog_can_replace_surface():
    catalog = resolve_status_phrase_catalog(
        {
            "display": {
                "platforms": {
                    "whatsapp": {
                        "status_phrases": {
                            "mode": "replace",
                            "tool": ["custom tool placeholder"],
                        }
                    }
                }
            }
        },
        "whatsapp",
    )

    assert catalog["tool"] == ["custom tool placeholder"]
    assert len(catalog["thinking"]) > 1


def test_choose_status_phrase_uses_custom_catalog_without_leaking_args():
    catalog = resolve_status_phrase_catalog(
        {"display": {"status_phrases": {"mode": "replace", "tool": ["custom safe tool text"]}}},
        "whatsapp",
    )

    msg = choose_status_phrase(
        "tool",
        tool_name="web_search",
        args={"query": "SECRET SEARCH"},
        catalog=catalog,
    )

    assert msg == "custom safe tool text"
    assert "SECRET" not in msg
