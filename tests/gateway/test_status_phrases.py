import random

from gateway.status_phrases import (
    classify_status_context,
    choose_status_phrase,
    resolve_status_phrase_catalog,
)


def test_long_running_context_uses_status_bucket():
    assert classify_status_context("status") == "status"
    assert classify_status_context("heartbeat") == "status"
    assert classify_status_context("long_running") == "status"


def test_non_status_context_falls_back_to_generic_bucket():
    assert classify_status_context("tool", tool_name="terminal") == "generic"
    assert classify_status_context("thinking") == "generic"
    assert classify_status_context("interim_assistant") == "generic"


def test_status_phrase_does_not_leak_raw_preview_or_args():
    msg = choose_status_phrase(
        "status",
        preview="actual private scratch text should not be sent",
        args={"secret": "SECRET-123"},
        rng=random.Random(4),
    )

    assert "actual private scratch" not in msg
    assert "SECRET-123" not in msg
    assert msg


def test_status_phrase_avoids_recent_repetition():
    recent: list[str] = []
    first = choose_status_phrase("status", rng=random.Random(2), recent=recent)
    second = choose_status_phrase("status", rng=random.Random(2), recent=recent)

    assert first != second
    assert recent[-2:] == [first, second]


def test_builtin_catalog_is_loaded_from_external_asset_and_is_status_only():
    catalog = resolve_status_phrase_catalog({}, "whatsapp")

    assert set(catalog) == {"status", "generic"}
    assert len(catalog["status"]) >= 25
    assert len(catalog["generic"]) >= 10


def test_relative_status_phrase_path_loads_from_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    phrase_file = tmp_path / "phrases.yaml"
    phrase_file.write_text("mode: replace\nstatus:\n  - relative safe status text\n", encoding="utf-8")

    catalog = resolve_status_phrase_catalog(
        {"display": {"status_phrases": {"path": "phrases.yaml"}}},
        "whatsapp",
    )

    assert catalog["status"] == ["relative safe status text"]


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
    outside.write_text("mode: replace\nstatus:\n  - should not load\n", encoding="utf-8")

    catalog = resolve_status_phrase_catalog(
        {"display": {"status_phrases": {"path": str(outside)}}},
        "whatsapp",
    )
    escaped = resolve_status_phrase_catalog(
        {"display": {"status_phrases": {"path": "../outside-phrases.yaml"}}},
        "whatsapp",
    )

    assert catalog["status"] != ["should not load"]
    assert escaped["status"] != ["should not load"]


def test_conventional_relative_status_phrase_file_is_loaded(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "status_phrases.yaml").write_text(
        "mode: replace\nstatus:\n  - conventional status text\n",
        encoding="utf-8",
    )

    catalog = resolve_status_phrase_catalog({}, "whatsapp")

    assert catalog["status"] == ["conventional status text"]


def test_global_custom_status_phrase_catalog_appends_to_builtin():
    catalog = resolve_status_phrase_catalog(
        {
            "display": {
                "status_phrases": {
                    "status": ["custom long-running placeholder"],
                }
            }
        },
        "whatsapp",
    )

    assert "custom long-running placeholder" in catalog["status"]
    assert len(catalog["status"]) > 1


def test_platform_custom_status_phrase_catalog_can_replace_surface():
    catalog = resolve_status_phrase_catalog(
        {
            "display": {
                "platforms": {
                    "whatsapp": {
                        "status_phrases": {
                            "mode": "replace",
                            "status": ["custom status placeholder"],
                        }
                    }
                }
            }
        },
        "whatsapp",
    )

    assert catalog["status"] == ["custom status placeholder"]
    assert len(catalog["generic"]) > 1


def test_choose_status_phrase_uses_custom_catalog_without_leaking_args():
    catalog = resolve_status_phrase_catalog(
        {"display": {"status_phrases": {"mode": "replace", "status": ["custom safe status text"]}}},
        "whatsapp",
    )

    msg = choose_status_phrase(
        "status",
        args={"query": "SECRET SEARCH"},
        catalog=catalog,
    )

    assert msg == "custom safe status text"
    assert "SECRET" not in msg
