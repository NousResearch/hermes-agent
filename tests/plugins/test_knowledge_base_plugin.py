import json

from hermes_cli.config import DEFAULT_CONFIG
from hermes_cli.tools_config import _DEFAULT_OFF_TOOLSETS
from plugins import knowledge_base as kb


def _use_home(monkeypatch, tmp_path):
    monkeypatch.setattr(kb, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(kb, "_load_kb_config", lambda: {"path": "knowledge-base"})


def test_knowledge_base_write_read_list_and_search(monkeypatch, tmp_path):
    _use_home(monkeypatch, tmp_path)

    written = json.loads(kb._kb_write({
        "path": "domain/widgets",
        "content": "# Widget Rules\n\nRoute premium widgets to the review queue.",
    }))
    assert written["success"] is True
    assert written["path"] == "domain/widgets.md"

    appended = json.loads(kb._kb_write({
        "path": "domain/widgets.md",
        "content": "Capture supplier notes separately.",
        "mode": "append",
    }))
    assert appended["success"] is True

    read = json.loads(kb._kb_read({"path": "domain/widgets"}))
    assert read["path"] == "domain/widgets.md"
    assert "review queue" in read["content"]
    assert "supplier notes" in read["content"]

    listed = json.loads(kb._kb_list({"path": "domain"}))
    assert listed["notes"] == ["domain/widgets.md"]

    searched = json.loads(kb._kb_search({"query": "premium widgets"}))
    assert searched["results"][0]["path"] == "domain/widgets.md"
    assert searched["results"][0]["title"] == "Widget Rules"


def test_knowledge_base_uses_configured_path(monkeypatch, tmp_path):
    configured = tmp_path / "vault"
    monkeypatch.setattr(kb, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
        "knowledge_base": {"path": "vault"},
    })

    payload = json.loads(kb._kb_write({"path": "decisions/launch", "content": "Ship it."}))

    assert payload["success"] is True
    assert (configured / "decisions" / "launch.md").read_text(encoding="utf-8") == "Ship it."


def test_knowledge_base_rejects_paths_outside_root(monkeypatch, tmp_path):
    _use_home(monkeypatch, tmp_path)

    payload = json.loads(kb._kb_read({"path": "../secrets"}))

    assert payload["success"] is False
    assert "must not contain" in payload["error"]


def test_knowledge_base_search_accepts_uppercase_markdown_suffix(monkeypatch, tmp_path):
    _use_home(monkeypatch, tmp_path)
    root = tmp_path / "knowledge-base"
    root.mkdir()
    (root / "UPPER.MD").write_text("# Upper\n\nSearchable knowledge.", encoding="utf-8")

    payload = json.loads(kb._kb_search({"query": "searchable"}))

    assert [item["path"] for item in payload["results"]] == ["UPPER.MD"]


def test_knowledge_base_config_is_registered():
    assert DEFAULT_CONFIG["knowledge_base"] == {"path": "knowledge-base"}


def test_knowledge_base_registers_opt_in_toolset():
    calls = []

    class Context:
        def register_tool(self, **kwargs):
            calls.append(kwargs)

    kb.register(Context())

    assert [call["name"] for call in calls] == ["kb_read", "kb_write", "kb_search", "kb_list"]
    assert {call["toolset"] for call in calls} == {"knowledge_base"}
    assert "knowledge_base" in _DEFAULT_OFF_TOOLSETS
