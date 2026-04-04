from argparse import Namespace

from hermes_cli import memory_cli


def test_memory_export_and_import_commands_roundtrip(tmp_path, monkeypatch, capsys):
    home = tmp_path / ".hermes"
    monkeypatch.setattr(memory_cli, "get_hermes_home", lambda: home)

    store = memory_cli._store()
    store.add_entry("memory", "Portable fact", kind="lesson")
    store.add_entry("user", "User prefers concise replies", kind="preference")

    output = tmp_path / "snapshot.json"
    memory_cli.memory_command(Namespace(memory_action="export", output=str(output), input=None))
    export_out = capsys.readouterr().out
    assert output.exists()
    assert "Exported memory snapshot" in export_out

    imported_home = tmp_path / ".hermes-imported"
    monkeypatch.setattr(memory_cli, "get_hermes_home", lambda: imported_home)
    memory_cli.memory_command(Namespace(memory_action="import", output=None, input=str(output)))
    import_out = capsys.readouterr().out
    assert "Imported memory snapshot" in import_out

    imported_store = memory_cli._store()
    assert [e["content"] for e in imported_store.list_entries("memory")] == ["Portable fact"]
    assert [e["content"] for e in imported_store.list_entries("user")] == ["User prefers concise replies"]


def test_memory_status_reports_db_and_exports(tmp_path, monkeypatch, capsys):
    home = tmp_path / ".hermes"
    monkeypatch.setattr(memory_cli, "get_hermes_home", lambda: home)
    store = memory_cli._store()
    store.add_entry("memory", "Fact A", kind="lesson")
    store.add_entry("user", "Fact B", kind="preference")

    memory_cli.memory_command(Namespace(memory_action="status", output=None, input=None))
    out = capsys.readouterr().out
    assert "memory.db" in out
    assert "active entries: 2" in out
    assert "MEMORY.md" in out
    assert "USER.md" in out


def test_memory_status_reports_hot_selection_reasons(tmp_path, monkeypatch, capsys):
    home = tmp_path / ".hermes"
    monkeypatch.setattr(memory_cli, "get_hermes_home", lambda: home)
    store = memory_cli._store()
    store.add_entry(
        "user",
        "Never use flattery or padding",
        entry_type="prohibition",
        strength="hard_rule",
        source="user_explicit",
        importance=0.6,
    )
    store.add_entry(
        "user",
        "User likes emoji sometimes",
        entry_type="user_preference",
        strength="soft_pref",
        source="user_explicit",
        importance=0.95,
    )

    memory_cli.memory_command(Namespace(memory_action="status", output=None, input=None))
    out = capsys.readouterr().out

    assert "Hot user memory selection:" in out
    assert "Never use flattery or padding" in out
    assert "path: hot_memory" in out
    assert "strength=hard_rule" in out
    assert "entry_type=prohibition" in out


def test_memory_status_reports_typed_entry_breakdown(tmp_path, monkeypatch, capsys):
    home = tmp_path / ".hermes"
    monkeypatch.setattr(memory_cli, "get_hermes_home", lambda: home)
    store = memory_cli._store()
    store.add_entry(
        "memory",
        "Project uses sqlite for local state",
        entry_type="project_convention",
        strength="contextual",
        source="user_explicit",
        importance=0.5,
    )
    store.add_entry(
        "user",
        "Never use flattery or padding",
        entry_type="prohibition",
        strength="hard_rule",
        source="user_explicit",
        importance=0.6,
    )

    memory_cli.memory_command(Namespace(memory_action="status", output=None, input=None))
    out = capsys.readouterr().out

    assert "Type breakdown:" in out
    assert "memory: project_convention=1" in out
    assert "user: prohibition=1" in out


def test_memory_status_reports_hot_memory_selection(tmp_path, monkeypatch, capsys):
    home = tmp_path / ".hermes"
    monkeypatch.setattr(memory_cli, "get_hermes_home", lambda: home)
    store = memory_cli._store()
    store.add_entry(
        "memory",
        "Project uses sqlite for local state",
        entry_type="project_convention",
        strength="contextual",
        source="user_explicit",
        importance=0.5,
    )
    store.add_entry(
        "memory",
        "Never expose secrets in chat",
        entry_type="constraint",
        strength="hard_rule",
        source="user_explicit",
        importance=0.4,
    )

    memory_cli.memory_command(Namespace(memory_action="status", output=None, input=None))
    out = capsys.readouterr().out

    assert "Hot memory selection:" in out
    assert "Never expose secrets in chat" in out
    assert "path: hot_memory" in out
    assert "entry_type=constraint" in out


def test_memory_status_reports_retrieval_split(tmp_path, monkeypatch, capsys):
    home = tmp_path / ".hermes"
    monkeypatch.setattr(memory_cli, "get_hermes_home", lambda: home)
    store = memory_cli._store()
    store.add_entry("user", "Never use flattery or padding")

    memory_cli.memory_command(Namespace(memory_action="status", output=None, input=None))
    out = capsys.readouterr().out

    assert "Recall paths:" in out
    assert "hot memory: persistent steering facts" in out
    assert "session_search: transcript/session history" in out
