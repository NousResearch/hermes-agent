from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.csrf import generate_csrf_token


def _headers() -> dict[str, str]:
    return {
        web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN,
        "X-CSRF-Token": generate_csrf_token(web_server._SESSION_TOKEN),
    }


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _client(tmp_path: Path, monkeypatch) -> TestClient:
    vault = tmp_path / "HermesAgent"
    _write(
        vault / "MOC.md",
        "---\n"
        "status: active\n"
        "owner: Nat\n"
        "tags:\n"
        "  - dashboard\n"
        "  - knowledge\n"
        "---\n"
        "# Hermes Agent\n\n"
        "## Operating Map\n\n"
        "See [[10-Knowledge/Operating Rules]].",
    )
    _write(
        vault / "10-Knowledge" / "Operating Rules.md",
        "# Operating Rules\n\nKnowledge rules for [[20-Departments/HR]].",
    )
    _write(vault / "20-Departments" / "HR.md", "# HR\n\nPeople systems.")
    _write(vault / "95-Inbox-Lab" / "review" / "candidate.md", "# Candidate\n")
    _write(vault / ".env", "TOKEN=secret")
    _write(vault / ".obsidian" / "workspace.json", "{}")
    _write(vault / "90-Owner-Private" / "profile" / "secret.md", "# private")
    monkeypatch.setenv("HERMES_OBSIDIAN_VAULT", str(vault))
    monkeypatch.delenv("HERMES_OWNER", raising=False)
    return TestClient(web_server.app)


def test_knowledge_status_reports_read_only_vault(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.get("/api/knowledge/status", headers=_headers())

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["read_only"] is True
    assert payload["vault_name"] == "HermesAgent"
    assert payload["safe_file_count"] == 4


def test_knowledge_tree_excludes_sensitive_and_owner_only_paths(
    tmp_path: Path, monkeypatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.get("/api/knowledge/tree", headers=_headers())
    assert response.status_code == 200
    names = {item["name"] for item in response.json()["items"]}

    assert "10-Knowledge" in names
    assert "20-Departments" in names
    assert "MOC.md" in names
    assert ".env" not in names
    assert ".obsidian" not in names
    assert "90-Owner-Private" not in names


def test_knowledge_file_reads_safe_note_and_extracts_links(
    tmp_path: Path, monkeypatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.get(
        "/api/knowledge/file",
        params={"path": "10-Knowledge/Operating Rules.md"},
        headers=_headers(),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["path"] == "10-Knowledge/Operating Rules.md"
    assert payload["title"] == "Operating Rules"
    assert "Knowledge rules" in payload["content"]
    assert payload["links"] == ["20-Departments/HR"]


def test_knowledge_file_extracts_frontmatter_and_headings(
    tmp_path: Path, monkeypatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.get(
        "/api/knowledge/file",
        params={"path": "MOC.md"},
        headers=_headers(),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["frontmatter"] == {
        "status": "active",
        "owner": "Nat",
        "tags": ["dashboard", "knowledge"],
    }
    assert payload["headings"] == [
        {"level": 1, "title": "Hermes Agent", "slug": "hermes-agent", "line": 8},
        {"level": 2, "title": "Operating Map", "slug": "operating-map", "line": 10},
    ]


def test_knowledge_file_blocks_path_traversal_and_sensitive_files(
    tmp_path: Path, monkeypatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    traversal = client.get(
        "/api/knowledge/file",
        params={"path": "../outside.md"},
        headers=_headers(),
    )
    env_file = client.get(
        "/api/knowledge/file",
        params={"path": ".env"},
        headers=_headers(),
    )
    owner_file = client.get(
        "/api/knowledge/file",
        params={"path": "90-Owner-Private/profile/secret.md"},
        headers=_headers(),
    )

    assert traversal.status_code == 400
    assert env_file.status_code == 400
    assert owner_file.status_code == 400


def test_knowledge_search_and_backlinks_stay_inside_safe_vault(
    tmp_path: Path, monkeypatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    search = client.get(
        "/api/knowledge/search",
        params={"q": "People"},
        headers=_headers(),
    )
    backlinks = client.get(
        "/api/knowledge/backlinks",
        params={"path": "20-Departments/HR.md"},
        headers=_headers(),
    )

    assert search.status_code == 200
    assert [item["path"] for item in search.json()["items"]] == ["20-Departments/HR.md"]
    assert backlinks.status_code == 200
    assert [item["path"] for item in backlinks.json()["items"]] == [
        "10-Knowledge/Operating Rules.md"
    ]


def test_knowledge_resolve_links_supports_stem_and_safe_paths(
    tmp_path: Path, monkeypatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    by_stem = client.get(
        "/api/knowledge/resolve",
        params={"link": "Operating Rules", "from_path": "MOC.md"},
        headers=_headers(),
    )
    by_path = client.get(
        "/api/knowledge/resolve",
        params={
            "link": "20-Departments/HR",
            "from_path": "10-Knowledge/Operating Rules.md",
        },
        headers=_headers(),
    )
    blocked = client.get(
        "/api/knowledge/resolve",
        params={"link": "90-Owner-Private/profile/secret"},
        headers=_headers(),
    )

    assert by_stem.status_code == 200
    assert by_stem.json()["path"] == "10-Knowledge/Operating Rules.md"
    assert by_path.status_code == 200
    assert by_path.json()["path"] == "20-Departments/HR.md"
    assert blocked.status_code == 404


def test_knowledge_save_is_disabled_by_default(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.put(
        "/api/knowledge/file",
        params={"path": "MOC.md"},
        json={"content": "# Changed"},
        headers=_headers(),
    )

    assert response.status_code == 403


def test_knowledge_save_requires_fresh_modified_and_creates_backup(
    tmp_path: Path, monkeypatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_KNOWLEDGE_WRITE", "1")
    backup_root = tmp_path / "knowledge-backups"
    monkeypatch.setenv("HERMES_KNOWLEDGE_BACKUP_PATH", str(backup_root))

    current = client.get(
        "/api/knowledge/file", params={"path": "MOC.md"}, headers=_headers()
    )
    assert current.status_code == 200
    modified = current.json()["modified"]

    stale = client.put(
        "/api/knowledge/file",
        params={"path": "MOC.md"},
        json={"content": "# Stale", "expected_modified": modified - 1},
        headers=_headers(),
    )
    assert stale.status_code == 409

    saved = client.put(
        "/api/knowledge/file",
        params={"path": "MOC.md"},
        json={"content": "# Changed\n\nSaved safely.", "expected_modified": modified},
        headers=_headers(),
    )

    assert saved.status_code == 200
    payload = saved.json()
    assert payload["content"].startswith("# Changed")
    assert payload["write_enabled"] is True
    assert payload["backup_path"].endswith(".bak")
    assert (backup_root / payload["backup_path"]).exists()
    assert "# Hermes Agent" in (backup_root / payload["backup_path"]).read_text(
        encoding="utf-8"
    )


def test_knowledge_graph_returns_local_note_neighborhood(
    tmp_path: Path, monkeypatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.get(
        "/api/knowledge/graph",
        params={"path": "20-Departments/HR.md"},
        headers=_headers(),
    )

    assert response.status_code == 200
    payload = response.json()
    node_ids = {node["id"] for node in payload["nodes"]}
    edge_pairs = {(edge["source"], edge["target"]) for edge in payload["edges"]}

    assert "20-Departments/HR.md" in node_ids
    assert "10-Knowledge/Operating Rules.md" in node_ids
    assert ("10-Knowledge/Operating Rules.md", "20-Departments/HR.md") in edge_pairs


def test_knowledge_global_graph_returns_vault_wide_map(
    tmp_path: Path, monkeypatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.get(
        "/api/knowledge/global-graph",
        params={"limit": 4, "edge_limit": 10},
        headers=_headers(),
    )

    assert response.status_code == 200
    payload = response.json()
    node_ids = {node["id"] for node in payload["nodes"]}
    edge_pairs = {(edge["source"], edge["target"]) for edge in payload["edges"]}

    assert payload["mode"] == "global"
    assert payload["path"] == ""
    assert payload["limit"] == 4
    assert payload["node_count"] == 4
    assert len(payload["nodes"]) <= 4
    assert "90-Owner-Private/profile/secret.md" not in node_ids
    assert "MOC.md" in node_ids
    assert "10-Knowledge/Operating Rules.md" in node_ids
    assert ("MOC.md", "10-Knowledge/Operating Rules.md") in edge_pairs


def test_knowledge_graph_and_backlinks_resolve_relative_links(
    tmp_path: Path, monkeypatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    vault = tmp_path / "HermesAgent"
    _write(
        vault / "10-Knowledge" / "lessons" / "daily.md",
        "# Daily\n\nSee [[../patterns/review-loop]].",
    )
    _write(
        vault / "10-Knowledge" / "patterns" / "review-loop.md",
        "# Review Loop\n\nUsed for daily work.",
    )

    graph = client.get(
        "/api/knowledge/graph",
        params={"path": "10-Knowledge/lessons/daily.md"},
        headers=_headers(),
    )
    backlinks = client.get(
        "/api/knowledge/backlinks",
        params={"path": "10-Knowledge/patterns/review-loop.md"},
        headers=_headers(),
    )

    assert graph.status_code == 200
    graph_payload = graph.json()
    edge_pairs = {(edge["source"], edge["target"]) for edge in graph_payload["edges"]}
    assert "10-Knowledge/patterns/review-loop.md" in {
        node["id"] for node in graph_payload["nodes"]
    }
    assert (
        "10-Knowledge/lessons/daily.md",
        "10-Knowledge/patterns/review-loop.md",
    ) in edge_pairs

    assert backlinks.status_code == 200
    assert [item["path"] for item in backlinks.json()["items"]] == [
        "10-Knowledge/lessons/daily.md"
    ]


def test_knowledge_search_matches_file_paths_with_hyphenated_tokens(
    tmp_path: Path, monkeypatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    vault = tmp_path / "HermesAgent"
    _write(
        vault / "10-Knowledge" / "lessons" / "auto-daily-2026-05-15.md",
        "# Daily Lesson\n\nCaptured lesson.",
    )

    response = client.get(
        "/api/knowledge/search",
        params={"q": "auto daily"},
        headers=_headers(),
    )

    assert response.status_code == 200
    assert [item["path"] for item in response.json()["items"]] == [
        "10-Knowledge/lessons/auto-daily-2026-05-15.md"
    ]


def test_knowledge_graph_depth_expands_second_hop_links(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    vault = tmp_path / "HermesAgent"
    _write(vault / "10-Knowledge" / "lessons" / "daily.md", "# Daily\n\nSee [[Router]].")
    _write(vault / "Router.md", "# Router\n\nSee [[Skill Graph]].")
    _write(vault / "Skill Graph.md", "# Skill Graph\n\nSecond hop target.")

    response = client.get(
        "/api/knowledge/graph",
        params={"path": "10-Knowledge/lessons/daily.md", "depth": 2},
        headers=_headers(),
    )

    assert response.status_code == 200
    payload = response.json()
    node_ids = {node["id"] for node in payload["nodes"]}
    edge_pairs = {(edge["source"], edge["target"]) for edge in payload["edges"]}

    assert payload["depth"] == 2
    assert "Router.md" in node_ids
    assert "Skill Graph.md" in node_ids
    assert ("10-Knowledge/lessons/daily.md", "Router.md") in edge_pairs
    assert ("Router.md", "Skill Graph.md") in edge_pairs


def test_knowledge_graph_respects_node_limit(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    vault = tmp_path / "HermesAgent"
    links = " ".join(f"[[Note {index}]]" for index in range(10))
    _write(vault / "Hub.md", f"# Hub\n\n{links}")
    for index in range(10):
        _write(vault / f"Note {index}.md", f"# Note {index}\n")

    response = client.get(
        "/api/knowledge/graph",
        params={"path": "Hub.md", "depth": 2, "limit": 6},
        headers=_headers(),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["limit"] == 6
    assert len(payload["nodes"]) <= 6
