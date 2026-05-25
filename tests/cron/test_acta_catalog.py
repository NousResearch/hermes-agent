import json
from pathlib import Path

from cron.acta_catalog import (
    import_acta_outputs,
    load_catalog,
    normalize_entry,
    promote_output,
    save_catalog,
    slugify,
    upsert_output,
)


def test_slugify_produces_stable_output_ids():
    assert slugify("Hermes Agent Lanes & Specialist Agents") == "hermes-agent-lanes-specialist-agents"
    assert slugify("  ---  ") == "output"


def test_save_and_load_catalog_uses_atomic_json_shape(tmp_path: Path):
    path = tmp_path / "nested" / "catalog.json"
    save_catalog(
        {
            "version": 999,
            "outputs": [
                {
                    "id": "Hermes Agent Lanes Decision Tree",
                    "title": "Hermes Agent Lanes Decision Tree",
                    "href": "/outputs/hermes-agent-lanes-decision-tree",
                    "source_ref": str(tmp_path / "secret" / "hermes-agent-lanes-decision-tree.html"),
                }
            ],
        },
        path,
    )

    raw = json.loads(path.read_text())
    loaded = load_catalog(path)

    assert raw["version"] == 1
    assert loaded["outputs"][0]["id"] == "hermes-agent-lanes-decision-tree"
    assert loaded["outputs"][0]["source_ref"] == {
        "kind": "acta-output",
        "name": "hermes-agent-lanes-decision-tree.html",
    }
    assert str(tmp_path) not in path.read_text()
    assert not list(path.parent.glob("*.tmp"))


def test_upsert_preserves_user_state_fields_and_created_at():
    catalog = {
        "version": 1,
        "outputs": [
            {
                "id": "hermes-agent-lanes-decision-tree",
                "title": "Old Title",
                "href": "/outputs/hermes-agent-lanes-decision-tree",
                "summary": "old",
                "created_at": "2026-05-23T00:00:00+00:00",
                "updated_at": "2026-05-23T00:00:00+00:00",
                "pinned": True,
                "read": True,
                "archived": True,
            }
        ],
    }

    updated = upsert_output(
        catalog,
        {
            "id": "hermes-agent-lanes-decision-tree",
            "title": "Hermes Agent Lanes & Specialist Agents",
            "href": "/outputs/hermes-agent-lanes-decision-tree",
            "summary": "new summary",
            "pinned": False,
            "read": False,
            "archived": False,
        },
    )

    assert len(catalog["outputs"]) == 1
    assert updated["title"] == "Hermes Agent Lanes & Specialist Agents"
    assert updated["summary"] == "new summary"
    assert updated["created_at"] == "2026-05-23T00:00:00+00:00"
    assert updated["pinned"] is True
    assert updated["read"] is True
    assert updated["archived"] is True


def test_promote_output_pins_and_unarchives_existing_entry():
    catalog = {
        "version": 1,
        "outputs": [
            {
                "id": "roadmap",
                "title": "Roadmap",
                "href": "/outputs/roadmap",
                "archived": True,
                "pinned": False,
                "read": False,
            }
        ],
    }

    promoted = promote_output(catalog, "roadmap", read=True)

    assert promoted["pinned"] is True
    assert promoted["archived"] is False
    assert promoted["read"] is True
    assert catalog["outputs"][0]["pinned"] is True


def test_normalize_entry_redacts_public_source_refs_and_rejects_unsafe_href(tmp_path: Path):
    entry = normalize_entry(
        {
            "id": "unsafe",
            "title": "Unsafe",
            "href": "javascript:alert(1)",
            "source_ref": {"path": str(tmp_path / "private" / "unsafe.html"), "label": "Generated"},
            "tags": "Hermes, Secret Stuff",
        }
    )

    assert entry["href"] == ""
    assert entry["source_ref"] == {"kind": "acta-output", "name": "unsafe.html", "label": "Generated"}
    assert str(tmp_path) not in json.dumps(entry)
    assert entry["tags"] == ["hermes", "secret", "stuff"]


def test_import_acta_outputs_detects_decision_tree_and_is_idempotent(tmp_path: Path):
    outputs = tmp_path / "artifacts" / "acta-outputs"
    outputs.mkdir(parents=True)
    (outputs / "index.html").write_text(
        """
        <section>
          <article class="card" data-href="/outputs/hermes-agent-lanes-decision-tree"
            data-id="hermes-agent-lanes-decision-tree"
            data-title="Hermes Agent Lanes & Specialist Agents"
            data-tags="hermes telegram topics profiles kanban decision tree agents">
            <h2>Fallback title</h2>
            <p>Visual decision tree for when to use Telegram topic lanes, separate Hermes profiles, Kanban swarms, cron jobs, or truly separate Telegram-facing bots.</p>
            <div class="meta">Published 2026-05-24 16:49 UTC · Hermes / Agents / Decision Tree</div>
          </article>
        </section>
        """,
        encoding="utf-8",
    )
    (outputs / "hermes-agent-lanes-decision-tree.html").write_text(
        """
        <!doctype html><html><head><title>Fallback document title</title></head>
        <body><h1>One gateway. Many lanes.</h1><p>Document paragraph.</p></body></html>
        """,
        encoding="utf-8",
    )
    catalog_path = tmp_path / "artifacts" / "catalog.json"

    first = import_acta_outputs(outputs, catalog_path=catalog_path)
    entry = first["outputs"][0]
    entry["pinned"] = True
    entry["read"] = True
    entry["archived"] = True
    save_catalog(first, catalog_path)
    second = import_acta_outputs(outputs, catalog_path=catalog_path)

    assert [item["id"] for item in second["outputs"]] == ["hermes-agent-lanes-decision-tree"]
    imported = second["outputs"][0]
    assert imported["title"] == "Hermes Agent Lanes & Specialist Agents"
    assert imported["summary"].startswith("Visual decision tree")
    assert imported["href"] == "/outputs/hermes-agent-lanes-decision-tree"
    assert imported["created_at"] == "2026-05-24T16:49:00+00:00"
    assert {"hermes", "telegram", "profiles", "decision", "tree", "agents"}.issubset(imported["tags"])
    assert imported["source_ref"] == {
        "kind": "acta-output",
        "name": "hermes-agent-lanes-decision-tree.html",
    }
    assert imported["pinned"] is True
    assert imported["read"] is True
    assert imported["archived"] is True
    assert len(second["outputs"]) == 1
    assert str(tmp_path) not in catalog_path.read_text()


def test_load_catalog_fails_closed_for_malformed_json_and_version(tmp_path: Path):
    broken = tmp_path / "catalog.json"
    broken.write_text("{not json", encoding="utf-8")
    assert load_catalog(broken) == {"version": 1, "outputs": []}

    broken.write_text(json.dumps({"version": "bad", "outputs": []}), encoding="utf-8")
    assert load_catalog(broken) == {"version": 1, "outputs": []}

    broken.write_text(json.dumps({"version": 999, "outputs": [{"id": "future", "href": "/outputs/future"}]}), encoding="utf-8")
    assert load_catalog(broken) == {"version": 1, "outputs": []}


def test_normalize_entry_rejects_catalog_path_escape_href():
    assert normalize_entry({"id": "x", "href": "/outputs/../runs"})["href"] == ""
    assert normalize_entry({"id": "x", "href": "/outputs/%2e%2e/runs"})["href"] == ""
    assert normalize_entry({"id": "x", "href": "/outputs/safe-slug"})["href"] == "/outputs/safe-slug"


def test_normalize_entry_preserves_no_link_state_for_missing_unsafe_and_root_hrefs(tmp_path: Path):
    path = tmp_path / "catalog.json"
    save_catalog(
        {
            "version": 1,
            "outputs": [
                {"id": "missing", "title": "Missing"},
                {"id": "unsafe", "title": "Unsafe", "href": "javascript:alert(1)"},
                {"id": "root", "title": "Root", "href": "/outputs"},
                {"id": "bare", "title": "Bare", "href": "bare-slug"},
            ],
        },
        path,
    )

    loaded = {entry["id"]: entry["href"] for entry in load_catalog(path)["outputs"]}

    assert loaded == {
        "bare": "/outputs/bare-slug",
        "missing": "",
        "root": "",
        "unsafe": "",
    }
