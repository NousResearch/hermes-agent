"""Tests for cron dataflow metadata (inputs / outputs / side_effects).

Phase 1 of the cron interflow graph: crons never dispatch to each other — they
communicate through data, so each job declares typed ``scheme:value`` resource
lists and cron→cron edges are inferred from ``cron-output:<id>`` inputs. These
tests cover normalization, the declared-vs-derived cross-checks, referential
integrity, acyclicity, backfill, and the store-wide ``validate_store`` sweep.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    """Isolated cron environment with temp HERMES_HOME."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "cron").mkdir()
    (hermes_home / "cron" / "output").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import cron.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")

    return hermes_home


class TestResourceListNormalization:
    def test_string_and_list_accepted_and_sorted_deduped(self):
        from cron.jobs import _normalize_resource_list, _INPUT_SCHEMES

        out = _normalize_resource_list(
            ["wiki:b", "file:a", "wiki:b", " file:a "],
            allowed_schemes=_INPUT_SCHEMES,
            field_name="inputs",
        )
        assert out == ["file:a", "wiki:b"]

        single = _normalize_resource_list(
            "https://example.com/x",
            allowed_schemes=_INPUT_SCHEMES,
            field_name="inputs",
        )
        assert single == ["https://example.com/x"]

    def test_none_and_empty_become_empty_list(self):
        from cron.jobs import _normalize_resource_list, _OUTPUT_SCHEMES

        assert _normalize_resource_list(
            None, allowed_schemes=_OUTPUT_SCHEMES, field_name="outputs"
        ) == []
        assert _normalize_resource_list(
            ["", "   "], allowed_schemes=_OUTPUT_SCHEMES, field_name="outputs"
        ) == []

    def test_scheme_lowercased(self):
        from cron.jobs import _normalize_resource_list, _OUTPUT_SCHEMES

        assert _normalize_resource_list(
            "WIKI:Reports/Daily", allowed_schemes=_OUTPUT_SCHEMES, field_name="outputs"
        ) == ["wiki:Reports/Daily"]

    def test_missing_colon_rejected(self):
        from cron.jobs import _normalize_resource_list, _INPUT_SCHEMES

        with pytest.raises(ValueError, match="not a typed reference"):
            _normalize_resource_list(
                "just-a-value", allowed_schemes=_INPUT_SCHEMES, field_name="inputs"
            )

    def test_unknown_scheme_rejected(self):
        from cron.jobs import _normalize_resource_list, _OUTPUT_SCHEMES

        with pytest.raises(ValueError, match="unknown scheme"):
            _normalize_resource_list(
                "telegram:me", allowed_schemes=_OUTPUT_SCHEMES, field_name="outputs"
            )

    def test_missing_value_rejected(self):
        from cron.jobs import _normalize_resource_list, _INPUT_SCHEMES

        with pytest.raises(ValueError, match="missing a value"):
            _normalize_resource_list(
                "wiki:   ", allowed_schemes=_INPUT_SCHEMES, field_name="inputs"
            )

    def test_non_string_entry_rejected(self):
        from cron.jobs import _normalize_resource_list, _INPUT_SCHEMES

        with pytest.raises(ValueError, match="must be 'scheme:value'"):
            _normalize_resource_list(
                [123], allowed_schemes=_INPUT_SCHEMES, field_name="inputs"
            )


class TestCreateStoresDataflow:
    def test_fields_stored_and_normalized(self, cron_env):
        from cron.jobs import create_job, get_job

        job = create_job(
            prompt="collect",
            schedule="every 1h",
            inputs=["https://api.example.com", "wiki:notes"],
            outputs="wiki:reports/daily",
            side_effects=["telegram:me"],
        )
        assert job["inputs"] == ["https://api.example.com", "wiki:notes"]
        assert job["outputs"] == ["wiki:reports/daily"]
        assert job["side_effects"] == ["telegram:me"]

        loaded = get_job(job["id"])
        assert loaded["inputs"] == ["https://api.example.com", "wiki:notes"]
        assert loaded["outputs"] == ["wiki:reports/daily"]
        assert loaded["side_effects"] == ["telegram:me"]

    def test_defaults_to_empty_lists(self, cron_env):
        from cron.jobs import create_job

        job = create_job(prompt="hello", schedule="every 1h")
        assert job["inputs"] == []
        assert job["outputs"] == []
        assert job["side_effects"] == []

    def test_output_scheme_rejected_in_inputs(self, cron_env):
        # telegram is a side-effect scheme; not valid as an input.
        from cron.jobs import create_job

        with pytest.raises(ValueError, match="unknown scheme 'telegram'"):
            create_job(prompt="x", schedule="every 1h", inputs=["telegram:me"])


class TestDeliverSideEffectCrossCheck:
    def test_external_deliver_requires_matching_side_effect(self, cron_env):
        from cron.jobs import create_job

        with pytest.raises(ValueError, match="side_effects declares no 'telegram:'"):
            create_job(
                prompt="x",
                schedule="every 1h",
                deliver="telegram",
                side_effects=["email:team@x.com"],
            )

    def test_matching_side_effect_passes(self, cron_env):
        from cron.jobs import create_job

        job = create_job(
            prompt="x",
            schedule="every 1h",
            deliver="telegram",
            side_effects=["telegram:me"],
        )
        assert job["deliver"] == "telegram"

    def test_undeclared_side_effects_not_blocked(self, cron_env):
        # Empty declaration is the legacy path — do not hard-block existing
        # callers that deliver externally without declaring dataflow.
        from cron.jobs import create_job

        job = create_job(prompt="x", schedule="every 1h", deliver="telegram")
        assert job["side_effects"] == []


class TestContextFromCrossCheck:
    def test_context_from_must_be_declared_input(self, cron_env):
        from cron.jobs import create_job

        upstream = create_job(prompt="produce", schedule="every 1h")
        with pytest.raises(ValueError, match="not declared as inputs"):
            create_job(
                prompt="consume",
                schedule="every 2h",
                context_from=upstream["id"],
                inputs=["wiki:something-else"],
            )

    def test_context_from_with_matching_input_passes(self, cron_env):
        from cron.jobs import create_job

        upstream = create_job(prompt="produce", schedule="every 1h")
        downstream = create_job(
            prompt="consume",
            schedule="every 2h",
            context_from=upstream["id"],
            inputs=[f"cron-output:{upstream['id']}"],
        )
        assert downstream["context_from"] == [upstream["id"]]

    def test_context_from_without_declared_inputs_not_blocked(self, cron_env):
        from cron.jobs import create_job

        upstream = create_job(prompt="produce", schedule="every 1h")
        downstream = create_job(
            prompt="consume", schedule="every 2h", context_from=upstream["id"]
        )
        assert downstream["context_from"] == [upstream["id"]]
        assert downstream["inputs"] == []


class TestReferentialIntegrityAndCycles:
    def test_cron_output_input_must_exist(self, cron_env):
        from cron.jobs import create_job

        with pytest.raises(ValueError, match="unknown job"):
            create_job(
                prompt="x",
                schedule="every 1h",
                inputs=["cron-output:doesnotexist"],
            )

    def test_self_reference_rejected_via_update(self, cron_env):
        from cron.jobs import create_job, update_job

        job = create_job(prompt="x", schedule="every 1h")
        with pytest.raises(ValueError, match="cycle"):
            update_job(job["id"], {"inputs": [f"cron-output:{job['id']}"]})

    def test_two_node_cycle_rejected(self, cron_env):
        from cron.jobs import create_job, update_job

        a = create_job(prompt="a", schedule="every 1h")
        b = create_job(
            prompt="b",
            schedule="every 1h",
            inputs=[f"cron-output:{a['id']}"],
        )
        # Now make A read B's output → A→B→A cycle.
        with pytest.raises(ValueError, match="cycle"):
            update_job(a["id"], {"inputs": [f"cron-output:{b['id']}"]})

    def test_linear_chain_allowed(self, cron_env):
        from cron.jobs import create_job

        a = create_job(prompt="a", schedule="every 1h")
        b = create_job(
            prompt="b", schedule="every 1h", inputs=[f"cron-output:{a['id']}"]
        )
        c = create_job(
            prompt="c", schedule="every 1h", inputs=[f"cron-output:{b['id']}"]
        )
        assert c["inputs"] == [f"cron-output:{b['id']}"]


class TestUpdateJobDataflow:
    def test_update_normalizes_and_stores(self, cron_env):
        from cron.jobs import create_job, update_job

        job = create_job(prompt="x", schedule="every 1h")
        updated = update_job(job["id"], {"outputs": ["WIKI:reports/x", "wiki:reports/x"]})
        assert updated["outputs"] == ["wiki:reports/x"]

    def test_update_clears_with_empty_list(self, cron_env):
        from cron.jobs import create_job, update_job

        job = create_job(
            prompt="x", schedule="every 1h", outputs=["wiki:reports/x"]
        )
        updated = update_job(job["id"], {"outputs": []})
        assert updated["outputs"] == []

    def test_update_bad_scheme_rejected(self, cron_env):
        from cron.jobs import create_job, update_job

        job = create_job(prompt="x", schedule="every 1h")
        with pytest.raises(ValueError, match="unknown scheme"):
            update_job(job["id"], {"inputs": ["bogus:thing"]})


class TestBackfill:
    def test_legacy_record_backfilled_on_read(self, cron_env):
        import json

        from cron.jobs import JOBS_FILE, create_job, get_job

        job = create_job(prompt="x", schedule="every 1h")
        # Simulate a pre-feature record by stripping the fields on disk.
        data = json.loads(JOBS_FILE.read_text())
        for record in data["jobs"]:
            record.pop("inputs", None)
            record.pop("outputs", None)
            record.pop("side_effects", None)
        JOBS_FILE.write_text(json.dumps(data))

        loaded = get_job(job["id"])
        assert loaded["inputs"] == []
        assert loaded["outputs"] == []
        assert loaded["side_effects"] == []


class TestValidateStore:
    def test_clean_store_reports_no_issues(self, cron_env):
        from cron.jobs import create_job, validate_store

        a = create_job(prompt="a", schedule="every 1h")
        create_job(prompt="b", schedule="every 1h", inputs=[f"cron-output:{a['id']}"])
        assert validate_store() == []

    def test_dangling_producer_reported(self, cron_env):
        import json

        from cron.jobs import JOBS_FILE, create_job, validate_store

        a = create_job(prompt="a", schedule="every 1h")
        create_job(prompt="b", schedule="every 1h", inputs=[f"cron-output:{a['id']}"])
        # Delete the producer directly on disk (drift create/update can't catch).
        data = json.loads(JOBS_FILE.read_text())
        data["jobs"] = [r for r in data["jobs"] if r["id"] != a["id"]]
        JOBS_FILE.write_text(json.dumps(data))

        issues = validate_store()
        assert any("unknown job" in i for i in issues)

    def test_cycle_formed_on_disk_reported(self, cron_env):
        import json

        from cron.jobs import JOBS_FILE, create_job, validate_store

        a = create_job(prompt="a", schedule="every 1h")
        b = create_job(prompt="b", schedule="every 1h")
        # Hand-edit both to reference each other → cycle in the stored graph.
        data = json.loads(JOBS_FILE.read_text())
        for record in data["jobs"]:
            if record["id"] == a["id"]:
                record["inputs"] = [f"cron-output:{b['id']}"]
            elif record["id"] == b["id"]:
                record["inputs"] = [f"cron-output:{a['id']}"]
        JOBS_FILE.write_text(json.dumps(data))

        issues = validate_store()
        assert any("cycle" in i for i in issues)


class TestBuildCronGraph:
    def test_empty_store(self, cron_env):
        from cron.jobs import build_cron_graph

        graph = build_cron_graph()
        assert graph == {"nodes": [], "edges": []}

    def test_cron_node_shape(self, cron_env):
        from cron.jobs import build_cron_graph, create_job

        job = create_job(name="collector", prompt="x", schedule="every 1h")
        graph = build_cron_graph()
        cron_nodes = [n for n in graph["nodes"] if n["kind"] == "cron"]
        assert len(cron_nodes) == 1
        node = cron_nodes[0]
        assert node["id"] == job["id"]
        assert node["type"] == "cron"
        assert node["label"] == "collector"
        assert node["uses_llm"] is True

    def test_no_agent_job_marks_uses_llm_false(self, cron_env):
        import os

        from cron.jobs import build_cron_graph, create_job

        script_dir = cron_env / "scripts"
        script_dir.mkdir(parents=True, exist_ok=True)
        (script_dir / "w.sh").write_text("echo hi\n")
        os.chmod(script_dir / "w.sh", 0o755)
        create_job(prompt="", schedule="every 1h", no_agent=True, script="w.sh")
        graph = build_cron_graph()
        cron_nodes = [n for n in graph["nodes"] if n["kind"] == "cron"]
        assert cron_nodes[0]["uses_llm"] is False

    def test_source_node_and_reads_edge(self, cron_env):
        from cron.jobs import build_cron_graph, create_job

        job = create_job(
            prompt="x", schedule="every 1h", inputs=["https://api.example.com"]
        )
        graph = build_cron_graph()
        source = [n for n in graph["nodes"] if n["kind"] == "source"]
        assert source == [
            {
                "id": "https://api.example.com",
                "kind": "source",
                "type": "https",
                "label": "//api.example.com",
            }
        ]
        assert {
            "source": "https://api.example.com",
            "target": job["id"],
            "type": "reads",
        } in graph["edges"]

    def test_artifact_links_producer_to_consumer(self, cron_env):
        # A writes wiki:reports/daily; B reads it → shared artifact node with a
        # writes edge in and a reads edge out (outputs make edges).
        from cron.jobs import build_cron_graph, create_job

        a = create_job(prompt="a", schedule="every 1h", outputs=["wiki:reports/daily"])
        b = create_job(prompt="b", schedule="every 2h", inputs=["wiki:reports/daily"])
        graph = build_cron_graph()

        artifacts = [n for n in graph["nodes"] if n["kind"] == "artifact"]
        assert [n["id"] for n in artifacts] == ["wiki:reports/daily"]

        edges = graph["edges"]
        assert {"source": a["id"], "target": "wiki:reports/daily", "type": "writes"} in edges
        assert {"source": "wiki:reports/daily", "target": b["id"], "type": "reads"} in edges

    def test_cron_output_input_makes_feeds_edge(self, cron_env):
        from cron.jobs import build_cron_graph, create_job

        a = create_job(prompt="a", schedule="every 1h")
        b = create_job(
            prompt="b", schedule="every 1h", inputs=[f"cron-output:{a['id']}"]
        )
        graph = build_cron_graph()
        assert {"source": a["id"], "target": b["id"], "type": "feeds"} in graph["edges"]
        # cron-output does not spawn a resource node — it's a direct cron→cron edge.
        assert all(n["kind"] == "cron" for n in graph["nodes"])

    def test_side_effect_makes_sink_and_scheme_typed_edge(self, cron_env):
        from cron.jobs import build_cron_graph, create_job

        job = create_job(
            prompt="x",
            schedule="every 1h",
            deliver="telegram",
            side_effects=["telegram:me"],
        )
        graph = build_cron_graph()
        sinks = [n for n in graph["nodes"] if n["kind"] == "sink"]
        assert sinks == [
            {"id": "telegram:me", "kind": "sink", "type": "telegram", "label": "me"}
        ]
        assert {
            "source": job["id"],
            "target": "telegram:me",
            "type": "telegram",
        } in graph["edges"]

    def test_postgres_output_links_writer_to_reader(self, cron_env):
        # A cron that writes a postgres table and another that reads it share one
        # artifact node — postgres is a first-class data store like wiki/file.
        from cron.jobs import build_cron_graph, create_job

        w = create_job(
            prompt="ingest", schedule="every 1h", outputs=["postgres:analytics.events"]
        )
        r = create_job(
            prompt="report", schedule="every 2h", inputs=["postgres:analytics.events"]
        )
        graph = build_cron_graph()

        artifacts = [n for n in graph["nodes"] if n["kind"] == "artifact"]
        assert artifacts == [
            {
                "id": "postgres:analytics.events",
                "kind": "artifact",
                "type": "postgres",
                "label": "analytics.events",
            }
        ]
        edges = graph["edges"]
        assert {
            "source": w["id"],
            "target": "postgres:analytics.events",
            "type": "writes",
        } in edges
        assert {
            "source": "postgres:analytics.events",
            "target": r["id"],
            "type": "reads",
        } in edges

    def test_produced_resource_outranks_source(self, cron_env):
        # If a ref is both read (by one job) and written (by another), the node
        # is an artifact, not a source.
        from cron.jobs import build_cron_graph, create_job

        create_job(prompt="reader", schedule="every 1h", inputs=["file:/data/x"])
        create_job(prompt="writer", schedule="every 2h", outputs=["file:/data/x"])
        graph = build_cron_graph()
        res = [n for n in graph["nodes"] if n["id"] == "file:/data/x"]
        assert len(res) == 1
        assert res[0]["kind"] == "artifact"

    def test_service_shares_store_node_with_cron(self, cron_env):
        # A live service reading postgres:analytics.events meets the cron that
        # writes it on ONE shared artifact node — service and cron converge.
        from cron.jobs import build_cron_graph, create_job

        writer = create_job(
            prompt="ingest",
            schedule="every 1h",
            outputs=["postgres:analytics.events"],
        )
        services = [{
            "id": "proc_dash1",
            "label": "Analytics Dashboard",
            "description": "# Dashboard\nRenders analytics from the events table.",
            "inputs": ["postgres:analytics.events"],
            "outputs": [],
            "side_effects": [],
        }]
        graph = build_cron_graph(services=services)

        assert [n for n in graph["nodes"] if n["kind"] == "service"] == [{
            "id": "proc_dash1",
            "kind": "service",
            "type": "service",
            "label": "Analytics Dashboard",
            "description": "# Dashboard\nRenders analytics from the events table.",
            "source_files": [],
        }]
        stores = [n for n in graph["nodes"] if n["id"] == "postgres:analytics.events"]
        assert len(stores) == 1 and stores[0]["kind"] == "artifact"
        edges = graph["edges"]
        assert {
            "source": writer["id"],
            "target": "postgres:analytics.events",
            "type": "writes",
        } in edges
        assert {
            "source": "postgres:analytics.events",
            "target": "proc_dash1",
            "type": "reads",
        } in edges

    def test_no_services_leaves_no_service_nodes(self, cron_env):
        from cron.jobs import build_cron_graph, create_job

        create_job(prompt="x", schedule="every 1h")
        graph = build_cron_graph(services=[])
        assert not any(n["kind"] == "service" for n in graph["nodes"])


class TestServiceDeclaration:
    def test_service_health_evidence_is_exposed_on_graph_node(self):
        from cron.jobs import build_cron_graph

        health = {
            "status": "unhealthy",
            "probe": "http",
            "target": "http://127.0.0.1:9120/health",
            "checked_at": "2026-08-23T22:00:00Z",
            "latency_ms": 2000.0,
            "message": "TimeoutError: timed out",
        }
        graph = build_cron_graph(jobs=[], services=[{
            "id": "svc-meet",
            "label": "Meet pipeline",
            "description": "Conversation service.",
            "inputs": [],
            "outputs": ["http://127.0.0.1:9120"],
            "side_effects": [],
            "health": health,
        }])

        node = next(n for n in graph["nodes"] if n["id"] == "svc-meet")
        assert node["health"] == health

    def test_valid_declaration_normalizes_and_dedupes(self):
        from cron.jobs import normalize_service_declaration

        decl = normalize_service_declaration(
            name="  Analytics Dashboard  ",
            description="  Renders analytics.  ",
            inputs=["postgres:analytics.events", "postgres:analytics.events"],
        )
        assert decl == {
            "name": "Analytics Dashboard",
            "description": "Renders analytics.",
            "inputs": ["postgres:analytics.events"],
            "outputs": [],
            "side_effects": [],
            "source_files": [],
        }

    def test_declaration_normalizes_source_files(self):
        # A service declares the code behind it exactly like a cron: the paths
        # are deduped, sorted, and the `file:` scheme is stripped.
        from cron.jobs import normalize_service_declaration

        decl = normalize_service_declaration(
            name="Dashboard",
            description="Renders analytics.",
            source_files=["app/server.py", " file:app/routes/ ", "app/server.py"],
        )
        assert decl["source_files"] == ["app/routes/", "app/server.py"]

    def test_relationships_emit_explicit_subject_predicate_object_edges(self):
        from cron.jobs import build_cron_graph, normalize_service_declaration

        gateway = normalize_service_declaration(
            name="PR review gateway",
            description="Spawns isolated review runs.",
            relationships=[
                {"predicate": "spawns", "object": "workflow:github-pr-review"},
                {"predicate": "runs_in", "object": "runtime:docker"},
                {"predicate": "spawns", "object": "workflow:github-pr-review"},
            ],
        )
        gateway.update(id="nomad:pr-review-gateway", label=gateway.pop("name"))

        graph = build_cron_graph(jobs=[], services=[gateway])

        assert {
            "id": "workflow:github-pr-review",
            "kind": "object",
            "type": "workflow",
            "label": "github-pr-review",
        } in graph["nodes"]
        assert {
            "id": "runtime:docker",
            "kind": "object",
            "type": "runtime",
            "label": "docker",
        } in graph["nodes"]
        assert graph["edges"].count({
            "source": "nomad:pr-review-gateway",
            "target": "workflow:github-pr-review",
            "type": "spawns",
            "class": "relationship",
        }) == 1
        assert {
            "source": "nomad:pr-review-gateway",
            "target": "runtime:docker",
            "type": "runs_in",
            "class": "relationship",
        } in graph["edges"]

    def test_relationship_predicate_and_object_are_validated(self):
        from cron.jobs import normalize_service_declaration

        with pytest.raises(ValueError, match="predicate"):
            normalize_service_declaration(
                name="gateway",
                description="review gateway",
                relationships=[{"predicate": "Runs In", "object": "runtime:docker"}],
            )
        with pytest.raises(ValueError, match="typed object ref"):
            normalize_service_declaration(
                name="gateway",
                description="review gateway",
                relationships=[{"predicate": "runs_in", "object": "docker"}],
            )
        with pytest.raises(ValueError, match="reserved"):
            normalize_service_declaration(
                name="gateway",
                description="review gateway",
                relationships=[{"predicate": "writes", "object": "runtime:docker"}],
            )

    def test_description_required(self):
        from cron.jobs import normalize_service_declaration

        for bad in (None, "", "   "):
            with pytest.raises(ValueError, match="description is required"):
                normalize_service_declaration(name="dash", description=bad)

    def test_name_required(self):
        from cron.jobs import normalize_service_declaration

        with pytest.raises(ValueError, match="name is required"):
            normalize_service_declaration(name="   ", description="a description")

    def test_service_api_output_can_feed_another_service(self):
        from cron.jobs import build_cron_graph, normalize_service_declaration

        endpoint = "http://127.0.0.1:8081/v1"
        producer = normalize_service_declaration(
            name="MLX inference",
            description="Hosts an OpenAI-compatible API.",
            outputs=[endpoint],
        )
        consumer = normalize_service_declaration(
            name="Meet pipeline",
            description="Consumes local inference.",
            inputs=[endpoint],
        )
        producer.update(id="svc-mlx", label=producer.pop("name"))
        consumer.update(id="svc-meet", label=consumer.pop("name"))

        graph = build_cron_graph(jobs=[], services=[producer, consumer])

        assert {
            "source": "svc-mlx",
            "target": endpoint,
            "type": "writes",
        } in graph["edges"]
        assert {
            "source": endpoint,
            "target": "svc-meet",
            "type": "reads",
        } in graph["edges"]
        assert {
            "id": endpoint,
            "kind": "artifact",
            "type": "http",
            "label": "//127.0.0.1:8081/v1",
        } in graph["nodes"]

    def test_custom_resource_schemes_are_declaration_local(self):
        from cron.jobs import normalize_service_declaration

        decl = normalize_service_declaration(
            name="Event worker",
            description="Moves events between infrastructure boundaries.",
            inputs=["kafka:events.raw", "redis:cache/session"],
            outputs=["s3:warehouse/events", "kafka:events.enriched"],
        )

        assert decl["inputs"] == ["kafka:events.raw", "redis:cache/session"]
        assert decl["outputs"] == ["kafka:events.enriched", "s3:warehouse/events"]

    def test_side_effect_scheme_stays_out_of_resource_fields(self):
        from cron.jobs import normalize_service_declaration

        with pytest.raises(ValueError, match="terminal side-effect scheme"):
            normalize_service_declaration(
                name="dash", description="d", inputs=["telegram:me"]
            )

    def test_reserved_cron_output_scheme_stays_out_of_outputs(self):
        from cron.jobs import normalize_service_declaration

        with pytest.raises(ValueError, match="reserved"):
            normalize_service_declaration(
                name="dash", description="d", outputs=["cron-output:job-1"]
            )

    def test_malformed_custom_scheme_rejected(self):
        from cron.jobs import normalize_service_declaration

        with pytest.raises(ValueError, match="invalid scheme"):
            normalize_service_declaration(
                name="dash", description="d", outputs=["not a scheme:value"]
            )


class TestCronGraphRPC:
    def test_handler_returns_ok_envelope(self, cron_env):
        from cron.jobs import create_job

        create_job(prompt="x", schedule="every 1h", side_effects=["notify:desktop"])

        # Invoke the registered handler directly with a stubbed _ok/_err.
        import tui_gateway.methods_tools as mt

        handler = dict(mt._registry._pending)["cron.graph"]
        captured = {}

        def _ok(rid, result):
            captured["rid"] = rid
            captured["result"] = result
            return {"result": result}

        def _err(rid, code, msg):  # pragma: no cover - failure path
            raise AssertionError(f"handler errored: {code} {msg}")

        handler.__globals__["_ok"] = _ok
        handler.__globals__["_err"] = _err
        handler.__globals__.setdefault("logger", __import__("logging").getLogger("t"))

        handler(7, {})
        assert captured["rid"] == 7
        assert set(captured["result"].keys()) == {"nodes", "edges"}
        assert any(n["kind"] == "cron" for n in captured["result"]["nodes"])


class TestSourceFiles:
    """`source_files`: the code behind a job, declared + mechanical, on its node."""

    def test_normalization_dedupes_sorts_and_strips_file_scheme(self):
        from cron.jobs import _normalize_source_files

        out = _normalize_source_files(
            ["ingest.py", " file:indexing/x.py ", "ingest.py", "", "~/.hermes/scripts/a.sh"]
        )
        assert out == ["indexing/x.py", "ingest.py", "~/.hermes/scripts/a.sh"]
        assert _normalize_source_files("one.py") == ["one.py"]
        assert _normalize_source_files(None) == []
        assert _normalize_source_files([]) == []

    def test_normalization_rejects_non_paths(self):
        from cron.jobs import _normalize_source_files

        with pytest.raises(ValueError, match="URL"):
            _normalize_source_files(["https://example.com/x.py"])
        with pytest.raises(ValueError, match="NUL"):
            _normalize_source_files(["bad\x00.py"])
        with pytest.raises(ValueError, match="path strings"):
            _normalize_source_files([42])
        with pytest.raises(ValueError, match="string or list"):
            _normalize_source_files({"a": 1})

    def test_create_stores_and_update_normalizes(self, cron_env):
        from cron.jobs import create_job, get_job, update_job

        job = create_job(prompt="x", schedule="every 1h", source_files=["b.py", "a.py"])
        assert job["source_files"] == ["a.py", "b.py"]

        update_job(job["id"], {"source_files": " file:c.py "})
        assert get_job(job["id"])["source_files"] == ["c.py"]

        update_job(job["id"], {"source_files": []})
        assert get_job(job["id"])["source_files"] == []

    def test_legacy_record_backfills_empty_list(self, cron_env):
        import json

        from cron.jobs import JOBS_FILE, create_job, get_job

        job = create_job(prompt="x", schedule="every 1h")
        data = json.loads(JOBS_FILE.read_text())
        for record in data["jobs"]:
            record.pop("source_files", None)
        JOBS_FILE.write_text(json.dumps(data))

        assert get_job(job["id"])["source_files"] == []

    def test_graph_node_merges_script_fields_with_declared(self, cron_env, monkeypatch):
        import os

        import cron.jobs as jobs_mod
        from cron.jobs import build_cron_graph, create_job

        scripts = cron_env / "scripts"
        scripts.mkdir(parents=True, exist_ok=True)
        (scripts / "w.sh").write_text("echo hi\n")
        os.chmod(scripts / "w.sh", 0o755)
        (scripts / "helper.py").write_text("print(1)\n")
        # Pin the browse roots so the assertion doesn't depend on where this
        # checkout lives; the nested `repo` root exercises deepest-root wins.
        repo = cron_env / "hermes-agent"
        (repo / "indexing").mkdir(parents=True)
        (repo / "indexing" / "x.py").write_text("pass\n")
        monkeypatch.setattr(
            jobs_mod, "_source_file_roots", lambda: {"hermes": cron_env, "repo": repo}
        )

        job = create_job(
            prompt="",
            schedule="every 1h",
            no_agent=True,
            script="w.sh",
            source_files=["w.sh", "helper.py", str(repo / "indexing" / "x.py"), "/tmp/elsewhere/gone.py"],
        )
        node = next(n for n in build_cron_graph()["nodes"] if n["id"] == job["id"])
        files = node["source_files"]

        by_path = {entry["path"]: entry for entry in files}
        # The script comes first with the mechanical role, and the declared
        # duplicate of it collapsed into that one entry.
        assert files[0]["role"] == "script"
        assert files[0]["declared"] == "w.sh"
        assert files[0]["root"] == "hermes"
        assert files[0]["rel"] == "scripts/w.sh"
        assert files[0]["exists"] is True
        assert sum(1 for e in files if e["path"].endswith("/scripts/w.sh")) == 1

        helper = by_path[str((scripts / "helper.py").resolve())]
        assert helper["role"] == "declared"
        assert helper["root"] == "hermes"
        assert helper["rel"] == "scripts/helper.py"

        nested = by_path[str((repo / "indexing" / "x.py").resolve())]
        assert nested["root"] == "repo"  # deepest containing root, not hermes
        assert nested["rel"] == "indexing/x.py"

        outside = next(e for e in files if e["declared"] == "/tmp/elsewhere/gone.py")
        assert outside["root"] is None
        assert outside["rel"] is None
        assert outside["exists"] is False

    def test_monitor_script_has_its_own_role(self, cron_env):
        import os

        from cron.jobs import build_cron_graph, create_job

        scripts = cron_env / "scripts"
        scripts.mkdir(parents=True, exist_ok=True)
        (scripts / "probe.sh").write_text("echo x\n")
        os.chmod(scripts / "probe.sh", 0o755)
        job = create_job(prompt="watch", schedule="every 1h", monitor_script="probe.sh")
        node = next(n for n in build_cron_graph()["nodes"] if n["id"] == job["id"])
        assert [e["role"] for e in node["source_files"]] == ["monitor"]

    def test_job_without_code_has_empty_list(self, cron_env):
        from cron.jobs import build_cron_graph, create_job

        job = create_job(prompt="x", schedule="every 1h")
        node = next(n for n in build_cron_graph()["nodes"] if n["id"] == job["id"])
        assert node["source_files"] == []

    def test_service_node_resolves_declared_source_files(self, cron_env, monkeypatch):
        # A service node carries the same resolved source_files as a cron node,
        # so its code is browsable in the graph: paths under a browse root map
        # to root/rel/exists via the identical resolver (deepest root wins).
        import cron.jobs as jobs_mod
        from cron.jobs import build_cron_graph

        repo = cron_env / "hermes-agent"
        (repo / "app").mkdir(parents=True)
        (repo / "app" / "server.py").write_text("pass\n")
        monkeypatch.setattr(
            jobs_mod, "_source_file_roots", lambda: {"hermes": cron_env, "repo": repo}
        )

        graph = build_cron_graph(services=[{
            "id": "proc_dash1",
            "label": "Dashboard",
            "description": "serves the app",
            "inputs": [],
            "outputs": [],
            "side_effects": [],
            "source_files": [str(repo / "app" / "server.py"), "/tmp/gone.py"],
        }])
        node = next(n for n in graph["nodes"] if n["id"] == "proc_dash1")
        files = node["source_files"]

        served = next(e for e in files if e["declared"].endswith("app/server.py"))
        assert served["role"] == "declared"
        assert served["root"] == "repo"  # deepest containing root, not hermes
        assert served["rel"] == "app/server.py"
        assert served["exists"] is True

        outside = next(e for e in files if e["declared"] == "/tmp/gone.py")
        assert outside["root"] is None
        assert outside["exists"] is False

    def test_default_roots_come_from_the_file_browser(self, cron_env):
        from cron.jobs import _source_file_roots

        roots = _source_file_roots()
        assert "repo" in roots
        assert all(root.is_absolute() for root in roots.values())

    def test_source_files_stay_out_of_the_commitment(self, cron_env):
        # Portal hashes the node row with the same fields as cron/changesets.py;
        # neither includes source_files, so declaring code must not move the digest.
        from cron.changesets import configuration_digest
        from cron.jobs import build_cron_graph, create_job, update_job

        job = create_job(prompt="x", schedule="every 1h")
        before = configuration_digest(build_cron_graph())
        update_job(job["id"], {"source_files": ["a.py"]})
        assert configuration_digest(build_cron_graph()) == before

    def test_tool_list_reports_declared_files(self, cron_env):
        import json

        from tools.cronjob_tools import cronjob

        created = json.loads(cronjob(action="create", prompt="x", schedule="every 1h", source_files=["a.py"]))
        assert created["success"] is True
        listed = json.loads(cronjob(action="list"))
        assert listed["jobs"][0]["source_files"] == ["a.py"]

        updated = json.loads(cronjob(action="update", job_id=created["job"]["job_id"], source_files=["b.py"]))
        assert updated["success"] is True
        assert json.loads(cronjob(action="list"))["jobs"][0]["source_files"] == ["b.py"]


def _cron_manage_handler():
    """The registered `cron.manage` handler with `_ok`/`_err` stubbed to return
    plain dicts, so a test can assert on either envelope."""
    import tui_gateway.methods_tools as mt

    handler = dict(mt._registry._pending)["cron.manage"]
    handler.__globals__["_ok"] = lambda rid, result: {"rid": rid, "result": result}
    handler.__globals__["_err"] = lambda rid, code, msg: {"rid": rid, "error": {"code": code, "message": msg}}
    handler.__globals__.setdefault("logger", __import__("logging").getLogger("t"))
    # The server rebinds handler globals at install time (that's where `json`
    # comes from in production); a direct invocation has to supply it.
    handler.__globals__.setdefault("json", __import__("json"))
    return handler


class TestCronManageRPC:
    """The person-facing door: describe / history / update, which `list` alone
    can't stand in for (it caps the prompt at a 100-char preview)."""

    def test_describe_returns_the_full_prompt(self, cron_env):
        from cron.jobs import create_job

        long_prompt = "p" * 240
        job = create_job(prompt=long_prompt, schedule="every 1h", source_files=["a.py"])
        out = _cron_manage_handler()(1, {"action": "describe", "name": job["id"]})

        detail = out["result"]["job"]
        assert out["result"]["success"] is True
        assert detail["job_id"] == job["id"]
        assert detail["prompt"] == long_prompt
        assert detail["prompt_preview"].endswith("...")
        assert detail["source_files"] == ["a.py"]
        assert detail["source_files_resolved"][0]["declared"] == "a.py"

    def test_describe_unknown_job_is_a_not_found_error(self, cron_env):
        out = _cron_manage_handler()(2, {"action": "describe", "name": "nope"})
        assert out["error"]["code"] == 4404

    def test_describe_requires_a_name(self, cron_env):
        out = _cron_manage_handler()(3, {"action": "describe"})
        assert out["error"]["code"] == 4001

    def test_update_prompt_persists(self, cron_env):
        from cron.jobs import create_job, get_job

        job = create_job(prompt="old", schedule="every 1h")
        out = _cron_manage_handler()(4, {"action": "update", "name": job["id"], "prompt": "new prompt text"})
        assert out["result"]["success"] is True
        assert get_job(job["id"])["prompt"] == "new prompt text"

    def test_update_renames_via_job_name(self, cron_env):
        from cron.jobs import create_job, get_job

        job = create_job(prompt="x", schedule="every 1h", name="db-backup")
        _cron_manage_handler()(5, {"action": "update", "name": job["id"], "job_name": "infra/db-backup"})
        assert get_job(job["id"])["name"] == "infra/db-backup"

    def test_update_source_files_and_rejects_empty_update(self, cron_env):
        from cron.jobs import create_job, get_job

        job = create_job(prompt="x", schedule="every 1h")
        _cron_manage_handler()(6, {"action": "update", "name": job["id"], "source_files": ["z.py"]})
        assert get_job(job["id"])["source_files"] == ["z.py"]

        out = _cron_manage_handler()(7, {"action": "update", "name": job["id"]})
        assert out["error"]["code"] == 4001

    def test_update_failure_surfaces_as_error(self, cron_env):
        out = _cron_manage_handler()(8, {"action": "update", "name": "missing", "prompt": "x"})
        assert out["error"]["code"] == 4017

    def test_history_returns_ledger_envelope(self, cron_env):
        from cron.executions import create_execution, finish_execution, mark_execution_running
        from cron.jobs import create_job

        job = create_job(prompt="x", schedule="every 1h", name="collector")
        execution = create_execution(job["id"], source="test")
        mark_execution_running(execution["id"])
        finish_execution(execution["id"], success=True)

        out = _cron_manage_handler()(9, {"action": "history", "name": job["id"], "limit": 5})
        result = out["result"]
        assert result["success"] is True
        assert result["job_id"] == job["id"]
        assert result["job_name"] == "collector"
        assert result["count"] == 1
        assert result["runs"][0]["status"] == "completed"
        assert result["runs"][0]["claimed_at"]

    def test_unknown_action_still_rejected(self, cron_env):
        out = _cron_manage_handler()(10, {"action": "explode", "name": "x"})
        assert out["error"]["code"] == 4016
