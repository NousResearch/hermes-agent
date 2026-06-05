"""Tests for scripts/code-scan/static_signals.py (TDD RED first).

Strictly follows bead: ua-tier1-001-static-signals-schema.
- Expect build_static_signals_artifact and signal record helpers.
- Stdlib only in impl.
- Core contract: heuristic_signal + not_validated unless Tier 0 fact.
- Empty artifact shape per spec.
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "code-scan"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Import will fail in RED phase
from static_signals import (
    build_static_signals_artifact,
    extract_edge_function_markers,
    extract_package_config_markers,
    extract_supabase_migration_markers,
    make_signal_record,
    SignalRecord,
)


class TestSchemaConstants:
    """Ensure core constants / enums are exposed per schema contract."""

    def test_claim_type_heuristic(self):
        # The returned claim_type must be exactly this for non-Tier0 signals
        artifact = build_static_signals_artifact([])
        assert artifact["claim_type"] == "heuristic_signal"

    def test_semantic_status_not_validated(self):
        artifact = build_static_signals_artifact([])
        assert artifact["semantic_status"] == "not_validated"


class TestEmptyArtifact:
    """Exact empty artifact shape per bead spec."""

    def test_empty_artifact_shape(self):
        artifact = build_static_signals_artifact([])
        expected = {
            "schema_version": "1.0.0",
            "claim_type": "heuristic_signal",
            "semantic_status": "not_validated",
            "signals": [],
            "summary": {"total_signals": 0, "by_surface": {}, "by_marker_type": {}},
            "boundaries": [
                "Tier 1 static signals are content markers only; they do not prove security, RLS correctness, auth correctness, runtime behavior, deployment readiness, CI success, or policy semantics."
            ],
        }
        assert artifact == expected
        # Also roundtrip stable
        assert json.loads(json.dumps(artifact)) == expected


class TestSignalRecordHelper:
    """Tests for make_signal_record and SignalRecord (if dataclass or dict helper)."""

    def test_make_signal_record_minimal(self):
        rec = make_signal_record(
            surface="code",
            path="src/foo.py",
            line=42,
            marker_type="supabase_rls_comment",
            marker="row_level_security",
            claim_type="heuristic_signal",
            semantic_status="not_validated",
            boundary="Tier 1 static signals are content markers only.",
        )
        assert isinstance(rec, dict)
        assert rec["surface"] == "code"
        assert rec["path"] == "src/foo.py"
        assert rec["line"] == 42
        assert rec["marker_type"] == "supabase_rls_comment"
        assert rec["marker"] == "row_level_security"
        assert rec["claim_type"] == "heuristic_signal"
        assert rec["semantic_status"] == "not_validated"
        assert "boundary" in rec
        # SignalRecord alias should work too
        rec2 = SignalRecord(
            surface="sql",
            path="migrations/001.sql",
            line=10,
            marker_type="migration_file",
            marker="-- rls",
            claim_type="heuristic_signal",
            semantic_status="not_validated",
            boundary="content marker",
        )
        assert rec2["surface"] == "sql"

    def test_make_signal_record_defaults(self):
        # Helper should accept minimal kwargs and supply safe defaults
        rec = make_signal_record(
            surface="config",
            path="supabase/config.toml",
            line=1,
            marker_type="edge_function_config",
            marker="functions",
        )
        assert rec["claim_type"] == "heuristic_signal"
        assert rec["semantic_status"] == "not_validated"
        assert rec["boundary"] is not None
        assert "content markers only" in rec["boundary"]
        assert "do not prove security" in rec["boundary"]


class TestArtifactWithSignals:
    """Building artifact with one or more signals."""

    def test_artifact_includes_signals(self):
        sig1 = make_signal_record(
            surface="code",
            path="app/models.py",
            line=15,
            marker_type="supabase_client_init",
            marker="create_client",
        )
        artifact = build_static_signals_artifact([sig1])
        assert artifact["schema_version"] == "1.0.0"
        assert artifact["claim_type"] == "heuristic_signal"
        assert artifact["semantic_status"] == "not_validated"
        assert len(artifact["signals"]) == 1
        assert artifact["signals"][0]["path"] == "app/models.py"
        assert artifact["summary"]["total_signals"] == 1
        assert "code" in artifact["summary"]["by_surface"]
        assert artifact["summary"]["by_surface"]["code"] == 1
        assert "supabase_client_init" in artifact["summary"]["by_marker_type"]
        assert len(artifact["boundaries"]) >= 1
        assert any("does not prove security" in b or "RLS correctness" in b for b in artifact["boundaries"])

    def test_multiple_signals_summary(self):
        sigs = [
            make_signal_record(surface="code", path="a.py", line=1, marker_type="m1", marker="x"),
            make_signal_record(surface="code", path="b.py", line=2, marker_type="m1", marker="y"),
            make_signal_record(surface="sql", path="s.sql", line=3, marker_type="m2", marker="z"),
        ]
        artifact = build_static_signals_artifact(sigs)
        assert artifact["summary"]["total_signals"] == 3
        assert artifact["summary"]["by_surface"]["code"] == 2
        assert artifact["summary"]["by_surface"]["sql"] == 1
        assert artifact["summary"]["by_marker_type"]["m1"] == 2
        assert artifact["summary"]["by_marker_type"]["m2"] == 1


class TestBoundaryContract:
    """Explicit evidence-boundary enforcement tests per bead."""

    def test_every_non_tier0_claim_is_heuristic_and_not_validated(self):
        sig = make_signal_record(
            surface="docs",
            path="README.md",
            line=5,
            marker_type="supabase_url_comment",
            marker="https://...supabase.co",
        )
        artifact = build_static_signals_artifact([sig])
        for s in artifact["signals"]:
            assert s["claim_type"] == "heuristic_signal"
            assert s["semantic_status"] == "not_validated"
        assert artifact["claim_type"] == "heuristic_signal"
        assert artifact["semantic_status"] == "not_validated"

    def test_boundaries_contains_required_disclaimer_text(self):
        artifact = build_static_signals_artifact([])
        boundaries_text = " ".join(artifact["boundaries"])
        assert "Tier 1 static signals are content markers only" in boundaries_text
        assert "do not prove security" in boundaries_text
        assert "RLS correctness" in boundaries_text
        assert "auth correctness" in boundaries_text
        assert "runtime behavior" in boundaries_text
        assert "deployment readiness" in boundaries_text
        assert "CI success" in boundaries_text
        assert "policy semantics" in boundaries_text

    def test_signals_never_contain_validated_claims(self):
        # Even if caller tries to pass something, canonical builder forces heuristic/not_validated
        bad_sig = {
            "surface": "code",
            "path": "evil.py",
            "line": 1,
            "marker_type": "fake",
            "marker": "x",
            "claim_type": "concrete_proof",  # attempt to overclaim
            "semantic_status": "validated",
            "boundary": "ignore",
        }
        artifact = build_static_signals_artifact([bad_sig])
        for s in artifact["signals"]:
            assert s["claim_type"] == "heuristic_signal"
            assert s["semantic_status"] == "not_validated"


class TestSupabaseMigrationMarkers:
    """Supabase SQL migration markers are inventory only, not validation claims."""

    def test_extracts_all_required_marker_types_with_lines(self):
        sql = "\n".join(
            [
                "create table public.todos (id uuid primary key);",
                "alter table public.todos enable row level security;",
                "create policy \"read todos\"",
                "on public.todos",
                "for select",
                "to authenticated",
                "using (auth.uid() = user_id);",
                "create policy \"insert todos\"",
                "on public.todos",
                "for insert",
                "to anon",
                "with check (auth.role() = 'authenticated');",
                "create policy \"public read\" on public.todos for select using (true);",
                "drop policy if exists \"old todos\" on public.todos;",
                "create function public.jwt_role() returns text",
                "language sql",
                "security definer",
                "as $$ select auth.role(); $$;",
                "grant select on public.todos to authenticated;",
                "revoke all on public.todos from anon;",
                "grant usage on schema private to service_role;",
            ]
        )

        signals = extract_supabase_migration_markers(
            "supabase/migrations/001_rls.sql",
            sql,
        )

        marker_types = {signal["marker_type"] for signal in signals}
        assert marker_types == {
            "enable_rls",
            "create_policy",
            "drop_policy",
            "using_clause",
            "with_check_clause",
            "auth_uid",
            "auth_role",
            "anon_role",
            "authenticated_role",
            "permissive_true",
            "security_definer",
            "service_role",
            "grant_statement",
            "revoke_statement",
            "create_function",
        }

        first_line_by_type = {}
        for signal in signals:
            first_line_by_type.setdefault(signal["marker_type"], signal["line"])

        assert first_line_by_type["enable_rls"] == 2
        assert first_line_by_type["create_policy"] == 3
        assert first_line_by_type["authenticated_role"] == 6
        assert first_line_by_type["using_clause"] == 7
        assert first_line_by_type["auth_uid"] == 7
        assert first_line_by_type["anon_role"] == 11
        assert first_line_by_type["with_check_clause"] == 12
        assert first_line_by_type["auth_role"] == 12
        assert first_line_by_type["permissive_true"] == 13
        assert first_line_by_type["drop_policy"] == 14
        assert first_line_by_type["create_function"] == 15
        assert first_line_by_type["security_definer"] == 17
        assert first_line_by_type["grant_statement"] == 19
        assert first_line_by_type["revoke_statement"] == 20
        assert first_line_by_type["service_role"] == 21

        for signal in signals:
            assert signal["surface"] == "supabase_migration"
            assert signal["path"] == "supabase/migrations/001_rls.sql"
            assert isinstance(signal["line"], int)
            assert signal["line"] >= 1
            assert signal["marker"].strip()
            assert signal["claim_type"] == "heuristic_signal"
            assert signal["semantic_status"] == "not_validated"
            assert "content markers only" in signal["boundary"]
            assert "do not prove security" in signal["boundary"]
            assert "policy semantics" in signal["boundary"]

    def test_equivalent_nested_supabase_migration_path_is_allowed(self):
        signals = extract_supabase_migration_markers(
            "apps/web/supabase/migrations/20260605010101_policy.sql",
            "create policy p on public.todos for select using (true);",
        )
        assert [signal["marker_type"] for signal in signals] == [
            "create_policy",
            "using_clause",
            "permissive_true",
        ]
        assert {signal["path"] for signal in signals} == {
            "apps/web/supabase/migrations/20260605010101_policy.sql"
        }

    def test_non_migration_paths_emit_no_markers(self):
        sql = "alter table public.todos enable row level security;"
        assert extract_supabase_migration_markers("supabase/schema.sql", sql) == []
        assert extract_supabase_migration_markers("migrations/001_rls.sql", sql) == []
        assert extract_supabase_migration_markers("supabase/migrations/readme.md", sql) == []

    def test_benign_sql_migration_emits_no_markers(self):
        sql = "\n".join(
            [
                "create table public.todos (id uuid primary key);",
                "insert into public.todos (id) values ('00000000-0000-0000-0000-000000000000');",
                "create index todos_id_idx on public.todos (id);",
            ]
        )
        assert extract_supabase_migration_markers("supabase/migrations/002_benign.sql", sql) == []

    def test_marker_cap_limits_each_marker_type_independently(self):
        sql = "\n".join(
            [
                "grant select on public.todos to authenticated;",
                "grant insert on public.todos to authenticated;",
                "grant update on public.todos to authenticated;",
                "revoke delete on public.todos from anon;",
                "revoke truncate on public.todos from anon;",
                "create policy p on public.todos for select using (true);",
            ]
        )

        signals = extract_supabase_migration_markers(
            "supabase/migrations/003_cap.sql",
            sql,
            max_per_type=2,
        )

        grant_lines = [signal["line"] for signal in signals if signal["marker_type"] == "grant_statement"]
        revoke_lines = [signal["line"] for signal in signals if signal["marker_type"] == "revoke_statement"]
        assert grant_lines == [1, 2]
        assert revoke_lines == [4, 5]


class TestEdgeFunctionMarkers:
    """Supabase Edge Function markers are static content inventory only."""

    def test_extracts_required_edge_function_marker_types_with_lines(self):
        content = "\n".join(
            [
                "import { serve } from 'https://deno.land/std/http/server.ts';",
                "const serviceKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY');",
                "const supabaseUrl = Deno.env.get('SUPABASE_URL');",
                "serve(async (req) => {",
                "  const authHeader = req.headers.get('Authorization');",
                "  const token = authHeader?.replace('Bearer ', '');",
                "  const jwt = token;",
                "  const { data: user } = await supabase.auth.getUser(jwt);",
                "  const payload = await req.json();",
                "  const upstream = await fetch('https://api.example.test/data');",
                "  return new Response(JSON.stringify({ payload, upstream }), {",
                "    headers: { 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Headers': 'authorization' },",
                "  });",
                "});",
            ]
        )

        signals = extract_edge_function_markers(
            "supabase/functions/example/index.ts",
            content,
        )

        marker_types = {signal["marker_type"] for signal in signals}
        assert marker_types == {
            "authorization_header",
            "bearer_token",
            "jwt",
            "get_user",
            "service_role_env",
            "deno_env",
            "cors_header",
            "cors_wildcard",
            "request_json",
            "external_fetch",
        }

        first_line_by_type = {}
        for signal in signals:
            first_line_by_type.setdefault(signal["marker_type"], signal["line"])

        assert first_line_by_type["service_role_env"] == 2
        assert first_line_by_type["deno_env"] == 2
        assert first_line_by_type["authorization_header"] == 5
        assert first_line_by_type["bearer_token"] == 6
        assert first_line_by_type["jwt"] == 7
        assert first_line_by_type["get_user"] == 8
        assert first_line_by_type["request_json"] == 9
        assert first_line_by_type["external_fetch"] == 10
        assert first_line_by_type["cors_header"] == 12
        assert first_line_by_type["cors_wildcard"] == 12

        artifact = build_static_signals_artifact(signals)
        assert artifact["summary"]["total_signals"] == len(signals)
        assert artifact["summary"]["by_surface"] == {"supabase_edge_function": len(signals)}
        assert artifact["summary"]["by_marker_type"]["authorization_header"] == 1
        assert artifact["summary"]["by_marker_type"]["deno_env"] == 2
        assert artifact["summary"]["by_marker_type"]["cors_header"] == 1

        for signal in artifact["signals"]:
            assert signal["surface"] == "supabase_edge_function"
            assert signal["path"] == "supabase/functions/example/index.ts"
            assert signal["claim_type"] == "heuristic_signal"
            assert signal["semantic_status"] == "not_validated"
            assert signal["semantic_status"] != "executed_external_gate"
            assert "content markers only" in signal["boundary"]

    def test_edge_markers_are_restricted_to_function_index_files(self):
        content = "Deno.env.get('SUPABASE_SERVICE_ROLE_KEY'); await req.json();"
        assert extract_edge_function_markers("supabase/functions/example/index.ts", content)
        assert extract_edge_function_markers("supabase/functions/example/index.js", content)
        assert extract_edge_function_markers("supabase/functions/example/helper.ts", content) == []
        assert extract_edge_function_markers("supabase/functions/example/nested/index.ts", content) == []
        assert extract_edge_function_markers("apps/web/supabase/functions/example/index.ts", content) == []
        assert extract_edge_function_markers("supabase/functions/example/index.tsx", content) == []


class TestPackageConfigMarkers:
    """Package/config markers identify declared gates, not executed gates."""

    def test_extracts_package_json_script_markers_with_lines(self):
        content = "\n".join(
            [
                "{",
                '  "scripts": {',
                '    "test": "vitest run",',
                '    "build": "vite build",',
                '    "lint": "eslint .",',
                '    "typecheck": "tsc --noEmit",',
                '    "audit": "npm audit --audit-level=high"',
                "  }",
                "}",
            ]
        )

        signals = extract_package_config_markers("package.json", content)

        assert [signal["marker_type"] for signal in signals] == [
            "script_test",
            "script_build",
            "script_lint",
            "script_typecheck",
            "script_audit",
        ]
        assert [signal["line"] for signal in signals] == [3, 4, 5, 6, 7]

        artifact = build_static_signals_artifact(signals)
        assert artifact["summary"]["total_signals"] == 5
        assert artifact["summary"]["by_surface"] == {"package_config": 5}
        assert artifact["summary"]["by_marker_type"] == {
            "script_test": 1,
            "script_build": 1,
            "script_lint": 1,
            "script_typecheck": 1,
            "script_audit": 1,
        }
        for signal in artifact["signals"]:
            assert signal["claim_type"] == "heuristic_signal"
            assert signal["semantic_status"] == "not_validated"
            assert signal["semantic_status"] != "executed_external_gate"
            assert "declared gates only" in signal["boundary"]
            assert "do not prove the gates were executed or passed" in signal["boundary"]

    def test_declared_npm_test_is_not_an_executed_external_gate(self):
        signals = extract_package_config_markers(
            "package.json",
            '{"scripts":{"test":"npm run unit"}}',
        )
        artifact = build_static_signals_artifact(signals)

        assert len(artifact["signals"]) == 1
        signal = artifact["signals"][0]
        assert signal["marker_type"] == "script_test"
        assert signal["claim_type"] == "heuristic_signal"
        assert signal["semantic_status"] == "not_validated"
        assert signal["semantic_status"] != "executed_external_gate"
        assert "executed_external_gate" not in json.dumps(artifact)

    def test_extracts_workflow_vite_and_hosting_config_markers(self):
        workflow = "\n".join(
            [
                "name: ci",
                "jobs:",
                "  test:",
                "    steps:",
                "      - run: npm ci",
                "      - run: npm test",
                "      - run: npm run build",
                "      - run: npm run typecheck",
            ]
        )
        vite = "\n".join(
            [
                "import { defineConfig } from 'vite';",
                "export default defineConfig({});",
                "console.log(import.meta.env.VITE_PUBLIC_SUPABASE_URL);",
            ]
        )

        signals = []
        signals.extend(extract_package_config_markers(".github/workflows/ci.yml", workflow))
        signals.extend(extract_package_config_markers("vite.config.ts", vite))
        signals.extend(extract_package_config_markers("vercel.json", '{"rewrites": []}'))
        signals.extend(extract_package_config_markers("netlify.toml", "[build]\ncommand = 'npm run build'"))

        marker_types = [signal["marker_type"] for signal in signals]
        assert marker_types == [
            "ci_npm_ci",
            "ci_test",
            "ci_build",
            "ci_typecheck",
            "vite_public_env",
            "vercel_config",
            "netlify_config",
        ]
        assert [signal["line"] for signal in signals] == [5, 6, 7, 8, 3, 1, 1]

        artifact = build_static_signals_artifact(signals)
        assert artifact["summary"]["total_signals"] == 7
        assert artifact["summary"]["by_surface"] == {"package_config": 7}
        assert artifact["summary"]["by_marker_type"]["ci_npm_ci"] == 1
        assert artifact["summary"]["by_marker_type"]["vite_public_env"] == 1
        assert artifact["summary"]["by_marker_type"]["vercel_config"] == 1
        assert artifact["summary"]["by_marker_type"]["netlify_config"] == 1

    def test_package_config_markers_are_restricted_to_allowed_paths(self):
        package_content = '{"scripts":{"test":"vitest"}}'
        workflow_content = "- run: npm ci"
        vite_content = "import.meta.env.VITE_SUPABASE_URL"

        assert extract_package_config_markers("package.json", package_content)
        assert extract_package_config_markers(".github/workflows/ci.yml", workflow_content)
        assert extract_package_config_markers(".github/workflows/ci.yaml", workflow_content)
        assert extract_package_config_markers("vite.config.js", vite_content)
        assert extract_package_config_markers("vite.config.mts", vite_content)
        assert extract_package_config_markers("vercel.json", "{}")
        assert extract_package_config_markers("netlify.toml", "[build]")

        assert extract_package_config_markers("apps/web/package.json", package_content) == []
        assert extract_package_config_markers(".github/actions/ci.yml", workflow_content) == []
        assert extract_package_config_markers(".github/workflows/ci.txt", workflow_content) == []
        assert extract_package_config_markers("src/vite.config.ts", vite_content) == []
        assert extract_package_config_markers("vercel.dev.json", "{}") == []
        assert extract_package_config_markers("netlify.ini", "[build]") == []

# ── TDD RED/GREEN for UA-T1-006-rust-agent-infra-signal-coverage ─────────

class TestRustAgentInfraMarkers:
    """Rust/coding-agent infrastructure repo static signal coverage (heuristic only).

    All emitted signals MUST be claim_type='heuristic_signal' and
    semantic_status='not_validated'. No security claims.
    """

    def test_extracts_all_required_rust_agent_infra_surfaces(self):
        """Detect deterministic line-oriented content markers for all 8 required surfaces."""
        from static_signals import extract_rust_agent_infra_markers

        fixture_root = PROJECT_ROOT / "tests" / "code_scan" / "fixtures" / "static_signals_rust_agent_infra"

        # Read files
        cargo = (fixture_root / "Cargo.toml").read_text(encoding="utf-8")
        robot_md = (fixture_root / "docs" / "ROBOT_MODE.md").read_text(encoding="utf-8")
        sec_md = (fixture_root / "docs" / "SECURITY_AUDIT_REPORT.md").read_text(encoding="utf-8")
        hermes_rs = (fixture_root / "src" / "connectors" / "hermes.rs").read_text(encoding="utf-8")
        pack_rs = (fixture_root / "src" / "search" / "pack_planner.rs").read_text(encoding="utf-8")
        sync_rs = (fixture_root / "src" / "sources" / "sync.rs").read_text(encoding="utf-8")
        models_rs = (fixture_root / "src" / "daemon" / "models.rs").read_text(encoding="utf-8")
        ci_yml = (fixture_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

        signals = []
        signals.extend(extract_rust_agent_infra_markers("Cargo.toml", cargo))
        signals.extend(extract_rust_agent_infra_markers("docs/ROBOT_MODE.md", robot_md))
        signals.extend(extract_rust_agent_infra_markers("docs/SECURITY_AUDIT_REPORT.md", sec_md))
        signals.extend(extract_rust_agent_infra_markers("src/connectors/hermes.rs", hermes_rs))
        signals.extend(extract_rust_agent_infra_markers("src/search/pack_planner.rs", pack_rs))
        signals.extend(extract_rust_agent_infra_markers("src/sources/sync.rs", sync_rs))
        signals.extend(extract_rust_agent_infra_markers("src/daemon/models.rs", models_rs))
        signals.extend(extract_rust_agent_infra_markers(".github/workflows/ci.yml", ci_yml))

        surface_set = {s["surface"] for s in signals}
        required_surfaces = {
            "agent_robot_api_surface",
            "session_history_privacy_surface",
            "remote_sync_surface",
            "model_embedding_surface",
            "crypto_security_surface",
            "multi_agent_connector_surface",
            "ci_supply_chain_surface",
            "custom_runtime_dependency_surface",
        }
        assert required_surfaces.issubset(surface_set), f"Missing surfaces: {required_surfaces - surface_set}"

        # All must be strict Tier-1 heuristic
        for s in signals:
            assert s["claim_type"] == "heuristic_signal"
            assert s["semantic_status"] == "not_validated"
            assert "content markers only" in s["boundary"]
            assert "do not prove security" in s["boundary"]

        # At least one marker per surface
        counts = {}
        for s in signals:
            counts[s["surface"]] = counts.get(s["surface"], 0) + 1
        for surf in required_surfaces:
            assert counts.get(surf, 0) >= 1, f"No signals for {surf}"

    def test_extract_rust_agent_infra_markers_respects_max_per_type(self):
        from static_signals import extract_rust_agent_infra_markers
        content = "ssh2\nssh2\nssh2\nsmtp\nsftp\nsftp\n"
        signals = extract_rust_agent_infra_markers("src/sync.rs", content, max_per_type=1)
        ssh_signals = [s for s in signals if "ssh" in s["marker"].lower() or s["marker_type"] == "remote_ssh"]
        assert len(ssh_signals) <= 1

    def test_non_rust_agent_paths_emit_no_rust_signals(self):
        from static_signals import extract_rust_agent_infra_markers
        assert extract_rust_agent_infra_markers("src/main.rs", "fn main() {}") == []
        assert extract_rust_agent_infra_markers("README.md", "Some docs") == []
        assert extract_rust_agent_infra_markers("pyproject.toml", "[tool]") == []
