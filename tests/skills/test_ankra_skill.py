"""Hermetic tests for the optional-skills/devops/ankra skill.

Covers SKILL.md frontmatter invariants (AGENTS.md hardline standards), the
dual-surface contract, and — critically — that the documented MCP tool names
are real. The Ankra MCP surface is generated from a fixed tool catalog
(cluster/go/internal/agentictools + mcpserverapi/toolsurface.go); this test
pins the SKILL.md doc to that ground truth so a fabricated tool name fails CI.

No live network calls — all assertions run against the SKILL.md source.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "devops" / "ankra"
SKILL_MD = SKILL_DIR / "SKILL.md"

yaml = pytest.importorskip("yaml")

# Ground truth: tool names that exist on the real Ankra MCP catalog. This is a
# curated subset of the ~125-entry catalog covering exactly the tools the doc
# is allowed to name. If the doc names anything outside this set, it is either
# fabricated or renamed upstream — either way the doc must be corrected.
REAL_MCP_TOOLS = {
    # read
    "list_clusters", "get_cluster_details", "get_cluster_status",
    "get_cluster_cost", "get_stack_cost", "get_security_reports",
    "list_recent_executions", "get_execution_details",
    "get_pods", "get_pod_logs", "get_deployments", "get_services",
    "get_events", "get_nodes", "get_secrets", "describe_resource",
    "list_addons", "get_addon_details", "get_addon_history",
    "list_stacks", "get_stack_details", "list_available_charts",
    "get_chart_schema", "list_provider_credentials", "list_instance_types",
    "query_prometheus", "query_prometheus_range",
    "github_list_workflow_runs", "github_get_job_logs",
    # write
    "restart_deployment", "scale_deployment", "scale_statefulset",
    "addon_install", "sync_addon", "patch_resource", "delete_pod",
    "apply_manifest", "delete_resource", "secret_rotation",
    "rollback_addon", "helm_release_rollback", "apply_stack_changes",
    "delete_stack", "uninstall_addon_from_stack", "create_hetzner_cluster",
    "add_helm_registry", "github_commit_files", "github_create_pull_request",
}

# Tool names that must NOT appear as MCP tools — they were fabricated in an
# earlier draft or belong to surfaces MCP does not expose (token/credential
# management, deprovision, cancel, cluster reconcile).
FORBIDDEN_MCP_TOOLS = {
    "reconcile_cluster", "deprovision_cluster", "cancel_operation",
    "create_token", "revoke_token", "delete_token",
    "create_credential", "delete_credential", "install_addon",
    "update_addon", "uninstall_addon", "create_stack", "clone_stack",
    "rollback_stack", "provision_cluster", "get_agent_status",
    "list_nodes", "list_operations", "get_operation", "list_manifests",
    "list_available_addons", "get_addon", "get_cluster",
    "list_helm_releases", "uninstall_helm_release", "list_tokens",
    "list_credentials",
}


def _parse_frontmatter(text: str) -> dict:
    match = re.search(r"^---\n(.*?)\n---", text, re.DOTALL)
    assert match, "SKILL.md has no valid YAML frontmatter block"
    return yaml.safe_load(match.group(1))


def _mcp_section(text: str) -> str:
    """The MCP surface section only (Surface 1, up to Surface 2)."""
    m = re.search(r"# Surface 1:(.*?)# Surface 2:", text, re.DOTALL)
    assert m, "MCP surface section not found"
    return m.group(1)


def _cli_section(text: str) -> str:
    m = re.search(r"# Surface 2:(.*)", text, re.DOTALL)
    assert m, "CLI surface section not found"
    return m.group(1)


# ---------------------------------------------------------------------------
# Frontmatter invariants (AGENTS.md hardline standards)
# ---------------------------------------------------------------------------

class TestFrontmatter:
    def setup_method(self):
        self.fm = _parse_frontmatter(SKILL_MD.read_text())

    def test_description_length(self):
        desc = self.fm["description"]
        assert len(desc) <= 60, f"description is {len(desc)} chars (limit 60): {desc!r}"

    def test_description_ends_with_period(self):
        assert self.fm["description"].endswith(".")

    def test_description_no_marketing_words(self):
        bad = {"powerful", "comprehensive", "seamless", "advanced", "robust"}
        found = bad & set(re.findall(r"[a-z]+", self.fm["description"].lower()))
        assert not found, f"description contains marketing words: {found}"

    def test_author_credits_human_first(self):
        first = self.fm["author"].split(",")[0].strip()
        assert first != "Hermes Agent"
        assert "@" in first or " " in first

    def test_platforms_present(self):
        assert "platforms" in self.fm and len(self.fm["platforms"]) >= 1

    def test_name_matches_directory(self):
        assert self.fm["name"] == SKILL_DIR.name

    def test_version_semver(self):
        parts = str(self.fm["version"]).split(".")
        assert len(parts) == 3 and all(p.isdigit() for p in parts)

    def test_hermes_tags_present(self):
        tags = self.fm.get("metadata", {}).get("hermes", {}).get("tags", [])
        assert len(tags) >= 1


# ---------------------------------------------------------------------------
# MCP tool names must be real, and forbidden names must never reappear
# ---------------------------------------------------------------------------

class TestMcpToolNamesAreReal:
    def setup_method(self):
        self.text = SKILL_MD.read_text()
        self.mcp = _mcp_section(self.text)
        # Bare tool names appear in table cells as `snake_case_ident`.
        self.named = set(re.findall(r"`([a-z][a-z0-9_]+)`", self.mcp))

    def test_no_forbidden_tool_names_anywhere(self):
        """Fabricated / non-MCP tool names must not appear in the whole doc."""
        # Strip the mcp_ankra_ prefix form too, in case someone re-adds it.
        hits = {t for t in FORBIDDEN_MCP_TOOLS
                if re.search(rf"`(mcp_ankra_)?{re.escape(t)}`", self.text)}
        assert not hits, f"Forbidden / fabricated MCP tool names present: {sorted(hits)}"

    def test_named_mcp_tools_are_all_real(self):
        """Every snake_case backticked token in the MCP section that looks like
        a tool must be a real catalog tool (or an allowed non-tool word)."""
        allowed_non_tools = {
            "cluster_id", "mcp_ankra", "tools_list", "save_memory",
            "ankra_mcp_token", "read", "write",
        }
        candidates = {n for n in self.named if "_" in n} - allowed_non_tools
        # A candidate is a claimed tool if it's not obviously prose.
        unknown = {n for n in candidates
                   if n not in REAL_MCP_TOOLS and not n.startswith("mcp_")}
        assert not unknown, f"MCP section names non-catalog tools: {sorted(unknown)}"

    def test_list_clusters_documented(self):
        assert "list_clusters" in self.named

    def test_high_risk_writes_documented(self):
        for tool in ["apply_manifest", "delete_resource", "delete_stack",
                     "create_hetzner_cluster", "secret_rotation"]:
            assert tool in self.named, f"expected high-risk write tool {tool!r} documented"


# ---------------------------------------------------------------------------
# Surface separation: token/credential/deprovision are CLI-only
# ---------------------------------------------------------------------------

class TestSurfaceSeparation:
    def setup_method(self):
        self.text = SKILL_MD.read_text()
        self.cli = _cli_section(self.text)
        self.mcp = _mcp_section(self.text)

    def test_token_rotation_is_cli_only(self):
        # Rotation flow lives in the CLI section...
        assert "ankra tokens create" in self.cli
        assert "ankra tokens revoke" in self.cli
        # ...and must not be presented as MCP tools.
        assert "create_token" not in self.mcp and "revoke_token" not in self.mcp

    def test_deprovision_is_cli(self):
        assert "ankra cluster deprovision" in self.cli

    def test_mcp_notes_what_it_lacks(self):
        # The MCP section must tell the reader token/credential/deprovision
        # are not on the MCP surface, so the agent doesn't hunt for them.
        assert re.search(r"[Nn]ot available over MCP|no MCP tool|not on the MCP",
                         self.mcp)

    def test_ordering_rule_documented(self):
        assert re.search(r"before .* revok", self.text, re.IGNORECASE | re.DOTALL)


# ---------------------------------------------------------------------------
# Structure & CLI reference
# ---------------------------------------------------------------------------

class TestStructure:
    def setup_method(self):
        self.text = SKILL_MD.read_text()

    def test_both_surfaces_declared(self):
        assert re.search(r"^# Surface 1: MCP", self.text, re.MULTILINE)
        assert re.search(r"^# Surface 2: .*CLI", self.text, re.MULTILINE)

    def test_mcp_endpoint(self):
        assert "https://platform.ankra.app/api/v1/mcp" in self.text

    def test_scope_gate_documented(self):
        assert "mcp:read" in self.text and "mcp:write" in self.text

    def test_when_to_use_and_checklist(self):
        assert "## When to Use" in self.text
        assert "## Verification Checklist" in self.text

    def test_cli_run_via_terminal(self):
        assert "`terminal`" in self.text

    def test_core_cli_commands(self):
        for cmd in ["ankra login", "ankra cluster list", "ankra tokens create",
                    "ankra credentials list"]:
            assert cmd in self.text, f"CLI reference missing: {cmd!r}"

    def test_pitfalls_cover_auth_and_rate(self):
        assert "401" in self.text and "403" in self.text
        assert re.search(r"rate limit", self.text, re.IGNORECASE)

    def test_audit_and_secret_rules(self):
        assert re.search(r"audit[\s-]+log", self.text)
        assert re.search(r"secret", self.text, re.IGNORECASE)
