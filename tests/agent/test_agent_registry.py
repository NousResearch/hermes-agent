"""Tests for agent/agent_registry.py — native agent registry discovery and validation."""

import os
import tempfile
from pathlib import Path

import pytest

# Import the module under test
from agent.agent_registry import (
    AgentDefinition,
    AgentRouting,
    AgentTools,
    AgentSkills,
    AgentLimits,
    AgentSecurity,
    AgentCompatibility,
    discover_agent_dirs,
    get_agent,
    list_agents,
    load_agent_file,
    validate_agent_name,
    AgentLoadError,
    SECRET_LIKE_FIELD_NAMES,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME for the duration of a test."""
    hh = tmp_path / ".hermes"
    hh.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hh))
    return hh


@pytest.fixture
def project_root(tmp_path):
    """A fake git-backed project root."""
    pr = tmp_path / "my-project"
    pr.mkdir()
    (pr / ".git").mkdir()
    return pr


@pytest.fixture
def global_agents_dir(hermes_home):
    """Global agents directory."""
    d = hermes_home / "agents"
    d.mkdir()
    return d


@pytest.fixture
def project_agents_dir(project_root):
    """Project-local agents directory."""
    d = project_root / ".hermes" / "agents"
    d.mkdir(parents=True)
    return d


def make_agent(path: Path, name: str, description: str = "A test agent", **frontmatter):
    """Write a minimal valid agent file."""
    fm = {"schema_version": 1, "name": name, "description": description}
    fm.update(frontmatter)
    lines = ["---", *[f"{k}: {v}" for k, v in fm.items()], "---", "", "This is the agent prompt."]
    path.write_text("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Name validation
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateAgentName:
    def test_valid_simple(self):
        assert validate_agent_name("qa-engineer") == "qa-engineer"

    def test_valid_with_numbers(self):
        assert validate_agent_name("code-explorer-2") == "code-explorer-2"

    def test_valid_min_length(self):
        assert validate_agent_name("a") == "a"

    def test_valid_max_length_64(self):
        assert validate_agent_name("a" * 64) == "a" * 64

    def test_rejects_empty(self):
        with pytest.raises(AgentLoadError):
            validate_agent_name("")

    def test_rejects_uppercase(self):
        with pytest.raises(AgentLoadError):
            validate_agent_name("BadName")

    def test_rejects_spaces(self):
        with pytest.raises(AgentLoadError):
            validate_agent_name("has space")

    def test_rejects_path_separator(self):
        with pytest.raises(AgentLoadError):
            validate_agent_name("foo/bar")

    def test_rejects_double_dot(self):
        with pytest.raises(AgentLoadError):
            validate_agent_name("..evil")

    def test_rejects_hidden(self):
        with pytest.raises(AgentLoadError):
            validate_agent_name(".hidden")

    def test_rejects_slashes(self):
        with pytest.raises(AgentLoadError):
            validate_agent_name("foo\\bar")

    def test_rejects_too_long(self):
        with pytest.raises(AgentLoadError):
            validate_agent_name("a" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Secret-like fields
# ─────────────────────────────────────────────────────────────────────────────

class TestSecretLikeFields:
    def test_rejects_api_key(self):
        content = (
            "---\n"
            "schema_version: 1\n"
            "name: secret-agent\n"
            "description: test\n"
            "routing:\n"
            "  api_key: sk-123456\n"
            "---\n"
            "prompt\n"
        )
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write(content)
            path = Path(f.name)

        try:
            with pytest.raises(AgentLoadError, match="secret"):
                load_agent_file(path, "global")
        finally:
            path.unlink()

    def test_rejects_token_field(self):
        content = (
            "---\n"
            "schema_version: 1\n"
            "name: token-agent\n"
            "description: test\n"
            "token: mytoken123\n"
            "---\n"
            "prompt\n"
        )
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write(content)
            path = Path(f.name)

        try:
            with pytest.raises(AgentLoadError, match="secret"):
                load_agent_file(path, "global")
        finally:
            path.unlink()

    def test_rejects_password_field(self):
        content = (
            "---\n"
            "schema_version: 1\n"
            "name: pw-agent\n"
            "description: test\n"
            "password: hunter2\n"
            "---\n"
            "prompt\n"
        )
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write(content)
            path = Path(f.name)

        try:
            with pytest.raises(AgentLoadError, match="secret"):
                load_agent_file(path, "global")
        finally:
            path.unlink()

    def test_rejects_suffixed_key(self):
        content = (
            "---\n"
            "schema_version: 1\n"
            "name: suffix-agent\n"
            "description: test\n"
            "routing:\n"
            "  anthropic_key: sk-ant-123\n"
            "---\n"
            "prompt\n"
        )
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write(content)
            path = Path(f.name)

        try:
            with pytest.raises(AgentLoadError, match="secret"):
                load_agent_file(path, "global")
        finally:
            path.unlink()

    def test_rejects_suffixed_token(self):
        content = (
            "---\n"
            "schema_version: 1\n"
            "name: sfx-token-agent\n"
            "description: test\n"
            "github_token: ghp_xxx\n"
            "---\n"
            "prompt\n"
        )
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write(content)
            path = Path(f.name)

        try:
            with pytest.raises(AgentLoadError, match="secret"):
                load_agent_file(path, "global")
        finally:
            path.unlink()


# ─────────────────────────────────────────────────────────────────────────────
# Empty directories
# ─────────────────────────────────────────────────────────────────────────────

class TestEmptyDirs:
    def test_empty_global_agents_returns_empty(self, hermes_home, monkeypatch):
        """list_agents on a HERMES_HOME with no agents dir returns [] without crash."""
        # Ensure no agents dir exists
        agents_dir = hermes_home / "agents"
        # Don't create it
        result = list_agents()
        assert result == []

    def test_no_global_dir_no_crash(self, tmp_path, monkeypatch):
        """When HERMES_HOME has no agents/ subdirectory at all, list_agents returns []. """
        hh = tmp_path / ".hermes"
        hh.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hh))
        result = list_agents()
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# Valid global agent loading
# ─────────────────────────────────────────────────────────────────────────────

class TestGlobalAgentLoading:
    def test_loads_valid_global_agent(self, global_agents_dir):
        """A valid global agent loads and fields normalize correctly."""
        path = global_agents_dir / "qa-engineer.md"
        make_agent(path, "qa-engineer", "Reviews code changes for correctness.")

        agents = list_agents()
        assert len(agents) == 1
        agent = agents[0]
        assert agent.name == "qa-engineer"
        assert agent.source == "global"
        assert agent.path == path
        assert agent.description == "Reviews code changes for correctness."
        assert agent.enabled is True
        assert agent.prompt == "This is the agent prompt."

    def test_get_agent_by_name(self, global_agents_dir):
        """get_agent returns the correct agent."""
        path = global_agents_dir / "code-analyst.md"
        make_agent(path, "code-analyst", "Analyzes code structure.")

        agent = get_agent("code-analyst")
        assert agent is not None
        assert agent.name == "code-analyst"
        assert agent.description == "Analyzes code structure."

    def test_nonexistent_agent_returns_none(self, global_agents_dir):
        """get_agent returns None for unknown name."""
        assert get_agent("does-not-exist") is None

    def test_display_name_defaults_to_name(self, global_agents_dir):
        """display_name falls back to name when not provided."""
        path = global_agents_dir / "simple.md"
        make_agent(path, "simple-agent", "A simple agent.")
        agent = get_agent("simple-agent")
        assert agent is not None
        assert agent.display_name == "simple-agent"

    def test_display_name_from_frontmatter(self, global_agents_dir):
        """display_name is used when provided."""
        path = global_agents_dir / "display-test.md"
        make_agent(path, "display-test", "A test agent.", display_name="My Custom Name")
        agent = get_agent("display-test")
        assert agent is not None
        assert agent.display_name == "My Custom Name"

    def test_tags_normalized(self, global_agents_dir):
        """tags are loaded as a list."""
        path = global_agents_dir / "tagged-agent.md"
        make_agent(path, "tagged-agent", "An agent with tags.", tags=["coding", "review"])
        agent = get_agent("tagged-agent")
        assert agent is not None
        assert agent.tags == ["coding", "review"]

    def test_routing_defaults_inherit(self, global_agents_dir):
        """routing defaults to mode=inherit when not specified."""
        path = global_agents_dir / "no-routing.md"
        make_agent(path, "no-routing", "No routing specified.")
        agent = get_agent("no-routing")
        assert agent is not None
        assert agent.routing.mode == "inherit"

    def test_tools_defaults_inherit(self, global_agents_dir):
        """tools defaults to mode=inherit when not specified."""
        path = global_agents_dir / "no-tools.md"
        make_agent(path, "no-tools", "No tools specified.")
        agent = get_agent("no-tools")
        assert agent is not None
        assert agent.tools.mode == "inherit"


# ─────────────────────────────────────────────────────────────────────────────
# Prompt body required
# ─────────────────────────────────────────────────────────────────────────────

class TestPromptBodyRequired:
    def test_rejects_empty_body(self, global_agents_dir):
        """Agent with no body after frontmatter is rejected."""
        path = global_agents_dir / "empty-body.md"
        path.write_text(
            "---\n"
            "schema_version: 1\n"
            "name: empty-body\n"
            "description: no body\n"
            "---\n"
            "\n"
        )
        with pytest.raises(AgentLoadError, match="prompt"):
            load_agent_file(path, "global")

    def test_rejects_whitespace_only_body(self, global_agents_dir):
        """Agent with only whitespace body is rejected."""
        path = global_agents_dir / "ws-body.md"
        path.write_text(
            "---\n"
            "schema_version: 1\n"
            "name: ws-body\n"
            "description: ws only\n"
            "---\n"
            "   \n"
        )
        with pytest.raises(AgentLoadError, match="prompt"):
            load_agent_file(path, "global")

    def test_accepts_minimal_body(self, global_agents_dir):
        """Agent with single-line body is accepted."""
        path = global_agents_dir / "tiny-body.md"
        path.write_text(
            "---\n"
            "schema_version: 1\n"
            "name: tiny-body\n"
            "description: tiny\n"
            "---\n"
            "x\n"
        )
        agent = load_agent_file(path, "global")
        assert agent.prompt.strip() == "x"


# ─────────────────────────────────────────────────────────────────────────────
# Symlink handling
# ─────────────────────────────────────────────────────────────────────────────

class TestSymlinksSkipped:
    def test_symlink_in_agents_dir_skipped(self, global_agents_dir):
        """A symlink inside the agents dir is not loaded."""
        real = global_agents_dir / "real-agent.md"
        make_agent(real, "real-agent", "Real agent.")

        link = global_agents_dir / "link-agent.md"
        link.symlink_to(real)

        agents = list_agents()
        names = [a.name for a in agents]
        assert "real-agent" in names
        # Symlink should be skipped (not crash)
        assert "link-agent" not in names


# ─────────────────────────────────────────────────────────────────────────────
# Malformed frontmatter
# ─────────────────────────────────────────────────────────────────────────────

class TestMalformedFrontmatter:
    def test_bad_yaml_is_warning_not_crash(self, global_agents_dir):
        """Bad YAML frontmatter produces a warning but doesn't crash list_agents."""
        path = global_agents_dir / "bad-yaml.md"
        path.write_text(
            "---\n"
            "schema_version: 1\n"
            "name: bad-yaml\n"
            "description: bad yaml\n"
            "  - this is not valid yaml list\n"
            "---\n"
            "prompt\n"
        )
        # Should not raise — warning is logged, agent skipped
        agents = list_agents()
        names = [a.name for a in agents]
        # bad-yaml may or may not load depending on fallback; at minimum no crash
        assert True  # reached here = no crash

    def test_missing_required_name_field(self, global_agents_dir):
        """Missing name field in frontmatter is rejected."""
        path = global_agents_dir / "no-name.md"
        path.write_text(
            "---\n"
            "schema_version: 1\n"
            "description: no name\n"
            "---\n"
            "prompt\n"
        )
        with pytest.raises(AgentLoadError, match="name"):
            load_agent_file(path, "global")

    def test_missing_required_description_field(self, global_agents_dir):
        """Missing description field in frontmatter is rejected."""
        path = global_agents_dir / "no-desc.md"
        path.write_text(
            "---\n"
            "schema_version: 1\n"
            "name: no-desc\n"
            "---\n"
            "prompt\n"
        )
        with pytest.raises(AgentLoadError, match="description"):
            load_agent_file(path, "global")

    def test_wrong_schema_version(self, global_agents_dir):
        """Wrong schema version is rejected."""
        path = global_agents_dir / "bad-schema.md"
        path.write_text(
            "---\n"
            "schema_version: 99\n"
            "name: bad-schema\n"
            "description: wrong schema\n"
            "---\n"
            "prompt\n"
        )
        with pytest.raises(AgentLoadError, match="schema"):
            load_agent_file(path, "global")


# ─────────────────────────────────────────────────────────────────────────────
# Disabled agents
# ─────────────────────────────────────────────────────────────────────────────

class TestDisabledAgents:
    def test_disabled_hidden_by_default(self, global_agents_dir):
        """enabled: false agents are hidden from default list_agents."""
        path = global_agents_dir / "disabled-agent.md"
        make_agent(path, "disabled-agent", "Should be hidden.", enabled=False)

        agents = list_agents()
        names = [a.name for a in agents]
        assert "disabled-agent" not in names

    def test_disabled_visible_with_include_disabled(self, global_agents_dir):
        """enabled: false agents appear when include_disabled=True."""
        path = global_agents_dir / "disabled-agent.md"
        make_agent(path, "disabled-agent", "Should be visible.", enabled=False)

        agents = list_agents(include_disabled=True)
        names = [a.name for a in agents]
        assert "disabled-agent" in names

    def test_disabled_agent_not_gettable(self, global_agents_dir):
        """get_agent still returns disabled agents (caller can check .enabled)."""
        path = global_agents_dir / "disabled-agent.md"
        make_agent(path, "disabled-agent", "A disabled agent.", enabled=False)

        agent = get_agent("disabled-agent")
        assert agent is not None
        assert agent.enabled is False


# ─────────────────────────────────────────────────────────────────────────────
# Project overrides global
# ─────────────────────────────────────────────────────────────────────────────

class TestProjectOverridesGlobal:
    def test_project_agent_overrides_global(self, global_agents_dir, project_agents_dir, project_root):
        """A project-local agent with the same name shadows the global one."""
        # Global agent
        gpath = global_agents_dir / "qa-engineer.md"
        make_agent(gpath, "qa-engineer", "Global QA Engineer")

        # Project agent with same name
        ppath = project_agents_dir / "qa-engineer.md"
        make_agent(ppath, "qa-engineer", "Project QA Engineer")

        # get_agent should return project source
        agent = get_agent("qa-engineer", workdir=str(project_root))
        assert agent is not None
        assert agent.source == "project"
        assert agent.description == "Project QA Engineer"
        assert agent.path == ppath

    def test_global_still_accessible_when_not_shadowed(self, global_agents_dir, project_agents_dir):
        """Agents that exist only globally are still found."""
        gpath = global_agents_dir / "code-analyst.md"
        make_agent(gpath, "code-analyst", "Global Code Analyst")

        agent = get_agent("code-analyst", workdir=str(project_root))
        assert agent is not None
        assert agent.source == "global"
        assert agent.description == "Global Code Analyst"

    def test_include_shadowed_shows_both(self, global_agents_dir, project_agents_dir, project_root):
        """include_shadowed=True shows both project-effective and global-shadowed."""
        gpath = global_agents_dir / "qa-engineer.md"
        make_agent(gpath, "qa-engineer", "Global QA Engineer")

        ppath = project_agents_dir / "qa-engineer.md"
        make_agent(ppath, "qa-engineer", "Project QA Engineer")

        agents = list_agents(workdir=str(project_root), include_shadowed=True)
        names = [a.name for a in agents]
        assert "qa-engineer" in names

        # Find the shadowed entry
        shadowed = [a for a in agents if a.shadowed]
        effective = [a for a in agents if not a.shadowed]

        assert len(shadowed) == 1
        assert shadowed[0].source == "global"
        assert shadowed[0].name == "qa-engineer"

        assert len(effective) == 1
        assert effective[0].source == "project"
        assert effective[0].description == "Project QA Engineer"

    def test_hermes_home_not_treated_as_project_duplicate(self, hermes_home, monkeypatch):
        """$HERMES_HOME agents dir should not be scanned as a project-local duplicate."""
        # Create agents in HERMES_HOME
        agents_dir = hermes_home / "agents"
        agents_dir.mkdir()
        make_agent(agents_dir / "global-only.md", "global-only", "Global only.")

        # Do not create .hermes in hermes_home (would make it look like a project root)
        # list_agents should return the global agent
        result = list_agents()
        names = [a.name for a in result]
        assert "global-only" in names

    def test_get_agent_source_global_returns_shadowed(self, global_agents_dir, project_agents_dir, project_root):
        """get_agent(name, source="global") returns the global agent even when shadowed."""
        gpath = global_agents_dir / "qa-engineer.md"
        make_agent(gpath, "qa-engineer", "Global QA Engineer")

        ppath = project_agents_dir / "qa-engineer.md"
        make_agent(ppath, "qa-engineer", "Project QA Engineer")

        # Default get_agent returns project (effective)
        agent = get_agent("qa-engineer", workdir=str(project_root))
        assert agent is not None
        assert agent.source == "project"
        assert agent.shadowed is False

        # get_agent with source="global" returns the shadowed global agent
        agent = get_agent("qa-engineer", workdir=str(project_root), source="global")
        assert agent is not None
        assert agent.source == "global"
        assert agent.shadowed is True
        assert agent.description == "Global QA Engineer"

    def test_get_agent_source_project_returns_project(self, global_agents_dir, project_agents_dir, project_root):
        """get_agent(name, source="project") returns the project agent."""
        gpath = global_agents_dir / "qa-engineer.md"
        make_agent(gpath, "qa-engineer", "Global QA Engineer")

        ppath = project_agents_dir / "qa-engineer.md"
        make_agent(ppath, "qa-engineer", "Project QA Engineer")

        agent = get_agent("qa-engineer", workdir=str(project_root), source="project")
        assert agent is not None
        assert agent.source == "project"
        assert agent.description == "Project QA Engineer"


# ─────────────────────────────────────────────────────────────────────────────
# Project-local agents — acp_command stripped without delegation block
# ─────────────────────────────────────────────────────────────────────────────

class TestProjectLocalAcpCommand:
    def test_project_agent_acp_command_stripped_no_delegation(self, project_agents_dir, project_root):
        """Project-local agents with routing.acp_command and no delegation block get acp_command=None and a warning."""
        path = project_agents_dir / "acp-test.md"
        path.write_text(
            "---\n"
            "schema_version: 1\n"
            "name: acp-test\n"
            "description: tests acp_command without delegation\n"
            "routing:\n"
            "  acp_command: my-cmd\n"
            "  acp_args:\n"
            "    - arg1\n"
            "---\n"
            "prompt\n"
        )

        agent = get_agent("acp-test", workdir=str(project_root))
        assert agent is not None
        assert agent.source == "project"
        assert agent.routing.acp_command is None
        assert agent.routing.acp_args is None
        warning_text = " ".join(agent.warnings)
        assert "routing.acp_command" in warning_text or "acp_command" in warning_text

    def test_project_agent_acp_command_allowed_in_delegation(self, project_agents_dir, project_root):
        """Project-local agents with delegation block can still have acp_command stripped with warning."""
        path = project_agents_dir / "acp-delegated.md"
        path.write_text(
            "---\n"
            "schema_version: 1\n"
            "name: acp-delegated\n"
            "description: tests acp_command with delegation\n"
            "routing:\n"
            "  acp_command: my-cmd\n"
            "delegation:\n"
            "  role: leaf\n"
            "---\n"
            "prompt\n"
        )

        agent = get_agent("acp-delegated", workdir=str(project_root))
        assert agent is not None
        assert agent.source == "project"
        assert agent.routing.acp_command is None


# ─────────────────────────────────────────────────────────────────────────────
# Unknown fields inside routing and security
# ─────────────────────────────────────────────────────────────────────────────

class TestUnknownRoutingSecurityFields:
    def test_unknown_routing_field_produces_warning(self, global_agents_dir):
        """Unknown routing fields produce a warning but don't fail loading."""
        path = global_agents_dir / "unknown-routing.md"
        path.write_text(
            "---\n"
            "schema_version: 1\n"
            "name: unknown-routing\n"
            "description: test unknown routing field\n"
            "routing:\n"
            "  mode: hermes\n"
            "  unknown_knob: some-value\n"
            "---\n"
            "prompt\n"
        )

        agent = get_agent("unknown-routing")
        assert agent is not None
        warning_text = " ".join(agent.warnings)
        assert "unknown_knob" in warning_text

    def test_unknown_security_field_produces_warning(self, global_agents_dir):
        """Unknown security fields produce a warning but don't fail loading."""
        path = global_agents_dir / "unknown-security.md"
        path.write_text(
            "---\n"
            "schema_version: 1\n"
            "name: unknown-security\n"
            "description: test unknown security field\n"
            "security:\n"
            "  require_approval: true\n"
            "  unknown_knob: some-value\n"
            "---\n"
            "prompt\n"
        )

        agent = get_agent("unknown-security")
        assert agent is not None
        warning_text = " ".join(agent.warnings)
        assert "unknown_knob" in warning_text

    def test_secret_like_routing_field_still_rejected(self, global_agents_dir):
        """Secret-like routing fields (e.g. api_key) are still rejected, not warned."""
        path = global_agents_dir / "secret-routing.md"
        path.write_text(
            "---\n"
            "schema_version: 1\n"
            "name: secret-routing\n"
            "description: test secret routing field\n"
            "routing:\n"
            "  api_key: sk-secret\n"
            "---\n"
            "prompt\n"
        )

        with pytest.raises(AgentLoadError, match="secret"):
            load_agent_file(path, "global")

    def test_secret_like_security_field_still_rejected(self, global_agents_dir):
        """Secret-like security fields are still rejected, not warned."""
        path = global_agents_dir / "secret-security.md"
        path.write_text(
            "---\n"
            "schema_version: 1\n"
            "name: secret-security\n"
            "description: test secret security field\n"
            "security:\n"
            "  api_key: sk-secret\n"
            "---\n"
            "prompt\n"
        )

        with pytest.raises(AgentLoadError, match="secret"):
            load_agent_file(path, "global")


# ─────────────────────────────────────────────────────────────────────────────
# AgentDefinition helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentDefinitionHelpers:
    def test_to_dict_returns_dict(self, global_agents_dir):
        """AgentDefinition.to_dict() returns a plain dict."""
        path = global_agents_dir / "to-dict-agent.md"
        make_agent(path, "to-dict-agent", "Testing to_dict.")

        agent = get_agent("to-dict-agent")
        d = agent.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "to-dict-agent"
        assert d["source"] == "global"

    def test_list_summary_returns_compact(self, global_agents_dir):
        """AgentDefinition.list_summary() returns compact metadata without full prompt."""
        path = global_agents_dir / "summary-agent.md"
        make_agent(path, "summary-agent", "Testing summary.")

        agent = get_agent("summary-agent")
        s = agent.list_summary()
        assert isinstance(s, dict)
        assert s["name"] == "summary-agent"
        assert "prompt" not in s  # no full prompt
        assert "description" in s  # description is included in list summary


# ─────────────────────────────────────────────────────────────────────────────
# Discovery helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestDiscoverAgentDirs:
    def test_global_dir_discovered(self, hermes_home):
        """discover_agent_dirs finds $HERMES_HOME/agents."""
        dirs = discover_agent_dirs()
        global_paths = [d.path for d in dirs if d.source == "global"]
        assert any("agents" in str(p) for p in global_paths)

    def test_no_crash_without_agents_subdir(self, tmp_path, monkeypatch):
        """discover_agent_dirs handles missing agents subdirectory gracefully."""
        hh = tmp_path / ".hermes"
        hh.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hh))

        dirs = discover_agent_dirs()
        # Should return list (may be empty for global-only when dir absent)
        assert isinstance(dirs, list)

    def test_project_dir_discovered(self, project_root):
        """discover_agent_dirs finds <project>/.hermes/agents when project root exists."""
        dirs = discover_agent_dirs(workdir=str(project_root))
        project_paths = [d.path for d in dirs if d.source == "project"]
        assert any(".hermes/agents" in str(p) for p in project_paths)

    def test_no_project_root_no_project_dirs(self, tmp_path):
        """When no project root found, only global dirs returned."""
        # tmp_path has no .git or .hermes
        dirs = discover_agent_dirs(workdir=str(tmp_path))
        project_dirs = [d for d in dirs if d.source == "project"]
        assert project_dirs == []


# ─────────────────────────────────────────────────────────────────────────────
# Unknown fields
# ─────────────────────────────────────────────────────────────────────────────

class TestUnknownFields:
    def test_unknown_top_level_field_produces_warning(self, global_agents_dir):
        """Unknown top-level frontmatter fields produce a warning, not an error."""
        path = global_agents_dir / "unknown-top.md"
        path.write_text(
            "---\n"
            "schema_version: 1\n"
            "name: unknown-top\n"
            "description: tests unknown field\n"
            "foobar: something\n"
            "---\n"
            "prompt\n"
        )
        agents = list_agents()
        names = [a.name for a in agents]
        assert "unknown-top" in names
        # Warning should be in the agent's warnings list
        agent = get_agent("unknown-top")
        assert any("unknown" in w.lower() or "foobar" in w.lower() for w in agent.warnings)


# ─────────────────────────────────────────────────────────────────────────────
# File size bound
# ─────────────────────────────────────────────────────────────────────────────

class TestFileSizeBound:
    def test_oversized_file_rejected(self, global_agents_dir):
        """Agent files over ~128 KiB are rejected."""
        path = global_agents_dir / "large-agent.md"
        large_content = (
            "---\n"
            "schema_version: 1\n"
            "name: large-agent\n"
            "description: too large\n"
            "---\n"
            + ("x" * (129 * 1024))  # 129 KiB
        )
        path.write_text(large_content)
        with pytest.raises(AgentLoadError, match="size"):
            load_agent_file(path, "global")


# ─────────────────────────────────────────────────────────────────────────────
# AgentDefinition dataclass fields
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentDefinitionFields:
    def test_all_expected_fields_present(self, global_agents_dir):
        """AgentDefinition has all expected dataclass fields."""
        path = global_agents_dir / "full-agent.md"
        make_agent(
            path,
            "full-agent",
            "Full agent.",
            tags=["test"],
            enabled=True,
            display_name="Full Agent",
        )

        agent = get_agent("full-agent")
        assert hasattr(agent, "schema_version")
        assert hasattr(agent, "name")
        assert hasattr(agent, "display_name")
        assert hasattr(agent, "description")
        assert hasattr(agent, "enabled")
        assert hasattr(agent, "source")
        assert hasattr(agent, "path")
        assert hasattr(agent, "prompt")
        assert hasattr(agent, "tags")
        assert hasattr(agent, "routing")
        assert hasattr(agent, "tools")
        assert hasattr(agent, "skills")
        assert hasattr(agent, "limits")
        assert hasattr(agent, "delegation_role")
        assert hasattr(agent, "security")
        assert hasattr(agent, "compatibility")
        assert hasattr(agent, "extensions")
        assert hasattr(agent, "warnings")
        assert hasattr(agent, "shadowed")
        assert hasattr(agent, "to_dict")
        assert hasattr(agent, "list_summary")


# ─────────────────────────────────────────────────────────────────────────────
# No duplicate loads
# ─────────────────────────────────────────────────────────────────────────────

class TestNoDuplicateLoads:
    def test_agented_md_pattern_not_duplicate(self, global_agents_dir):
        """An agent file at agents/foo/AGENT.md should not be loaded twice."""
        # Create agents/foo/AGENT.md
        subdir = global_agents_dir / "subcategory"
        subdir.mkdir()
        path = subdir / "AGENT.md"
        make_agent(path, "sub-agent", "A subdir agent.")

        agents = list_agents()
        names = [a.name for a in agents]
        assert "sub-agent" in names
        # Should appear exactly once
        assert names.count("sub-agent") == 1

    def test_nested_md_files_found(self, global_agents_dir):
        """agents/category/nested.md is discovered."""
        subdir = global_agents_dir / "category"
        subdir.mkdir()
        path = subdir / "nested.md"
        make_agent(path, "nested-agent", "A nested agent.")

        agents = list_agents()
        names = [a.name for a in agents]
        assert "nested-agent" in names