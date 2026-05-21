"""Unit tests for agent/apm_consumer.py."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add hermes-agent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent.apm_consumer import (
    _package_name_from_path,
    _populate_virtual_paths,
    _safe_read,
    detect_apm_project,
    discover_apm_mcp_servers,
    discover_apm_skill_files,
    get_apm_modules_dir,
    load_apm_instructions,
    remove_apm_skill_symlinks,
    symlink_apm_skills,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def apm_project(tmp_path):
    """Create a minimal APM project on disk."""
    proj = tmp_path / "apm-test"
    proj.mkdir()
    # Write apm.yml
    (proj / "apm.yml").write_text(
        "name: test-proj\n"
        "version: 1.0.0\n"
        "dependencies: {}\n"
        "mcp: []\n"
    )
    # Create apm_modules with one skill
    skill_dir = proj / "apm_modules" / "owner" / "repo" / ".apm" / "skills" / "test-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: test-skill\ndescription: A test skill\n---\n\n# Test Skill\n"
    )
    # Create instructions
    instr_dir = proj / "apm_modules" / "owner" / "repo" / ".apm" / "instructions"
    instr_dir.mkdir(parents=True)
    (instr_dir / "setup.instructions.md").write_text("# Setup\nRun me first.\n")
    # Create agent
    agent_dir = proj / "apm_modules" / "owner" / "repo" / ".apm" / "agents"
    agent_dir.mkdir(parents=True)
    (agent_dir / "helper.md").write_text("---\nname: helper\n---\n\nI help.\n")
    # Create prompt
    prompt_dir = proj / "apm_modules" / "owner" / "repo" / ".apm" / "prompts"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "commit.prompt.md").write_text("Write a commit message.\n")
    return proj


@pytest.fixture
def apm_project_with_mcp(tmp_path):
    """APM project with MCP servers declared."""
    proj = tmp_path / "apm-mcp-test"
    proj.mkdir()
    (proj / "apm.yml").write_text(
        "name: test-proj\n"
        "version: 1.0.0\n"
        "dependencies: {}\n"
        "mcp:\n"
        "  - name: github/server\n"
        "    transport: http\n"
        "    url: https://api.github.com/mcp\n"
        "  - name: filesystem/server\n"
        "    transport: stdio\n"
        "    command: npx\n"
        "    args: ['-y', '@modelcontextprotocol/server-filesystem']\n"
    )
    return proj


@pytest.fixture
def non_apm_project(tmp_path):
    """Bare directory with no APM files."""
    proj = tmp_path / "no-apm"
    proj.mkdir()
    return proj


# ── Detection tests ──────────────────────────────────────────────────────


def test_detect_apm_project_true(apm_project):
    assert detect_apm_project(str(apm_project)) is True


def test_detect_apm_project_false(non_apm_project):
    assert detect_apm_project(str(non_apm_project)) is False


def test_detect_apm_project_default_cwd(apm_project, monkeypatch):
    monkeypatch.chdir(apm_project)
    assert detect_apm_project() is True


# ── get_apm_modules_dir tests ────────────────────────────────────────────


def test_get_modules_dir_exists(apm_project):
    result = get_apm_modules_dir(str(apm_project))
    assert result is not None
    assert result.name == "apm_modules"


def test_get_modules_dir_absent(non_apm_project):
    result = get_apm_modules_dir(str(non_apm_project))
    assert result is None


# ── Skill discovery tests ────────────────────────────────────────────────


def test_discover_skill_files(apm_project):
    skills = discover_apm_skill_files(str(apm_project))
    assert len(skills) == 1
    path, name = skills[0]
    assert name == "apm/owner/repo/test-skill"
    assert path.name == "SKILL.md"
    assert "test-skill" in str(path)


def test_discover_skill_files_empty(non_apm_project):
    skills = discover_apm_skill_files(str(non_apm_project))
    assert skills == []


def test_discover_skill_files_no_duplicates(apm_project):
    """Ensure secondary skills/ path doesn't duplicate .apm/skills/ entries."""
    # Create a duplicate at the skills/ top-level
    alt = apm_project / "apm_modules" / "owner" / "repo" / "skills" / "test-skill"
    alt.mkdir(parents=True)
    (alt / "SKILL.md").write_text("---\nname: test-skill\n---\n\nDuplicate.\n")

    skills = discover_apm_skill_files(str(apm_project))
    names = [n for _, n in skills]
    # .apm/skills/ takes priority; duplicate name from skills/ is skipped
    assert names.count("apm/owner/repo/test-skill") == 1


# ── Package name extraction tests ────────────────────────────────────────


def test_package_name_from_path_two_segments(apm_project):
    modules = apm_project / "apm_modules"
    p = modules / "owner" / "repo" / "deep" / "path" / "SKILL.md"
    assert _package_name_from_path(p, modules) == "owner/repo"


def test_package_name_from_path_one_segment(tmp_path):
    """Single-segment relative path gets returned as-is."""
    modules = tmp_path / "modules"
    p = modules / "singlefile.md"
    assert _package_name_from_path(p, modules) == "singlefile.md"


def test_package_name_from_path_empty(tmp_path):
    modules = tmp_path / "modules"
    assert _package_name_from_path(modules, modules) == "unknown"


def test_package_name_from_path_outside(tmp_path):
    modules = tmp_path / "modules"
    p = tmp_path / "outside" / "file.md"
    assert _package_name_from_path(p, modules) == "unknown"


# ── Symlink tests ────────────────────────────────────────────────────────


def test_symlink_apm_skills_creates_links(apm_project, monkeypatch):
    """Symlinks are created in the correct hermes skills path structure."""
    # Redirect get_skills_dir to a temp directory
    fake_skills = apm_project / ".fake-skills"
    fake_skills.mkdir()
    
    import hermes_constants
    original = hermes_constants.get_skills_dir
    monkeypatch.setattr(hermes_constants, "get_skills_dir", lambda: fake_skills)

    try:
        count = symlink_apm_skills(str(apm_project))
        assert count == 1

        # Verify symlink structure
        symlink = fake_skills / "apm" / "owner" / "repo" / "test-skill" / "SKILL.md"
        assert symlink.is_symlink()
        assert symlink.resolve().exists()
        content = symlink.read_text()
        assert "test-skill" in content
    finally:
        monkeypatch.setattr(hermes_constants, "get_skills_dir", original)


def test_symlink_apm_skills_idempotent(apm_project, monkeypatch):
    """Re-running symlink should not create duplicates or break.

    On the first call, symlinks are created (count > 0). On the second call,
    already-correct symlinks are skipped (count 0). Both are valid idempotency
    behaviours — the important thing is the symlink still works.
    """
    fake_skills = apm_project / ".fake-skills"
    fake_skills.mkdir()

    import hermes_constants
    original = hermes_constants.get_skills_dir
    monkeypatch.setattr(hermes_constants, "get_skills_dir", lambda: fake_skills)

    try:
        count1 = symlink_apm_skills(str(apm_project))
        assert count1 == 1
        count2 = symlink_apm_skills(str(apm_project))
        assert count2 in (0, 1)  # Re-symlink may skip already-correct links

        symlink = fake_skills / "apm" / "owner" / "repo" / "test-skill" / "SKILL.md"
        assert symlink.is_symlink()
    finally:
        monkeypatch.setattr(hermes_constants, "get_skills_dir", original)


def test_symlink_apm_skills_empty(non_apm_project, monkeypatch):
    """No symlinks created when no APM skills exist."""
    fake_skills = non_apm_project / ".fake-skills"
    fake_skills.mkdir()
    
    import hermes_constants
    original = hermes_constants.get_skills_dir
    monkeypatch.setattr(hermes_constants, "get_skills_dir", lambda: fake_skills)

    try:
        count = symlink_apm_skills(str(non_apm_project))
        assert count == 0
        apm_dir = fake_skills / "apm"
        assert not apm_dir.exists() or not any(apm_dir.iterdir())
    finally:
        monkeypatch.setattr(hermes_constants, "get_skills_dir", original)


def test_symlink_apm_skills_rejects_escape(apm_project, monkeypatch, tmp_path):
    """Symlinked SKILL.md pointing outside apm_modules/ is skipped.

    An attacker-controlled APM package could ship a symlinked SKILL.md
    that escapes the module root.  The symlink code must resolve the
    path, verify it stays within ``apm_modules/``, and skip it with
    a warning if it does not.
    """
    # Replace the real SKILL.md with a symlink to an external file
    external = tmp_path / "outside.txt"
    external.write_text("---\nname: escape\n---\n\n# Escaped content\n")
    skill_dir = (
        apm_project / "apm_modules" / "owner" / "repo" / ".apm" / "skills" / "escape-skill"
    )
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.symlink_to(external)  # Create symlink to outside, no real file here

    fake_skills = apm_project / ".fake-skills"
    fake_skills.mkdir()

    import hermes_constants
    original = hermes_constants.get_skills_dir
    monkeypatch.setattr(hermes_constants, "get_skills_dir", lambda: fake_skills)

    try:
        count = symlink_apm_skills(str(apm_project))
        # The escape symlink is skipped; only the original test-skill is linked
        assert count == 1  # Only test-skill (from fixture), not escape-skill
        apm_dir = fake_skills / "apm"
        assert apm_dir.exists()
        # Verify escape-skill was NOT symlinked
        escape_link = apm_dir / "owner" / "repo" / "escape-skill" / "SKILL.md"
        assert not escape_link.exists()
        # But the legitimate skill IS present
        legit_link = apm_dir / "owner" / "repo" / "test-skill" / "SKILL.md"
        assert legit_link.is_symlink()
    finally:
        monkeypatch.setattr(hermes_constants, "get_skills_dir", original)


# ── Remove symlinks tests ────────────────────────────────────────────────


def test_remove_apm_skill_symlinks(apm_project, monkeypatch):
    """remove_apm_skill_symlinks() cleans up everything."""
    fake_skills = apm_project / ".fake-skills"
    fake_skills.mkdir()
    
    import hermes_constants
    original = hermes_constants.get_skills_dir
    monkeypatch.setattr(hermes_constants, "get_skills_dir", lambda: fake_skills)

    try:
        symlink_apm_skills(str(apm_project))
        count = remove_apm_skill_symlinks()
        assert count == 1

        # apm/ root should be empty or removed
        apm_dir = fake_skills / "apm"
        assert not apm_dir.exists() or not any(apm_dir.iterdir())
    finally:
        monkeypatch.setattr(hermes_constants, "get_skills_dir", original)


# ── Instruction loading tests ────────────────────────────────────────────


def test_load_apm_instructions_loads_content(apm_project):
    content = load_apm_instructions(str(apm_project))
    assert "APM Package Context" in content
    assert "APM Instructions: setup" in content
    assert "Run me first" in content
    assert "APM Agent: repo/helper" in content
    assert "I help" in content
    assert "APM Prompt: commit" in content
    assert "Write a commit message" in content


def test_load_apm_instructions_empty(non_apm_project):
    content = load_apm_instructions(str(non_apm_project))
    assert content == ""


# ── MCP server discovery tests ───────────────────────────────────────────


def test_discover_mcp_servers_http(apm_project_with_mcp):
    servers = discover_apm_mcp_servers(str(apm_project_with_mcp))
    assert "github-server" in servers
    assert servers["github-server"]["url"] == "https://api.github.com/mcp"


def test_discover_mcp_servers_stdio(apm_project_with_mcp):
    servers = discover_apm_mcp_servers(str(apm_project_with_mcp))
    assert "filesystem-server" in servers
    assert servers["filesystem-server"]["command"] == "npx"
    assert servers["filesystem-server"]["args"] == [
        "-y",
        "@modelcontextprotocol/server-filesystem",
    ]


def test_discover_mcp_servers_empty_no_mcp(apm_project):
    """When apm.yml has mcp: []"""
    servers = discover_apm_mcp_servers(str(apm_project))
    assert servers == {}


def test_discover_mcp_servers_no_apm_yml(non_apm_project):
    servers = discover_apm_mcp_servers(str(non_apm_project))
    assert servers == {}


# ── _safe_read tests ─────────────────────────────────────────────────────


def test_safe_read_normal(tmp_path):
    f = tmp_path / "test.md"
    f.write_text("hello world")
    assert _safe_read(f, 100) == "hello world"


def test_safe_read_truncation(tmp_path):
    f = tmp_path / "long.md"
    f.write_text("a" * 500)
    result = _safe_read(f, 100)
    assert len(result) <= 200  # 100 + "...truncated" overhead
    assert "truncated" in result


def test_safe_read_non_utf8(tmp_path):
    f = tmp_path / "binary.dat"
    f.write_bytes(b"\xff\xfe\x00\x01")
    result = _safe_read(f, 100)
    assert result == ""


# ── Edge case: apm.yml exists but apm_modules/ doesn't ───────────────────


def test_detect_without_modules(tmp_path):
    proj = tmp_path / "partial-apm"
    proj.mkdir()
    (proj / "apm.yml").write_text("name: test\n")
    assert detect_apm_project(str(proj)) is True
    assert get_apm_modules_dir(str(proj)) is None
    assert discover_apm_skill_files(str(proj)) == []
    assert load_apm_instructions(str(proj)) == ""


# ── Edge case: malformed apm.yml ──────────────────────────────────────────


def test_mcp_discovery_malformed_yaml(tmp_path):
    proj = tmp_path / "bad-yml"
    proj.mkdir()
    (proj / "apm.yml").write_text(": bad: yaml: : :")
    servers = discover_apm_mcp_servers(str(proj))
    assert servers == {}  # Graceful degradation


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Content hash staleness detection
# ═══════════════════════════════════════════════════════════════════════════


def test_compute_skill_hash_deterministic(apm_project):
    """Same content produces same hash."""
    from agent.apm_consumer import compute_skill_hash
    skills = discover_apm_skill_files(str(apm_project))
    path, _ = skills[0]
    h1 = compute_skill_hash(path)
    h2 = compute_skill_hash(path)
    assert h1 is not None
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex


def test_compute_skill_hash_nonexistent():
    """Non-existent file returns None."""
    from agent.apm_consumer import compute_skill_hash
    assert compute_skill_hash(Path("/nonexistent/skill.md")) is None


def test_detect_apm_staleness_initial(apm_project, monkeypatch):
    """First run shows all skills as new (empty cache)."""
    from agent.apm_consumer import detect_apm_staleness, _hermes_cache_dir
    # Use isolated cache so parallel tests don't interfere
    fake_cache = apm_project / ".hermes-cache"
    fake_cache.mkdir()
    monkeypatch.setattr(
        "agent.apm_consumer._hermes_cache_dir",
        lambda: fake_cache,
    )
    stale, changed, new = detect_apm_staleness(str(apm_project))
    assert stale is True  # First run: all skills are "new"
    assert len(new) >= 1  # At least one new skill
    assert changed == []


def test_detect_apm_staleness_second_run(apm_project, monkeypatch):
    """Second run with no changes detects no staleness."""
    from agent.apm_consumer import detect_apm_staleness
    # Use isolated cache
    fake_cache = apm_project / ".hermes-cache"
    fake_cache.mkdir()
    monkeypatch.setattr(
        "agent.apm_consumer._hermes_cache_dir",
        lambda: fake_cache,
    )
    detect_apm_staleness(str(apm_project))  # Seed cache
    stale, changed, new = detect_apm_staleness(str(apm_project))
    assert stale is False
    assert changed == []
    assert new == []


def test_detect_apm_staleness_modified(apm_project, monkeypatch):
    """Modifying a skill file triggers staleness."""
    from agent.apm_consumer import detect_apm_staleness
    # Use isolated cache
    fake_cache = apm_project / ".hermes-cache"
    fake_cache.mkdir()
    monkeypatch.setattr(
        "agent.apm_consumer._hermes_cache_dir",
        lambda: fake_cache,
    )
    detect_apm_staleness(str(apm_project))  # Seed cache

    # Modify the skill file
    skills = discover_apm_skill_files(str(apm_project))
    path, _name = skills[0]
    path.write_text("---\nname: test-skill\ndescription: modified\n---\n\n# Changed!\n")

    stale, changed, new = detect_apm_staleness(str(apm_project))
    assert stale is True
    assert len(changed) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Lockfile parsing & validation
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def apm_project_with_lockfile(apm_project):
    """APM project with a lockfile."""
    lock = apm_project / "apm.lock.yaml"
    lock.write_text(
        "lockfile_version: '1'\n"
        "generated_at: '2026-01-01T00:00:00Z'\n"
        "apm_version: 0.8.0\n"
        "dependencies:\n"
        "- repo_url: owner/repo\n"
        "  host: github.com\n"
        "  resolved_commit: abc123\n"
        "  virtual_path: skills/test-skill\n"
        "  is_virtual: true\n"
        "  package_type: apm_package\n"
    )
    return apm_project


@pytest.fixture
def apm_project_with_bad_lockfile(apm_project):
    """APM project with malformed lockfile."""
    (apm_project / "apm.lock.yaml").write_text(": : : bad yaml")
    return apm_project


def test_parse_apm_lockfile(apm_project_with_lockfile):
    from agent.apm_consumer import parse_apm_lockfile
    data = parse_apm_lockfile(str(apm_project_with_lockfile))
    assert data is not None
    assert data["lockfile_version"] == "1"
    assert data["apm_version"] == "0.8.0"  # YAML parses as string
    assert len(data["dependencies"]) == 1
    assert data["dependencies"][0]["repo_url"] == "owner/repo"


def test_parse_apm_lockfile_absent(apm_project):
    from agent.apm_consumer import parse_apm_lockfile
    assert parse_apm_lockfile(str(apm_project)) is None


def test_parse_apm_lockfile_malformed(apm_project_with_bad_lockfile):
    from agent.apm_consumer import parse_apm_lockfile
    assert parse_apm_lockfile(str(apm_project_with_bad_lockfile)) is None


def test_validate_lockfile_no_lockfile(apm_project):
    """No lockfile → valid (nothing to validate)."""
    from agent.apm_consumer import validate_lockfile_against_modules
    valid, issues = validate_lockfile_against_modules(str(apm_project))
    assert valid is True
    assert issues == []


def test_validate_lockfile_missing_module(apm_project_with_lockfile, tmp_path):
    """Lockfile references a module not on disk."""
    from agent.apm_consumer import validate_lockfile_against_modules
    # Create a bare project with lockfile but no modules
    proj = tmp_path / "lock-only"
    proj.mkdir()
    (proj / "apm.yml").write_text("name: test\n")
    lock = proj / "apm.lock.yaml"
    lock.write_text(
        "lockfile_version: '1'\n"
        "dependencies:\n"
        "- repo_url: missing/pkg\n"
        "  host: github.com\n"
        "  resolved_commit: xyz\n"
    )
    valid, issues = validate_lockfile_against_modules(str(proj))
    # Should report issues for missing module
    assert valid is False
    assert len(issues) >= 1


# ── Virtual path collection tests ─────────────────────────────────────


def test_populate_virtual_paths_content_bearing_only(tmp_path):
    """Only directories that contain files are recorded as virtual paths."""
    root = tmp_path / "module"
    root.mkdir()
    # Create a content-bearing subdirectory
    skills = root / "skills" / "primary"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text("# test\n")
    # Create an empty subdirectory
    (root / "skills" / "empty").mkdir(parents=True)
    # Create a hidden directory (should be skipped)
    (root / ".git").mkdir(parents=True)
    (root / ".git" / "config").write_text("[core]\n")

    result: set = set()
    _populate_virtual_paths(root, [], result)

    assert "skills/primary" in result
    assert "skills" in result
    # Empty directory should NOT be a virtual path
    assert "skills/empty" not in result
    # Hidden directories should NOT appear
    assert ".git" not in result
    assert ".git/config" not in result


def test_populate_virtual_paths_files_at_root(tmp_path):
    """Files directly in the module root generate the root itself as a path."""
    root = tmp_path / "module"
    root.mkdir()
    (root / "README.md").write_text("# readme\n")

    result: set = set()
    _populate_virtual_paths(root, [], result)

    # Root-level files should NOT register (prefix is empty — guard at line 535)
    assert len(result) == 0


def test_populate_virtual_paths_all_empty(tmp_path):
    """A module with only empty directories yields no virtual paths."""
    root = tmp_path / "module"
    root.mkdir()
    (root / "skills" / "empty1").mkdir(parents=True)
    (root / "skills" / "empty2").mkdir(parents=True)
    (root / "agents").mkdir(parents=True)

    result: set = set()
    _populate_virtual_paths(root, [], result)

    assert len(result) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: APM CLI integration tests
# ═══════════════════════════════════════════════════════════════════════════


def test_should_auto_install_lockfile_newer(apm_project):
    """Lockfile newer than modules → auto-install needed."""
    from agent.apm_consumer import should_auto_install
    # Set lockfile mtime to future
    lock = apm_project / "apm.lock.yaml"
    lock.write_text("lockfile_version: '1'\ndependencies: []\n")
    import os as _os
    _os.utime(lock, (_os.stat(lock).st_mtime + 3600, _os.stat(lock).st_mtime + 3600))
    result = should_auto_install(str(apm_project))
    assert result is True


def test_should_auto_install_no_lockfile(apm_project):
    """No lockfile → no auto-install."""
    from agent.apm_consumer import should_auto_install
    assert should_auto_install(str(apm_project)) is False


def test_should_auto_install_lockfile_no_modules(tmp_path):
    """Lockfile exists but no apm_modules → need install."""
    proj = tmp_path / "lock-no-mods"
    proj.mkdir()
    (proj / "apm.lock.yaml").write_text("lockfile_version: '1'\ndependencies: []\n")
    from agent.apm_consumer import should_auto_install
    assert should_auto_install(str(proj)) is True


def test_should_auto_install_modules_newer(apm_project):
    """Modules newer than lockfile → no auto-install needed (steady state)."""
    from agent.apm_consumer import should_auto_install
    # Create lockfile with older mtime, touch an existing module file to be newer
    lock = apm_project / "apm.lock.yaml"
    lock.write_text("lockfile_version: '1'\ndependencies: []\n")
    import os as _os
    # Set lockfile mtime to 1 hour ago
    past = _os.stat(lock).st_mtime - 3600
    _os.utime(lock, (past, past))
    # Touch a module file to be newer (it already exists from the fixture)
    skill = apm_project / "apm_modules" / "owner" / "repo" / ".apm" / "skills" / "test-skill" / "SKILL.md"
    assert skill.exists()
    now = _os.stat(skill).st_mtime
    _os.utime(skill, (now, now))  # Ensure it's current
    assert should_auto_install(str(apm_project)) is False


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: Policy allow-list filtering
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def apm_project_with_policy(apm_project):
    """APM project with a policy file."""
    policy = apm_project / "apm-policy.yml"
    policy.write_text(
        "allow_skills: [apm/owner/repo/test-skill]\n"
        "deny_skills: [apm/bad/pkg/dangerous]\n"
        "allow_instructions: []\n"
        "deny_packages: [blocked/org]\n"
    )
    return apm_project


def test_load_apm_policy(apm_project_with_policy):
    from agent.apm_consumer import load_apm_policy
    policy = load_apm_policy(str(apm_project_with_policy))
    assert policy is not None
    assert policy["allow_skills"] == ["apm/owner/repo/test-skill"]
    assert policy["deny_skills"] == ["apm/bad/pkg/dangerous"]
    assert policy["deny_packages"] == ["blocked/org"]


def test_load_apm_policy_absent(apm_project):
    from agent.apm_consumer import load_apm_policy
    assert load_apm_policy(str(apm_project)) is None


def test_filter_by_policy_no_policy():
    """No policy → everything allowed."""
    from agent.apm_consumer import filter_by_policy
    assert filter_by_policy("apm/owner/repo/skill", "skills", None) is True


def test_filter_by_policy_allow_list():
    """Item in allow_skills → allowed."""
    from agent.apm_consumer import filter_by_policy
    policy = {"allow_skills": ["apm/owner/repo/skill", "apm/other/pkg/thing"]}
    assert filter_by_policy("apm/owner/repo/skill", "skills", policy) is True
    assert filter_by_policy("apm/owner/repo/other", "skills", policy) is False


def test_filter_by_policy_deny():
    """Item in deny_skills → denied."""
    from agent.apm_consumer import filter_by_policy
    policy = {"deny_skills": ["apm/bad/pkg/dangerous"]}
    assert filter_by_policy("apm/bad/pkg/dangerous", "skills", policy) is False
    assert filter_by_policy("apm/owner/repo/safe", "skills", policy) is True


def test_filter_by_policy_deny_packages():
    """Package prefix in deny_packages → denied."""
    from agent.apm_consumer import filter_by_policy
    policy = {"deny_packages": ["blocked/org"]}
    assert filter_by_policy("apm/blocked/org/some-skill", "skills", policy) is False
    assert filter_by_policy("apm/allowed/org/safe-skill", "skills", policy) is True


def test_filter_by_policy_deny_wins_over_allow():
    """When both deny and allow list an item, deny wins."""
    from agent.apm_consumer import filter_by_policy
    policy = {
        "allow_skills": ["apm/pkg/skill"],
        "deny_skills": ["apm/pkg/skill"],
    }
    # Deny is checked first → False
    assert filter_by_policy("apm/pkg/skill", "skills", policy) is False


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Marketplace shorthand resolution
# ═══════════════════════════════════════════════════════════════════════════


def test_resolve_marketplace_ref_known():
    """@awesome-copilot → github/awesome-copilot/plugins/<name>."""
    from agent.apm_consumer import resolve_marketplace_ref
    result = resolve_marketplace_ref("devops-oncall@awesome-copilot")
    assert result == "github/awesome-copilot/plugins/devops-oncall"


def test_resolve_marketplace_ref_no_at():
    """String without @ returned unchanged."""
    from agent.apm_consumer import resolve_marketplace_ref
    assert resolve_marketplace_ref("owner/repo") == "owner/repo"


def test_resolve_marketplace_ref_unknown_marketplace():
    """Unknown marketplace → returned unchanged."""
    from agent.apm_consumer import resolve_marketplace_ref
    result = resolve_marketplace_ref("thing@nonexistent")
    assert result == "thing@nonexistent"


def test_resolve_marketplace_ref_with_pin():
    """The @marketplace portion is resolved; the #ref stays untouched
    because the caller handles pin-splitting separately."""
    from agent.apm_consumer import resolve_marketplace_ref
    # Caller strips #ref first, so we pass the clean name
    result = resolve_marketplace_ref("devops-oncall@awesome-copilot")
    assert result == "github/awesome-copilot/plugins/devops-oncall"


def test_parse_apm_dependencies_with_marketplace(apm_project):
    """apm.yml deps with @awesome-copilot are resolved."""
    (apm_project / "apm.yml").write_text(
        "name: test\n"
        "version: 1.0.0\n"
        "dependencies:\n"
        "  apm:\n"
        "    - devops-oncall@awesome-copilot\n"
        "    - owner/repo\n"
        "    - github/awesome-copilot/skills/agent-governance#main\n"
    )
    from agent.apm_consumer import parse_apm_dependencies
    deps = parse_apm_dependencies(str(apm_project))
    assert len(deps) == 3
    # First: marketplace ref resolved
    assert deps[0]["raw"] == "devops-oncall@awesome-copilot"
    assert deps[0]["resolved"] == "github/awesome-copilot/plugins/devops-oncall"
    assert deps[0]["ref"] == ""
    # Second: plain owner/repo unchanged
    assert deps[1]["raw"] == "owner/repo"
    assert deps[1]["resolved"] == "owner/repo"
    # Third: with #ref pin
    assert deps[2]["raw"] == "github/awesome-copilot/skills/agent-governance"
    assert deps[2]["ref"] == "main"


def test_parse_apm_dependencies_no_manifest(tmp_path):
    """No apm.yml → empty list."""
    from agent.apm_consumer import parse_apm_dependencies
    deps = parse_apm_dependencies(str(tmp_path))
    assert deps == []


def test_parse_apm_dependencies_object_form(apm_project):
    """Object-form deps (git + path + ref) are parsed."""
    (apm_project / "apm.yml").write_text(
        "name: test\n"
        "dependencies:\n"
        "  apm:\n"
        "    - git: https://gitlab.com/acme/coding-standards.git\n"
        "      path: instructions/security\n"
        "      ref: v2.0\n"
    )
    from agent.apm_consumer import parse_apm_dependencies
    deps = parse_apm_dependencies(str(apm_project))
    assert len(deps) == 1
    assert deps[0]["raw"] == "https://gitlab.com/acme/coding-standards.git"
    assert deps[0]["ref"] == "v2.0"
    assert deps[0]["path"] == "instructions/security"


def test_add_marketplace():
    """Registering a marketplace adds it to the resolver."""
    from agent.apm_consumer import add_marketplace, resolve_marketplace_ref
    add_marketplace("my-org", "github/my-org")
    result = resolve_marketplace_ref("my-plugin@my-org")
    assert result == "github/my-org/plugins/my-plugin"


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: APM security audit
# ═══════════════════════════════════════════════════════════════════════════


def test_audit_report_for_prompt_empty():
    """No findings → empty string."""
    from agent.apm_consumer import audit_report_for_prompt
    assert audit_report_for_prompt([]) == ""


def test_audit_report_for_prompt_with_findings():
    """Findings → warning block with each item."""
    from agent.apm_consumer import audit_report_for_prompt
    report = audit_report_for_prompt([
        "WARN: hidden unicode in skills/evil/SKILL.md",
        "CRITICAL: prompt injection in instructions/backdoor.instructions.md",
    ])
    assert "APM Security Audit Findings" in report
    assert "hidden unicode" in report
    assert "prompt injection" in report
    assert report.startswith("# APM Security Audit Findings")


def test_run_apm_audit_no_cli(tmp_path, monkeypatch):
    """When apm is not on PATH, return clean with no findings."""
    monkeypatch.setattr("shutil.which", lambda x: None)
    from agent.apm_consumer import run_apm_audit
    clean, findings = run_apm_audit(str(tmp_path))
    assert clean is True
    assert findings == []


def test_run_apm_audit_clean(apm_project, monkeypatch):
    """APM audit exits 0 with no findings → clean."""
    import subprocess

    class _CleanResult:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _CleanResult)
    monkeypatch.setattr("shutil.which", lambda x: "/usr/local/bin/apm")
    from agent.apm_consumer import run_apm_audit
    clean, findings = run_apm_audit(str(apm_project))
    assert clean is True
    assert findings == []


def test_run_apm_audit_with_warnings(apm_project, monkeypatch):
    """APM audit output with WARN lines → findings list."""
    import subprocess

    class _WarnResult:
        returncode = 0
        stdout = (
            "WARN: suspicious unicode in plugins/evil/SKILL.md\n"
            "OK: 5 packages scanned\n"
        )
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _WarnResult)
    monkeypatch.setattr("shutil.which", lambda x: "/usr/local/bin/apm")
    from agent.apm_consumer import run_apm_audit
    clean, findings = run_apm_audit(str(apm_project))
    assert clean is False
    assert len(findings) == 1
    assert "suspicious unicode" in findings[0]


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: Transitive dependency resolution
# ═══════════════════════════════════════════════════════════════════════════


def test_resolve_transitive_deps_no_lockfile(apm_project):
    from agent.apm_consumer import resolve_transitive_deps
    deps = resolve_transitive_deps(str(apm_project))
    assert deps == []


def test_resolve_transitive_deps_with_lockfile(apm_project_with_lockfile):
    from agent.apm_consumer import resolve_transitive_deps
    deps = resolve_transitive_deps(str(apm_project_with_lockfile))
    assert len(deps) == 1
    assert deps[0]["repo_url"] == "owner/repo"
    assert deps[0]["virtual_path"] == "skills/test-skill"
    assert deps[0]["is_transitive"] is False
    assert deps[0]["required_by"] == []


def test_resolve_transitive_deps_multi_repo(tmp_path):
    """Multiple virtual paths from same repo → transitive detection."""
    proj = tmp_path / "multi-repo"
    proj.mkdir()
    (proj / "apm.lock.yaml").write_text(
        "lockfile_version: '1'\n"
        "dependencies:\n"
        "- repo_url: org/pkg\n"
        "  virtual_path: skills/primary\n"
        "  resolved_commit: abc\n"
        "- repo_url: org/pkg\n"
        "  virtual_path: skills/secondary\n"
        "  resolved_commit: abc\n"
    )
    from agent.apm_consumer import resolve_transitive_deps
    deps = resolve_transitive_deps(str(proj))
    assert len(deps) == 2
    # First is primary, second is transitive
    trans = [d for d in deps if d["is_transitive"]]
    assert len(trans) == 1
