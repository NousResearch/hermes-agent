"""Tests for hermes_cli.agent_import — ``hermes import-agent``.

Covers: source detection, Claude Code and Codex parsing, mapping into the
real Hermes stores (memories/MEMORY.md, config.yaml command_allowlist /
approvals.deny / mcp_servers, skills/), dry-run write-nothing guarantees,
malformed-input skip reports, and the never-import-secrets rule.

Uses the profile_env fixture pattern from tests/hermes_cli/test_profiles.py:
Path.home() and HERMES_HOME are redirected to tmp_path so nothing touches
the real ~/.hermes.
"""

import json
from pathlib import Path

import pytest
import yaml

from hermes_cli.agent_import import (
    ENTRY_DELIMITER,
    AgentImporter,
    claude_rule_to_command_pattern,
    detect_agents,
    extract_markdown_entries,
    is_secret_key,
    parse_existing_memory_entries,
    sanitize_mcp_env,
)


# ---------------------------------------------------------------------------
# Shared fixture: redirect Path.home() and HERMES_HOME (profile_env pattern)
# ---------------------------------------------------------------------------

@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    """Isolated environment: Path.home() -> tmp_path, HERMES_HOME -> tmp/.hermes."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return tmp_path


@pytest.fixture()
def hermes_home(profile_env):
    return profile_env / ".hermes"


# ---------------------------------------------------------------------------
# Fake source trees
# ---------------------------------------------------------------------------

CLAUDE_MD = """# Global instructions

## Style
- Always use type hints
- Prefer pathlib over os.path

Run the linter before committing.
"""

AGENTS_MD = """# Codex rules

- Never force-push to main
- Keep commits atomic
"""


@pytest.fixture()
def claude_tree(profile_env):
    """Build a fake ~/.claude tree (plus sibling ~/.claude.json)."""
    root = profile_env / ".claude"
    root.mkdir()
    (root / "CLAUDE.md").write_text(CLAUDE_MD, encoding="utf-8")
    (root / "settings.json").write_text(json.dumps({
        "permissions": {
            "allow": [
                "Bash(npm run build)",
                "Bash(npm run test:*)",
                "Bash(git diff *)",
                "Read(~/.zshrc)",     # non-Bash → unmapped
            ],
            "deny": [
                "Bash(rm -rf *)",
                "WebFetch",           # non-Bash → dropped
            ],
        },
        "mcpServers": {
            "settings-server": {"command": "uvx", "args": ["settings-mcp"]},
        },
    }), encoding="utf-8")
    # mcpServers in the sibling ~/.claude.json (Claude's primary MCP store)
    (profile_env / ".claude.json").write_text(json.dumps({
        "mcpServers": {
            "github": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "env": {
                    "GITHUB_TOKEN": "ghp_SECRET123",
                    "GITHUB_HOST": "github.example.com",
                },
            },
            "remote": {
                "url": "https://mcp.example.com/sse",
                "headers": {
                    "Authorization": "Bearer abc123",
                    "X-Region": "us-east",
                },
            },
        },
    }), encoding="utf-8")
    # A credentials file that must never be read/imported
    (root / ".credentials.json").write_text(
        json.dumps({"api_key": "sk-ant-SUPERSECRET"}), encoding="utf-8")
    # Skills
    skill = root / "skills" / "deploy-helper"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: deploy-helper\n---\n\nDeploy things.\n", encoding="utf-8")
    (root / "skills" / "not-a-skill").mkdir()  # no SKILL.md → ignored
    # Slash commands (reported as skipped)
    commands = root / "commands"
    commands.mkdir()
    (commands / "review.md").write_text("Review this PR", encoding="utf-8")
    return root


@pytest.fixture()
def codex_tree(profile_env):
    """Build a fake ~/.codex tree."""
    root = profile_env / ".codex"
    root.mkdir()
    (root / "AGENTS.md").write_text(AGENTS_MD, encoding="utf-8")
    (root / "config.toml").write_text(
        'model = "gpt-5"\n'
        'approval_policy = "on-request"\n'
        "\n"
        "[mcp_servers.docs]\n"
        'command = "uvx"\n'
        'args = ["docs-mcp"]\n'
        "\n"
        "[mcp_servers.docs.env]\n"
        'DOCS_API_KEY = "secret-value"\n'
        'DOCS_REGION = "eu"\n',
        encoding="utf-8",
    )
    (root / "auth.json").write_text(
        json.dumps({"OPENAI_API_KEY": "sk-SECRET"}), encoding="utf-8")
    memories = root / "memories"
    memories.mkdir()
    (memories / "2026-01-01.md").write_text(
        "- User prefers tabs over spaces\n- Project uses PostgreSQL\n",
        encoding="utf-8",
    )
    skill = root / "skills" / "db-migrate"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: db-migrate\n---\n\nMigrate databases.\n", encoding="utf-8")
    return root


def snapshot_tree(root: Path) -> dict:
    """Map of relative-path -> bytes for every file under root."""
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*")) if p.is_file()
    }


def run_import(agent, source, hermes_home, execute, overwrite=False):
    return AgentImporter(
        agent=agent,
        source_root=source,
        target_root=hermes_home,
        execute=execute,
        overwrite=overwrite,
    ).run()


# ---------------------------------------------------------------------------
# Detection & helpers
# ---------------------------------------------------------------------------

class TestDetection:
    def test_detects_claude_and_codex(self, claude_tree, codex_tree):
        assert detect_agents() == ["claude-code", "codex"]

    def test_detects_nothing_when_absent(self, profile_env):
        assert detect_agents() == []

    def test_unsupported_agent_raises(self, hermes_home, tmp_path):
        with pytest.raises(ValueError):
            AgentImporter("cursor", tmp_path, hermes_home)


class TestRuleMapping:
    def test_bash_rule_plain(self):
        assert claude_rule_to_command_pattern("Bash(npm run build)") == "npm run build"

    def test_bash_rule_colon_star_prefix(self):
        assert claude_rule_to_command_pattern("Bash(npm run test:*)") == "npm run test*"

    def test_non_bash_rule_is_none(self):
        assert claude_rule_to_command_pattern("Read(~/.zshrc)") is None
        assert claude_rule_to_command_pattern("WebFetch") is None

    def test_blanket_bash_is_none(self):
        assert claude_rule_to_command_pattern("Bash()") is None


class TestSecretDetection:
    @pytest.mark.parametrize("key", [
        "GITHUB_TOKEN", "OPENAI_API_KEY", "MY_SECRET", "DB_PASSWORD",
        "AWS_ACCESS_KEY", "AUTH_HEADER", "APIKEY",
    ])
    def test_secret_keys(self, key):
        assert is_secret_key(key)

    @pytest.mark.parametrize("key", ["GITHUB_HOST", "REGION", "DEBUG", "PORT"])
    def test_non_secret_keys(self, key):
        assert not is_secret_key(key)

    def test_sanitize_env_splits(self):
        kept, stripped = sanitize_mcp_env(
            {"API_TOKEN": "x", "HOST": "h", "MY_KEY": "k"})
        assert kept == {"HOST": "h"}
        assert sorted(stripped) == ["API_TOKEN", "MY_KEY"]


class TestMarkdownEntries:
    def test_extracts_bullets_and_paragraphs(self):
        entries = extract_markdown_entries(CLAUDE_MD)
        assert any("type hints" in e for e in entries)
        assert any("linter" in e for e in entries)

    def test_heading_context_prefix(self):
        entries = extract_markdown_entries(CLAUDE_MD)
        assert any(e.startswith("Global instructions > Style:") for e in entries)


# ---------------------------------------------------------------------------
# Dry run writes NOTHING
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_claude_dry_run_writes_nothing(self, claude_tree, hermes_home):
        before = snapshot_tree(hermes_home)
        report = run_import("claude-code", claude_tree, hermes_home, execute=False)
        assert snapshot_tree(hermes_home) == before
        assert report["dry_run"] is True
        assert report["summary"]["imported"] > 0

    def test_codex_dry_run_writes_nothing(self, codex_tree, hermes_home):
        before = snapshot_tree(hermes_home)
        report = run_import("codex", codex_tree, hermes_home, execute=False)
        assert snapshot_tree(hermes_home) == before
        assert report["dry_run"] is True
        assert report["summary"]["imported"] > 0

    def test_dry_run_and_real_run_plan_same_items(self, claude_tree, hermes_home):
        preview = run_import("claude-code", claude_tree, hermes_home, execute=False)
        applied = run_import("claude-code", claude_tree, hermes_home, execute=True)
        pk = [(i["kind"], i["status"]) for i in preview["items"]]
        ak = [(i["kind"], i["status"]) for i in applied["items"]]
        assert pk == ak


# ---------------------------------------------------------------------------
# Claude Code real run — every item lands in the right store
# ---------------------------------------------------------------------------

class TestClaudeCodeImport:
    @pytest.fixture()
    def report(self, claude_tree, hermes_home):
        return run_import("claude-code", claude_tree, hermes_home, execute=True)

    def test_claude_md_becomes_memory_entries(self, report, hermes_home):
        memory = (hermes_home / "memories" / "MEMORY.md").read_text(encoding="utf-8")
        assert "type hints" in memory
        assert "§" in memory  # entry-delimited store format

    def test_allowlist_lands_in_config_yaml(self, report, hermes_home):
        config = yaml.safe_load((hermes_home / "config.yaml").read_text())
        allow = config["command_allowlist"]
        assert "npm run build" in allow
        assert "npm run test*" in allow
        assert "git diff *" in allow
        # non-Bash rules must not leak in
        assert not any("Read(" in p for p in allow)

    def test_denylist_lands_in_approvals_deny(self, report, hermes_home):
        config = yaml.safe_load((hermes_home / "config.yaml").read_text())
        assert "rm -rf *" in config["approvals"]["deny"]

    def test_mcp_servers_from_claude_json_and_settings(self, report, hermes_home):
        config = yaml.safe_load((hermes_home / "config.yaml").read_text())
        servers = config["mcp_servers"]
        assert servers["github"]["command"] == "npx"
        assert servers["remote"]["url"] == "https://mcp.example.com/sse"
        assert servers["settings-server"]["command"] == "uvx"

    def test_skill_copied_into_category_dir(self, report, hermes_home):
        dest = hermes_home / "skills" / "claude-code-imports" / "deploy-helper" / "SKILL.md"
        assert dest.exists()
        assert "Deploy things" in dest.read_text(encoding="utf-8")

    def test_dir_without_skill_md_not_copied(self, report, hermes_home):
        assert not (hermes_home / "skills" / "claude-code-imports" / "not-a-skill").exists()

    def test_slash_commands_reported_skipped(self, report):
        items = {i["kind"]: i for i in report["items"]}
        assert items["slash-commands"]["status"] == "skipped"


# ---------------------------------------------------------------------------
# Codex real run
# ---------------------------------------------------------------------------

class TestCodexImport:
    @pytest.fixture()
    def report(self, codex_tree, hermes_home):
        return run_import("codex", codex_tree, hermes_home, execute=True)

    def test_agents_md_becomes_memory_entries(self, report, hermes_home):
        memory = (hermes_home / "memories" / "MEMORY.md").read_text(encoding="utf-8")
        assert "force-push" in memory

    def test_memories_dir_merged(self, report, hermes_home):
        memory = (hermes_home / "memories" / "MEMORY.md").read_text(encoding="utf-8")
        assert "tabs over spaces" in memory
        assert "PostgreSQL" in memory

    def test_mcp_servers_from_config_toml(self, report, hermes_home):
        config = yaml.safe_load((hermes_home / "config.yaml").read_text())
        docs = config["mcp_servers"]["docs"]
        assert docs["command"] == "uvx"
        assert docs["args"] == ["docs-mcp"]
        # non-secret env survives, secret is stripped
        assert docs["env"] == {"DOCS_REGION": "eu"}

    def test_skill_copied(self, report, hermes_home):
        assert (hermes_home / "skills" / "codex-imports" / "db-migrate" / "SKILL.md").exists()


# ---------------------------------------------------------------------------
# Secrets are never copied
# ---------------------------------------------------------------------------

class TestSecretsNeverImported:
    def test_no_secret_values_anywhere_claude(self, claude_tree, hermes_home):
        run_import("claude-code", claude_tree, hermes_home, execute=True)
        blob = "".join(
            p.read_text(encoding="utf-8", errors="replace")
            for p in hermes_home.rglob("*") if p.is_file()
        )
        assert "ghp_SECRET123" not in blob
        assert "Bearer abc123" not in blob
        assert "sk-ant-SUPERSECRET" not in blob

    def test_no_secret_values_anywhere_codex(self, codex_tree, hermes_home):
        run_import("codex", codex_tree, hermes_home, execute=True)
        blob = "".join(
            p.read_text(encoding="utf-8", errors="replace")
            for p in hermes_home.rglob("*") if p.is_file()
        )
        assert "secret-value" not in blob
        assert "sk-SECRET" not in blob

    def test_stripped_secrets_reported(self, claude_tree, hermes_home):
        report = run_import("claude-code", claude_tree, hermes_home, execute=True)
        stripped = report.get("stripped_secrets", [])
        assert "mcp_servers.github.env.GITHUB_TOKEN" in stripped
        assert any("Authorization" in s for s in stripped)

    def test_non_secret_header_kept(self, claude_tree, hermes_home):
        run_import("claude-code", claude_tree, hermes_home, execute=True)
        config = yaml.safe_load((hermes_home / "config.yaml").read_text())
        assert config["mcp_servers"]["remote"]["headers"] == {"X-Region": "us-east"}


# ---------------------------------------------------------------------------
# Malformed inputs: per-item skip/error reports, no crashes
# ---------------------------------------------------------------------------

class TestMalformedInputs:
    def test_bad_settings_json_reports_error(self, profile_env, hermes_home):
        root = profile_env / ".claude"
        root.mkdir()
        (root / "settings.json").write_text("{not json!!", encoding="utf-8")
        (root / "CLAUDE.md").write_text("- still importable\n", encoding="utf-8")
        report = run_import("claude-code", root, hermes_home, execute=True)
        errors = [i for i in report["items"] if i["status"] == "error"]
        assert any(i["kind"] == "settings" for i in errors)
        # CLAUDE.md still imported despite bad settings.json
        assert "still importable" in (
            hermes_home / "memories" / "MEMORY.md").read_text()

    def test_bad_claude_json_reports_error(self, profile_env, hermes_home):
        root = profile_env / ".claude"
        root.mkdir()
        (profile_env / ".claude.json").write_text("][", encoding="utf-8")
        (root / "CLAUDE.md").write_text("- an entry\n", encoding="utf-8")
        report = run_import("claude-code", root, hermes_home, execute=True)
        assert any(
            i["status"] == "error" and ".claude.json" in (i["source"] or "")
            for i in report["items"]
        )
        assert report["summary"]["imported"] >= 1

    def test_bad_config_toml_reports_error(self, profile_env, hermes_home):
        root = profile_env / ".codex"
        root.mkdir()
        (root / "config.toml").write_text("[[[[not toml", encoding="utf-8")
        (root / "AGENTS.md").write_text("- rule one\n", encoding="utf-8")
        report = run_import("codex", root, hermes_home, execute=True)
        errors = [i for i in report["items"] if i["status"] == "error"]
        assert any(i["kind"] == "config" for i in errors)
        assert "rule one" in (hermes_home / "memories" / "MEMORY.md").read_text()

    def test_missing_source_dir_is_error_not_crash(self, profile_env, hermes_home):
        report = run_import(
            "claude-code", profile_env / "nope", hermes_home, execute=True)
        assert report["summary"]["error"] == 1
        assert report["summary"]["imported"] == 0

    def test_empty_tree_all_skipped(self, profile_env, hermes_home):
        root = profile_env / ".codex"
        root.mkdir()
        report = run_import("codex", root, hermes_home, execute=True)
        assert report["summary"]["imported"] == 0
        assert report["summary"]["error"] == 0


# ---------------------------------------------------------------------------
# Merge semantics & conflicts
# ---------------------------------------------------------------------------

class TestMergeSemantics:
    def test_existing_allowlist_preserved(self, claude_tree, hermes_home):
        (hermes_home / "config.yaml").write_text(
            yaml.safe_dump({"command_allowlist": ["docker ps"]}), encoding="utf-8")
        run_import("claude-code", claude_tree, hermes_home, execute=True)
        config = yaml.safe_load((hermes_home / "config.yaml").read_text())
        assert "docker ps" in config["command_allowlist"]
        assert "npm run build" in config["command_allowlist"]

    def test_existing_mcp_server_conflicts_without_overwrite(
            self, claude_tree, hermes_home):
        (hermes_home / "config.yaml").write_text(
            yaml.safe_dump({"mcp_servers": {"github": {"command": "mine"}}}),
            encoding="utf-8")
        report = run_import("claude-code", claude_tree, hermes_home, execute=True)
        config = yaml.safe_load((hermes_home / "config.yaml").read_text())
        assert config["mcp_servers"]["github"]["command"] == "mine"
        assert any(
            i["status"] == "conflict" and i["source"] == "github"
            for i in report["items"]
        )

    def test_overwrite_replaces_mcp_server(self, claude_tree, hermes_home):
        (hermes_home / "config.yaml").write_text(
            yaml.safe_dump({"mcp_servers": {"github": {"command": "mine"}}}),
            encoding="utf-8")
        run_import("claude-code", claude_tree, hermes_home, execute=True,
                   overwrite=True)
        config = yaml.safe_load((hermes_home / "config.yaml").read_text())
        assert config["mcp_servers"]["github"]["command"] == "npx"

    def test_existing_skill_conflicts_without_overwrite(
            self, claude_tree, hermes_home):
        dest = hermes_home / "skills" / "claude-code-imports" / "deploy-helper"
        dest.mkdir(parents=True)
        (dest / "SKILL.md").write_text("mine\n", encoding="utf-8")
        report = run_import("claude-code", claude_tree, hermes_home, execute=True)
        assert (dest / "SKILL.md").read_text() == "mine\n"
        assert any(
            i["kind"] == "skill" and i["status"] == "conflict"
            for i in report["items"]
        )

    def test_reimport_is_idempotent_for_memory(self, claude_tree, hermes_home):
        run_import("claude-code", claude_tree, hermes_home, execute=True)
        first = (hermes_home / "memories" / "MEMORY.md").read_text()
        report = run_import("claude-code", claude_tree, hermes_home, execute=True)
        assert (hermes_home / "memories" / "MEMORY.md").read_text() == first
        memory_items = [i for i in report["items"] if i["kind"] == "claude-md"]
        assert memory_items[0]["status"] == "skipped"


# ---------------------------------------------------------------------------
# The DESTINATION memories/MEMORY.md is a §-delimited store, not a document
# ---------------------------------------------------------------------------

# A realistic hand-edited store: one entry, no "§", with a fenced code block
# and a markdown table — exactly the content extract_markdown_entries() drops.
EXISTING_MEMORY = """Homelab runbook. Restart the ingress controller with:

```bash
kubectl -n ingress rollout restart deploy/nginx
```

Escalation ladder:

| Severity | Contact | Window |
|----------|---------|--------|
| SEV1     | on-call | 15m    |
| SEV2     | #ops    | 4h     |

Never page for SEV3.
"""


class TestExistingMemoryStorePreserved:
    """An import must not shred the memory store it merges into.

    ``memories/MEMORY.md`` is the entry-delimited store written by
    ``MemoryStore._write_file``; a single-entry or hand-edited store contains
    no ``§`` delimiter.  Parsing it with the *source* markdown extractor drops
    code blocks and table rows and splits one entry into fragments, and the
    merged result is written straight back over the file.
    """

    @pytest.fixture()
    def seeded_home(self, hermes_home):
        memory = hermes_home / "memories" / "MEMORY.md"
        memory.parent.mkdir(parents=True, exist_ok=True)
        memory.write_text(EXISTING_MEMORY, encoding="utf-8")
        return hermes_home

    def test_undelimited_store_is_one_entry(self, seeded_home):
        path = seeded_home / "memories" / "MEMORY.md"
        assert ENTRY_DELIMITER not in path.read_text(encoding="utf-8")
        assert parse_existing_memory_entries(path) == [EXISTING_MEMORY.strip()]

    def test_agrees_with_memory_store_parser(self, seeded_home):
        from tools.memory_tool import MemoryStore

        path = seeded_home / "memories" / "MEMORY.md"
        raw = path.read_text(encoding="utf-8")
        assert parse_existing_memory_entries(path) == MemoryStore._parse_entries(raw)

    def test_import_preserves_existing_entry_verbatim(
            self, claude_tree, seeded_home):
        path = seeded_home / "memories" / "MEMORY.md"
        run_import("claude-code", claude_tree, seeded_home, execute=True)
        entries = path.read_text(encoding="utf-8").split(ENTRY_DELIMITER)
        # The pre-existing store survives byte-intact as a SINGLE entry ...
        assert entries[0] == EXISTING_MEMORY.strip()
        # ... including the parts the markdown extractor would have dropped.
        assert "kubectl -n ingress rollout restart deploy/nginx" in entries[0]
        assert "| SEV1     | on-call | 15m    |" in entries[0]
        # ... and the imported entries are still appended after it.
        assert any("type hints" in e for e in entries[1:])

    def test_import_backs_up_the_previous_store(self, claude_tree, seeded_home):
        memories = seeded_home / "memories"
        run_import("claude-code", claude_tree, seeded_home, execute=True)
        backups = sorted(memories.glob("MEMORY.md.bak.*"))
        assert len(backups) == 1
        assert backups[0].read_text(encoding="utf-8") == EXISTING_MEMORY

    def test_dry_run_leaves_the_store_untouched(self, claude_tree, seeded_home):
        memories = seeded_home / "memories"
        run_import("claude-code", claude_tree, seeded_home, execute=False)
        assert (memories / "MEMORY.md").read_text(
            encoding="utf-8") == EXISTING_MEMORY
        assert not list(memories.glob("MEMORY.md.bak.*"))


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------

class TestCliWiring:
    def test_parser_builds_and_parses(self):
        import argparse
        from hermes_cli.subcommands.import_agent import build_import_agent_parser

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        called = {}
        build_import_agent_parser(
            subparsers, cmd_import_agent=lambda a: called.setdefault("ok", a))
        args = parser.parse_args(
            ["import-agent", "claude-code", "--dry-run", "--source", "/tmp/x"])
        assert args.agent == "claude-code"
        assert args.dry_run is True
        assert args.source == "/tmp/x"
        args.func(args)
        assert "ok" in called

    def test_rejects_unknown_agent(self):
        import argparse
        from hermes_cli.subcommands.import_agent import build_import_agent_parser

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        build_import_agent_parser(subparsers, cmd_import_agent=lambda a: None)
        with pytest.raises(SystemExit):
            parser.parse_args(["import-agent", "cursor"])

    def test_command_dry_run_via_cli_writes_nothing(
            self, claude_tree, hermes_home, capsys):
        """End-to-end through import_agent_command with --dry-run."""
        import types
        from hermes_cli.agent_import import import_agent_command

        args = types.SimpleNamespace(
            agent="claude-code", source=str(claude_tree), dry_run=True,
            overwrite=False, yes=False)
        import_agent_command(args)
        out = capsys.readouterr().out
        assert "Dry Run Results" in out
        assert "command-allowlist" in out
        # Baseline config.yaml/SOUL.md may be seeded by save_config() before
        # the preview runs — but nothing from the IMPORT itself may land:
        assert not (hermes_home / "memories" / "MEMORY.md").exists()
        assert not (hermes_home / "skills" / "claude-code-imports").exists()
        config_text = (hermes_home / "config.yaml").read_text(encoding="utf-8") \
            if (hermes_home / "config.yaml").exists() else ""
        assert "npm run build" not in config_text
        assert "github" not in config_text
