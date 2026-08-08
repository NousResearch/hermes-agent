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
import logging
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


    def test_unsupported_agent_raises(self, hermes_home, tmp_path):
        with pytest.raises(ValueError):
            AgentImporter("cursor", tmp_path, hermes_home)


class TestRuleMapping:
    def test_bash_rule_plain(self):
        assert claude_rule_to_command_pattern("Bash(npm run build)") == "npm run build"


    def test_non_bash_rule_is_none(self):
        assert claude_rule_to_command_pattern("Read(~/.zshrc)") is None
        assert claude_rule_to_command_pattern("WebFetch") is None


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


    def test_allowlist_lands_in_config_yaml(self, report, hermes_home):
        config = yaml.safe_load((hermes_home / "config.yaml").read_text())
        allow = config["command_allowlist"]
        assert "npm run build" in allow
        assert "npm run test*" in allow
        assert "git diff *" in allow
        # non-Bash rules must not leak in
        assert not any("Read(" in p for p in allow)




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

    # -- credential files INSIDE a skill directory -------------------------
    #
    # The cases above hold by accident: every fixture plants its credential
    # file at the ROOT of the source tree, and no importer reads root-level
    # files.  skills/ is the one destination that copies a whole subtree, and
    # it copied every member verbatim -- so `_CREDENTIAL_FILENAMES`, the list
    # the module docstring promises is "NEVER imported", was enforced nowhere.

    SKILL_SECRETS = {
        ".credentials.json": "sk-ant-INSIDE-SKILL",
        "auth.json": "sk-oai-INSIDE-SKILL",
        "credentials.json": "pw-INSIDE-SKILL",
    }

    @pytest.fixture()
    def skill_with_credentials(self, claude_tree):
        """A source skill the user keeps their agent credentials inside."""
        skill = claude_tree / "skills" / "deploy-helper"
        for name, secret in self.SKILL_SECRETS.items():
            (skill / name).write_text(
                json.dumps({"key": secret}), encoding="utf-8")
        # ... and one a directory deeper, beside a file that must still copy.
        nested = skill / "helpers"
        nested.mkdir()
        (nested / "auth.json").write_text(
            json.dumps({"key": "sk-oai-NESTED"}), encoding="utf-8")
        (nested / "run.sh").write_text("echo deploying\n", encoding="utf-8")
        return skill

    def test_skill_imports_without_its_credential_files(
            self, claude_tree, skill_with_credentials, hermes_home):
        report = run_import("claude-code", claude_tree, hermes_home,
                            execute=True)

        assert [i["status"] for i in skill_items(report)] == ["imported"]
        dest = hermes_home / "skills" / "claude-code-imports" / "deploy-helper"
        # The skill itself lands, in full ...
        assert "Deploy things." in (dest / "SKILL.md").read_text(encoding="utf-8")
        assert (dest / "helpers" / "run.sh").exists()
        # ... minus every credential file, at the top level ...
        for name in self.SKILL_SECRETS:
            assert not (dest / name).exists()
        # ... and nested, which is what makes this a name match and not a
        # top-level-only one.
        assert not (dest / "helpers" / "auth.json").exists()

    def test_no_skill_credential_values_anywhere(
            self, claude_tree, skill_with_credentials, hermes_home):
        run_import("claude-code", claude_tree, hermes_home, execute=True)
        blob = "".join(
            p.read_text(encoding="utf-8", errors="replace")
            for p in hermes_home.rglob("*") if p.is_file()
        )
        for secret in self.SKILL_SECRETS.values():
            assert secret not in blob
        assert "sk-oai-NESTED" not in blob

    def test_symlinked_credential_file_is_excluded_not_followed(
            self, claude_tree, hermes_home, tmp_path):
        """``symlinks=True`` reproduces links AS links, so a link named
        ``auth.json`` would otherwise still resolve to the real secret."""
        real = tmp_path / "real-auth.json"
        real.write_text(json.dumps({"key": "sk-oai-BEHIND-LINK"}),
                        encoding="utf-8")
        (claude_tree / "skills" / "deploy-helper" / "auth.json").symlink_to(real)

        run_import("claude-code", claude_tree, hermes_home, execute=True)

        dest = hermes_home / "skills" / "claude-code-imports" / "deploy-helper"
        assert not (dest / "auth.json").is_symlink()
        assert not (dest / "auth.json").exists()
        blob = "".join(
            p.read_text(encoding="utf-8", errors="replace")
            for p in hermes_home.rglob("*") if p.is_file()
        )
        assert "sk-oai-BEHIND-LINK" not in blob

    def test_skipped_skill_credentials_are_reported(
            self, claude_tree, skill_with_credentials, hermes_home):
        """Strip *and* tell the user -- the same contract the MCP env-var
        stripper holds, so a file left behind can be moved across
        deliberately if it was never a credential."""
        report = run_import("claude-code", claude_tree, hermes_home,
                            execute=True)

        item = skill_items(report)[0]
        assert sorted(Path(p).name for p in item["skipped_credentials"]) == [
            ".credentials.json", "auth.json", "auth.json", "credentials.json",
        ]
        assert (str(skill_with_credentials / "helpers" / "auth.json")
                in item["skipped_credentials"])
        # They surface in the same report section as the stripped env vars.
        for name in self.SKILL_SECRETS:
            assert str(skill_with_credentials / name) in report["stripped_secrets"]


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


# ---------------------------------------------------------------------------
# config.yaml preservation (sibling of the MEMORY.md case above)
# ---------------------------------------------------------------------------

EXISTING_CONFIG = """\
model: hermes-4-405b
api_key_env: OPENROUTER_API_KEY
command_allowlist:
  - ls *
  - cat *
approvals:
  deny:
    - shutdown *
mcp_servers:
  local-notes:
    command: uvx
    args:
      - notes-mcp
telegram:
  enabled: true
  chat_id: 12345
"""

MALFORMED_CONFIG = """\
model: hermes-4-405b
command_allowlist:
  - ls *
   - cat *
approvals: [unclosed
"""

CONFIG_WRITING_KINDS = ("command-allowlist", "command-denylist", "mcp-servers")


class TestExistingConfigPreserved:
    """An import must not destroy the config.yaml it merges into.

    ``load_yaml_file`` returned ``{}`` for an absent file AND for a present
    file it could not read or parse.  The three importers below read
    config.yaml, merge one section into it, and write the whole mapping back —
    so a YAML syntax error or a permission problem meant every existing
    setting was replaced by just the merged section, reported as ``imported``.
    """

    @pytest.fixture()
    def config_path(self, hermes_home):
        return hermes_home / "config.yaml"

    @staticmethod
    def items_for(report, kind):
        return [i for i in report["items"] if i["kind"] == kind]

    # -- present but unreadable: refuse, change nothing --------------------

    def test_malformed_config_is_left_byte_identical(
            self, claude_tree, hermes_home, config_path):
        config_path.write_text(MALFORMED_CONFIG, encoding="utf-8")
        before = config_path.read_bytes()

        run_import("claude-code", claude_tree, hermes_home, execute=True)

        assert config_path.read_bytes() == before

    def test_malformed_config_records_an_error_not_an_import(
            self, claude_tree, hermes_home, config_path):
        config_path.write_text(MALFORMED_CONFIG, encoding="utf-8")

        report = run_import("claude-code", claude_tree, hermes_home, execute=True)

        for kind in CONFIG_WRITING_KINDS:
            items = self.items_for(report, kind)
            assert items, f"no {kind} item recorded"
            assert [i["status"] for i in items] == ["error"], kind
            reason = items[0]["reason"]
            assert str(config_path) in reason
            assert "not valid YAML" in reason
            # Points the user at a way out.
            assert "hermes config edit" in reason

    def test_unreadable_config_is_left_byte_identical(
            self, claude_tree, hermes_home, config_path):
        """A permission problem is the other half of the same root cause."""
        import os

        if os.name != "posix":
            pytest.skip("chmod-based permission denial is POSIX-only")
        if getattr(os, "geteuid", lambda: 1)() == 0:
            pytest.skip("root bypasses file permissions")
        config_path.write_text(EXISTING_CONFIG, encoding="utf-8")
        before = config_path.read_bytes()
        config_path.chmod(0o000)
        try:
            report = run_import("claude-code", claude_tree, hermes_home,
                                execute=True)
            statuses = {
                i["status"]
                for kind in CONFIG_WRITING_KINDS
                for i in self.items_for(report, kind)
            }
            assert statuses == {"error"}
        finally:
            config_path.chmod(0o600)
        assert config_path.read_bytes() == before

    def test_non_mapping_config_is_refused(
            self, claude_tree, hermes_home, config_path):
        """Valid YAML that is not a mapping was also collapsed to {}."""
        config_path.write_text("- just\n- a\n- list\n", encoding="utf-8")
        before = config_path.read_bytes()

        report = run_import("claude-code", claude_tree, hermes_home, execute=True)

        assert config_path.read_bytes() == before
        reasons = [i["reason"] for i in self.items_for(report, "command-allowlist")]
        assert any("YAML mapping" in r for r in reasons)

    def test_dry_run_reports_the_refusal_rather_than_a_preview(
            self, claude_tree, hermes_home, config_path):
        """--dry-run must not promise an import that would destroy the file."""
        config_path.write_text(MALFORMED_CONFIG, encoding="utf-8")

        report = run_import("claude-code", claude_tree, hermes_home, execute=False)

        for kind in CONFIG_WRITING_KINDS:
            statuses = [i["status"] for i in self.items_for(report, kind)]
            assert statuses == ["error"], kind
            assert "imported" not in statuses

    # -- the regression that actually pins the data loss -------------------

    def test_import_preserves_every_pre_existing_config_key(
            self, claude_tree, hermes_home, config_path):
        config_path.write_text(EXISTING_CONFIG, encoding="utf-8")

        run_import("claude-code", claude_tree, hermes_home, execute=True)

        merged = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        # Untouched sections survive verbatim.
        assert merged["model"] == "hermes-4-405b"
        assert merged["api_key_env"] == "OPENROUTER_API_KEY"
        assert merged["telegram"] == {"enabled": True, "chat_id": 12345}
        # Merged-into sections keep their existing members ...
        assert "ls *" in merged["command_allowlist"]
        assert "shutdown *" in merged["approvals"]["deny"]
        assert "local-notes" in merged["mcp_servers"]
        # ... alongside the imported ones.
        assert "npm run build" in merged["command_allowlist"]
        assert "github" in merged["mcp_servers"]

    def test_absent_config_is_still_created(
            self, claude_tree, hermes_home, config_path):
        """The guard must not break first-time creation."""
        assert not config_path.exists()

        run_import("claude-code", claude_tree, hermes_home, execute=True)

        created = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert "npm run build" in created["command_allowlist"]

    def test_empty_config_is_treated_as_absent(
            self, claude_tree, hermes_home, config_path):
        config_path.write_text("", encoding="utf-8")

        run_import("claude-code", claude_tree, hermes_home, execute=True)

        created = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert "npm run build" in created["command_allowlist"]

    # -- the write itself --------------------------------------------------

    def test_config_write_is_atomic_and_leaves_no_temp_files(
            self, claude_tree, hermes_home, config_path):
        config_path.write_text(EXISTING_CONFIG, encoding="utf-8")

        run_import("claude-code", claude_tree, hermes_home, execute=True)

        leftovers = [p.name for p in hermes_home.glob(".tmp*")]
        assert leftovers == []

    def test_symlinked_config_stays_a_symlink(
            self, claude_tree, hermes_home, tmp_path, config_path):
        """A non-atomic ``write_text`` would follow it; ``os.replace`` would
        clobber it.  ``atomic_replace`` keeps the link and writes the target."""
        real = tmp_path / "real-config.yaml"
        real.write_text(EXISTING_CONFIG, encoding="utf-8")
        config_path.symlink_to(real)

        run_import("claude-code", claude_tree, hermes_home, execute=True)

        assert config_path.is_symlink()
        assert "npm run build" in real.read_text(encoding="utf-8")

    def test_failed_write_does_not_truncate_the_existing_config(
            self, hermes_home, config_path, monkeypatch):
        """An interrupted dump leaves the previous file complete."""
        from hermes_cli import agent_import

        config_path.write_text(EXISTING_CONFIG, encoding="utf-8")
        before = config_path.read_bytes()

        def boom(*_args, **_kwargs):
            raise OSError("no space left on device")

        monkeypatch.setattr(agent_import, "atomic_yaml_write", boom)
        with pytest.raises(OSError):
            agent_import.dump_yaml_file(config_path, {"model": "replacement"})

        assert config_path.read_bytes() == before


# ---------------------------------------------------------------------------
# skills/ durability (sibling of the MEMORY.md and config.yaml cases above)
# ---------------------------------------------------------------------------

SENTINEL = "keep me: hand-edited notes that exist nowhere else\n"


def skill_items(report):
    return [i for i in report["items"] if i["kind"] == "skill"]


class TestImportSkillsDurability:
    """An import must not destroy the skill directory it is replacing.

    ``import_skills`` was the last of import-agent's four destinations with no
    read-safety, no backup and no atomic swap: it called
    ``shutil.rmtree(destination)`` and *then* ``shutil.copytree`` into the hole
    it had just made.  Any failure in between left the user's skill deleted and
    half-rebuilt, and the exception escaped past the per-item report — so the
    user was not even told which items had already landed in config.yaml and
    memories/MEMORY.md.
    """

    @pytest.fixture()
    def dest_root(self, hermes_home):
        root = hermes_home / "skills" / "claude-code-imports"
        root.mkdir(parents=True)
        return root

    @pytest.fixture()
    def seeded_skill(self, dest_root):
        """A previously-imported skill the user has since edited."""
        dest = dest_root / "deploy-helper"
        dest.mkdir()
        (dest / "SKILL.md").write_text("mine\n", encoding="utf-8")
        (dest / "notes.md").write_text(SENTINEL, encoding="utf-8")
        return dest

    # -- destroy-then-copy: the copy fails, the skill is already gone ------

    def test_failed_copy_leaves_the_existing_skill_intact(
            self, claude_tree, hermes_home, seeded_skill, monkeypatch):
        from hermes_cli import agent_import

        def boom(*_args, **_kwargs):
            raise OSError("no space left on device")

        monkeypatch.setattr(agent_import.shutil, "copytree", boom)

        report = run_import("claude-code", claude_tree, hermes_home,
                            execute=True, overwrite=True)

        # The user's skill survives, byte for byte.
        assert (seeded_skill / "SKILL.md").read_text(encoding="utf-8") == "mine\n"
        assert (seeded_skill / "notes.md").read_text(encoding="utf-8") == SENTINEL
        # ... and the failure is reported per item rather than aborting the run.
        assert [i["status"] for i in skill_items(report)] == ["error"]
        assert "no space left on device" in skill_items(report)[0]["reason"]

    def test_failed_copy_leaves_no_staging_directory_behind(
            self, claude_tree, hermes_home, dest_root, seeded_skill,
            monkeypatch):
        from hermes_cli import agent_import

        def boom(*_args, **_kwargs):
            raise OSError("no space left on device")

        monkeypatch.setattr(agent_import.shutil, "copytree", boom)

        run_import("claude-code", claude_tree, hermes_home,
                   execute=True, overwrite=True)

        assert sorted(p.name for p in dest_root.iterdir()) == ["deploy-helper"]

    # -- symlinks in the SOURCE skill --------------------------------------

    def test_source_skill_with_a_dangling_symlink_is_imported(
            self, claude_tree, hermes_home):
        """``copytree`` dereferenced source links, so one dangling link
        aborted the copy — on a FIRST import, with no --overwrite involved."""
        (claude_tree / "skills" / "deploy-helper" / "missing-link").symlink_to(
            "does-not-exist.md")

        report = run_import("claude-code", claude_tree, hermes_home,
                            execute=True)

        assert [i["status"] for i in skill_items(report)] == ["imported"]
        dest = hermes_home / "skills" / "claude-code-imports" / "deploy-helper"
        assert (dest / "SKILL.md").exists()
        # The link is reproduced as a link, still dangling — not dereferenced.
        assert (dest / "missing-link").is_symlink()
        assert not (dest / "missing-link").exists()

    # -- symlinks at the DESTINATION ---------------------------------------

    def test_symlinked_destination_is_written_through_not_clobbered(
            self, claude_tree, hermes_home, dest_root, tmp_path):
        """``rmtree`` refuses a symlink outright ("Cannot call rmtree on a
        symbolic link"), so overwriting a symlinked skill always crashed."""
        real = tmp_path / "external-deploy-helper"
        real.mkdir()
        (real / "SKILL.md").write_text("mine\n", encoding="utf-8")
        link = dest_root / "deploy-helper"
        link.symlink_to(real, target_is_directory=True)

        report = run_import("claude-code", claude_tree, hermes_home,
                            execute=True, overwrite=True)

        assert [i["status"] for i in skill_items(report)] == ["imported"]
        # The link the user set up deliberately survives ...
        assert link.is_symlink()
        assert link.resolve() == real.resolve()
        # ... and the skill behind it is the imported one.
        assert "Deploy things." in (real / "SKILL.md").read_text(encoding="utf-8")

    def test_dangling_destination_symlink_is_a_conflict_without_overwrite(
            self, claude_tree, hermes_home, dest_root):
        """``exists()`` follows links, so a dangling one read as "nothing
        here": no conflict was reported and ``copytree`` died on the link."""
        link = dest_root / "deploy-helper"
        link.symlink_to(dest_root / "gone")

        report = run_import("claude-code", claude_tree, hermes_home,
                            execute=True)

        assert [i["status"] for i in skill_items(report)] == ["conflict"]
        assert link.is_symlink()

    def test_dangling_destination_symlink_is_replaced_with_overwrite(
            self, claude_tree, hermes_home, dest_root):
        link = dest_root / "deploy-helper"
        link.symlink_to(dest_root / "gone")

        report = run_import("claude-code", claude_tree, hermes_home,
                            execute=True, overwrite=True)

        assert [i["status"] for i in skill_items(report)] == ["imported"]
        assert not link.is_symlink()
        assert "Deploy things." in (link / "SKILL.md").read_text(encoding="utf-8")

    # -- cleanup failures are reported, not swallowed ----------------------

    def test_failed_cleanup_is_logged_rather_than_swallowed(
            self, tmp_path, caplog):
        """``rmtree(..., ignore_errors=True)`` would drop a permission problem
        on the floor, leaving a hidden ``.import-*`` sibling with no signal."""
        import os

        if os.name != "posix":
            pytest.skip("chmod-based permission denial is POSIX-only")
        if getattr(os, "geteuid", lambda: 1)() == 0:
            pytest.skip("root bypasses file permissions")

        from hermes_cli import agent_import

        stray = tmp_path / ".deploy-helper.import-abcd1234"
        locked = stray / "locked"
        locked.mkdir(parents=True)
        (locked / "SKILL.md").write_text("staged\n", encoding="utf-8")
        # r-x: the child is readable but the directory is not writable, so it
        # cannot be unlinked.
        locked.chmod(0o500)
        try:
            with caplog.at_level(logging.DEBUG, logger=agent_import.__name__):
                agent_import.discard_path(stray)  # must not raise
        finally:
            locked.chmod(0o700)

        assert "Could not remove temporary path" in caplog.text
        assert "SKILL.md" in caplog.text

    # -- the ordinary overwrite still works --------------------------------

    def test_overwrite_replaces_the_skill_and_leaves_no_leftovers(
            self, claude_tree, hermes_home, dest_root, seeded_skill):
        report = run_import("claude-code", claude_tree, hermes_home,
                            execute=True, overwrite=True)

        assert [i["status"] for i in skill_items(report)] == ["imported"]
        assert "Deploy things." in (
            seeded_skill / "SKILL.md").read_text(encoding="utf-8")
        # The stale member of the old skill is gone (a replace, not a merge) ...
        assert not (seeded_skill / "notes.md").exists()
        # ... and no staging/backup sibling is left behind.
        assert sorted(p.name for p in dest_root.iterdir()) == ["deploy-helper"]


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
