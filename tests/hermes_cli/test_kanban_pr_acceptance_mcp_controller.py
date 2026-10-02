"""Parent semantic regressions; all inputs/provider material SYNTHETIC."""
import pytest
from mcp.types import CallToolResult
from hermes_cli import kanban_pr_acceptance_mcp as m
from tests.hermes_cli.test_kanban_pr_acceptance_mcp import _evidence


def test_sdk2_native_result_preserves_structured_wire_mirror():
    import json
    evidence = _evidence(11)
    wire = {"content": [{"type": "text", "text": json.dumps(evidence)}],
            "structuredContent": evidence, "isError": False}
    native = CallToolResult.model_validate(wire)
    assert m._decode_evidence(native) == evidence


def test_active_home_uses_its_own_scoped_values(tmp_path, monkeypatch):
    from tests.hermes_cli.test_kanban_pr_acceptance_mcp import _write_profile
    home = tmp_path / "default"
    _write_profile(home, 99)
    monkeypatch.setenv("HERMES_HOME", str(home))
    server = m._raw_server_entry(None)
    env = m._server_env(server, None)
    assert env["GITHUB_APP_ID"] == "99"


def test_mcp_only_completer_cannot_downgrade_when_target_has_no_selector(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_pr_acceptance as native
    from hermes_cli import kanban_pr_acceptance_store as store
    from tests.hermes_cli.test_kanban_pr_acceptance_mcp import _write_profile, PR_URL
    from hermes_cli.kanban_db_connect import connect
    home = tmp_path / "default"
    _write_profile(home, 99)
    target = home / "profiles" / "sandbox"
    target.mkdir(parents=True)
    (target / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(native, "_assignee_profile_home", lambda _: str(target))
    gh_calls = []
    def forbidden_gh(*args, **kwargs):
        gh_calls.append(True)
        return {"ok": False, "classification": "auth", "head_sha": None,
                "pr_url": PR_URL, "checks": [], "detail": "synthetic gh path",
                "recovery": "synthetic only"}
    monkeypatch.setattr(store, "collect_acceptance", forbidden_gh)
    kb.init_db()
    with connect() as conn:
        task = kb.create_task(conn, title="fixture", completion_contract="acme/repo", assignee="sandbox")
        assert not kb.complete_task(conn, task, result="done", metadata={"published_pr": PR_URL})
        assert not gh_calls
        assert "MCP" in kb.get_task(conn, task).last_failure_error


def test_external_owner_secret_source_works_without_plaintext_dotenv(tmp_path, monkeypatch):
    from tools import mcp_tool_discovery
    own = tmp_path / "owned-profile"
    own.mkdir()
    visited = []
    def mapping(home):
        visited.append(home.resolve())
        assert home.resolve() == own.resolve()
        return {"APP_ID": "41", "INSTALL_ID": "1041", "APP_KEY": "synthetic-private-marker"}
    monkeypatch.setattr(mcp_tool_discovery, "_owner_secret_mapping", mapping)
    monkeypatch.setenv("APP_ID", "ambient-must-not-be-used")
    cfg = {"env": {"GITHUB_APP_ID": "${APP_ID}",
                   "GITHUB_APP_INSTALLATION_ID": "${INSTALL_ID}",
                   "GITHUB_APP_PRIVATE_KEY": "${APP_KEY}"}}
    env = m._server_env(cfg, str(own))
    assert env["GITHUB_APP_ID"] == "41"
    assert env["GITHUB_APP_INSTALLATION_ID"] == "1041"
    assert len(visited) == 1
    assert not (own / ".env").exists()


def test_malformed_config_cannot_downgrade_to_gh(tmp_path):
    own = tmp_path / "own"
    own.mkdir()
    (own / "config.yaml").write_text("kanban: [\n", encoding="utf-8")
    with pytest.raises(m._McpConfigError):
        m.read_transport(str(own))


@pytest.mark.parametrize("value", ["MCP", " mcp", "mcp ", "mcp\n"])
def test_transport_enum_is_preserved_not_repaired(tmp_path, value):
    import json
    own = tmp_path / "own"
    own.mkdir()
    (own / "config.yaml").write_text(json.dumps({"kanban": {"pr_acceptance": {"transport": value}}}), encoding="utf-8")
    with pytest.raises(m._McpConfigError):
        m.read_transport(str(own))


def test_disabled_server_is_refused_before_own_secret_hydration(tmp_path):
    import json
    own = tmp_path / "own"
    own.mkdir()
    entry = {"enabled": False, "command": "must-not-run", "args": []}
    (own / "config.yaml").write_text(json.dumps({"mcp_servers": {"github_acceptance": entry}}), encoding="utf-8")
    with pytest.raises(m._McpConfigError):
        m._raw_server_entry(str(own))
