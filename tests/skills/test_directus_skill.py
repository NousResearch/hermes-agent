"""Invariant tests for the directus optional skill.

Covers optional-skills/web-development/directus — Directus 11 CRUD, schema, and the
policy-based permission chain. Every HTTP call is faked; the tests pin the contracts
the SKILL.md promises (the v11 policy chain is walked in order, `fields: []` is not
"all fields", destructive commands need --yes, static tokens go through the user
endpoint, settings resolve flag > env > default) rather than any live server's replies.
"""
from __future__ import annotations

import ast
import importlib.util
import io
import json
import re
import sys
import urllib.error
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO / "optional-skills" / "web-development" / "directus"
SCRIPT = SKILL_DIR / "scripts" / "directus_admin.py"


@pytest.fixture(scope="module")
def directus():
    spec = importlib.util.spec_from_file_location("directus_admin", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def frontmatter() -> dict:
    src = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    m = re.search(r"^---\n(.*?)\n---", src, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("DIRECTUS_URL", "DIRECTUS_TOKEN", "DIRECTUS_EMAIL", "DIRECTUS_PASSWORD", "DIRECTUS_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class FakeHTTP:
    """Records requests and replies from a {(METHOD, path): body} routing table."""

    def __init__(self, routes: dict, default=None):
        self.routes = routes
        self.default = default if default is not None else {"data": {}}
        self.calls: list[tuple[str, str, dict | None]] = []

    def __call__(self, request, timeout=None):
        method = request.get_method()
        full = request.full_url
        path = full.split("://", 1)[1].split("/", 1)[1]
        path = "/" + path.split("?", 1)[0]
        payload = json.loads(request.data.decode()) if request.data else None
        self.calls.append((method, path, payload))
        self.last_request = request
        body = self.routes.get((method, path), self.default)
        if callable(body):
            body = body(len([c for c in self.calls if c[0] == method and c[1] == path]))
        return _Response(json.dumps(body).encode())


def http_error(code: int, body: dict | str = "") -> urllib.error.HTTPError:
    raw = json.dumps(body) if isinstance(body, dict) else body
    return urllib.error.HTTPError("http://d", code, "err", {}, io.BytesIO(raw.encode()))


# ---------------------------------------------------------------------------
# Skill packaging
# ---------------------------------------------------------------------------


def test_frontmatter_contract(frontmatter):
    assert frontmatter["name"] == SKILL_DIR.name
    assert len(frontmatter["description"]) <= 60
    assert frontmatter["description"].endswith(".")
    hermes = frontmatter["metadata"]["hermes"]
    assert {entry["key"] for entry in hermes["config"]} == {"directus.url"}
    assert hermes["category"] == "web-development"


def test_script_is_stdlib_only():
    """The skill promises no pip installs and no Directus SDK."""
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    stdlib = set(sys.stdlib_module_names)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in stdlib, alias.name
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] in stdlib, node.module


def test_skill_md_names_the_shipped_files():
    text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    assert "scripts/directus_admin.py" in text
    assert "references/directus-11-permissions.md" in text
    assert (SKILL_DIR / "references" / "directus-11-permissions.md").is_file()


def test_credentials_are_documented_as_secrets_only():
    """The URL is config; only the credentials belong in .env (root AGENTS.md)."""
    env_example = (REPO / ".env.example").read_text(encoding="utf-8")
    assert "DIRECTUS_TOKEN=" in env_example
    block = env_example.split("directus skill (optional)", 1)[1]
    assert "DIRECTUS_URL=" not in block.split("\n\n")[0]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_setting_precedence_flag_env_default(directus, monkeypatch):
    monkeypatch.setenv("DIRECTUS_URL", "http://from-env:8055")
    assert directus.resolve_setting("http://flag:8055", "DIRECTUS_URL", "") == "http://flag:8055"
    assert directus.resolve_setting(None, "DIRECTUS_URL", "") == "http://from-env:8055"
    monkeypatch.delenv("DIRECTUS_URL")
    assert directus.resolve_setting(None, "DIRECTUS_TIMEOUT", 30.0, float) == 30.0
    monkeypatch.setenv("DIRECTUS_TIMEOUT", "  ")
    assert directus.resolve_setting(None, "DIRECTUS_TIMEOUT", 30.0, float) == 30.0


def test_url_normalisation_keeps_localhost_on_http(directus):
    assert directus.normalise_url("localhost:8055/") == "http://localhost:8055"
    assert directus.normalise_url("127.0.0.1:8055") == "http://127.0.0.1:8055"
    assert directus.normalise_url("cms.example.com") == "https://cms.example.com"
    assert directus.normalise_url("https://cms.example.com/") == "https://cms.example.com"
    with pytest.raises(directus.DirectusError):
        directus.normalise_url("  ")


def test_missing_credentials_is_an_actionable_error(directus):
    with pytest.raises(directus.DirectusError) as exc:
        directus.DirectusClient(url="http://localhost:8055")
    assert "DIRECTUS_TOKEN" in str(exc.value)


def test_json_argument_accepts_inline_file_and_stdin(directus, tmp_path, monkeypatch):
    assert directus.parse_json_arg('{"a": 1}', "--data") == {"a": 1}
    path = tmp_path / "payload.json"
    path.write_text('{"b": 2}', encoding="utf-8")
    assert directus.parse_json_arg(f"@{path}", "--data") == {"b": 2}
    monkeypatch.setattr("sys.stdin", io.StringIO('{"c": 3}'))
    assert directus.parse_json_arg("-", "--data") == {"c": 3}
    assert directus.parse_json_arg(None, "--data") is None
    with pytest.raises(directus.DirectusError) as exc:
        directus.parse_json_arg("{nope", "--filter")
    assert "--filter is not valid JSON" in str(exc.value)


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------


def test_static_token_is_sent_as_a_bearer_header(directus):
    http = FakeHTTP({("GET", "/server/info"): {"data": {"version": "11.2.0"}}})
    client = directus.DirectusClient("http://localhost:8055", token="tok", opener=http)
    assert client.server_info()["version"] == "11.2.0"
    assert http.last_request.get_header("Authorization") == "Bearer tok"


def test_login_is_exchanged_for_an_access_token_once(directus):
    http = FakeHTTP({
        ("POST", "/auth/login"): {"data": {"access_token": "temp-123"}},
        ("GET", "/server/info"): {"data": {"version": "11.0.0"}},
    })
    client = directus.DirectusClient("http://localhost:8055", email="a@b.co", password="pw", opener=http)
    client.server_info()
    client.server_info()
    logins = [call for call in http.calls if call[1] == "/auth/login"]
    assert len(logins) == 1
    assert logins[0][2] == {"email": "a@b.co", "password": "pw"}


def test_403_explains_the_policy_chain(directus):
    def boom(request, timeout=None):
        raise http_error(403, {"errors": [{"message": "You don't have permission", "extensions": {"code": "FORBIDDEN"}}]})

    client = directus.DirectusClient("http://localhost:8055", token="t", opener=boom)
    with pytest.raises(directus.DirectusError) as exc:
        client.request("GET", "/items/agent_tasks")
    message = str(exc.value)
    assert "FORBIDDEN" in message and "directus_access" in message


def test_429_retries_with_backoff_then_succeeds(directus):
    calls = {"n": 0}
    slept: list[float] = []

    def flaky(request, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise http_error(429, "rate limited")
        return _Response(json.dumps({"data": [{"id": 1}]}).encode())

    client = directus.DirectusClient(
        "http://localhost:8055", token="t", opener=flaky, sleep=slept.append
    )
    assert client.request("GET", "/items/x") == [{"id": 1}]
    assert slept == [1.0, 3.0]


def test_non_json_body_names_the_likely_cause(directus):
    def html(request, timeout=None):
        return _Response(b"<html>502 Bad Gateway</html>")

    client = directus.DirectusClient("http://localhost:8055", token="t", opener=html)
    with pytest.raises(directus.DirectusError) as exc:
        client.server_info()
    assert "non-JSON" in str(exc.value)


def test_pagination_stops_on_a_short_page(directus):
    page_size = 3

    def pages(request, timeout=None):
        query = dict(part.split("=", 1) for part in request.full_url.split("?", 1)[1].split("&"))
        offset = int(query["offset"])
        rows = [{"id": offset + i} for i in range(page_size if offset < 6 else 1)]
        return _Response(json.dumps({"data": rows}).encode())

    client = directus.DirectusClient("http://localhost:8055", token="t", opener=pages)
    rows = client.paginate("/items/x", {}, page_size=page_size)
    assert [row["id"] for row in rows] == [0, 1, 2, 3, 4, 5, 6]


# ---------------------------------------------------------------------------
# Directus 11 model
# ---------------------------------------------------------------------------


def test_version_guard_names_the_v10_difference(directus):
    http = FakeHTTP({("GET", "/server/info"): {"data": {"version": "10.13.1"}}})
    client = directus.DirectusClient("http://localhost:8055", token="t", opener=http)
    with pytest.raises(directus.DirectusError) as exc:
        client.require_v11("Access policies")
    assert "Directus 11+" in str(exc.value) and "10.13.1" in str(exc.value)


def test_version_guard_passes_on_11_and_above(directus):
    http = FakeHTTP({("GET", "/server/info"): {"data": {"version": "11.5.0"}}})
    client = directus.DirectusClient("http://localhost:8055", token="t", opener=http)
    client.require_v11("Access policies")  # does not raise
    assert directus.major_version("v12.0.0") == 12
    assert directus.major_version("") is None


def test_permission_fields_default_to_all_not_empty(directus):
    payload = directus.permission_payload("p1", "agent_tasks", "read", None, None, None)
    assert payload["fields"] == ["*"]
    assert payload["permissions"] == {} and payload["validation"] == {}


def test_empty_fields_is_warned_about_because_it_means_primary_key_only(directus, capsys):
    directus.check_fields_trap([], "read")
    assert "primary key ONLY" in capsys.readouterr().err
    directus.check_fields_trap(["*"], "read")
    assert capsys.readouterr().err == ""


def test_unknown_action_is_rejected_before_the_request(directus):
    with pytest.raises(directus.DirectusError) as exc:
        directus.permission_payload("p1", "c", "upsert", None, None, None)
    assert "create, read, update, delete, share" in str(exc.value)


def test_bootstrap_plan_walks_policy_permissions_role_access_user_token(directus):
    steps = [s["step"] for s in directus.bootstrap_plan("dao", ["a", "b"], ["read", "create"], "x@y.co", False)]
    assert steps == [
        "create_policy",
        "create_permission", "create_permission", "create_permission", "create_permission",
        "create_role",
        "link_access",
        "create_user",
        "set_static_token",
    ]
    assert "create_user" not in [s["step"] for s in directus.bootstrap_plan("dao", ["a"], ["read"], None, False)]


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def test_field_spec_parsing_and_required_marker(directus):
    assert directus.parse_field_spec("agent_id:string!") == {"field": "agent_id", "type": "string", "required": True}
    assert directus.parse_field_spec("payload:json")["type"] == "json"
    assert directus.parse_field_spec("name")["type"] == "string"
    with pytest.raises(directus.DirectusError):
        directus.parse_field_spec("score:money")
    with pytest.raises(directus.DirectusError):
        directus.parse_field_spec(":string")


def test_required_field_is_not_nullable(directus):
    payload = directus.field_payload(directus.parse_field_spec("agent_id:string!"))
    assert payload["meta"]["required"] is True
    assert payload["schema"]["is_nullable"] is False


def test_collection_payload_always_declares_a_primary_key(directus):
    payload = directus.collection_payload("agent_tasks", [directus.parse_field_spec("status:string")], "uuid", None, False)
    pk = payload["fields"][0]
    assert pk["field"] == "id" and pk["schema"]["is_primary_key"] is True
    assert directus.collection_payload("c", [], "auto", None, False)["fields"][0]["schema"]["has_auto_increment"] is True


def test_system_collections_are_refused(directus):
    with pytest.raises(directus.DirectusError) as exc:
        directus.guard_system_collection("directus_users", "delete")
    assert "system collection" in str(exc.value)
    directus.guard_system_collection("agent_tasks", "delete")  # does not raise


# ---------------------------------------------------------------------------
# CLI end to end
# ---------------------------------------------------------------------------


def _run(directus, monkeypatch, http, argv):
    monkeypatch.setattr(directus.urllib.request, "urlopen", http)
    monkeypatch.setenv("DIRECTUS_URL", "http://localhost:8055")
    monkeypatch.setenv("DIRECTUS_TOKEN", "tok")
    return directus.main(argv)


def test_cli_check_reports_the_policy_model(directus, monkeypatch, capsys):
    http = FakeHTTP({
        ("GET", "/server/info"): {"data": {"version": "11.4.0", "project": {"project_name": "DAO"}}},
        ("GET", "/users/me"): {"data": {"email": "admin@example.com", "role": {"name": "Admin", "admin_access": True}}},
    })
    assert _run(directus, monkeypatch, http, ["check"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["version"] == "11.4.0"
    assert out["policy_model"].startswith("policy-based")
    assert out["project"] == "DAO" and out["admin_access"] is True


def test_cli_check_exits_nonzero_on_directus_10(directus, monkeypatch, capsys):
    http = FakeHTTP({
        ("GET", "/server/info"): {"data": {"version": "10.10.0"}},
        ("GET", "/users/me"): {"data": {"email": "a@b.co", "role": {"admin_access": True}}},
    })
    assert _run(directus, monkeypatch, http, ["check"]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["policy_model"].startswith("role-based")
    assert "targets Directus 11+" in captured.err


def test_cli_items_list_sends_filter_and_fields_as_query_params(directus, monkeypatch, capsys):
    http = FakeHTTP({("GET", "/items/agent_tasks"): {"data": [{"id": 1}], "meta": {"total_count": 1}}})
    code = _run(directus, monkeypatch, http, [
        "items", "list", "agent_tasks",
        "--filter", '{"status":{"_eq":"queued"}}',
        "--fields", "id, status",
        "--sort=-date_created",
        "--limit", "5",
    ])
    assert code == 0
    query = dict(part.split("=", 1) for part in http.last_request.full_url.split("?", 1)[1].split("&"))
    from urllib.parse import unquote_plus
    assert json.loads(unquote_plus(query["filter"])) == {"status": {"_eq": "queued"}}
    assert unquote_plus(query["fields"]) == "id,status"
    assert query["limit"] == "5" and unquote_plus(query["sort"]) == "-date_created"
    assert json.loads(capsys.readouterr().out)["returned"] == 1


def test_cli_item_delete_requires_yes(directus, monkeypatch, capsys):
    http = FakeHTTP({})
    assert _run(directus, monkeypatch, http, ["items", "delete", "agent_tasks", "7"]) == 2
    assert "--yes" in capsys.readouterr().err
    assert http.calls == []
    assert _run(directus, monkeypatch, http, ["items", "delete", "agent_tasks", "7", "--yes"]) == 0
    assert http.calls[-1][:2] == ("DELETE", "/items/agent_tasks/7")


def test_cli_dry_run_sends_nothing(directus, monkeypatch, capsys):
    http = FakeHTTP({})
    code = _run(directus, monkeypatch, http, [
        "collections", "create", "agent_heartbeats", "--field", "agent_id:string!", "--dry-run",
    ])
    assert code == 0 and http.calls == []
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] is True and out["endpoint"] == "POST /collections"
    assert [f["field"] for f in out["payload"]["fields"]] == ["id", "agent_id"]


def test_cli_token_set_uses_the_user_endpoint_and_prints_the_value_once(directus, monkeypatch, capsys):
    http = FakeHTTP({("PATCH", "/users/u-1"): {"data": {"id": "u-1"}}})
    assert _run(directus, monkeypatch, http, ["token", "set", "u-1"]) == 0
    method, path, payload = http.calls[-1]
    assert (method, path) == ("PATCH", "/users/u-1")
    captured = capsys.readouterr()
    token = json.loads(captured.out)["static_token"]
    assert payload == {"token": token} and len(token) > 20
    assert "never shows it again" in captured.err


def test_cli_bootstrap_agent_creates_the_whole_chain_in_order(directus, monkeypatch, capsys):
    http = FakeHTTP({
        ("GET", "/server/info"): {"data": {"version": "11.4.0"}},
        ("POST", "/policies"): {"data": {"id": "pol-1"}},
        ("POST", "/permissions"): {"data": {"id": "perm"}},
        ("POST", "/roles"): {"data": {"id": "role-1"}},
        ("POST", "/access"): {"data": {"id": "acc-1"}},
        ("POST", "/users"): {"data": {"id": "usr-1"}},
        ("PATCH", "/users/usr-1"): {"data": {"id": "usr-1"}},
    })
    code = _run(directus, monkeypatch, http, [
        "bootstrap-agent", "dao-07",
        "--collections", "agent_tasks,agent_heartbeats",
        "--actions", "read,create",
        "--email", "dao-07@example.com",
    ])
    assert code == 0
    ordered = [(method, path) for method, path, _ in http.calls if method != "GET"]
    assert ordered == [
        ("POST", "/policies"),
        ("POST", "/permissions"), ("POST", "/permissions"),
        ("POST", "/permissions"), ("POST", "/permissions"),
        ("POST", "/roles"),
        ("POST", "/access"),
        ("POST", "/users"),
        ("PATCH", "/users/usr-1"),
    ]
    access_payload = next(p for m, path, p in http.calls if path == "/access")
    assert access_payload == {"role": "role-1", "policy": "pol-1"}
    result = json.loads(capsys.readouterr().out)
    assert result["policy"] == "pol-1" and result["role"] == "role-1" and result["access"] == "acc-1"
    assert len(result["permissions"]) == 4
    assert result["static_token"]


def test_cli_bootstrap_agent_never_grants_admin_access(directus, monkeypatch, capsys):
    http = FakeHTTP({
        ("GET", "/server/info"): {"data": {"version": "11.4.0"}},
        ("POST", "/policies"): {"data": {"id": "pol-1"}},
        ("POST", "/permissions"): {"data": {"id": "perm"}},
        ("POST", "/roles"): {"data": {"id": "role-1"}},
        ("POST", "/access"): {"data": {"id": "acc-1"}},
    })
    assert _run(directus, monkeypatch, http, ["bootstrap-agent", "dao", "--collections", "a"]) == 0
    policy_payload = next(p for _m, path, p in http.calls if path == "/policies")
    assert policy_payload["admin_access"] is False
    assert policy_payload["app_access"] is False
    permission_payload = next(p for _m, path, p in http.calls if path == "/permissions")
    assert permission_payload["fields"] == ["*"]
    capsys.readouterr()


def test_cli_bootstrap_agent_rejects_an_unknown_action_before_connecting(directus, monkeypatch, capsys):
    http = FakeHTTP({})
    code = _run(directus, monkeypatch, http, ["bootstrap-agent", "dao", "--collections", "a", "--actions", "purge"])
    assert code == 2 and http.calls == []
    assert "Unknown action" in capsys.readouterr().err
