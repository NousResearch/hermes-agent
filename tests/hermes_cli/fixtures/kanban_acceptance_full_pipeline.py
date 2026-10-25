"""SYNTHETIC provider transport driving the ACTUAL production MCP factory.

Only the HTTP transport is replaced. App auth, provider normalization, collection,
MCP stdio, native collector and SQLite completion all remain real code paths.
No real credential or GitHub request occurs. Never use this entrypoint in a profile.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "policy-mcp"))
from github_acceptance_mcp import github as g, server

scenario = sys.argv[1] if len(sys.argv) == 2 else "invalid"
if scenario not in {"success", "failed", "head_moved", "none_required"}:
    raise SystemExit(2)
app_id = int(os.environ["GITHUB_APP_ID"])
installation_id = int(os.environ["GITHUB_APP_INSTALLATION_ID"])
head = "a" * 40
required = [] if scenario == "none_required" else [
    {"context": "required", "app": {"databaseId": app_id}}]

def snapshot(sha):
    return {"data": {"repository": {"id": "SYNTHETIC_REPO", "pullRequest": {
        "id": "SYNTHETIC_PR", "state": "MERGED", "headRefOid": sha,
        "baseRefName": "main", "baseRef": {
            "branchProtectionRule": {"requiredStatusChecks": required}}}}}}

rules = [] if scenario == "none_required" else [{
    "type": "required_status_checks", "parameters": {
        "required_status_checks": [{"context": "status-required", "integration_id": None}]}}]
checks = [
    {"id": 42, "name": "required", "head_sha": head, "app": {"id": app_id},
     "status": "completed", "conclusion": "failure" if scenario == "failed" else "success",
     "html_url": "https://github.com/acme/repo/checks/42"},
    {"id": 43, "name": "optional-failure", "head_sha": head, "app": {"id": app_id},
     "status": "completed", "conclusion": "failure",
     "html_url": "https://github.com/acme/repo/checks/43"}]
statuses = [{"id": 44, "context": "status-required", "state": "success",
             "target_url": "https://ci.example.test/synthetic/44"}]
script = {
    "GET /app": [(200, {"id": app_id})],
    f"GET /app/installations/{installation_id}": [(200, {
        "id": installation_id, "app_id": app_id, "account": {"login": "acme"}})],
    f"POST /app/installations/{installation_id}/access_tokens": [(201, {
        "token": "synthetic-only-never-real", "permissions": g.MINIMAL_READ_PERMISSIONS})],
    "GET /installation/repositories": [(200, {"total_count": 1, "repositories": [{"full_name": "acme/repo"}]})],
    "POST /graphql": [(200, snapshot(head)),
                     (200, snapshot("b" * 40 if scenario == "head_moved" else head))],
    "GET /repos/acme/repo/rules/branches/main": [(200, rules)],
    f"GET /repos/acme/repo/commits/{head}/check-runs": [(200, {"total_count": 2, "check_runs": checks})],
    f"GET /repos/acme/repo/commits/{head}/statuses": [(200, statuses)],
}
transport = g.FakeHTTP(script)
server.HttpxTransport = lambda: transport  # TEST-ONLY; production origin remains hardcoded.
os.environ["GITHUB_APP_PRIVATE_KEY"] = g.generate_synthetic_private_key_pem()
instance = server.create_server()  # actual production env factory + App client
try:
    asyncio.run(server.run_server(instance))
finally:
    trace = os.environ.get("PIPELINE_TRACE")
    if trace:
        Path(trace).write_text(json.dumps({
            "synthetic": True, "scenario": scenario,
            "call_paths": [c["method"] + " " + c["path"] for c in transport.calls],
            "jwt_algorithm": "RS256", "real_network": False,
            "ambient_token_forwarded": "GITHUB_TOKEN" in os.environ or "GH_TOKEN" in os.environ,
        }), encoding="utf-8")
