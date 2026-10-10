"""Explicit MCP acceptance transport (interface v1): scoped stdio client.

Opted in per profile with ``kanban.pr_acceptance.transport=mcp``. In this mode PR
acceptance evidence must come from the named ``github_acceptance`` stdio MCP server
defined in the completing profile's OWN config — the default scope's server, the
ambient process credentials, ``gh``, the REST client and any other transport are never
consulted, and a missing/incomplete server fails closed instead of falling back.

The server's environment is built strictly: OS plumbing keys plus the server's
configured env, whose ``${VAR}`` references expand ONLY against that profile's own
``.env`` secret scope. Ambient ``GITHUB_*``/``GH_*`` residue and plaintext config
secrets never reach the child; secret values are never printed, logged or persisted.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from pathlib import Path

ACCEPTANCE_SERVER_NAME = "github_acceptance"
ACCEPTANCE_TOOL_NAME = "get_pr_acceptance_evidence"
MCP_TRANSPORT = "mcp"
GH_TRANSPORT = "gh"
_CALL_TIMEOUT_SECONDS = 60

_MCP_FIXED_RECOVERY = ("Fix the acceptance transport configuration for the completing profile, "
                       "then retry completion. Use kanban_block if human input is needed; "
                       "receipts remain on the task event log.")
MCP_CONFIG_AUTH_DETAIL = ("The profile's own github_acceptance MCP server is not configured or "
                          "incomplete; MCP acceptance mode never falls back to gh or any other "
                          "transport. Configure the named stdio server on this profile, then retry.")
MCP_EVIDENCE_INVALID_DETAIL = ("The github_acceptance MCP server returned incomplete or malformed "
                               "acceptance evidence; retry completion.")
MCP_UNAVAILABLE_DETAIL = ("The github_acceptance MCP server could not be reached or did not answer "
                          "in time; retry completion.")

_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_PR = re.compile(r"https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/pull/([1-9][0-9]*)")
_SHA = re.compile(r"[0-9a-f]{40}")
_ENV_REF = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
# Non-secret OS plumbing a stdio child needs to run; everything credential-shaped
# must come from the profile's own .env scope via the configured env references.
_SAFE_BASE_KEYS = ("PATH", "SYSTEMROOT", "SYSTEMDRIVE", "COMSPEC", "PATHEXT", "TEMP", "TMP",
                   "APPDATA", "LOCALAPPDATA", "PROGRAMFILES", "PROGRAMW6432", "WINDIR")
_APP_CREDENTIAL_KEYS = ("GITHUB_APP_ID", "GITHUB_APP_INSTALLATION_ID", "GITHUB_APP_PRIVATE_KEY")


class _McpConfigError(RuntimeError):
    """The transport setting is unusable (unknown value) or the named server is
    missing/incomplete on the completing profile: an authority/config problem,
    never something to work around with another transport."""


class _McpUnavailableError(RuntimeError):
    """The configured server could not be spawned, timed out or crashed."""


class _McpInvalidError(RuntimeError):
    """The server answered with isError, malformed, mismatched or duplicated
    material: hostile output is denied whole, never partially trusted."""


def _native_receipt(published_pr: str | None) -> dict:
    return {"ok": False, "classification": "missing", "head_sha": None,
            "pr_url": published_pr, "checks": [], "required": [],
            "recovery": ("Fix required failures, rerun infrastructure checks or wait, then retry "
                         "completion. Use kanban_block if human input is needed; receipts remain "
                         "on the task event log.")}


def _profile_config_path(profile_home: str | None) -> Path:
    if profile_home:
        return Path(profile_home) / "config.yaml"
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "config.yaml"


def read_transport(profile_home: str | None = None) -> str | None:
    """The completing profile's own ``kanban.pr_acceptance.transport``.

    ``None``/``gh`` keeps the native collector; ``mcp`` opts into MCP-only mode. The
    profile has no config section at all -> None (native behavior unchanged). An
    unreadable config fails closed. An UNKNOWN value also fails closed —
    neither may silently degrade to gh.
    """
    from hermes_cli.config_effective import load_user_config_effective
    try:
        config = load_user_config_effective(_profile_config_path(profile_home), fail_closed=True)
    except Exception:
        raise _McpConfigError("acceptance config is unreadable") from None
    section = (config.get("kanban") or {}).get("pr_acceptance") or {}
    value = section.get("transport")
    if value is None:
        return None
    if not isinstance(value, str):
        raise _McpConfigError("transport must be an exact enum string")
    if value == GH_TRANSPORT:
        return GH_TRANSPORT
    if value == MCP_TRANSPORT:
        return MCP_TRANSPORT
    raise _McpConfigError("kanban.pr_acceptance.transport supports only 'gh' or 'mcp'")


def unsupported_transport_receipt(published_pr: str | None) -> dict:
    receipt = _native_receipt(published_pr)
    receipt.update(classification="auth", detail=MCP_CONFIG_AUTH_DETAIL)
    return receipt


def _raw_server_entry(profile_home: str | None) -> dict:
    """The profile's own ``mcp_servers[github_acceptance]`` entry, read raw.

    Deliberately not the effective loader: it expands ``${VAR}`` against the ambient
    process env, which would leak the launching profile's credentials into the child.
    """
    path = _profile_config_path(profile_home)
    try:
        from hermes_yaml import safe_load
        config = safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        raise _McpConfigError("no github_acceptance MCP server is configured for this profile")
    except Exception:
        raise _McpConfigError("the completing profile's config.yaml is unreadable") from None
    entry = (config.get("mcp_servers") or {}).get(ACCEPTANCE_SERVER_NAME)
    if not isinstance(entry, dict):
        raise _McpConfigError("no github_acceptance MCP server is configured for this profile")
    if entry.get("enabled", True) is not True:
        raise _McpConfigError("the acceptance server is disabled")
    command = entry.get("command")
    if isinstance(command, str):
        command = [command]
    args = entry.get("args") or []
    if not (isinstance(command, list) and command and all(isinstance(part, str) for part in command)
            and isinstance(args, list) and all(isinstance(part, str) for part in args)):
        raise _McpConfigError("the github_acceptance MCP server entry is malformed")
    return {"command": command, "args": args, "env": entry.get("env") or {}}


def _profile_dotenv(profile_home: str | None) -> dict[str, str]:
    """Resolve this owner's native secret scope, including external sources."""
    from tools.mcp_tool_discovery import _owner_secret_mapping
    home = _profile_config_path(profile_home).parent
    try:
        values = _owner_secret_mapping(home)
    except Exception:
        raise _McpConfigError("owner secret source unavailable") from None
    if not isinstance(values, dict):
        raise _McpConfigError("owner secret source invalid")
    return values


def _server_env(server_config: dict, profile_home: str | None) -> dict[str, str]:
    dotenv = _profile_dotenv(profile_home)
    env = {key: os.environ[key] for key in _SAFE_BASE_KEYS if key in os.environ}
    configured = server_config.get("env") or {}
    if not isinstance(configured, dict):
        raise _McpConfigError("the github_acceptance server env must be a mapping")
    for key, value in configured.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise _McpConfigError("the github_acceptance server env must be string-valued")
        ref = _ENV_REF.fullmatch(value.strip())
        if ref:
            if ref[1] not in dotenv:
                raise _McpConfigError(
                    f"server env {key} references {ref[1]}, which this profile's own .env "
                    "does not provide")
            env[key] = dotenv[ref[1]]
        else:
            env[key] = value
        if key in _APP_CREDENTIAL_KEYS and not ref:
            # App credentials reach the server only through the profile's own secret
            # scope; a plaintext value in config.yaml is refused.
            raise _McpConfigError(
                f"server env {key} must reference this profile's own .env via ${{VAR}}")
    missing = [key for key in _APP_CREDENTIAL_KEYS if key not in env]
    if missing:
        raise _McpConfigError(
            "the github_acceptance server env must provide the GitHub App credentials: "
            + ", ".join(missing))
    return env


def _call_tool(server_config: dict, profile_home: str | None, arguments: dict):
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        raise _McpUnavailableError("the MCP client SDK is unavailable") from None
    params = StdioServerParameters(command=server_config["command"][0],
                                   args=server_config["command"][1:] + server_config["args"],
                                   env=_server_env(server_config, profile_home))
    holder: dict = {}

    def run():
        async def session_call():
            # The timeout cancels the whole async-with stack, so stdio_client tears
            # down the streams and terminates the child: bounded, reliable cleanup.
            async with asyncio.timeout(_CALL_TIMEOUT_SECONDS):
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        try:
                            return await session.call_tool(ACCEPTANCE_TOOL_NAME, arguments)
                        except RuntimeError:
                            # SDK output-schema violations are invalid evidence,
                            # not permission to accept a text-only reply.
                            raise _McpInvalidError("invalid tool result") from None

        try:
            holder["result"] = asyncio.run(session_call())
        except BaseException as exc:  # noqa: BLE001 — sanitized into a fixed code below
            holder["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(_CALL_TIMEOUT_SECONDS + 5)
    if thread.is_alive():
        raise _McpUnavailableError("the acceptance server did not answer in time")
    if "error" in holder:
        def invalid(exc):
            if isinstance(exc, _McpInvalidError):
                return True
            return isinstance(exc, BaseExceptionGroup) and any(invalid(x) for x in exc.exceptions)
        if invalid(holder["error"]):
            raise _McpInvalidError("invalid tool result") from None
        raise _McpUnavailableError("the acceptance server session failed")
    return holder["result"]


def _decode_evidence(result) -> dict:
    # MCP 2 exposes snake_case Python attributes while retaining camelCase
    # protocol aliases. Keep compatibility with SDK 1 without relaxing v1.
    if getattr(result, "is_error", getattr(result, "isError", False)):
        raise _McpInvalidError("the server reported a tool error")
    structured = getattr(result, "structured_content", getattr(result, "structuredContent", None))
    if not isinstance(structured, dict) or structured.get("schema_version") != 1:
        raise _McpInvalidError("missing or unversioned structuredContent")
    texts = [block for block in (result.content or [])
             if getattr(block, "type", None) == "text"]
    if len(texts) != 1:
        raise _McpInvalidError("expected exactly one JSON text block")
    try:
        mirrored = json.loads(texts[0].text)
    except (TypeError, ValueError):
        raise _McpInvalidError("the text block is not JSON") from None
    if mirrored != structured:
        raise _McpInvalidError("the text block does not mirror the structured content")
    return structured


def _int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_evidence(evidence: dict, owner: str, repo: str, number: int) -> None:
    """Full v1 shape/identity/count validation before any field is trusted."""
    def bad(reason):
        raise _McpInvalidError(reason)

    if evidence.get("repository") != f"{owner}/{repo}" or evidence.get("pull_number") != number:
        bad("evidence does not match the requested repository/PR")
    identity = evidence.get("identity")
    if not isinstance(identity, dict) or identity.get("kind") != "github_app" \
            or not _int(identity.get("app_id")) or not _int(identity.get("installation_id")):
        bad("invalid identity")
    for phase in ("pr_before", "pr_after"):
        snapshot = evidence.get(phase)
        if not isinstance(snapshot, dict) or not _SHA.fullmatch(str(snapshot.get("head_sha"))) \
                or snapshot.get("state") not in {"OPEN", "MERGED", "CLOSED"} \
                or not str(snapshot.get("base_ref") or ""):
            bad(f"invalid {phase}")
    for field in ("classic_required", "ruleset_required"):
        entries = evidence.get(field)
        if not isinstance(entries, list) or not all(
                isinstance(entry, dict) and isinstance(entry.get("context"), str)
                and (entry.get("app_id") is None or _int(entry.get("app_id")))
                for entry in entries):
            bad(f"invalid {field}")
    runs = evidence.get("check_runs")
    head = evidence["pr_before"]["head_sha"]
    if not isinstance(runs, list) or not all(
            isinstance(run, dict) and _int(run.get("id")) and isinstance(run.get("name"), str)
            and run.get("head_sha") == head and _int(run.get("app_id"))
            and isinstance(run.get("status"), str)
            and isinstance(run.get("conclusion"), str) and isinstance(run.get("url"), str)
            for run in runs):
        bad("invalid check_runs")
    statuses = evidence.get("statuses")
    if not isinstance(statuses, list) or not all(
            isinstance(status, dict) and _int(status.get("id"))
            and isinstance(status.get("context"), str) and status.get("sha") == head
            and isinstance(status.get("state"), str) for status in statuses):
        bad("invalid statuses")
    counts = evidence.get("counts")
    if not isinstance(counts, dict) or not _int(counts.get("check_runs")) \
            or not _int(counts.get("statuses")) or counts["check_runs"] != len(runs) \
            or counts["statuses"] != len(statuses):
        bad("counts do not match the reported lists")
    complete = evidence.get("complete")
    if not isinstance(complete, dict) or set(complete) != {"classic", "rulesets",
                                                          "check_runs", "statuses"} \
            or not all(value is True for value in complete.values()):
        # Partial evidence is denied whole: an incomplete policy read is never a PASS.
        bad("evidence is incomplete")
    if evidence["pr_before"] != evidence["pr_after"]:
        bad("PR head/base/state changed while collecting evidence")
    if evidence["pr_after"]["state"] not in {"OPEN", "MERGED"}:
        bad("PR is closed")


_CONCLUSION_CLASSIFICATION = {"success": "success", "failure": "failure",
                              "error": "infra", "pending": "pending"}


def _classify(item: dict, head_sha: str, is_run: bool) -> str:
    if item.get("head_sha", item.get("sha")) != head_sha:
        return "stale"
    if is_run and item.get("status") != "completed":
        return "pending"
    return _CONCLUSION_CLASSIFICATION.get(
        item.get("conclusion") if is_run else item.get("state"), "infra")


def _evaluate(evidence: dict, owner: str, repo: str, number: int) -> dict:
    receipt = _native_receipt(None)
    head_sha = evidence["pr_before"]["head_sha"]
    receipt["head_sha"] = head_sha
    receipt["pr_url"] = f"https://github.com/{owner}/{repo}/pull/{number}"
    required = {(entry["context"], entry.get("app_id"))
                for entry in evidence["classic_required"] + evidence["ruleset_required"]}
    receipt["required"] = [{"context": context, "app_id": app_id}
                           for context, app_id in sorted(required, key=str)]
    if not required:
        receipt["detail"] = ("No repository-required checks are configured; explicitly use a "
                             "local-only contract for non-CI tasks.")
        return receipt
    outcomes = []
    for context, app_id in sorted(required, key=str):
        matching = [run for run in evidence["check_runs"]
                    if run["name"] == context and (app_id in (None, -1) or run["app_id"] == app_id)]
        # A legacy status can satisfy an unpinned context, but never a check pinned to an app.
        legacy = ([max((status for status in evidence["statuses"]
                        if status["context"] == context), key=lambda s: s["id"])]
                  if app_id in (None, -1) and any(
                      status["context"] == context for status in evidence["statuses"]) else [])
        selected = matching + legacy
        if not selected:
            outcomes.append("missing")
            receipt["checks"].append({"name": context, "classification": "missing",
                                      "head_sha": head_sha})
        for item in selected:
            is_run = "conclusion" in item
            outcome = item.get("conclusion") if is_run else item.get("state")
            classification = _classify(item, head_sha, is_run)
            outcomes.append(classification)
            receipt["checks"].append({"name": context, "id": item["id"], "url": item.get("url"),
                                      "head_sha": item.get("head_sha", item.get("sha")),
                                      "classification": classification, "conclusion": outcome})
    receipt["classification"] = next((x for x in outcomes if x != "success"),
                                     "missing" if not outcomes else "success")
    receipt["ok"] = receipt["classification"] == "success"
    return receipt


def collect_acceptance_mcp(contract: str, published_pr: str | None,
                           profile_home: str | None = None) -> dict:
    """Collect acceptance evidence through the profile's own ``github_acceptance``
    server. Native receipt semantics; the only failures are fixed codes — a
    missing server is ``auth`` and never a gh/REST/PAT fallback."""
    receipt = _native_receipt(published_pr)
    try:
        declared = _PR.fullmatch(contract)
        url = contract if declared else published_pr
        match = _PR.fullmatch(url or "")
        if not match or (not declared and match[1] != contract) \
                or (declared and published_pr and published_pr != contract):
            receipt["detail"] = "Supply metadata.published_pr matching the persisted completion contract."
            return receipt
        owner, name = match[1].split("/")
        number = int(match[2])
        server_config = _raw_server_entry(profile_home)
        result = _call_tool(server_config, profile_home,
                            {"owner": owner, "repo": name, "pullNumber": number})
        evidence = _decode_evidence(result)
        _validate_evidence(evidence, owner, name, number)
        return _evaluate(evidence, owner, name, number)
    except _McpConfigError:
        receipt.update(classification="auth", detail=MCP_CONFIG_AUTH_DETAIL)
        return receipt
    except _McpUnavailableError:
        receipt.update(classification="infra", detail=MCP_UNAVAILABLE_DETAIL)
        return receipt
    except (_McpInvalidError, ValueError, KeyError, TypeError, IndexError, OSError):
        receipt.update(classification="infra", detail=MCP_EVIDENCE_INVALID_DETAIL)
        return receipt