"""Antigravity Code Assist control-plane helpers.

The new Antigravity CLI uses the same v1internal Code Assist family as
gemini-cli, but with Antigravity OAuth scopes, metadata and model catalog. This
module keeps that provider-specific surface separate from
``agent.google_code_assist``.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from agent.google_code_assist import CodeAssistError

logger = logging.getLogger(__name__)

ANTIGRAVITY_CODE_ASSIST_ENDPOINT = "https://daily-cloudcode-pa.sandbox.googleapis.com"
ANTIGRAVITY_MODEL_ENDPOINTS = [
    ANTIGRAVITY_CODE_ASSIST_ENDPOINT,
    "https://cloudcode-pa.googleapis.com",
    "https://autopush-cloudcode-pa.sandbox.googleapis.com",
]

ANTIGRAVITY_CLIENT_METADATA = {
    "ideType": "ANTIGRAVITY",
    "platform": "PLATFORM_UNSPECIFIED",
    "pluginType": "GEMINI",
}
ANTIGRAVITY_USER_AGENT = "antigravity/1.0.0 windows/amd64"
ANTIGRAVITY_X_GOOG_API_CLIENT = "google-cloud-sdk vscode_cloudshelleditor/0.1"

DEFAULT_AGENT_MODEL_IDS = [
    "gemini-3-flash-agent",
    "gemini-3.5-flash-low",
    "gemini-pro-agent",
    "gemini-3.1-pro-low",
    "claude-sonnet-4-6",
    "claude-opus-4-6-thinking",
    "gpt-oss-120b-medium",
]

DEPRECATED_MODEL_REPLACEMENTS = {
    "gemini-3.1-pro-high": "gemini-pro-agent",
}


@dataclass
class AntigravityProjectInfo:
    project_id: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectContext:
    project_id: str = ""
    managed_project_id: str = ""
    tier_id: str = ""
    source: str = ""


def _client_metadata() -> Dict[str, str]:
    return dict(ANTIGRAVITY_CLIENT_METADATA)


def build_headers(access_token: str, *, accept: str = "application/json") -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Accept": accept,
        "Authorization": f"Bearer {access_token}",
        "User-Agent": ANTIGRAVITY_USER_AGENT,
        "X-Goog-Api-Client": ANTIGRAVITY_X_GOOG_API_CLIENT,
        "Client-Metadata": json.dumps(_client_metadata(), separators=(",", ":")),
        "x-activity-request-id": str(uuid.uuid4()),
    }


def _post_json(
    url: str,
    body: Dict[str, Any],
    access_token: str,
    *,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers=build_headers(access_token),
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise CodeAssistError(
            f"Antigravity Code Assist HTTP {exc.code}: {detail or exc.reason}",
            code=f"antigravity_code_assist_http_{exc.code}",
        ) from exc
    except urllib.error.URLError as exc:
        raise CodeAssistError(
            f"Antigravity Code Assist request failed: {exc}",
            code="antigravity_code_assist_network_error",
        ) from exc


def load_code_assist(
    access_token: str,
    *,
    project_id: str = "",
    endpoint: str = ANTIGRAVITY_CODE_ASSIST_ENDPOINT,
) -> AntigravityProjectInfo:
    metadata = _client_metadata()
    if project_id:
        metadata["duetProject"] = project_id
    body: Dict[str, Any] = {"metadata": metadata}
    if project_id:
        body["cloudaicompanionProject"] = project_id
    resp = _post_json(f"{endpoint}/v1internal:loadCodeAssist", body, access_token)
    project = (
        str(resp.get("cloudaicompanionProject") or "").strip()
        or str(resp.get("project") or "").strip()
    )
    return AntigravityProjectInfo(project_id=project, raw=resp)


def resolve_project_context(
    access_token: str,
    *,
    configured_project_id: str = "",
    env_project_id: str = "",
) -> ProjectContext:
    if configured_project_id:
        return ProjectContext(project_id=configured_project_id, source="config")
    if env_project_id:
        return ProjectContext(project_id=env_project_id, source="env")
    info = load_code_assist(access_token)
    return ProjectContext(
        project_id=info.project_id,
        managed_project_id=info.project_id,
        source="discovered" if info.project_id else "unknown",
    )


def fetch_available_models(
    access_token: str,
    *,
    project_id: str = "",
    endpoint: str = ANTIGRAVITY_CODE_ASSIST_ENDPOINT,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {}
    if project_id:
        body["project"] = project_id
    return _post_json(f"{endpoint}/v1internal:fetchAvailableModels", body, access_token)


def fetch_available_models_with_fallbacks(
    access_token: str,
    *,
    project_id: str = "",
    endpoints: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for endpoint in endpoints or ANTIGRAVITY_MODEL_ENDPOINTS:
        try:
            return fetch_available_models(
                access_token,
                project_id=project_id,
                endpoint=endpoint,
            )
        except Exception as exc:
            last_err = exc
            logger.debug("Antigravity fetchAvailableModels failed on %s: %s", endpoint, exc)
    if last_err:
        raise last_err
    return {}


def _model_id_from_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("modelId", "model_id", "id", "name"):
            candidate = str(value.get(key) or "").strip()
            if candidate:
                return candidate
    return ""


def _ids_from_sort(sort: Dict[str, Any]) -> List[str]:
    ids: List[str] = []
    for key in ("modelIds", "model_ids", "models", "modelSorts"):
        value = sort.get(key)
        if isinstance(value, list):
            for item in value:
                mid = _model_id_from_value(item)
                if mid:
                    ids.append(mid)
        elif isinstance(value, dict):
            mid = _model_id_from_value(value)
            if mid:
                ids.append(mid)
    return ids


def _is_recommended_sort(sort: Dict[str, Any]) -> bool:
    label = " ".join(
        str(sort.get(key) or "")
        for key in ("name", "displayName", "title", "category", "group")
    ).lower()
    return "recommended" in label


def _raw_model_ids(payload: Dict[str, Any]) -> List[str]:
    ids: List[str] = []
    models = payload.get("models")
    if isinstance(models, list):
        for item in models:
            mid = _model_id_from_value(item)
            if mid:
                ids.append(mid)
    return ids


def filter_agent_model_ids(ids: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    filtered: List[str] = []
    raw = [str(mid).strip() for mid in ids if str(mid).strip()]
    replacements = set(DEPRECATED_MODEL_REPLACEMENTS.values())
    for mid in raw:
        if mid in seen:
            continue
        if mid.startswith(("chat_", "tab_")):
            continue
        if mid in DEPRECATED_MODEL_REPLACEMENTS and DEPRECATED_MODEL_REPLACEMENTS[mid] in raw:
            continue
        if mid in replacements and mid in seen:
            continue
        seen.add(mid)
        filtered.append(mid)
    return filtered


def parse_agent_model_ids(payload: Dict[str, Any]) -> List[str]:
    """Return the user-facing Antigravity agent model list in display order."""
    sorts = payload.get("agentModelSorts")
    ordered: List[str] = []
    if isinstance(sorts, list):
        recommended = [s for s in sorts if isinstance(s, dict) and _is_recommended_sort(s)]
        rest = [s for s in sorts if isinstance(s, dict) and not _is_recommended_sort(s)]
        for sort in recommended + rest:
            ordered.extend(_ids_from_sort(sort))

    if not ordered:
        default_id = str(payload.get("defaultAgentModelId") or "").strip()
        if default_id:
            ordered.append(default_id)
        for mid in DEFAULT_AGENT_MODEL_IDS:
            ordered.append(mid)
        ordered.extend(_raw_model_ids(payload))

    filtered = filter_agent_model_ids(ordered)
    if filtered:
        return filtered
    return list(DEFAULT_AGENT_MODEL_IDS)
