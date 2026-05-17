from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)


def _find_git_root(start: Path) -> Optional[Path]:
    current = start.resolve()
    for parent in [current, *current.parents]:
        if (parent / '.git').exists():
            return parent
    return None


def _discover_repo_root(start: Path) -> Optional[Path]:
    direct = _find_git_root(start)
    if direct:
        return direct

    try:
        children = [child for child in start.resolve().iterdir() if child.is_dir()]
    except OSError:
        return None

    repo_roots = [child for child in children if (child / '.git').exists()]
    if len(repo_roots) == 1:
        return repo_roots[0]
    return None


def _sanitize_scope_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-._")
    return cleaned or "default"


def _run_git(args: List[str], cwd: Path) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def _run_multica(args: List[str], cwd: Path) -> str:
    try:
        result = subprocess.run(
            ["multica", *args],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def _extract_issue_id_from_agent_context(start: Path) -> str:
    issue_id = os.getenv("MULTICA_ISSUE_ID", "").strip()
    if issue_id:
        return issue_id
    issue_context = start.resolve() / ".agent_context" / "issue_context.md"
    try:
        text = issue_context.read_text(encoding="utf-8")
    except OSError:
        return ""
    match = re.search(r"\*\*Issue ID:\*\*\s*([0-9a-fA-F-]{36})", text)
    return match.group(1) if match else ""


def _project_scope_from_multica_context(start: Path) -> str:
    issue_id = _extract_issue_id_from_agent_context(start)
    if not issue_id:
        return ""
    raw = _run_multica(["issue", "get", issue_id, "--output", "json"], start)
    if not raw:
        return ""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    project_id = str(data.get("project_id", "") or "").strip()
    if not project_id:
        return ""
    return f"project:{_sanitize_scope_part(project_id)}"


def _project_scope_from_multica_workspace(start: Path, canonical_origin: str) -> str:
    if not canonical_origin:
        return ""
    workspace_raw = _run_multica(["workspace", "get", "--output", "json"], start)
    if not workspace_raw:
        return ""
    try:
        workspace = json.loads(workspace_raw)
    except json.JSONDecodeError:
        return ""

    repos = workspace.get("repos") or []
    matching_repos = []
    for repo in repos:
        repo_url = _canonicalize_git_remote(str(repo.get("url", "") or ""))
        if repo_url == canonical_origin:
            matching_repos.append(repo)
    if len(matching_repos) != 1:
        return ""

    projects_raw = _run_multica(["project", "list", "--output", "json"], start)
    if not projects_raw:
        return ""
    try:
        projects = json.loads(projects_raw)
    except json.JSONDecodeError:
        return ""
    if not isinstance(projects, list) or len(projects) != 1:
        return ""

    project_id = str(projects[0].get("id", "") or "").strip()
    if not project_id:
        return ""
    return f"project:{_sanitize_scope_part(project_id)}"


def _canonicalize_git_remote(url: str) -> str:
    value = (url or "").strip()
    if not value:
        return ""
    if value.startswith("git@") and ":" in value:
        host, path = value[4:].split(":", 1)
        value = f"ssh://{host}/{path}"
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()
    path = (parsed.path or value).strip()
    if path.startswith("/"):
        path = path[1:]
    if path.endswith(".git"):
        path = path[:-4]
    path = path.strip("/")
    if host and path:
        return f"{host}/{path}"
    return value.removesuffix(".git").strip()


def _project_scope_from_cwd(start: Path) -> Optional[str]:
    explicit_scope = os.getenv("HERMES_MEMORY_PROJECT_SCOPE", "").strip()
    if explicit_scope:
        return f"project:{_sanitize_scope_part(explicit_scope)}"

    explicit_project_id = (
        os.getenv("HERMES_MEMORY_PROJECT_ID", "").strip()
        or os.getenv("MULTICA_PROJECT_ID", "").strip()
    )
    if explicit_project_id:
        return f"project:{_sanitize_scope_part(explicit_project_id)}"

    multica_scope = _project_scope_from_multica_context(start)
    if multica_scope:
        return multica_scope

    root = _discover_repo_root(start)
    if not root:
        return None

    repo_name = _sanitize_scope_part(root.name)

    origin_url = _run_git(["remote", "get-url", "origin"], root)
    canonical_origin = _canonicalize_git_remote(origin_url)
    multica_workspace_scope = _project_scope_from_multica_workspace(start, canonical_origin)
    if multica_workspace_scope:
        return multica_workspace_scope
    if canonical_origin:
        origin_hash = hashlib.sha1(canonical_origin.encode("utf-8")).hexdigest()[:8]
        return f"project:{repo_name}-{origin_hash}"

    common_dir = _run_git(["rev-parse", "--git-common-dir"], root)
    if common_dir:
        common_path = Path(common_dir)
        if not common_path.is_absolute():
            common_path = (root / common_path).resolve()
        common_hash = hashlib.sha1(str(common_path).encode("utf-8")).hexdigest()[:8]
        return f"project:{repo_name}-{common_hash}"

    root_hash = hashlib.sha1(str(root.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"project:{repo_name}-{root_hash}"


class HermesMemoryProvider(MemoryProvider):
    def __init__(self):
        self._base_url = os.getenv("HERMES_MEMORY_BASE_URL", "http://127.0.0.1:8790").rstrip("/")
        self._allow_remote = str(os.getenv("HERMES_MEMORY_ALLOW_REMOTE", "")).lower() in {"1", "true", "yes", "on"}
        self._session_id = ""
        self._platform = "cli"
        self._user_id = ""
        self._agent_context = "primary"
        self._project_scope: Optional[str] = None
        self._user_scope: Optional[str] = None
        self._global_scope = "global"
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None

    @property
    def name(self) -> str:
        return "hermes-memory"

    def is_available(self) -> bool:
        if not self._base_url:
            return False
        parsed = urlparse(self._base_url)
        host = (parsed.hostname or "").lower()
        if self._allow_remote:
            return True
        return host in {"127.0.0.1", "localhost", "::1"}

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._platform = kwargs.get("platform", "cli") or "cli"
        self._agent_context = kwargs.get("agent_context", "primary") or "primary"
        self._user_id = str(kwargs.get("user_id", "") or "")
        if self._user_id:
            self._user_scope = f"user:{_sanitize_scope_part(self._user_id)}"
        cwd = os.getenv("TERMINAL_CWD") or os.getcwd()
        self._project_scope = _project_scope_from_cwd(Path(cwd))

    def system_prompt_block(self) -> str:
        scopes = ", ".join(self._build_scopes())
        return (
            "# hermes-memory\n"
            f"External memory provider active at {self._base_url}. "
            f"Default recall scopes: {scopes}. "
            "This provider injects recalled durable context before each turn and mirrors durable memory writes after explicit memory updates."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not query.strip():
            return ""
        with self._prefetch_lock:
            cached = self._prefetch_result
            self._prefetch_result = ""
        if cached:
            return cached
        return self._perform_prefetch(query)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not query.strip():
            return None
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            return None

        def _runner() -> None:
            result = self._perform_prefetch(query)
            with self._prefetch_lock:
                self._prefetch_result = result

        self._prefetch_thread = threading.Thread(target=_runner, daemon=True)
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if self._agent_context != "primary":
            return
        content = assistant_content.strip()
        if not content:
            return
        if not self._looks_durable(content):
            return
        if self._sync_thread and self._sync_thread.is_alive():
            return

        def _runner() -> None:
            self._ingest(scope=self._preferred_write_scope(), content=content, kind=self._classify_kind(content), summary=content[:280])

        self._sync_thread = threading.Thread(target=_runner, daemon=True)
        self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        if self._agent_context != "primary":
            return
        if action not in {"add", "replace"}:
            return
        scope = self._user_scope if target == "user" and self._user_scope else self._preferred_write_scope()
        kind = "preference" if target == "user" else self._classify_kind(content)
        self._ingest(scope=scope, content=content, kind=kind, summary=content[:280])

    def shutdown(self) -> None:
        return None

    def _build_scopes(self) -> List[str]:
        scopes: List[str] = []
        if self._project_scope:
            scopes.append(self._project_scope)
        if self._user_scope:
            scopes.append(self._user_scope)
        scopes.append(self._global_scope)
        return scopes

    def _preferred_write_scope(self) -> str:
        return self._project_scope or self._user_scope or self._global_scope

    def _looks_durable(self, text: str) -> bool:
        durable_patterns = [
            r"\bprefer(?:s|ence)?\b",
            r"\blikes?\b",
            r"\bmust\b",
            r"\brequired\b",
            r"\bconstraint\b",
            r"\bverified\b",
            r"\broot cause\b",
            r"\bfix(?:ed)?\b",
            r"\bdeploy(?:ment)?\b",
            r"偏好|喜欢|必须|约束|修复|已验证|部署",
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in durable_patterns)

    def _classify_kind(self, text: str) -> str:
        lower = text.lower()
        if re.search(r"prefer|likes|偏好|喜欢", lower):
            return "preference"
        if re.search(r"verified|fix|root cause|修复|已验证", lower):
            return "solution"
        if re.search(r"must|required|constraint|必须|约束", lower):
            return "decision"
        if re.search(r"deploy|deployment|部署", lower):
            return "decision"
        return "fact"

    def _perform_prefetch(self, query: str) -> str:
        payload = {
            "query": query,
            "scopes": self._build_scopes(),
            "limit": 8,
            "budgetChars": 1200,
        }
        try:
            data = self._post_json("/memories/recall", payload)
        except Exception as exc:
            logger.debug("hermes-memory prefetch failed: %s", exc)
            return ""
        context = str(data.get("context", "") or "").strip()
        return f"## hermes-memory recall\n{context}" if context else ""

    def _ingest(self, *, scope: str, content: str, kind: str, summary: str) -> None:
        payload = {
            "scope": scope,
            "items": [{
                "kind": kind,
                "title": f"Hermes memory {self._platform}",
                "content": content,
                "summary": summary,
                "source": "hermes-runtime",
                "sourceRef": self._session_id,
                "importance": 0.85,
                "confidence": 0.9,
                "tags": ["hermes", self._platform, kind],
                "metadata": {
                    "sessionId": self._session_id,
                    "platform": self._platform,
                },
            }],
        }
        try:
            self._post_json("/memories/ingest", payload)
        except Exception as exc:
            logger.debug("hermes-memory ingest failed: %s", exc)

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        request = Request(
            f"{self._base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=5) as response:
                return json.loads(response.read().decode("utf-8"))
        except URLError as exc:
            raise RuntimeError(f"HTTP request failed: {exc}") from exc


def register(ctx):
    ctx.register_memory_provider(HermesMemoryProvider())
