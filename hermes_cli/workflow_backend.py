"""本地 Langflow 启动器(Python)—— 浏览器仪表盘(hermes dashboard)用。

对标 Electron 的 ``apps/desktop/electron/workflow-backend.cjs``,让**非 Electron 的本地 Web 端**
也能启动/连上本地 langflow 并登录云端:

- 读**同一个** ``~/.hermes/workflow-secrets.json``(与 Electron 共享登录态),把 ``kari.{token,
  cloudBaseURL}`` 注入为 ``KARI_HUB_URL`` + ``KARI_WORKSPACE_TOKEN``,节点据此 relay 计费/出图。
- ``start()`` 幂等:本地 langflow 已在跑(健康探测通过)→ 直接 attach,不重复拉起;否则 spawn。
- ``login()``:POST 云端 ``/auth/login`` 拿 per-user token,写入 secrets,再 ``start()``。

**真实 KIE_API_KEY 不在本地**;只注入云端地址 + token。
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HOST = os.environ.get("HERMES_DESKTOP_LANGFLOW_HOST", "127.0.0.1")
PORT = int(os.environ.get("HERMES_DESKTOP_LANGFLOW_PORT", "7860"))
FRONTEND_REL = "src/backend/base/langflow/frontend"
DEFAULT_KARI_CLOUD_BASE_URL = (os.environ.get("VITE_KARI_CLOUD_URL") or "https://lotjc.com/hermes").strip().rstrip("/")
_READY_TIMEOUT_S = 10 * 60
_PROBE_TIMEOUT_S = 1.5
_STOP_TIMEOUT_S = 8


def _hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home())
    except Exception:  # noqa: BLE001 - 退回默认,保证可独立测试
        return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))


def secrets_path() -> Path:
    return _hermes_home() / "workflow-secrets.json"


def read_secrets() -> dict:
    try:
        raw = secrets_path().read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError):
        return {}
    try:
        data = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def write_secrets(secrets: dict) -> None:
    p = secrets_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(secrets, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        p.chmod(0o600)
    except OSError:
        pass


def _assign(env: dict, key: str, value) -> None:
    if value is None:
        return
    v = str(value).strip()
    if v:
        env[key] = v


def secrets_env(secrets: dict) -> dict:
    """secrets → langflow 进程环境变量(与 workflow-backend.cjs:workflowSecretsEnv 对齐)。"""
    out: dict[str, str] = {}
    if not isinstance(secrets, dict):
        return out
    extra = secrets.get("env")
    if isinstance(extra, dict):
        for k, v in extra.items():
            _assign(out, str(k), v)
    kari = secrets.get("kari")
    if isinstance(kari, dict):
        # 桌面端/网页端只注入云端地址 + per-user token;真实 KIE_API_KEY 只在云端。
        _assign(out, "KARI_HUB_URL", kari.get("cloudBaseURL") or kari.get("cloudBaseUrl"))
        _assign(out, "KARI_WORKSPACE_TOKEN", kari.get("token"))
    openai = secrets.get("openai")
    if isinstance(openai, dict):
        _assign(out, "OPENAI_API_KEY", openai.get("apiKey"))
        _assign(out, "OPENAI_BASE_URL", openai.get("baseURL") or openai.get("baseUrl"))
        _assign(out, "KARI_LLM_PERFORMANCE_API_KEY", openai.get("apiKey"))
        _assign(out, "KARI_LLM_PERFORMANCE_BASE_URL", openai.get("baseURL") or openai.get("baseUrl"))
        _assign(out, "KARI_LLM_PERFORMANCE_MODEL", openai.get("model") or "性能")
    anthropic = secrets.get("anthropic")
    if isinstance(anthropic, dict):
        _assign(out, "ANTHROPIC_API_KEY", anthropic.get("apiKey"))
        _assign(out, "ANTHROPIC_BASE_URL", anthropic.get("baseURL") or anthropic.get("baseUrl"))
        _assign(out, "KARI_LLM_EXTREME_API_KEY", anthropic.get("apiKey"))
        _assign(out, "KARI_LLM_EXTREME_BASE_URL", anthropic.get("baseURL") or anthropic.get("baseUrl"))
        _assign(out, "KARI_LLM_EXTREME_MODEL", anthropic.get("model") or "极致")
    return out


def _repo_root() -> Path:
    # hermes_cli/workflow_backend.py → 仓库根 = 上一级目录
    return Path(__file__).resolve().parent.parent


def _is_langflow_root(root: Path) -> bool:
    return (root / "src" / "backend" / "base" / "langflow").is_dir() or (root / FRONTEND_REL / "index.html").exists()


def resolve_langflow_root() -> str:
    configured = (os.environ.get("HERMES_DESKTOP_LANGFLOW_ROOT") or "").strip()
    if configured:
        return str(Path(configured).resolve())
    repo = _repo_root()
    candidates = [repo.parent.parent / "kari-all" / "langflow", repo.parent / "langflow", _hermes_home() / "langflow"]
    for c in candidates:
        rc = c.resolve()
        if _is_langflow_root(rc):
            return str(rc)
    return ""


def has_built_frontend(root: str) -> bool:
    return bool(root) and (Path(root) / FRONTEND_REL / "index.html").exists()


def build_runtime_env(secrets: dict | None = None, base_env: dict | None = None) -> dict:
    env = dict(base_env if base_env is not None else os.environ)
    env.update(secrets_env(read_secrets() if secrets is None else secrets))
    env["LANGFLOW_AUTO_LOGIN"] = "true"
    env["LANGFLOW_SKIP_AUTH_AUTO_LOGIN"] = "true"
    # 本地为主:workspace 模式 → 跳过 hub 的"创建办公空间"门槛,直接进画布。
    env["KARI_MODE"] = "workspace"
    env["PYTHONUNBUFFERED"] = "1"
    config_dir = _hermes_home() / "langflow"
    env["LANGFLOW_CONFIG_DIR"] = str(config_dir)
    env["KARI_PERMS_DB"] = str(config_dir / ".kari_perms.sqlite")
    # Kari 自定义节点进画布组件面板。
    root = resolve_langflow_root()
    if root:
        env["LANGFLOW_COMPONENTS_PATH"] = str(Path(root) / "kari_components")
    return env


def build_launch(root: str) -> list[str]:
    """返回启动 langflow 的命令(优先 uv run;前端没构建则回退 make run_cli)。"""
    if has_built_frontend(root):
        return [
            "uv", "run", "langflow", "run",
            "--frontend-path", FRONTEND_REL,
            "--host", HOST, "--port", str(PORT),
            "--env-file", ".env", "--no-open-browser",
        ]
    return ["make", "run_cli", f"host={HOST}", f"port={PORT}", "open_browser=false"]


def langflow_url() -> str:
    return f"http://{HOST}:{PORT}"


def langflow_capable() -> bool:
    """本节点是否「能力承载节点」—— 有 langflow,能跑工作流 / 当 MCP 提供者(给下级配 MCP)。
    判据:本机装了 langflow 源码(懒启动也算有能力,优先;文件系统判定快)**或** langflow 正在跑。
    轻 Hermes(没装 langflow)= False —— 权限面板据此不显示「配 MCP 给下级」。"""
    try:
        if resolve_langflow_root():
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        return is_reachable()
    except Exception:  # noqa: BLE001
        return False


def is_reachable(timeout: float = _PROBE_TIMEOUT_S) -> bool:
    try:
        with urllib.request.urlopen(langflow_url(), timeout=timeout) as r:
            return 200 <= r.status < 500
    except urllib.error.HTTPError as e:
        return 200 <= e.code < 500  # 任何 HTTP 响应都说明端口活着
    except Exception:  # noqa: BLE001
        return False


def _normalize_base_url(value) -> str:
    return str(value or "").strip().rstrip("/")


def is_cloud_reachable(cloud_base_url: str) -> bool:
    url = _normalize_base_url(cloud_base_url)
    if not url:
        return False
    try:
        with urllib.request.urlopen(url, timeout=_PROBE_TIMEOUT_S) as r:
            return 200 <= r.status < 500
    except urllib.error.HTTPError as e:
        return 200 <= e.code < 500
    except Exception:  # noqa: BLE001
        return False


class WorkflowBackend:
    """单例:管理本地 langflow 子进程。start() 幂等(已在跑则 attach)。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._state = "stopped"  # stopped|starting|ready|error|exited
        self._error: str | None = None
        self._root = ""

    def status(self) -> dict:
        if self._state in ("starting", "ready") and is_reachable():
            self._state = "ready"
        return {
            "state": self._state,
            "url": langflow_url(),
            "pid": self._proc.pid if self._proc else None,
            "root": self._root,
            "external": self._proc is None and self._state == "ready",
            "error": self._error,
        }

    def start(self) -> dict:
        with self._lock:
            if is_reachable():  # 已在跑(可能是 Electron 起的)→ attach
                self._state = "ready"
                self._error = None
                return self.status()
            if self._state == "starting" and self._proc and self._proc.poll() is None:
                return self.status()
            root = resolve_langflow_root()
            if not root:
                self._state = "error"
                self._error = "找不到本地 langflow 源码(设 HERMES_DESKTOP_LANGFLOW_ROOT)"
                return self.status()
            self._root = root
            try:
                self._proc = subprocess.Popen(  # noqa: S603 - 命令固定,非 shell
                    build_launch(root), cwd=root, env=build_runtime_env(),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except Exception as e:  # noqa: BLE001
                self._state = "error"
                self._error = f"启动 langflow 失败: {e}"
                return self.status()
            self._state = "starting"
            self._error = None
            threading.Thread(target=self._await_ready, daemon=True).start()
            return self.status()

    def _await_ready(self) -> None:
        deadline = time.time() + _READY_TIMEOUT_S
        while time.time() < deadline:
            if self._proc and self._proc.poll() is not None:
                self._state = "exited"
                self._error = f"langflow 进程退出(code {self._proc.returncode})"
                return
            if is_reachable():
                self._state = "ready"
                self._error = None
                return
            time.sleep(1.0)
        self._state = "error"
        self._error = "langflow 启动超时"

    def stop(self) -> dict:
        """终止本管理器自己启动的 langflow 子进程,等它真正退出再返回。"""
        with self._lock:
            proc = self._proc
            self._proc = None
            self._state = "stopped"
            self._error = None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=_STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=2.0)
        return self.status()

    def restart(self) -> dict:
        """先停再起,确保新写入的 secrets(token/云端模型 key)注入进 langflow env。

        只能重启本管理器拥有的进程;若 langflow 是外部(如 Electron 桌面端)启动的,
        本侧 ``self._proc`` 为 None,无法重启它——由该启动方各自负责重启。等进程真正
        退出后再 start(),才能让 ``is_reachable()`` 返回 False、从而带新 env 重新 spawn。
        """
        if self._proc is not None:
            self.stop()
        return self.start()


_manager: WorkflowBackend | None = None


def manager() -> WorkflowBackend:
    global _manager
    if _manager is None:
        _manager = WorkflowBackend()
    return _manager


# --------------------------- 云端登录(账号/计费/发 key) ---------------------------
def auth_status() -> dict:
    kari = read_secrets().get("kari") or {}
    token = str(kari.get("token") or "").strip()
    cloud = _normalize_base_url(kari.get("cloudBaseURL") or kari.get("cloudBaseUrl") or "")
    fallback_cloud = DEFAULT_KARI_CLOUD_BASE_URL
    if not token:
        return {"loggedIn": False, "cloudBaseUrl": cloud or fallback_cloud, "cloudReachable": False, "error": None}
    if not cloud:
        return {
            "loggedIn": False,
            "cloudBaseUrl": fallback_cloud,
            "cloudReachable": False,
            "error": "Kari hub is not configured. Please log in to Workflow again.",
        }
    if not is_cloud_reachable(cloud):
        return {
            "loggedIn": False,
            "cloudBaseUrl": cloud,
            "cloudReachable": False,
            "error": f"Kari hub is not reachable: {cloud}. Please log in to Workflow again or start the local hub.",
        }
    return {"loggedIn": True, "cloudBaseUrl": cloud, "cloudReachable": True, "error": None}


def login(cloud_base_url: str, email: str, password: str) -> dict:
    """POST 云端 /auth/login 拿 token,写入 secrets,再启动本地 langflow。"""
    cloud = (cloud_base_url or "").strip().rstrip("/")
    email = (email or "").strip()
    if not cloud or not email or not password:
        return {"ok": False, "error": "请填写云端地址、邮箱和密码"}
    body = json.dumps({"email": email, "password": password}).encode()
    req = urllib.request.Request(
        f"{cloud}/auth/login", data=body, method="POST", headers={"content-type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            resp = json.loads(r.read() or "{}")
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read() or "{}").get("detail")
        except Exception:  # noqa: BLE001
            detail = None
        return {"ok": False, "error": detail or f"登录失败(HTTP {e.code})"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"无法连接云端:{e}"}
    token = resp.get("token")
    if not token:
        return {"ok": False, "error": "云端未返回 token"}
    secrets = read_secrets()
    secrets["kari"] = {"token": token, "cloudBaseURL": cloud}
    write_secrets(secrets)
    # restart (not start): if langflow was already started pre-login, a plain
    # start() just attaches to the running process and the new token never lands
    # in its env — login would succeed yet Kari nodes stay unavailable.
    manager().restart()
    return {"ok": True, "balance": resp.get("balance"), "email": resp.get("email")}


def logout() -> dict:
    secrets = read_secrets()
    secrets.pop("kari", None)
    write_secrets(secrets)
    # restart so the running langflow drops the cleared token and the Kari nodes
    # go unavailable; the canvas still boots without an account.
    manager().restart()
    return {"ok": True}
