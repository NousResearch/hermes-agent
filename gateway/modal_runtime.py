# ABOUTME: Helpers for running the Hermes gateway inside a hosted Modal web container.
# ABOUTME: Sanitizes local Hermes config for remote use, bootstraps HERMES_HOME, and renders a status dashboard.

from __future__ import annotations

import asyncio
import base64
import binascii
import copy
import html
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlparse

import yaml

BOOTSTRAP_ENV_MAP = {
    "config": "HERMES_MODAL_CONFIG_B64",
    "env": "HERMES_MODAL_ENV_B64",
    "auth": "HERMES_MODAL_AUTH_B64",
}

_LOCAL_ONLY_ENV_KEYS = {
    "HERMES_HOME",
    "MESSAGING_CWD",
    "MODAL_CONFIG_PATH",
}


def named_modal_secret_names() -> list[str]:
    """Return deploy-time named Modal secrets to attach to the gateway container."""
    secret_names: list[str] = []
    for value in (
        os.getenv("HERMES_MODAL_GITHUB_TOKEN_SECRET", "github-token").strip(),
        os.getenv("HERMES_MODAL_PRIME_API_KEY_SECRET", "PRIME_API_KEY").strip(),
    ):
        if value and value not in secret_names:
            secret_names.append(value)
    return secret_names


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_env_key(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _is_local_url(value: str) -> bool:
    if not value:
        return False
    parsed = urlparse(value.strip())
    host = (parsed.hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def _drop_local_base_urls(node: Any) -> None:
    if isinstance(node, dict):
        for key in list(node.keys()):
            value = node[key]
            if key == "base_url" and isinstance(value, str) and _is_local_url(value):
                node.pop(key, None)
                continue
            _drop_local_base_urls(value)
    elif isinstance(node, list):
        for item in node:
            _drop_local_base_urls(item)


def sanitize_config_for_modal(config: dict[str, Any], *, project_root: str) -> dict[str, Any]:
    """Return a remote-safe copy of config.yaml for the hosted gateway."""
    sanitized = copy.deepcopy(config or {})
    _drop_local_base_urls(sanitized)

    terminal = sanitized.get("terminal")
    if not isinstance(terminal, dict):
        terminal = {}
    terminal["backend"] = "local"
    terminal["cwd"] = project_root
    sanitized["terminal"] = terminal

    return sanitized


def sanitize_env_text_for_modal(raw_text: str) -> str:
    """Drop env vars that only make sense on the deploy host."""
    if not raw_text:
        return ""

    lines: list[str] = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            lines.append(line)
            continue

        key, _, value = line.partition("=")
        env_key = key.strip()
        env_value = _strip_wrapping_quotes(value.strip())

        if env_key.startswith("MODAL_") or env_key.startswith("TERMINAL_"):
            continue
        if env_key in _LOCAL_ONLY_ENV_KEYS:
            continue
        if env_key == "OPENAI_BASE_URL" and _is_local_url(env_value):
            continue

        lines.append(line)

    return "\n".join(lines).strip() + ("\n" if lines else "")


def _decode_base64_env(name: str) -> Optional[str]:
    payload = os.getenv(name, "").strip()
    if not payload:
        return None
    try:
        decoded = base64.b64decode(payload, validate=True)
        return decoded.decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return None


def normalize_github_token_env() -> None:
    """Alias common GitHub token secret keys to the names Hermes and gh expect."""
    github_token = os.environ.get("GITHUB_TOKEN")
    gh_token = os.environ.get("GH_TOKEN")
    if github_token and gh_token:
        return
    if github_token:
        os.environ.setdefault("GH_TOKEN", github_token)
        return
    if gh_token:
        os.environ.setdefault("GITHUB_TOKEN", gh_token)
        return

    token_value: Optional[str] = None
    for key, value in os.environ.items():
        if not value:
            continue
        normalized = _normalize_env_key(key)
        if normalized in {"githubtoken", "ghtoken"}:
            token_value = value
            break

    if not token_value:
        return

    os.environ.setdefault("GITHUB_TOKEN", token_value)
    os.environ.setdefault("GH_TOKEN", token_value)


def _read_json_file(path: Path) -> Optional[dict[str, Any]]:
    try:
        raw = path.read_text().strip()
    except OSError:
        return None
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _tail_file(path: Path, *, max_lines: int = 120) -> str:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return ""
    if max_lines <= 0:
        return ""
    return "\n".join(lines[-max_lines:])


def _write_text_file(path: Path, text: str, *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if private:
        path.chmod(0o600)


def bootstrap_modal_home(
    home_dir: Path,
    *,
    project_root: str,
    commit_fn: Optional[Callable[[], None]] = None,
) -> None:
    """Materialize a remote-safe Hermes home directory from deploy-time secrets."""
    home_dir.mkdir(parents=True, exist_ok=True)
    (home_dir / "logs").mkdir(parents=True, exist_ok=True)
    (home_dir / "sessions").mkdir(parents=True, exist_ok=True)

    config_text = _decode_base64_env(BOOTSTRAP_ENV_MAP["config"])
    config_payload: dict[str, Any] = {}
    config_error: Optional[str] = None
    if config_text:
        try:
            loaded = yaml.safe_load(config_text) or {}
        except yaml.YAMLError as exc:
            config_error = str(exc)
        else:
            if isinstance(loaded, dict):
                config_payload = loaded
    sanitized_config = sanitize_config_for_modal(config_payload, project_root=project_root)
    _write_text_file(
        home_dir / "config.yaml",
        yaml.safe_dump(sanitized_config, sort_keys=False),
    )

    env_text = sanitize_env_text_for_modal(_decode_base64_env(BOOTSTRAP_ENV_MAP["env"]) or "")
    _write_text_file(home_dir / ".env", env_text, private=True)

    auth_text = _decode_base64_env(BOOTSTRAP_ENV_MAP["auth"])
    if auth_text:
        _write_text_file(home_dir / "auth.json", auth_text, private=True)

    bootstrap_metadata = {
        "bootstrapped_at": _utc_now_iso(),
        "project_root": project_root,
        "has_auth": bool(auth_text),
    }
    if config_error:
        bootstrap_metadata["config_error"] = config_error
    _write_text_file(
        home_dir / "modal-bootstrap.json",
        json.dumps(bootstrap_metadata, indent=2),
    )

    if commit_fn is not None:
        try:
            commit_fn()
        except Exception:
            pass


class ModalGatewayService:
    """Own the hosted gateway lifecycle for the Modal web container."""

    def __init__(
        self,
        *,
        hermes_home: Path,
        project_root: str,
        commit_fn: Optional[Callable[[], None]] = None,
        commit_interval_seconds: int = 30,
    ) -> None:
        self.hermes_home = hermes_home
        self.project_root = project_root
        self.commit_fn = commit_fn
        self.commit_interval_seconds = commit_interval_seconds
        self.started_at = time.time()
        self._gateway_thread: Optional[threading.Thread] = None
        self._commit_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._last_error: Optional[str] = None

    def start(self) -> None:
        bootstrap_modal_home(
            self.hermes_home,
            project_root=self.project_root,
            commit_fn=self.commit_fn,
        )

        with self._lock:
            if self._gateway_thread and self._gateway_thread.is_alive():
                return

            self._stop_event.clear()
            self._last_error = None
            self._gateway_thread = threading.Thread(
                target=self._run_gateway,
                name="modal-hermes-gateway",
                daemon=True,
            )
            self._gateway_thread.start()

            if self.commit_fn and (self._commit_thread is None or not self._commit_thread.is_alive()):
                self._commit_thread = threading.Thread(
                    target=self._commit_loop,
                    name="modal-hermes-volume-commit",
                    daemon=True,
                )
                self._commit_thread.start()

    def wait_until_ready(self, timeout_seconds: int = 45, poll_interval: float = 0.5) -> dict[str, Any]:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            snapshot = self.snapshot()
            gateway_state = (snapshot.get("runtime_status") or {}).get("gateway_state")
            if gateway_state in {"running", "startup_failed", "stopped"}:
                return snapshot
            if snapshot.get("last_error"):
                return snapshot
            time.sleep(poll_interval)
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        runtime_status = _read_json_file(self.hermes_home / "gateway_state.json")
        pid_record = _read_json_file(self.hermes_home / "gateway.pid")
        return {
            "captured_at": _utc_now_iso(),
            "hermes_home": str(self.hermes_home),
            "project_root": self.project_root,
            "thread_alive": bool(self._gateway_thread and self._gateway_thread.is_alive()),
            "pid_record": pid_record,
            "runtime_status": runtime_status,
            "gateway_log_tail": _tail_file(self.hermes_home / "logs" / "gateway.log"),
            "error_log_tail": _tail_file(self.hermes_home / "logs" / "errors.log"),
            "last_error": self._last_error,
            "uptime_seconds": max(0, int(time.time() - self.started_at)),
        }

    def _commit_loop(self) -> None:
        while not self._stop_event.wait(self.commit_interval_seconds):
            if self.commit_fn is None:
                return
            try:
                self.commit_fn()
            except Exception:
                pass

    def _run_gateway(self) -> None:
        os.environ["HERMES_HOME"] = str(self.hermes_home)
        os.environ["HERMES_MODAL_HOSTED"] = "1"
        normalize_github_token_env()

        try:
            os.chdir(self.project_root)
            from gateway.run import start_gateway

            result = asyncio.run(start_gateway(replace=True))
            if result is False and not self._stop_event.is_set():
                self._last_error = "Gateway exited before reporting readiness."
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
        finally:
            if self.commit_fn is not None:
                try:
                    self.commit_fn()
                except Exception:
                    pass


def render_dashboard_html(snapshot: dict[str, Any], *, request_url: Optional[str] = None) -> str:
    """Render a lightweight hosted status dashboard."""
    runtime_status = snapshot.get("runtime_status") or {}
    pid_record = snapshot.get("pid_record") or {}
    platforms = runtime_status.get("platforms") or {}
    gateway_state = runtime_status.get("gateway_state") or ("starting" if snapshot.get("thread_alive") else "stopped")
    exit_reason = runtime_status.get("exit_reason") or ""

    platform_rows = []
    for name, pdata in sorted(platforms.items()):
        state = html.escape(str(pdata.get("state", "unknown")))
        message = html.escape(str(pdata.get("error_message", "")))
        platform_rows.append(
            f"<tr><td>{html.escape(name)}</td><td>{state}</td><td>{message}</td></tr>"
        )
    if not platform_rows:
        platform_rows.append("<tr><td colspan='3'>No platform state reported yet.</td></tr>")

    gateway_log = html.escape(snapshot.get("gateway_log_tail") or "(no gateway log output yet)")
    error_log = html.escape(snapshot.get("error_log_tail") or "(no error log output yet)")
    last_error = html.escape(snapshot.get("last_error") or "none")
    dashboard_url = html.escape(request_url or "")
    exit_reason_html = html.escape(exit_reason or "none")
    pid_html = html.escape(str(pid_record.get("pid", "unknown")))
    started_at = html.escape(str(pid_record.get("start_time", "unknown")))

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="10">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Hermes Gateway Dashboard</title>
    <style>
      :root {{
        color-scheme: light;
        --ink: #1f2937;
        --muted: #6b7280;
        --line: #d1d5db;
        --panel: #ffffff;
        --page: #f3f4f6;
        --accent: #0f766e;
        --danger: #b91c1c;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: "SF Mono", "JetBrains Mono", Menlo, monospace;
        background: linear-gradient(180deg, #f8fafc 0%, var(--page) 100%);
        color: var(--ink);
      }}
      main {{
        max-width: 1100px;
        margin: 0 auto;
        padding: 24px;
      }}
      h1 {{
        margin: 0 0 8px;
        font-size: 28px;
      }}
      p {{
        margin: 0 0 16px;
        color: var(--muted);
      }}
      .grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 16px;
        margin-bottom: 16px;
      }}
      .card {{
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
      }}
      .label {{
        display: block;
        margin-bottom: 8px;
        font-size: 12px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
      }}
      .value {{
        font-size: 18px;
        word-break: break-word;
      }}
      .ok {{ color: var(--accent); }}
      .danger {{ color: var(--danger); }}
      table {{
        width: 100%;
        border-collapse: collapse;
      }}
      th, td {{
        padding: 10px 8px;
        border-bottom: 1px solid var(--line);
        text-align: left;
        vertical-align: top;
      }}
      pre {{
        margin: 0;
        padding: 14px;
        background: #0f172a;
        color: #e2e8f0;
        border-radius: 12px;
        overflow-x: auto;
        white-space: pre-wrap;
        word-break: break-word;
      }}
    </style>
  </head>
  <body>
    <main>
      <h1>Hermes Gateway Dashboard</h1>
      <p>Hosted Modal gateway status for Discord access, remote health checks, and log inspection.</p>

      <section class="grid">
        <article class="card">
          <span class="label">Gateway State</span>
          <div class="value {'ok' if gateway_state == 'running' else 'danger' if gateway_state in {'startup_failed', 'stopped'} else ''}">{html.escape(gateway_state)}</div>
        </article>
        <article class="card">
          <span class="label">Gateway PID</span>
          <div class="value">{pid_html}</div>
        </article>
        <article class="card">
          <span class="label">Process Start Token</span>
          <div class="value">{started_at}</div>
        </article>
        <article class="card">
          <span class="label">Last Error</span>
          <div class="value">{last_error}</div>
        </article>
        <article class="card">
          <span class="label">Exit Reason</span>
          <div class="value">{exit_reason_html}</div>
        </article>
        <article class="card">
          <span class="label">Dashboard URL</span>
          <div class="value">{dashboard_url or 'unavailable'}</div>
        </article>
      </section>

      <section class="card" style="margin-bottom: 16px;">
        <span class="label">Platform Status</span>
        <table>
          <thead>
            <tr>
              <th>Platform</th>
              <th>State</th>
              <th>Details</th>
            </tr>
          </thead>
          <tbody>
            {''.join(platform_rows)}
          </tbody>
        </table>
      </section>

      <section class="grid">
        <article class="card">
          <span class="label">Gateway Log</span>
          <pre>{gateway_log}</pre>
        </article>
        <article class="card">
          <span class="label">Error Log</span>
          <pre>{error_log}</pre>
        </article>
      </section>
    </main>
  </body>
</html>"""
