"""
Minimal Railway admin wrapper for Hermes.

- Serves a small admin UI at /
- Health check at /health
- Manages `hermes gateway` as a subprocess
- Stores config in /data/.hermes/.env
- Writes a minimal config.yaml so Hermes picks up the selected model/provider
"""

import asyncio
import base64
import os
import re
import secrets
import signal
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

from starlette.applications import Starlette
from starlette.authentication import (
    AuthCredentials,
    AuthenticationBackend,
    AuthenticationError,
    SimpleUser,
)
from starlette.middleware import Middleware
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse
from starlette.routing import Route

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

HERMES_HOME = os.environ.get("HERMES_HOME", "/data/.hermes")
ENV_FILE = Path(HERMES_HOME) / ".env"
CONFIG_FILE = Path(HERMES_HOME) / "config.yaml"

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
if not ADMIN_PASSWORD:
    ADMIN_PASSWORD = secrets.token_urlsafe(16)
    print(
        f"[server] Admin credentials generated - username: {ADMIN_USERNAME} password: {ADMIN_PASSWORD}",
        flush=True,
    )
else:
    print(f"[server] Admin username: {ADMIN_USERNAME}", flush=True)

ENV_VARS = [
    ("LLM_PROVIDER", "LLM Provider", "model", False),
    ("LLM_MODEL", "Model", "model", False),
    ("OPENROUTER_API_KEY", "OpenRouter", "provider", True),
    ("GOOGLE_API_KEY", "Google / Gemini", "provider", True),
    ("DEEPSEEK_API_KEY", "DeepSeek", "provider", True),
    ("DASHSCOPE_API_KEY", "DashScope", "provider", True),
    ("GLM_API_KEY", "GLM / Z.AI", "provider", True),
    ("KIMI_API_KEY", "Kimi", "provider", True),
    ("MINIMAX_API_KEY", "MiniMax", "provider", True),
    ("HF_TOKEN", "Hugging Face", "provider", True),
    ("TELEGRAM_BOT_TOKEN", "Telegram Bot Token", "channel", True),
    ("DISCORD_BOT_TOKEN", "Discord Bot Token", "channel", True),
    ("SLACK_BOT_TOKEN", "Slack Bot Token", "channel", True),
    ("SLACK_APP_TOKEN", "Slack App Token", "channel", True),
    ("GITHUB_TOKEN", "GitHub Token", "tool", True),
    ("PARALLEL_API_KEY", "Parallel API Key", "tool", True),
    ("FIRECRAWL_API_KEY", "Firecrawl API Key", "tool", True),
    ("TAVILY_API_KEY", "Tavily API Key", "tool", True),
    ("FAL_KEY", "FAL Key", "tool", True),
    ("BROWSERBASE_API_KEY", "Browserbase API Key", "tool", True),
    ("BROWSERBASE_PROJECT_ID", "Browserbase Project ID", "tool", False),
    ("VOICE_TOOLS_OPENAI_KEY", "OpenAI Voice/TTS Key", "tool", True),
    ("HONCHO_API_KEY", "Honcho API Key", "tool", True),
    ("GATEWAY_ALLOW_ALL_USERS", "Allow all users", "gateway", False),
    ("ADMIN_USERNAME", "Admin username", "admin", False),
    ("ADMIN_PASSWORD", "Admin password", "admin", True),
]

SECRET_KEYS = {k for k, _, _, s in ENV_VARS if s}
PROVIDER_KEYS = [k for k, _, category, _ in ENV_VARS if category == "provider"]


def ensure_dirs() -> None:
    for rel in [
        "",
        "cron",
        "sessions",
        "logs",
        "memories",
        "skills",
        "pairing",
        "hooks",
        "image_cache",
        "audio_cache",
        "workspace",
    ]:
        (Path(HERMES_HOME) / rel).mkdir(parents=True, exist_ok=True)


def read_env(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    out: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            v = v[1:-1]
        out[k.strip()] = v
    return out


def write_env(path: Path, data: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for key in sorted(data.keys()):
        value = data[key]
        if value is None:
            continue
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def write_config_yaml(data: Dict[str, str]) -> None:
    provider = (data.get("LLM_PROVIDER", "").strip() or "openrouter").lower()
    model = data.get("LLM_MODEL", "").strip()

    if not model:
        if provider == "gemini":
            model = "gemini-2.5-flash"
        elif provider == "openrouter":
            model = "openrouter/auto"
        else:
            model = "openrouter/auto"

    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(
        f"""model:
  default: "{model}"
  provider: "{provider}"

terminal:
  backend: "local"
  timeout: 60
  cwd: "/tmp"

agent:
  max_iterations: 50
  data_dir: "{HERMES_HOME}"
"""
    )


def mask(data: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in data.items():
        if k in SECRET_KEYS and v:
            out[k] = (v[:8] + "***") if len(v) > 8 else "***"
        else:
            out[k] = v
    return out


def unmask(new: Dict[str, str], existing: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in new.items():
        if k in SECRET_KEYS and isinstance(v, str) and v.endswith("***"):
            out[k] = existing.get(k, "")
        else:
            out[k] = v
    return out


class BasicAuth(AuthenticationBackend):
    async def authenticate(self, conn):
        header = conn.headers.get("Authorization")
        if not header:
            return None
        try:
            scheme, creds = header.split()
            if scheme.lower() != "basic":
                return None
            user, _, pw = base64.b64decode(creds).decode().partition(":")
        except Exception as exc:
            raise AuthenticationError("Invalid credentials") from exc

        if user == ADMIN_USERNAME and pw == ADMIN_PASSWORD:
            return AuthCredentials(["authenticated"]), SimpleUser(user)
        raise AuthenticationError("Invalid credentials")


def guard(request: Request):
    if not request.user.is_authenticated:
        return PlainTextResponse(
            "Unauthorized",
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="hermes-admin"'},
        )


INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Hermes Admin</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root {
      --bg:#0b1020; --panel:#121a2f; --muted:#94a3b8; --text:#e5e7eb;
      --accent:#60a5fa; --ok:#22c55e; --warn:#f59e0b; --bad:#ef4444; --border:#24314f;
    }
    body { font-family: Inter, ui-sans-serif, system-ui, Arial; margin:0; background:var(--bg); color:var(--text); }
    .wrap { max-width: 1200px; margin: 0 auto; padding: 24px; }
    .grid { display:grid; grid-template-columns: 1fr 1fr; gap:16px; }
    .card { background:var(--panel); border:1px solid var(--border); border-radius:16px; padding:16px; box-shadow: 0 8px 30px rgba(0,0,0,.15); }
    .full { grid-column:1 / -1; }
    h1, h2, h3 { margin-top:0; }
    .muted { color:var(--muted); }
    .row { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
    input, textarea, select {
      width:100%; background:#0f172a; color:var(--text); border:1px solid var(--border);
      border-radius:10px; padding:10px; box-sizing:border-box;
    }
    textarea { min-height: 120px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    button {
      background:#1d4ed8; color:white; border:none; border-radius:10px; padding:10px 14px; cursor:pointer;
    }
    button.secondary { background:#334155; }
    button.warn { background:#b45309; }
    button.bad { background:#b91c1c; }
    .pill { display:inline-block; padding:6px 10px; border-radius:999px; font-size:12px; background:#0f172a; border:1px solid var(--border); }
    .ok { color:var(--ok); }
    .badc { color:var(--bad); }
    pre {
      background:#0a0f1f; border:1px solid var(--border); border-radius:12px; padding:12px;
      max-height: 420px; overflow:auto; white-space:pre-wrap;
    }
    .field { margin-bottom:12px; }
    .small { font-size:12px; }
    .spacer { height:12px; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="row" style="justify-content:space-between; margin-bottom:16px;">
      <div>
        <h1>Hermes Admin</h1>
        <div class="muted">Railway wrapper for Hermes gateway</div>
      </div>
      <div class="row">
        <span class="pill">/health</span>
        <span class="pill">/api/status</span>
        <span class="pill">/api/logs</span>
      </div>
    </div>

    <div class="grid">
      <div class="card">
        <h3>Status</h3>
        <div id="status" class="muted">Loading...</div>
        <div class="spacer"></div>
        <div class="row">
          <button onclick="gatewayAction('start')">Start</button>
          <button class="warn" onclick="gatewayAction('restart')">Restart</button>
          <button class="bad" onclick="gatewayAction('stop')">Stop</button>
          <button class="secondary" onclick="refreshAll()">Refresh</button>
        </div>
      </div>

      <div class="card">
        <h3>Quick Config</h3>

        <div class="field">
          <label class="small muted">Provider</label>
          <select id="LLM_PROVIDER" onchange="updateProviderFields()">
            <option value="openrouter">OpenRouter</option>
            <option value="gemini">Google Gemini</option>
          </select>
        </div>

        <div class="field">
          <label class="small muted">Model</label>
          <input id="LLM_MODEL" placeholder="e.g. gemini-2.5-flash" />
        </div>

        <div class="field" id="openrouterField">
          <label class="small muted">OpenRouter API Key</label>
          <input id="OPENROUTER_API_KEY" type="password" placeholder="sk-or-..." />
        </div>

        <div class="field" id="googleField">
          <label class="small muted">Google API Key</label>
          <input id="GOOGLE_API_KEY" type="password" placeholder="AIza..." />
        </div>

        <div class="field">
          <label class="small muted">Telegram Bot Token</label>
          <input id="TELEGRAM_BOT_TOKEN" type="password" placeholder="123456:ABC..." />
        </div>

        <div class="row">
          <button onclick="saveConfig(true)">Save & Restart</button>
          <button class="secondary" onclick="loadConfig()">Reload</button>
        </div>
      </div>

      <div class="card full">
        <h3>Raw Environment Values</h3>
        <div class="muted small">One KEY=VALUE per line. Secrets are masked when reloaded.</div>
        <div class="spacer"></div>
        <textarea id="envText"></textarea>
        <div class="spacer"></div>
        <div class="row">
          <button onclick="saveRawEnv(true)">Save Raw & Restart</button>
          <button class="secondary" onclick="loadConfig()">Reload</button>
        </div>
      </div>

      <div class="card full">
        <h3>Gateway Logs</h3>
        <pre id="logs">Loading logs...</pre>
      </div>
    </div>
  </div>

<script>
async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(txt || ("HTTP " + res.status));
  }
  const contentType = res.headers.get("content-type") || "";
  if (contentType.includes("application/json")) return res.json();
  return res.text();
}

function envObjectFromTextarea(text) {
  const out = {};
  const lines = text.split(/\\r?\\n/);
  for (const raw of lines) {
    const line = raw.trim();
    if (!line || line.startsWith("#") || !line.includes("=")) continue;
    const idx = line.indexOf("=");
    const key = line.slice(0, idx).trim();
    const val = line.slice(idx + 1).trim();
    out[key] = val;
  }
  return out;
}

function textareaFromEnvObject(obj) {
  return Object.keys(obj).sort().map(k => `${k}=${obj[k] ?? ""}`).join("\\n");
}

function updateProviderFields() {
  const provider = document.getElementById("LLM_PROVIDER").value;

  document.getElementById("openrouterField").style.display =
    provider === "openrouter" ? "block" : "none";

  document.getElementById("googleField").style.display =
    provider === "gemini" ? "block" : "none";

  const modelInput = document.getElementById("LLM_MODEL");
  if (!modelInput.value.trim()) {
    if (provider === "gemini") {
      modelInput.placeholder = "e.g. gemini-2.5-flash";
    } else {
      modelInput.placeholder = "e.g. openrouter/auto";
    }
  }
}

async function loadStatus() {
  const data = await api("/api/status");
  const gw = data.gateway || {};
  const state = gw.state || "unknown";
  const stateClass = state === "running" ? "ok" : (state === "error" ? "badc" : "");
  document.getElementById("status").innerHTML = `
    <div>State: <strong class="${stateClass}">${state}</strong></div>
    <div>PID: <strong>${gw.pid ?? "-"}</strong></div>
    <div>Uptime: <strong>${gw.uptime ?? "-"}</strong></div>
    <div>Restarts: <strong>${gw.restarts ?? 0}</strong></div>
  `;
}

async function loadLogs() {
  const data = await api("/api/logs");
  document.getElementById("logs").textContent = (data.lines || []).join("\\n") || "No logs yet.";
}

async function loadConfig() {
  const data = await api("/api/config");
  const vars = data.vars || {};

  document.getElementById("LLM_PROVIDER").value = vars.LLM_PROVIDER || "openrouter";
  document.getElementById("LLM_MODEL").value = vars.LLM_MODEL || "";
  document.getElementById("OPENROUTER_API_KEY").value = vars.OPENROUTER_API_KEY || "";
  document.getElementById("GOOGLE_API_KEY").value = vars.GOOGLE_API_KEY || "";
  document.getElementById("TELEGRAM_BOT_TOKEN").value = vars.TELEGRAM_BOT_TOKEN || "";
  document.getElementById("envText").value = textareaFromEnvObject(vars);

  updateProviderFields();
}

async function saveConfig(restart) {
  const current = envObjectFromTextarea(document.getElementById("envText").value);

  current.LLM_PROVIDER = document.getElementById("LLM_PROVIDER").value.trim();
  current.LLM_MODEL = document.getElementById("LLM_MODEL").value.trim();

  current.OPENROUTER_API_KEY =
    document.getElementById("OPENROUTER_API_KEY").value.trim() ||
    current.OPENROUTER_API_KEY ||
    "";

  current.GOOGLE_API_KEY =
    document.getElementById("GOOGLE_API_KEY").value.trim() ||
    current.GOOGLE_API_KEY ||
    "";

  current.TELEGRAM_BOT_TOKEN =
    document.getElementById("TELEGRAM_BOT_TOKEN").value.trim() ||
    current.TELEGRAM_BOT_TOKEN ||
    "";

  await api("/api/config", {
    method: "PUT",
    body: JSON.stringify({ vars: current, _restart: !!restart })
  });
  await refreshAll();
}

async function saveRawEnv(restart) {
  const vars = envObjectFromTextarea(document.getElementById("envText").value);
  await api("/api/config", {
    method: "PUT",
    body: JSON.stringify({ vars, _restart: !!restart })
  });
  await refreshAll();
}

async function gatewayAction(action) {
  await api(`/api/gateway/${action}`, { method: "POST" });
  setTimeout(refreshAll, 1000);
}

async function refreshAll() {
  await Promise.all([loadStatus(), loadLogs(), loadConfig()]);
}

updateProviderFields();
refreshAll();
setInterval(loadStatus, 5000);
setInterval(loadLogs, 3000);
</script>
</body>
</html>
"""


class Gateway:
    def __init__(self):
        self.proc: asyncio.subprocess.Process | None = None
        self.state = "stopped"
        self.logs: deque[str] = deque(maxlen=1000)
        self.started_at: float | None = None
        self.restarts = 0

    async def start(self):
        if self.proc and self.proc.returncode is None:
            return
        ensure_dirs()
        self.state = "starting"
        try:
            env = {**os.environ, "HERMES_HOME": HERMES_HOME, "HOME": "/data"}
            file_env = read_env(ENV_FILE)
            env.update(file_env)

            write_config_yaml(file_env)

            model = file_env.get("LLM_MODEL", "")
            provider = file_env.get("LLM_PROVIDER", "")
            provider_present = any(file_env.get(k) for k in PROVIDER_KEYS)

            print(
                f"[gateway] starting - provider={provider or 'NOT SET'} model={model or 'NOT SET'} provider_key={'set' if provider_present else 'NOT SET'}",
                flush=True,
            )

            self.proc = await asyncio.create_subprocess_exec(
                "hermes",
                "gateway",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )
            self.state = "running"
            self.started_at = time.time()
            asyncio.create_task(self._drain())
        except Exception as exc:
            self.state = "error"
            self.logs.append(f"[error] Failed to start gateway: {exc!r}")

    async def stop(self):
        if not self.proc or self.proc.returncode is not None:
            self.state = "stopped"
            return
        self.state = "stopping"
        self.proc.terminate()
        try:
            await asyncio.wait_for(self.proc.wait(), timeout=10)
        except asyncio.TimeoutError:
            self.proc.kill()
            await self.proc.wait()
        self.state = "stopped"
        self.started_at = None

    async def restart(self):
        await self.stop()
        self.restarts += 1
        await self.start()

    async def _drain(self):
        assert self.proc and self.proc.stdout
        async for raw in self.proc.stdout:
            line = ANSI_ESCAPE.sub("", raw.decode(errors="replace").rstrip())
            self.logs.append(line)

        if self.state == "running":
            code = self.proc.returncode
            self.state = "error"
            self.logs.append(f"[error] Gateway exited (code {code})")

    def status(self):
        uptime = (
            int(time.time() - self.started_at)
            if self.started_at and self.state == "running"
            else None
        )
        return {
            "state": self.state,
            "pid": self.proc.pid if self.proc and self.proc.returncode is None else None,
            "uptime": uptime,
            "restarts": self.restarts,
        }


gw = Gateway()
cfg_lock = asyncio.Lock()


async def page_index(request: Request):
    if err := guard(request):
        return err
    return HTMLResponse(INDEX_HTML)


async def route_health(request: Request):
    return JSONResponse({"status": "ok", "gateway": gw.state})


async def api_config_get(request: Request):
    if err := guard(request):
        return err
    async with cfg_lock:
        data = read_env(ENV_FILE)
        return JSONResponse({"vars": mask(data), "defs": ENV_VARS})


async def api_config_put(request: Request):
    if err := guard(request):
        return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    restart = bool(body.get("_restart", False))
    new_vars = body.get("vars", {})
    if not isinstance(new_vars, dict):
        return JSONResponse({"error": "vars must be an object"}, status_code=400)

    async with cfg_lock:
        existing = read_env(ENV_FILE)
        merged = unmask(new_vars, existing)

        for k, v in existing.items():
            if k not in merged:
                merged[k] = v

        provider = (merged.get("LLM_PROVIDER", "") or "").strip().lower()

        if not provider:
            if merged.get("GOOGLE_API_KEY"):
                provider = "gemini"
            else:
                provider = "openrouter"
            merged["LLM_PROVIDER"] = provider

        if provider == "gemini":
            merged.pop("OPENROUTER_API_KEY", None)
        elif provider == "openrouter":
            merged.pop("GOOGLE_API_KEY", None)

        merged["ADMIN_USERNAME"] = ADMIN_USERNAME
        merged["ADMIN_PASSWORD"] = ADMIN_PASSWORD

        write_env(ENV_FILE, merged)
        write_config_yaml(merged)

    if restart:
        asyncio.create_task(gw.restart())

    return JSONResponse({"ok": True, "restarting": restart})


async def api_status(request: Request):
    if err := guard(request):
        return err
    return JSONResponse({"gateway": gw.status()})


async def api_logs(request: Request):
    if err := guard(request):
        return err
    return JSONResponse({"lines": list(gw.logs)})


async def api_gw_start(request: Request):
    if err := guard(request):
        return err
    asyncio.create_task(gw.start())
    return JSONResponse({"ok": True})


async def api_gw_stop(request: Request):
    if err := guard(request):
        return err
    asyncio.create_task(gw.stop())
    return JSONResponse({"ok": True})


async def api_gw_restart(request: Request):
    if err := guard(request):
        return err
    asyncio.create_task(gw.restart())
    return JSONResponse({"ok": True})


async def api_config_reset(request: Request):
    if err := guard(request):
        return err
    asyncio.create_task(gw.stop())
    async with cfg_lock:
        if ENV_FILE.exists():
            ENV_FILE.unlink()
        write_config_yaml({})
    return JSONResponse({"ok": True})


async def auto_start():
    ensure_dirs()
    data = read_env(ENV_FILE)
    if any(data.get(k) for k in PROVIDER_KEYS):
        asyncio.create_task(gw.start())
    else:
        print(
            "[server] No provider key found in /data/.hermes/.env - gateway not auto-started.",
            flush=True,
        )


@asynccontextmanager
async def lifespan(app):
    await auto_start()
    yield
    await gw.stop()


routes = [
    Route("/", page_index),
    Route("/health", route_health),
    Route("/api/config", api_config_get, methods=["GET"]),
    Route("/api/config", api_config_put, methods=["PUT"]),
    Route("/api/status", api_status, methods=["GET"]),
    Route("/api/logs", api_logs, methods=["GET"]),
    Route("/api/gateway/start", api_gw_start, methods=["POST"]),
    Route("/api/gateway/stop", api_gw_stop, methods=["POST"]),
    Route("/api/gateway/restart", api_gw_restart, methods=["POST"]),
    Route("/api/config/reset", api_config_reset, methods=["POST"]),
]

app = Starlette(
    routes=routes,
    middleware=[Middleware(AuthenticationMiddleware, backend=BasicAuth())],
    lifespan=lifespan,
)

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        loop="asyncio",
    )
    server = uvicorn.Server(config)

    def _shutdown():
        loop.create_task(gw.stop())
        server.should_exit = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _shutdown)
        except NotImplementedError:
            pass

    loop.run_until_complete(server.serve())
