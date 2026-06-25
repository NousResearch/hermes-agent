import os
import base64
import hashlib
import hmac
import json
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse


app = FastAPI(title="Company AI Gateway", version="0.1.0")

KNOWLEDGE_API_URL = os.environ.get("KNOWLEDGE_API_URL", "http://knowledge-api:8010")
GATEWAY_API_KEY = os.environ.get("GATEWAY_API_KEY", "")
ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "")
GATEWAY_TOKEN_SECRET = os.environ.get("GATEWAY_TOKEN_SECRET") or GATEWAY_API_KEY
STATIC_DIR = Path(__file__).resolve().parent / "static"
ADMIN_SSH_TARGET = os.environ.get("ADMIN_SSH_TARGET", "root@YOUR_SERVER_IP")
DEPLOY_ROOT = os.environ.get("DEPLOY_ROOT", "/opt/second-brain")
INSTALL_ASSET_VERSION = os.environ.get("SECOND_BRAIN_INSTALL_VERSION", "20260624-generic")
MAX_TEXT_UPLOAD_BYTES = int(os.environ.get("MAX_TEXT_UPLOAD_BYTES", str(10 * 1024 * 1024)))
PUBLIC_WORKSPACE = "company_public"
C_LEVEL_WORKSPACE = "department_c_level"


def resolve_public_base_url() -> str:
    configured = os.environ.get("PUBLIC_BASE_URL", "").strip()
    if configured:
        return configured.rstrip("/")
    domain = os.environ.get("SECOND_BRAIN_DOMAIN", "").strip()
    if domain and domain not in {"example.com", "second-brain.example.com"}:
        if domain.startswith(("http://", "https://")):
            return domain.rstrip("/")
        return f"https://{domain}"
    return "http://localhost:8000"


PUBLIC_BASE_URL = resolve_public_base_url()

ALLOWED_GROUPS = {
    "company_all",
    "role_admin",
}


def require_key(x_api_key: str | None) -> None:
    if GATEWAY_API_KEY and x_api_key != GATEWAY_API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")


def b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def sign_token(header: dict[str, Any], payload: dict[str, Any]) -> str:
    encoded_header = b64url_encode(json.dumps(header, separators=(",", ":")).encode())
    encoded_payload = b64url_encode(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{encoded_header}.{encoded_payload}".encode()
    signature = hmac.new(GATEWAY_TOKEN_SECRET.encode(), signing_input, hashlib.sha256).digest()
    return f"{encoded_header}.{encoded_payload}.{b64url_encode(signature)}"


def parse_bearer(authorization: str | None) -> dict[str, Any] | None:
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    parts = token.split(".")
    if len(parts) != 3:
        raise HTTPException(status_code=401, detail="invalid bearer token")
    signing_input = f"{parts[0]}.{parts[1]}".encode()
    expected = hmac.new(GATEWAY_TOKEN_SECRET.encode(), signing_input, hashlib.sha256).digest()
    try:
        actual = b64url_decode(parts[2])
        payload = json.loads(b64url_decode(parts[1]))
    except Exception as exc:
        raise HTTPException(status_code=401, detail="invalid bearer token") from exc
    if not hmac.compare_digest(expected, actual):
        raise HTTPException(status_code=401, detail="invalid bearer token")
    if int(payload.get("exp", 0)) < int(time.time()):
        raise HTTPException(status_code=401, detail="token expired")
    return payload


def require_auth(
    x_api_key: str | None,
    authorization: str | None,
    *,
    admin_only: bool = False,
) -> dict[str, Any]:
    if GATEWAY_API_KEY and x_api_key == GATEWAY_API_KEY:
        return {
            "type": "admin_api_key",
            "email": ADMIN_EMAIL,
            "groups": sorted(ALLOWED_GROUPS),
            "role": "admin",
        }
    token_payload = parse_bearer(authorization)
    if token_payload:
        if admin_only and token_payload.get("role") != "admin":
            raise HTTPException(status_code=403, detail="admin role required")
        return token_payload
    raise HTTPException(status_code=401, detail="missing or invalid credentials")


def normalize_groups(groups: list[Any]) -> list[str]:
    normalized = []
    for group in groups:
        value = str(group).strip().lower().replace(" ", "_")
        if value in ALLOWED_GROUPS:
            normalized.append(value)
    return sorted(set(normalized))


def is_admin_auth(auth: dict[str, Any]) -> bool:
    groups = set(auth.get("groups") or [])
    return auth.get("type") == "admin_api_key" or auth.get("role") == "admin" or "role_admin" in groups


def query_groups_for_auth(auth: dict[str, Any]) -> list[str]:
    if is_admin_auth(auth):
        return ["role_admin"]
    return ["company_all"]


def visible_workspace_slugs(auth: dict[str, Any]) -> set[str]:
    if is_admin_auth(auth):
        return {PUBLIC_WORKSPACE, C_LEVEL_WORKSPACE}
    return {PUBLIC_WORKSPACE}


def ide_config(token: str) -> dict[str, Any]:
    return {
        "base_url": PUBLIC_BASE_URL,
        "headers": {"Authorization": f"Bearer {token}"},
        "query_endpoint": f"{PUBLIC_BASE_URL}/api/query",
        "workspaces_endpoint": f"{PUBLIC_BASE_URL}/api/workspaces",
        "curl_example": (
            "curl -X POST "
            f"{PUBLIC_BASE_URL}/api/query "
            f"-H 'Authorization: Bearer {token}' "
            "-H 'Content-Type: application/json' "
            "-d '{\"query\":\"What documents can I access?\",\"mode\":\"mix\"}'"
        ),
    }


def render_static_template(path: Path) -> str:
    text = path.read_text()
    return (
        text.replace("__PUBLIC_BASE_URL__", PUBLIC_BASE_URL)
        .replace("__ADMIN_SSH_TARGET__", ADMIN_SSH_TARGET)
        .replace("__DEPLOY_ROOT__", DEPLOY_ROOT)
    )


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "service": "company-ai-gateway",
        "status": "ok",
        "admin_email": ADMIN_EMAIL,
        "docs": "/docs",
        "install": "/install",
        "admin": "/admin",
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/install", response_class=HTMLResponse)
async def install_page() -> str:
    install_command = (
        f"curl -fsSL {PUBLIC_BASE_URL}/install.sh "
        "-o install-company-second-brain.sh && bash install-company-second-brain.sh"
    )
    bundle_url = f"{PUBLIC_BASE_URL}/download/company-second-brain-skill.tar.gz?v={INSTALL_ASSET_VERSION}"
    installer_url = f"{PUBLIC_BASE_URL}/install.sh"
    return f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Company Second Brain Install</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #111827;
      --muted: #5b6472;
      --border: #d9dee7;
      --accent: #116149;
      --accent-2: #1f6feb;
      --code: #151b23;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }}
    main {{
      width: min(980px, calc(100% - 32px));
      margin: 0 auto;
      padding: 40px 0;
    }}
    header {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 24px;
      padding-bottom: 24px;
      border-bottom: 1px solid var(--border);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(30px, 5vw, 52px);
      line-height: 1.04;
      letter-spacing: 0;
    }}
    p {{ margin: 0; color: var(--muted); }}
    .status {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      min-width: 230px;
      justify-content: flex-end;
    }}
    .pill {{
      border: 1px solid var(--border);
      background: var(--panel);
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 13px;
      color: #243042;
      white-space: nowrap;
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.15fr) minmax(280px, .85fr);
      gap: 18px;
      margin-top: 24px;
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 22px;
    }}
    h2 {{
      margin: 0 0 14px;
      font-size: 20px;
      letter-spacing: 0;
    }}
    ol {{
      margin: 0;
      padding-left: 22px;
      color: #253044;
    }}
    li {{ margin: 10px 0; }}
    .actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }}
    a.button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 40px;
      padding: 0 14px;
      border-radius: 7px;
      border: 1px solid transparent;
      background: var(--accent);
      color: white;
      text-decoration: none;
      font-weight: 650;
    }}
    a.button.secondary {{
      background: white;
      color: var(--accent-2);
      border-color: var(--border);
    }}
    pre {{
      margin: 14px 0 0;
      overflow-x: auto;
      white-space: pre-wrap;
      word-break: break-word;
      background: var(--code);
      color: #f2f4f8;
      border-radius: 8px;
      padding: 14px;
      font-size: 13px;
    }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .note {{
      margin-top: 16px;
      padding: 12px;
      border-left: 3px solid var(--accent);
      background: #f3faf7;
      color: #244136;
      border-radius: 4px;
    }}
    .table {{
      display: grid;
      gap: 8px;
      margin-top: 10px;
      font-size: 14px;
    }}
    .row {{
      display: grid;
      grid-template-columns: 110px 1fr;
      gap: 10px;
      padding: 8px 0;
      border-bottom: 1px solid #eef1f5;
    }}
    .key {{ color: var(--muted); }}
    footer {{
      margin-top: 18px;
      color: var(--muted);
      font-size: 13px;
    }}
    @media (max-width: 760px) {{
      main {{ width: min(100% - 24px, 980px); padding: 24px 0; }}
      header {{ display: block; }}
      .status {{ justify-content: flex-start; margin-top: 16px; }}
      .grid {{ grid-template-columns: 1fr; }}
      .row {{ grid-template-columns: 1fr; gap: 2px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Company Second Brain</h1>
        <p>Cài skill cho agent hoặc IDE, nhập token nhân sự, rồi query tài liệu public của công ty. C-Level chỉ mở cho admin.</p>
      </div>
      <div class="status" aria-label="Compatibility">
        <span class="pill">Hermes skill</span>
        <span class="pill">Codex CLI</span>
        <span class="pill">IDE/agent API</span>
      </div>
    </header>

    <div class="grid">
      <section>
        <h2>Cài nhanh</h2>
        <ol>
          <li>Admin tạo token nhân sự và gửi token cho user.</li>
          <li>Nhân sự chạy lệnh dưới đây trong terminal của agent hoặc IDE.</li>
          <li>Installer sẽ hỏi token, xác minh với gateway, rồi lưu cấu hình local.</li>
        </ol>
        <pre><code>{install_command}</code></pre>
        <div class="actions">
          <a class="button" href="{installer_url}">Tải installer</a>
          <a class="button secondary" href="{bundle_url}">Tải skill bundle</a>
        </div>
        <p class="note">Nhân sự luôn query workspace <code>company_public</code>. Admin query thêm <code>department_c_level</code>.</p>
      </section>

      <section>
        <h2>Thông số kết nối</h2>
        <div class="table">
          <div class="row"><div class="key">Gateway</div><div><code>{PUBLIC_BASE_URL}</code></div></div>
          <div class="row"><div class="key">Query API</div><div><code>POST /api/query</code></div></div>
          <div class="row"><div class="key">Identity</div><div><code>GET /api/me</code></div></div>
          <div class="row"><div class="key">Workspaces</div><div><code>GET /api/workspaces</code></div></div>
        </div>
        <pre><code>second-brain query "tom tat tai lieu cong ty moi nhat"</code></pre>
        <pre><code>Authorization: Bearer USER_TOKEN</code></pre>
      </section>
    </div>
    <footer>Health check: <a href="/health">/health</a> · API docs: <a href="/docs">/docs</a></footer>
  </main>
</body>
</html>"""


@app.get("/admin", response_class=HTMLResponse)
async def admin_page() -> str:
    install_command = (
        f"curl -fsSL {PUBLIC_BASE_URL}/install.sh "
        "-o install-company-second-brain.sh && bash install-company-second-brain.sh"
    )
    return f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Company Second Brain Admin</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #111827;
      --muted: #5b6472;
      --border: #d9dee7;
      --accent: #116149;
      --code: #151b23;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }}
    main {{ width: min(1040px, calc(100% - 32px)); margin: 0 auto; padding: 36px 0; }}
    header {{ padding-bottom: 22px; border-bottom: 1px solid var(--border); }}
    h1 {{ margin: 0 0 8px; font-size: clamp(30px, 5vw, 48px); line-height: 1.05; letter-spacing: 0; }}
    h2 {{ margin: 0 0 12px; font-size: 20px; letter-spacing: 0; }}
    p {{ margin: 0; color: var(--muted); }}
    section {{ background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-top: 16px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    ol, ul {{ margin: 0; padding-left: 22px; color: #253044; }}
    li {{ margin: 8px 0; }}
    pre {{
      margin: 12px 0 0;
      overflow-x: auto;
      white-space: pre-wrap;
      word-break: break-word;
      background: var(--code);
      color: #f2f4f8;
      border-radius: 8px;
      padding: 14px;
      font-size: 13px;
    }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    a {{ color: #1f6feb; }}
    .note {{ margin-top: 12px; padding: 12px; border-left: 3px solid var(--accent); background: #f3faf7; color: #244136; border-radius: 4px; }}
    @media (max-width: 820px) {{ main {{ width: min(100% - 24px, 1040px); }} .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Admin Setup</h1>
      <p>Vận hành Company Second Brain: tạo token, upload tài liệu Public hoặc C-Level, nâng cấp skill/gateway và handoff cho agent khác.</p>
    </header>

    <section>
      <h2>1. Cài CLI/skill cho admin</h2>
      <ol>
        <li>Chạy installer như user bình thường.</li>
        <li>Khi installer hỏi token, dùng token admin bearer nếu muốn query C-Level; các lệnh quản trị dùng admin key.</li>
      </ol>
      <pre><code>{install_command}</code></pre>
    </section>

    <section>
      <h2>2. Lấy admin key trên VPS và cấu hình máy admin</h2>
      <p>Admin key không hiển thị trên web. Lấy trực tiếp từ VPS rồi lưu vào máy admin.</p>
      <pre><code>ssh {ADMIN_SSH_TARGET} 'cd {DEPLOY_ROOT} && grep "^GATEWAY_API_KEY=" .env'

~/.hermes/skills/productivity/company-second-brain/scripts/second-brain config \\
  --base-url {PUBLIC_BASE_URL} \\
  --admin-key "PASTE_GATEWAY_API_KEY"</code></pre>
    </section>

    <div class="grid">
      <section>
        <h2>3. Tạo token nhân sự</h2>
        <pre><code>second-brain admin-token-create \\
  --email user@company.com \\
  --name "Company User" \\
  --group company_all \\
  --expires-days 90</code></pre>
        <p class="note">Gửi cho nhân sự: link <a href="/install">/install</a> + token vừa tạo. Không gửi admin key.</p>
      </section>

      <section>
        <h2>4. Tạo token admin bearer</h2>
        <pre><code>second-brain admin-token-create \\
  --email admin@company.com \\
  --name "Admin" \\
  --group role_admin \\
  --admin \\
  --expires-days 365</code></pre>
      </section>
    </div>

    <section>
      <h2>5. Upload tài liệu</h2>
      <p>CLI ingest file text-friendly hoặc gọi API multipart. Upload trả <code>queued</code>, worker sẽ index nền vào LightRAG.</p>
      <pre><code>second-brain ingest-text \\
  --file ./company-handbook.md \\
  --title "Company Handbook" \\
  --target public

second-brain ingest-text \\
  --file ./board-plan.md \\
  --title "Board Plan" \\
  --target c_level \\
  --classification restricted

curl -X POST {PUBLIC_BASE_URL}/api/documents/file \\
  -H "X-API-Key: PASTE_GATEWAY_API_KEY" \\
  -F "file=@./company-handbook.md" \\
  -F "title=Company Handbook" \\
  -F "target=public"

second-brain document-status DOCUMENT_ID
second-brain queue-status</code></pre>
      <p class="note">Tài liệu lớn: text trên 1MB, file trên 10MB, PDF/DOCX trên 50 trang, hoặc ước tính trên 200 chunks. Với nhóm này nên extract, làm sạch, chia section rồi upload.</p>
    </section>

    <section>
      <h2>6. Nguồn tự động: Notion và Drive public</h2>
      <p>Admin cấu hình source một lần, worker sẽ scan theo chu kỳ. Có thể chạy manual scan bất kỳ lúc nào.</p>
      <pre><code>second-brain source-create \\
  --type notion \\
  --name "Company Notion" \\
  --notion-api-key "PASTE_NOTION_API_KEY" \\
  --notion-page-url "https://www.notion.so/..." \\
  --target public \\
  --interval-minutes 360

second-brain source-create \\
  --type drive_public \\
  --name "Public Drive Doc" \\
  --drive-url "https://docs.google.com/document/d/.../edit" \\
  --target public \\
  --interval-minutes 720

second-brain sources-list
second-brain source-scan SOURCE_ID
second-brain source-runs SOURCE_ID
second-brain source-update SOURCE_ID --interval-minutes 1440 --reset-schedule</code></pre>
      <p class="note">Drive public MVP hỗ trợ link Google Docs/Sheets/Slides hoặc file public. Public folder cần Drive API/OAuth nên chưa tự list folder trong MVP.</p>
    </section>

    <section>
      <h2>7. Mở rộng workspace sau MVP</h2>
      <ul>
        <li>MVP hiện tại chỉ chạy 2 LightRAG: <code>company_public</code> và <code>department_c_level</code>.</li>
        <li>Muốn thêm phòng ban riêng thì thêm service LightRAG mới, URL trong <code>knowledge-api</code>, row <code>rag_workspaces</code>, và policy gateway.</li>
        <li>Chỉ mở rộng khi có nhu cầu cách ly thật sự; càng nhiều workspace thì query càng fan-out chậm hơn.</li>
      </ul>
      <pre><code>cd {DEPLOY_ROOT}
docker compose up -d --build --remove-orphans company-ai-gateway knowledge-api knowledge-worker lightrag-company-public lightrag-c-level</code></pre>
    </section>

    <section>
      <h2>8. Nâng cấp và handoff</h2>
      <pre><code>cd {DEPLOY_ROOT}
docker compose ps
docker compose logs -f company-ai-gateway knowledge-api knowledge-worker
docker compose up -d --build company-ai-gateway knowledge-api knowledge-worker</code></pre>
      <p class="note">Tải spec/handoff cho agent khác tại <a href="/download/admin-handoff.md">/download/admin-handoff.md</a>.</p>
    </section>
  </main>
</body>
</html>"""


@app.get("/install.sh", response_class=PlainTextResponse)
async def install_script() -> PlainTextResponse:
    script = STATIC_DIR / "install-company-second-brain-skill.sh"
    if not script.exists():
        raise HTTPException(status_code=404, detail="installer not found")
    return PlainTextResponse(
        render_static_template(script),
        media_type="text/x-shellscript",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/download/company-second-brain-skill.tar.gz")
async def download_skill() -> FileResponse:
    bundle = STATIC_DIR / "company-second-brain-skill.tar.gz"
    if not bundle.exists():
        raise HTTPException(status_code=404, detail="skill bundle not found")
    return FileResponse(
        bundle,
        media_type="application/gzip",
        filename="company-second-brain-skill.tar.gz",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/download/install-company-second-brain-skill.sh")
async def download_installer() -> PlainTextResponse:
    script = STATIC_DIR / "install-company-second-brain-skill.sh"
    if not script.exists():
        raise HTTPException(status_code=404, detail="installer not found")
    return PlainTextResponse(
        render_static_template(script),
        media_type="text/x-shellscript",
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": 'attachment; filename="install-company-second-brain-skill.sh"',
        },
    )


@app.post("/api/admin/tokens")
async def create_user_token(
    payload: dict[str, Any],
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> Any:
    require_auth(x_api_key, authorization, admin_only=True)
    email = str(payload.get("email", "")).strip().lower()
    if "@" not in email:
        raise HTTPException(status_code=400, detail="valid email is required")
    groups = normalize_groups(payload.get("groups") or [])
    role = "admin" if "role_admin" in groups or payload.get("role") == "admin" else "member"
    expires_days = int(payload.get("expires_days", 30))
    expires_days = max(1, min(expires_days, 365))
    now = int(time.time())
    token_payload = {
        "iss": "company-ai-gateway",
        "sub": email,
        "email": email,
        "name": payload.get("name") or email,
        "groups": groups,
        "role": role,
        "iat": now,
        "exp": now + expires_days * 86400,
    }
    token = sign_token({"alg": "HS256", "typ": "JWT"}, token_payload)
    return {
        "token_type": "Bearer",
        "token": token,
        "expires_at": token_payload["exp"],
        "user": {
            "email": email,
            "name": token_payload["name"],
            "groups": groups,
            "role": role,
        },
        "setup": ide_config(token),
    }


@app.get("/api/me")
async def me(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    auth = require_auth(x_api_key, authorization)
    return {
        "email": auth.get("email"),
        "groups": auth.get("groups", []),
        "role": auth.get("role", "member"),
        "type": auth.get("type", "bearer_token"),
    }


@app.get("/api/workspaces")
async def workspaces(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> Any:
    auth = require_auth(x_api_key, authorization)
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(f"{KNOWLEDGE_API_URL}/workspaces")
        response.raise_for_status()
        body = response.json()
    allowed = visible_workspace_slugs(auth)
    body["workspaces"] = [
        workspace for workspace in body.get("workspaces", [])
        if workspace.get("slug") in allowed
    ]
    return body


@app.post("/api/query")
async def query(
    payload: dict[str, Any],
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> Any:
    auth = require_auth(x_api_key, authorization)
    payload = dict(payload)
    payload["groups"] = query_groups_for_auth(auth)
    payload["actor_email"] = auth.get("email")
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(f"{KNOWLEDGE_API_URL}/query", json=payload)
        response.raise_for_status()
        return response.json()


@app.post("/api/documents/text")
async def ingest_text(
    payload: dict[str, Any],
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> Any:
    auth = require_auth(x_api_key, authorization)
    if auth.get("role") != "admin":
        raise HTTPException(status_code=403, detail="admin role required")
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(f"{KNOWLEDGE_API_URL}/documents/text", json=payload)
        response.raise_for_status()
        return response.json()


@app.post("/api/documents/file")
async def ingest_file(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    target: str = Form(default="public"),
    department: str | None = Form(default=None),
    visibility: str = Form(default="public"),
    classification: str = Form(default="internal"),
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> Any:
    auth = require_auth(x_api_key, authorization)
    if auth.get("role") != "admin":
        raise HTTPException(status_code=403, detail="admin role required")
    data = await file.read()
    if len(data) > MAX_TEXT_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"file is too large for upload limit: {MAX_TEXT_UPLOAD_BYTES} bytes")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="MVP file upload supports UTF-8 text files. Use text extraction before uploading PDF/DOCX.",
        ) from exc
    payload = {
        "title": title or file.filename or "uploaded-document",
        "target": target,
        "department": department or target,
        "visibility": visibility,
        "classification": classification,
        "text": text,
    }
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(f"{KNOWLEDGE_API_URL}/documents/text", json=payload)
        response.raise_for_status()
        return response.json()


@app.get("/api/documents/{document_id}")
async def document_status(
    document_id: str,
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> Any:
    auth = require_auth(x_api_key, authorization)
    if auth.get("role") != "admin":
        raise HTTPException(status_code=403, detail="admin role required")
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"{KNOWLEDGE_API_URL}/documents/{document_id}")
        response.raise_for_status()
        return response.json()


@app.get("/api/queue/status")
async def queue_status(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> Any:
    auth = require_auth(x_api_key, authorization)
    if auth.get("role") != "admin":
        raise HTTPException(status_code=403, detail="admin role required")
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"{KNOWLEDGE_API_URL}/queue/status")
        response.raise_for_status()
        return response.json()


@app.get("/api/sources")
async def list_sources(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> Any:
    auth = require_auth(x_api_key, authorization)
    if auth.get("role") != "admin":
        raise HTTPException(status_code=403, detail="admin role required")
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"{KNOWLEDGE_API_URL}/sources")
        response.raise_for_status()
        return response.json()


@app.post("/api/sources")
async def create_source(
    payload: dict[str, Any],
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> Any:
    auth = require_auth(x_api_key, authorization)
    if auth.get("role") != "admin":
        raise HTTPException(status_code=403, detail="admin role required")
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(f"{KNOWLEDGE_API_URL}/sources", json=payload)
        response.raise_for_status()
        return response.json()


@app.patch("/api/sources/{source_id}")
async def update_source(
    source_id: str,
    payload: dict[str, Any],
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> Any:
    auth = require_auth(x_api_key, authorization)
    if auth.get("role") != "admin":
        raise HTTPException(status_code=403, detail="admin role required")
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.patch(f"{KNOWLEDGE_API_URL}/sources/{source_id}", json=payload)
        response.raise_for_status()
        return response.json()


@app.post("/api/sources/{source_id}/scan")
async def scan_source(
    source_id: str,
    payload: dict[str, Any] | None = None,
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> Any:
    auth = require_auth(x_api_key, authorization)
    if auth.get("role") != "admin":
        raise HTTPException(status_code=403, detail="admin role required")
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(f"{KNOWLEDGE_API_URL}/sources/{source_id}/scan", json=payload or {})
        response.raise_for_status()
        return response.json()


@app.get("/api/sources/{source_id}/runs")
async def source_runs(
    source_id: str,
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> Any:
    auth = require_auth(x_api_key, authorization)
    if auth.get("role") != "admin":
        raise HTTPException(status_code=403, detail="admin role required")
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"{KNOWLEDGE_API_URL}/sources/{source_id}/runs")
        response.raise_for_status()
        return response.json()


@app.get("/download/admin-handoff.md")
async def download_admin_handoff() -> PlainTextResponse:
    doc = STATIC_DIR / "admin-handoff.md"
    if not doc.exists():
        raise HTTPException(status_code=404, detail="handoff document not found")
    return PlainTextResponse(
        render_static_template(doc),
        media_type="text/markdown",
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": 'attachment; filename="admin-handoff.md"',
        },
    )
