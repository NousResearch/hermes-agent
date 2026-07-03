"""Admin HTTP API + static UI for the control-plane.

Runs the supervisor in-process so the UI can provision, wake/sleep, delete, and
send test messages to any employee's isolated worker.

SECURITY: this is a privileged admin surface (it can act as any tenant). Bind to
localhost only; put real auth + TLS in front before exposing it. Same posture as
Hermes' own dashboard.
"""
from __future__ import annotations

import time
from pathlib import Path

from aiohttp import web

from .backends import make_backend
from .config import Settings
from .models import Employee
from .provisioner import deprovision, scaffold_home
from .registry import Registry
from .secrets import LinkStore, make_store
from .skills import secret_status
from . import integrations as integrations_mod
from .supervisor import CapacityFull, Supervisor

_WEB = Path(__file__).parent / "web"


class OrchardAPI:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.registry = Registry(settings.paths.registry_db)
        self.backend = make_backend(settings)
        self.supervisor = Supervisor(settings, self.backend)
        self.secrets = make_store(settings)
        self.links = LinkStore(settings.paths.links_db())

    def build_app(self) -> web.Application:
        app = web.Application(client_max_size=1 << 20)
        app.router.add_get("/", self.index)
        app.router.add_get("/api/status", self.status)
        app.router.add_get("/api/employees", self.list_employees)
        app.router.add_post("/api/employees", self.create_employee)
        app.router.add_delete("/api/employees/{id}", self.delete_employee)
        app.router.add_post("/api/employees/{id}/wake", self.wake)
        app.router.add_post("/api/employees/{id}/sleep", self.sleep)
        app.router.add_post("/api/employees/{id}/chat", self.chat)
        # secret management
        app.router.add_post("/mm/secret", self.mm_secret)          # Mattermost slash command
        app.router.add_get("/secret-entry", self.secret_entry_form)  # one-time link form
        app.router.add_post("/secret-entry", self.secret_entry_submit)
        app.router.add_get("/api/employees/{id}/secrets", self.list_secrets)
        app.router.add_delete("/api/employees/{id}/secrets/{name}", self.delete_secret)
        # integrations (dashboard cards)
        app.router.add_get("/api/employees/{id}/integrations", self.list_integrations)
        app.router.add_post("/api/employees/{id}/integrations/{iid}/link", self.integration_link)
        app.router.add_delete("/api/employees/{id}/integrations/{iid}", self.delete_integration)
        app.on_startup.append(lambda _a: self.supervisor.start())
        app.on_cleanup.append(self._cleanup)
        return app

    async def _cleanup(self, _app) -> None:
        await self.supervisor.stop()
        self.registry.close()
        self.links.close()

    # --- static --------------------------------------------------------------
    async def index(self, _req: web.Request) -> web.Response:
        return web.FileResponse(_WEB / "index.html")

    # --- api -----------------------------------------------------------------
    async def status(self, _req: web.Request) -> web.Response:
        awake = self.supervisor.snapshot()
        return web.json_response({
            "backend": self.settings.backend,
            "sandbox": self.settings.security.sandbox or "none",
            "model": f"{self.settings.llm.provider}:{self.settings.llm.model}",
            "endpoint": self.settings.llm.base_url,
            "active": len(awake),
            "max_active": self.settings.supervisor.max_active_workers,
            "employees": len(self.registry.all()),
        })

    def _employee_view(self, emp: Employee) -> dict:
        info = next((w for w in self.supervisor.snapshot() if w.employee_id == emp.id), None)
        home = self.settings.paths.home_for(emp.id)
        return {
            "id": emp.id,
            "name": emp.display_name,
            "mm_user": emp.mm_user_id,
            "status": self.supervisor.status_of(emp.id).value,
            "last_used": info.last_used if info else None,
            "home_ok": home.exists(),
        }

    async def list_employees(self, _req: web.Request) -> web.Response:
        return web.json_response([self._employee_view(e) for e in self.registry.all()])

    async def create_employee(self, req: web.Request) -> web.Response:
        body = await req.json()
        try:
            emp = self.registry.add(
                body["id"].strip(),
                (body.get("name") or body["id"]).strip(),
                body["mm_user"].strip(),
            )
            scaffold_home(self.settings, emp)
        except (KeyError, ValueError) as e:
            return web.json_response({"error": str(e)}, status=400)
        return web.json_response(self._employee_view(emp), status=201)

    async def delete_employee(self, req: web.Request) -> web.Response:
        eid = req.match_info["id"]
        emp = self.registry.get(eid)
        if emp:
            await self.supervisor.put_to_sleep(emp)
        self.registry.remove(eid)
        deprovision(self.settings, eid)
        return web.json_response({"ok": True})

    async def wake(self, req: web.Request) -> web.Response:
        emp = self._require(req)
        if isinstance(emp, web.Response):
            return emp
        try:
            await self.supervisor.ensure(emp)
        except CapacityFull as e:
            return web.json_response({"error": str(e)}, status=503)
        return web.json_response(self._employee_view(emp))

    async def sleep(self, req: web.Request) -> web.Response:
        emp = self._require(req)
        if isinstance(emp, web.Response):
            return emp
        was = await self.supervisor.put_to_sleep(emp)
        return web.json_response({"slept": was, **self._employee_view(emp)})

    async def chat(self, req: web.Request) -> web.Response:
        emp = self._require(req)
        if isinstance(emp, web.Response):
            return emp
        body = await req.json()
        message = (body.get("message") or "").strip()
        if not message:
            return web.json_response({"error": "empty message"}, status=400)
        session = body.get("session") or "web"
        started = time.monotonic()
        try:
            reply = await self.supervisor.handle(emp, session, message)
        except CapacityFull as e:
            return web.json_response({"error": str(e)}, status=503)
        except Exception as e:
            return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=500)
        return web.json_response({"reply": reply, "elapsed": round(time.monotonic() - started, 1)})

    def _require(self, req: web.Request):
        emp = self.registry.get(req.match_info["id"])
        if not emp:
            return web.json_response({"error": "no such employee"}, status=404)
        return emp

    # --- secrets: Mattermost slash command -----------------------------------
    def _link_url(self, token: str) -> str:
        return f"{self.settings.secrets.form_base_url.rstrip('/')}/secret-entry?t={token}"

    def _ephemeral(self, text: str):
        return web.json_response({"response_type": "ephemeral", "text": text})

    async def mm_secret(self, req: web.Request) -> web.Response:
        """Mattermost slash command `/secret`. Args are NOT posted to the channel;
        we reply ephemerally. Secret VALUES are never accepted here — only a
        one-time entry link is returned."""
        data = await req.post()
        expected = self.settings.mattermost.slash_token
        if expected and data.get("token") != expected:
            return web.json_response({"error": "bad slash token"}, status=403)
        emp = self.registry.by_mm_user((data.get("user_id") or "").strip())
        if not emp:
            return self._ephemeral("⛔ You don't have a workspace yet — contact IT.")

        parts = (data.get("text") or "").strip().split()
        usage = ("Usage: `/secret set NAME` (opens a private link — never paste the value here), "
                 "`/secret list`, `/secret rm NAME`")
        if not parts or parts[0] == "help":
            return self._ephemeral(usage)
        cmd = parts[0].lower()

        if cmd == "list":
            rows = secret_status(self.settings, self.secrets, emp.id)
            if not rows:
                return self._ephemeral("No skill secrets are required by your skills yet.")
            lines = [f"- `{r['env']}` — {'✅ set' if r['set'] else '❌ missing'} ({r['label']})" for r in rows]
            return self._ephemeral("Your skill secrets:\n" + "\n".join(lines))

        if cmd == "rm" and len(parts) == 2:
            ok = self.secrets.delete(emp.id, parts[1])
            return self._ephemeral(f"{'🗑️ removed' if ok else 'not set'}: `{parts[1]}`")

        if cmd == "set" and len(parts) > 2:
            # They pasted a value inline — refuse and treat it as compromised.
            return self._ephemeral(
                f"⚠️ Don't paste secret values in chat — it stays in Mattermost history/logs. "
                f"Run `/secret set {parts[1]}` (name only); I'll DM a private link. "
                f"If `{parts[1]}` was a real token, rotate it now.")

        if cmd == "set" and len(parts) == 2:
            name = parts[1]
            token = self.links.mint(
                emp.id, f"secret:{name}", name,
                float(self.settings.secrets.link_ttl_seconds), time.time())
            ttl_min = self.settings.secrets.link_ttl_seconds // 60
            return self._ephemeral(
                f"🔐 Enter `{name}` here (one-time, expires in ~{ttl_min} min):\n{self._link_url(token)}")

        return self._ephemeral(usage)

    # --- secrets: one-time entry form (single secret OR a whole integration) --
    def _target_fields(self, target: str, tenant_id: str):
        """Resolve a link target to (title, [token field]). Employees only ever
        enter TOKENS here — shared/org config (URLs) lives in the catalog."""
        if target.startswith("integration:"):
            iid = target.split(":", 1)[1]
            it = integrations_mod.get(self.settings, iid)
            if not it:
                return None, []
            fields = [{"env": f["env"], "label": f["label"], "secret": True, "value": ""}
                      for f in integrations_mod.secret_fields(self.settings, iid)]
            return it.get("name", iid), fields
        if target.startswith("secret:"):
            env = target.split(":", 1)[1]
            return env, [{"env": env, "label": env, "secret": True, "value": ""}]
        return None, []

    async def secret_entry_form(self, req: web.Request) -> web.Response:
        row = self.links.peek(req.query.get("t", ""), time.time())
        if not row:
            return web.Response(text=_page("Link invalid or expired",
                "This entry link is invalid, already used, or expired. Request a new one."),
                content_type="text/html")
        title, fields = self._target_fields(row["target"], row["tenant_id"])
        if not fields:
            return web.Response(text=_page("Unknown integration",
                "This link points at something no longer configured."), content_type="text/html")
        return web.Response(
            text=_form_page(req.query.get("t", ""), title, fields, row["tenant_id"]),
            content_type="text/html")

    async def secret_entry_submit(self, req: web.Request) -> web.Response:
        data = await req.post()
        row = self.links.consume(data.get("t", ""), time.time())
        if not row:
            return web.Response(text=_page("Link invalid or expired",
                "This link is no longer valid. Request a new one."), content_type="text/html")
        _title, fields = self._target_fields(row["target"], row["tenant_id"])
        saved = []
        for f in fields:
            val = (data.get(f"value_{f['env']}") or "").strip()
            if val:
                self.secrets.set(row["tenant_id"], f["env"], val)  # value never logged
                saved.append(f["env"])
        if not saved:
            return web.Response(text=_page("Nothing saved",
                "No values were entered. Request a new link and try again."), content_type="text/html")
        return web.Response(text=_page("Saved ✓",
            "Saved securely for your workspace: " + ", ".join(f"<code>{s}</code>" for s in saved)
            + ". You can close this tab."), content_type="text/html")

    # --- secrets: admin API (names + status only, never values) --------------
    async def list_secrets(self, req: web.Request) -> web.Response:
        emp = self._require(req)
        if isinstance(emp, web.Response):
            return emp
        return web.json_response(secret_status(self.settings, self.secrets, emp.id))

    async def delete_secret(self, req: web.Request) -> web.Response:
        emp = self._require(req)
        if isinstance(emp, web.Response):
            return emp
        ok = self.secrets.delete(emp.id, req.match_info["name"])
        return web.json_response({"deleted": ok})

    # --- integrations (dashboard cards) --------------------------------------
    async def list_integrations(self, req: web.Request) -> web.Response:
        emp = self._require(req)
        if isinstance(emp, web.Response):
            return emp
        return web.json_response(integrations_mod.status(self.settings, self.secrets, emp.id))

    async def integration_link(self, req: web.Request) -> web.Response:
        emp = self._require(req)
        if isinstance(emp, web.Response):
            return emp
        iid = req.match_info["iid"]
        it = integrations_mod.get(self.settings, iid)
        if not it:
            return web.json_response({"error": "no such integration"}, status=404)
        token = self.links.mint(
            emp.id, f"integration:{iid}", it.get("name", iid),
            float(self.settings.secrets.link_ttl_seconds), time.time())
        return web.json_response({"url": self._link_url(token)})

    async def delete_integration(self, req: web.Request) -> web.Response:
        emp = self._require(req)
        if isinstance(emp, web.Response):
            return emp
        n = integrations_mod.delete(self.settings, self.secrets, emp.id, req.match_info["iid"])
        return web.json_response({"deleted_fields": n})


def _page(title: str, body: str) -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title>
<style>body{{font:15px/1.6 ui-monospace,Menlo,monospace;background:#0f1216;color:#e6e9ef;
display:flex;min-height:100vh;margin:0;align-items:center;justify-content:center}}
.card{{background:#171b21;border:1px solid #262c36;border-radius:12px;padding:28px;max-width:460px;width:90%}}
h1{{font-size:17px;margin:0 0 12px;color:#ffd23f}}code{{background:#0e1116;padding:1px 6px;border-radius:4px;color:#ffd23f}}
input{{width:100%;box-sizing:border-box;font:inherit;background:#0e1116;color:#e6e9ef;border:1px solid #262c36;
border-radius:8px;padding:10px;margin:10px 0}}button{{font:inherit;background:#ffd23f;color:#111;border:0;
border-radius:8px;padding:10px 16px;font-weight:600;cursor:pointer;margin-top:8px}}.muted{{color:#8a94a6;font-size:13px}}
label{{display:block;margin-top:12px;font-size:13px;color:#8a94a6}}</style>
</head><body><div class="card"><h1>{title}</h1>{body}</div></body></html>"""


def _form_page(token: str, title: str, fields: list, tenant_id: str) -> str:
    import html
    tt, tn, tok = html.escape(title or "secret"), html.escape(tenant_id), html.escape(token)
    inputs = []
    for f in fields:
        env = html.escape(f["env"])
        lb = html.escape(f.get("label", f["env"]))
        val = html.escape(f.get("value", "") or "")
        typ = "password" if f.get("secret", True) else "text"
        hint = "" if f.get("secret", True) else ' <span class="muted">(not secret)</span>'
        inputs.append(
            f'<label>{lb} <code>{env}</code>{hint}</label>'
            f'<input type="{typ}" name="value_{env}" value="{val}" '
            f'placeholder="{"paste token" if typ=="password" else "value"}" '
            f'autocomplete="off" spellcheck="false">'
        )
    body = (
        f'<p class="muted">Workspace: <code>{tn}</code></p>'
        f'<p>Configure <b>{tt}</b>. Values are stored only for your workspace and never shown in chat.</p>'
        f'<form method="post" action="/secret-entry" autocomplete="off">'
        f'<input type="hidden" name="t" value="{tok}">'
        + "".join(inputs)
        + '<button type="submit">Save securely</button></form>'
        f'<p class="muted">This link is one-time and expires shortly.</p>'
    )
    return _page(f"Configure {tt}", body)


def run(settings: Settings, host: str, port: int) -> None:
    api = OrchardAPI(settings)
    web.run_app(api.build_app(), host=host, port=port)
