"""Public customer commerce workspace surface for Sales Core.

The owner operates through the agent, but customers need a lightweight public URL
for quote/catalog/invoice review. These routes intentionally use opaque public
tokens rather than dashboard session auth and only expose the document tied to
that token.
"""
from __future__ import annotations

import html
import json
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from hermes_cli import agent_core_sql as sql
from tools import sales_tool

router = APIRouter()


def _user() -> str:
    return sql.runtime_env().get("SALES_DB_RUNTIME_USER", "sales_runtime")


def _q(value: Any) -> str:
    return sql.quote_literal(value)


def _j(value: Any) -> str:
    return sql.quote_jsonb(value)


def _money(value: Any, currency: str = "USD") -> str:
    try:
        amount = float(value or 0)
    except (TypeError, ValueError):
        amount = 0.0
    return f"{currency} {amount:,.2f}"


def _e(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def _get_workspace(public_token: str) -> dict[str, Any]:
    workspace = sql.one(
        f"""
        SELECT *, expires_at IS NOT NULL AND expires_at < now() AS is_expired
        FROM sales.customer_workspaces
        WHERE public_token={_q(public_token)}
        """,
        user=_user(),
    )
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    if workspace.get("is_expired") or workspace.get("status") == "expired":
        raise HTTPException(status_code=410, detail="Workspace expired")
    return workspace


def render_agent_workspace_placeholder_html(lang: str = "es") -> str:
    """Render the generic public landing page for inactive agent workspace URLs."""
    selected = "en" if str(lang).lower().startswith("en") else "es"
    es_active = " active" if selected == "es" else ""
    en_active = " active" if selected == "en" else ""
    return f"""<!doctype html>
<html lang="{_e(selected)}">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Espacio para agentes personalizados | Personalized agent space</title>
  <meta name="description" content="Un espacio seguro para revisar cotizaciones, catálogos, facturas, aprobaciones, firmas y pagos." />
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #f3f7f2;
      --ink: #111712;
      --muted: #526255;
      --line: rgba(17, 23, 18, .14);
      --panel: rgba(255, 255, 255, .72);
      --panel-strong: rgba(255, 255, 255, .92);
      --accent: #1f7a4d;
      --accent-ink: #ffffff;
      --field: #dceee2;
      --shadow: rgba(23, 53, 32, .16);
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #0d120f;
        --ink: #edf7ef;
        --muted: #a7b9ab;
        --line: rgba(237, 247, 239, .16);
        --panel: rgba(18, 28, 21, .70);
        --panel-strong: rgba(20, 31, 24, .94);
        --accent: #79d99f;
        --accent-ink: #07100a;
        --field: #17271c;
        --shadow: rgba(0, 0, 0, .42);
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; min-height: 100dvh; background: var(--bg); color: var(--ink); font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    body::before {{ content: ""; position: fixed; inset: 0; pointer-events: none; background: radial-gradient(circle at 10% 10%, rgba(31, 122, 77, .24), transparent 30%), radial-gradient(circle at 90% 20%, rgba(121, 217, 159, .18), transparent 28%), linear-gradient(135deg, transparent, rgba(31, 122, 77, .10)); }}
    a {{ color: inherit; }}
    .shell {{ position: relative; width: min(1180px, calc(100% - 32px)); margin: 0 auto; padding: 28px 0 34px; }}
    .nav {{ height: 68px; display: flex; align-items: center; justify-content: space-between; gap: 18px; }}
    .brand {{ display: flex; align-items: center; gap: 12px; font-weight: 800; letter-spacing: -.02em; }}
    .mark {{ width: 38px; height: 38px; border-radius: 14px; background: var(--ink); color: var(--bg); display: grid; place-items: center; font-weight: 900; box-shadow: 0 12px 30px var(--shadow); }}
    .lang {{ display: flex; gap: 8px; padding: 5px; border: 1px solid var(--line); border-radius: 999px; background: var(--panel); backdrop-filter: blur(16px); }}
    .lang a {{ text-decoration: none; padding: 8px 13px; border-radius: 999px; color: var(--muted); font-size: 14px; font-weight: 800; }}
    .lang a.active {{ background: var(--ink); color: var(--bg); }}
    .hero {{ min-height: calc(100dvh - 130px); display: grid; grid-template-columns: minmax(0, 1.02fr) minmax(320px, .98fr); gap: clamp(28px, 6vw, 70px); align-items: center; padding: 30px 0 54px; }}
    .content {{ max-width: 650px; }}
    .kicker {{ display: inline-flex; align-items: center; gap: 10px; color: var(--accent); font-size: 13px; font-weight: 900; letter-spacing: .12em; text-transform: uppercase; }}
    .kicker::before {{ content: ""; width: 42px; height: 1px; background: var(--accent); }}
    h1 {{ margin: 16px 0 18px; font-size: clamp(46px, 8vw, 92px); line-height: .92; letter-spacing: -.075em; max-width: 12ch; }}
    .lead {{ color: var(--muted); font-size: clamp(18px, 2.2vw, 23px); line-height: 1.45; max-width: 31em; margin: 0 0 28px; }}
    .actions {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 28px; }}
    .button {{ display: inline-flex; align-items: center; justify-content: center; min-height: 48px; padding: 0 20px; border-radius: 999px; border: 1px solid var(--line); text-decoration: none; font-weight: 900; transition: transform .2s ease, background .2s ease; }}
    .button:active {{ transform: translateY(1px) scale(.99); }}
    .button.primary {{ background: var(--accent); color: var(--accent-ink); border-color: transparent; }}
    .button.secondary {{ background: var(--panel); color: var(--ink); }}
    .micro {{ color: var(--muted); font-size: 14px; line-height: 1.55; max-width: 45em; }}
    .micro a {{ font-weight: 900; text-decoration-thickness: 2px; text-underline-offset: 3px; }}
    .visual {{ position: relative; min-height: 560px; border-radius: 34px; overflow: hidden; border: 1px solid var(--line); background: linear-gradient(145deg, var(--panel-strong), var(--field)); box-shadow: 0 36px 100px var(--shadow); isolation: isolate; }}
    .visual::before {{ content: ""; position: absolute; inset: -24%; background: conic-gradient(from 180deg, transparent, rgba(31, 122, 77, .24), transparent, rgba(121, 217, 159, .35), transparent); animation: turn 22s linear infinite; }}
    .visual::after {{ content: ""; position: absolute; inset: 1px; border-radius: 33px; background: radial-gradient(circle at 50% 22%, rgba(255,255,255,.42), transparent 30%), linear-gradient(180deg, transparent, var(--panel-strong)); }}
    .orbit {{ position: absolute; z-index: 2; inset: 48px; border: 1px solid var(--line); border-radius: 30px; display: grid; grid-template-rows: auto 1fr auto; padding: 28px; background: rgba(255,255,255,.10); backdrop-filter: blur(18px); }}
    .status {{ display: flex; align-items: center; justify-content: space-between; gap: 16px; color: var(--muted); font-weight: 800; }}
    .status span:last-child {{ color: var(--accent); }}
    .node-wrap {{ position: relative; display: grid; place-items: center; }}
    .node {{ width: 190px; height: 190px; border-radius: 42px; display: grid; place-items: center; text-align: center; padding: 24px; background: var(--ink); color: var(--bg); font-size: 18px; line-height: 1.15; font-weight: 900; box-shadow: 0 30px 80px var(--shadow); }}
    .ring {{ position: absolute; border: 1px solid var(--line); border-radius: 999px; }}
    .ring.one {{ width: 340px; height: 340px; }}
    .ring.two {{ width: 460px; height: 460px; opacity: .55; }}
    .capabilities {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
    .capabilities span {{ min-height: 52px; border: 1px solid var(--line); border-radius: 18px; display: flex; align-items: center; padding: 0 14px; background: var(--panel); color: var(--muted); font-size: 14px; font-weight: 800; }}
    .panel {{ display: none; }}
    .panel.active {{ display: block; }}
    @keyframes turn {{ to {{ transform: rotate(360deg); }} }}
    @media (prefers-reduced-motion: reduce) {{ .visual::before {{ animation: none; }} .button {{ transition: none; }} }}
    @media (max-width: 860px) {{
      .shell {{ width: min(100% - 24px, 680px); }}
      .nav {{ height: auto; padding-top: 4px; align-items: flex-start; }}
      .hero {{ grid-template-columns: 1fr; min-height: auto; padding-top: 28px; }}
      h1 {{ font-size: clamp(42px, 14vw, 66px); max-width: 11ch; }}
      .visual {{ min-height: 440px; border-radius: 28px; }}
      .orbit {{ inset: 22px; padding: 18px; }}
      .ring.one {{ width: 250px; height: 250px; }}
      .ring.two {{ width: 330px; height: 330px; }}
      .node {{ width: 158px; height: 158px; border-radius: 34px; font-size: 16px; }}
      .capabilities {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <nav class="nav" aria-label="{_e('Language selector' if selected == 'en' else 'Selector de idioma')}">
      <div class="brand"><div class="mark">A</div><span>Agent Space</span></div>
      <div class="lang"><a class="{'active' if selected == 'es' else ''}" href="?lang=es">ES</a><a class="{'active' if selected == 'en' else ''}" href="?lang=en">EN</a></div>
    </nav>
    <section class="hero">
      <div class="content">
        <div class="panel{es_active}">
          <div class="kicker">Enlace inactivo</div>
          <h1>Espacio para agentes personalizados</h1>
          <p class="lead">Este enlace todavía no tiene un documento activo. Aquí podrás revisar cotizaciones, catálogos, aprobaciones, firmas y pagos.</p>
          <div class="actions"><a class="button primary" href="https://ear.app">Más información en ear.app</a><a class="button secondary" href="https://sitiouno.us">sitiouno.us</a></div>
          <p class="micro">Los agentes personalizados trabajan por WhatsApp, correo electrónico, llamadas o canales web. Pueden gestionar ventas, agenda, inventario, facturas, seguimiento de clientes y automatizaciones conectadas a sistemas como Odoo. Desarrollado por <a href="https://sitiouno.us">sitiouno.us</a>.</p>
        </div>
        <div class="panel{en_active}">
          <div class="kicker">Inactive link</div>
          <h1>Personalized agent space</h1>
          <p class="lead">This link is not connected to an active document yet. You’ll be able to review quotes, catalogs, approvals, signatures and payments here.</p>
          <div class="actions"><a class="button primary" href="https://ear.app">Learn more at ear.app</a><a class="button secondary" href="https://sitiouno.us">sitiouno.us</a></div>
          <p class="micro">Personalized agents can support customers and teams on WhatsApp, email, voice calls or web channels. They help manage sales, scheduling, inventory, invoicing, customer follow-up and automations connected to systems like Odoo. Developed by <a href="https://sitiouno.us">sitiouno.us</a>.</p>
        </div>
      </div>
      <aside class="visual" aria-label="{_e('Agent space illustration' if selected == 'en' else 'Ilustración del espacio para agentes')}">
        <div class="orbit">
          <div class="status"><span>{_e('Space' if selected == 'en' else 'Espacio')}</span><span>{_e('Ready when needed' if selected == 'en' else 'Listo cuando lo necesites')}</span></div>
          <div class="node-wrap"><div class="ring two"></div><div class="ring one"></div><div class="node">{_e('Guided business tasks' if selected == 'en' else 'Gestiones guiadas para tu negocio')}</div></div>
          <div class="capabilities"><span>{_e('Quotes and approvals' if selected == 'en' else 'Cotizaciones y aprobaciones')}</span><span>{_e('Catalogs and payments' if selected == 'en' else 'Catálogos y pagos')}</span><span>{_e('Customer follow-up' if selected == 'en' else 'Seguimiento de clientes')}</span><span>{_e('Odoo-connected automations' if selected == 'en' else 'Automatizaciones conectadas a Odoo')}</span></div>
        </div>
      </aside>
    </section>
  </main>
</body>
</html>"""


def _workspace_events(workspace_id: str) -> list[dict[str, Any]]:
    return sql.rows(
        f"""
        SELECT event_type, actor_type, actor_ref, comment, metadata, occurred_at
        FROM sales.customer_workspace_events
        WHERE workspace_id={_q(workspace_id)}
        ORDER BY occurred_at ASC, workspace_event_id ASC
        """,
        user=_user(),
    )


def _record_event(
    workspace_id: str,
    event_type: str,
    *,
    actor_type: str = "customer",
    actor_ref: str | None = None,
    comment: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    return sql.statement_one(
        f"""
        INSERT INTO sales.customer_workspace_events (workspace_id, event_type, actor_type, actor_ref, comment, metadata, occurred_at)
        VALUES ({_q(workspace_id)}, {_q(event_type)}, {_q(actor_type)}, {_q(actor_ref)}, {_q(comment)}, {_j(metadata or {})}, now())
        RETURNING *
        """,
        user=_user(),
    )


def _customer_actor_ref(workspace: dict[str, Any]) -> str | None:
    return workspace.get("customer_email") or workspace.get("customer_name")


def _customer_signature(workspace: dict[str, Any], signature: str | None = None) -> str | None:
    return signature or workspace.get("customer_name") or workspace.get("customer_email")


def _notify_agent_workspace_interaction(
    workspace: dict[str, Any],
    event_type: str,
    *,
    comment: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record a pending owner follow-up and optionally email the owner/agent."""
    labels = {"commented": "comentó", "approved": "aprobó", "rejected": "rechazó", "signed": "firmó"}
    action = labels.get(event_type, event_type)
    customer = workspace.get("customer_name") or workspace.get("customer_email") or "Cliente"
    document_type = workspace.get("document_type") or "documento"
    document_id = workspace.get("document_id") or ""
    summary = f"{customer} {action} {document_type} {document_id}".strip()
    if comment:
        summary = f"{summary}: {comment}"
    follow_up = sql.statement_one(
        f"""
        INSERT INTO crm.follow_ups (organization_id, contact_id, opportunity_id, due_at, summary, status, priority, assignee, metadata, created_at, updated_at)
        VALUES (NULL, NULL, NULL, now(), {_q(summary)}, 'open', {_q('high' if event_type in {'approved', 'rejected'} else 'normal')}, {_q('agent')}, {_j({'workspace_id': workspace.get('workspace_id'), 'document_type': document_type, 'document_id': document_id, 'event_type': event_type, **(metadata or {})})}, now(), now())
        RETURNING *
        """,
        user=os.getenv("CRM_DB_RUNTIME_USER") or sql.runtime_env().get("CRM_DB_RUNTIME_USER", "crm_runtime"),
    )
    email = None
    owner_email = (
        sql.runtime_env().get("WORKSPACE_OWNER_EMAIL")
        or sql.runtime_env().get("AGENT_NOTIFICATION_EMAIL")
        or os.getenv("WORKSPACE_OWNER_EMAIL")
        or os.getenv("AGENT_NOTIFICATION_EMAIL")
    )
    if owner_email:
        from tools import notification_tool

        email = notification_tool._email_adapter_send({
            "to_email": owner_email,
            "to_name": "Agente",
            "subject": f"Interacción en workspace: {summary[:90]}",
            "text": f"{summary}\n\nWorkspace: {workspace.get('public_url') or workspace.get('workspace_id')}",
            "html": f"<p>{_e(summary)}</p><p><strong>Workspace:</strong> {_e(workspace.get('public_url') or workspace.get('workspace_id'))}</p>",
            "metadata": {"workspace_id": workspace.get("workspace_id"), "event_type": event_type},
        })
    return {"ok": True, "follow_up": follow_up, "email": email}


def _set_workspace_status(workspace_id: str, status: str) -> dict[str, Any] | None:
    return sql.statement_one(
        f"""
        UPDATE sales.customer_workspaces SET status='{status}', updated_at=now()
        WHERE workspace_id={_q(workspace_id)}
        RETURNING *
        """,
        user=_user(),
    )


def _document_for_workspace(workspace: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    document_type = workspace.get("document_type")
    document_id = workspace.get("document_id")
    if document_type == "quote":
        document = sql.one(f"SELECT * FROM sales.quotes WHERE quote_id={_q(document_id)}", user=_user())
        items = sql.rows(
            f"SELECT * FROM sales.quote_items WHERE quote_id={_q(document_id)} ORDER BY quote_item_id ASC",
            user=_user(),
        )
    elif document_type == "invoice":
        document = sql.one(f"SELECT * FROM sales.invoices WHERE invoice_id={_q(document_id)}", user=_user())
        items = sql.rows(
            f"SELECT * FROM sales.invoice_items WHERE invoice_id={_q(document_id)} ORDER BY invoice_item_id ASC",
            user=_user(),
        )
    elif document_type == "catalog":
        document = {"title": "Catálogo", "currency": "USD", "total": None, "status": workspace.get("status")}
        items = sql.rows(
            "SELECT product_id, sku, name AS description, 1 AS quantity, unit_price, unit_price AS line_total, currency FROM sales.products WHERE status='active' ORDER BY name ASC",
            user=_user(),
        )
    else:
        raise HTTPException(status_code=400, detail="Unsupported document type")
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document, items


def _mark_opened(workspace: dict[str, Any]) -> None:
    if workspace.get("status") == "pending":
        _set_workspace_status(workspace["workspace_id"], "viewed")
    _record_event(
        workspace["workspace_id"],
        "opened",
        actor_type="customer",
        metadata={"document_type": workspace.get("document_type"), "document_id": workspace.get("document_id")},
    )


def _items_html(items: list[dict[str, Any]], currency: str) -> str:
    rows = []
    for item in items:
        description = item.get("description") or item.get("name") or item.get("product_id") or "Item"
        rows.append(
            "<tr>"
            f"<td>{_e(description)}</td>"
            f"<td>{_e(item.get('quantity') or 1)}</td>"
            f"<td>{_money(item.get('unit_price'), item.get('currency') or currency)}</td>"
            f"<td>{_money(item.get('line_total') or item.get('unit_price'), item.get('currency') or currency)}</td>"
            "</tr>"
        )
    return "".join(rows) or "<tr><td colspan='4'>Sin ítems registrados.</td></tr>"


def _events_html(events: list[dict[str, Any]]) -> str:
    if not events:
        return "<p class='muted'>Todavía no hay comentarios.</p>"
    parts = []
    for event in events:
        comment = event.get("comment")
        if not comment and event.get("event_type") not in {"commented", "approved", "rejected", "signed"}:
            continue
        parts.append(
            "<div class='event'>"
            f"<strong>{_e(event.get('event_type'))}</strong>"
            f"<p>{_e(comment or '')}</p>"
            "</div>"
        )
    return "".join(parts) or "<p class='muted'>Todavía no hay comentarios.</p>"


def render_workspace_html(public_token: str, banner: str | None = None) -> str:
    workspace = _get_workspace(public_token)
    document, items = _document_for_workspace(workspace)
    events = _workspace_events(workspace["workspace_id"])
    _mark_opened(workspace)

    document_type = str(workspace.get("document_type") or "")
    document_type_label = {"quote": "Cotización", "catalog": "Catálogo", "invoice": "Factura"}.get(
        document_type, "Documento"
    )
    title = document.get("title") or workspace.get("document_id")
    currency = document.get("currency") or "USD"
    total = document.get("total")
    customer_identity = workspace.get("customer_email") or workspace.get("customer_name") or "este enlace seguro"
    action_buttons = ""
    if workspace.get("status") not in {"approved", "rejected", "paid", "cancelled"}:
        action_buttons = f"""
          <div class="decision-grid" aria-label="Opciones de respuesta">
            <form method="post" action="/w/{_e(public_token)}/approve" class="decision-card approve-card">
              <p class="decision-label">Aceptar propuesta</p>
              <p class="decision-copy">Confirmo esta cotización y autorizo al agente a continuar con la orden y la factura.</p>
              <button class="button approve" type="submit">Aprobar cotización</button>
            </form>
            <form method="post" action="/w/{_e(public_token)}/reject" class="decision-card reject-card">
              <label for="reject-comment">Motivo del rechazo</label>
              <textarea id="reject-comment" name="comment" placeholder="Cuéntanos qué debemos ajustar"></textarea>
              <button class="button reject" type="submit">Rechazar</button>
            </form>
          </div>
        """

    return f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{_e(document_type_label)} - {_e(title)}</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #f4f7f3;
      --surface: rgba(255,255,255,.86);
      --surface-strong: #ffffff;
      --ink: #101510;
      --muted: #5c675f;
      --line: rgba(16,21,16,.13);
      --soft: #e8efe8;
      --green: #15803d;
      --green-ink: #ffffff;
      --red: #b42318;
      --red-ink: #ffffff;
      --shadow: rgba(21, 64, 37, .14);
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #0c110d;
        --surface: rgba(18, 27, 20, .86);
        --surface-strong: #121b14;
        --ink: #f0f7f0;
        --muted: #a8b5aa;
        --line: rgba(240,247,240,.15);
        --soft: #17231a;
        --green: #72d991;
        --green-ink: #061108;
        --red: #ff8b82;
        --red-ink: #210403;
        --shadow: rgba(0,0,0,.42);
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; min-height: 100dvh; background: var(--bg); color: var(--ink); font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    body::before {{ content: ""; position: fixed; inset: 0; pointer-events: none; background: radial-gradient(circle at 12% 0%, rgba(21,128,61,.20), transparent 34%), radial-gradient(circle at 92% 12%, rgba(114,217,145,.14), transparent 30%); }}
    main {{ position: relative; width: min(1120px, calc(100% - 28px)); margin: 0 auto; padding: 34px 0 44px; }}
    .card {{ background: var(--surface); border: 1px solid var(--line); border-radius: 32px; box-shadow: 0 30px 90px var(--shadow); overflow: hidden; backdrop-filter: blur(16px); }}
    .quote-header {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(220px, .34fr); gap: 24px; padding: clamp(26px, 5vw, 52px); border-bottom: 1px solid var(--line); background: linear-gradient(135deg, var(--surface-strong), transparent); }}
    .eyebrow {{ color: var(--green); font-weight: 900; letter-spacing: .12em; text-transform: uppercase; font-size: 12px; }}
    h1 {{ font-size: clamp(34px, 6vw, 68px); line-height: .95; margin: 12px 0 16px; letter-spacing: -.055em; max-width: 13ch; }}
    .muted {{ color: var(--muted); line-height: 1.55; }}
    .summary {{ display: grid; gap: 12px; align-content: start; }}
    .pill {{ border: 1px solid var(--line); border-radius: 20px; padding: 16px; background: var(--soft); }}
    .pill strong {{ display: block; font-size: 12px; color: var(--muted); margin-bottom: 8px; text-transform: uppercase; letter-spacing: .08em; }}
    .body-grid {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(280px, .45fr); gap: 24px; padding: clamp(22px, 4vw, 42px); }}
    .section-title {{ margin: 0 0 14px; font-size: 20px; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 22px; background: var(--surface-strong); }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 16px 14px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }}
    tr:last-child td {{ border-bottom: 0; }}
    .actions, .comment-box {{ display: grid; gap: 14px; }}
    .decision-grid {{ display: grid; gap: 14px; }}
    .decision-card {{ border: 1px solid var(--line); border-radius: 24px; padding: 18px; background: var(--surface-strong); display: grid; gap: 12px; }}
    .approve-card {{ border-color: color-mix(in srgb, var(--green), transparent 58%); }}
    .reject-card {{ border-color: color-mix(in srgb, var(--red), transparent 62%); }}
    .decision-label {{ margin: 0; font-weight: 900; }}
    .decision-copy {{ margin: 0; color: var(--muted); line-height: 1.45; }}
    label {{ font-weight: 800; }}
    textarea {{ width: 100%; min-height: 96px; border: 1px solid var(--line); border-radius: 18px; padding: 13px 14px; font: inherit; color: var(--ink); background: var(--surface); resize: vertical; }}
    textarea::placeholder {{ color: color-mix(in srgb, var(--muted), transparent 10%); }}
    .button {{ border: 0; border-radius: 999px; min-height: 46px; padding: 0 18px; font-weight: 900; cursor: pointer; transition: transform .18s ease, filter .18s ease; }}
    .button:active {{ transform: translateY(1px) scale(.99); }}
    .approve {{ background: var(--green); color: var(--green-ink); }}
    .reject {{ background: var(--red); color: var(--red-ink); }}
    .secondary {{ background: var(--ink); color: var(--bg); width: fit-content; }}
    .banner {{ background: color-mix(in srgb, var(--green), transparent 84%); border: 1px solid color-mix(in srgb, var(--green), transparent 52%); border-radius: 18px; padding: 14px 16px; margin-bottom: 18px; }}
    .event {{ border-left: 3px solid var(--green); padding-left: 12px; margin: 12px 0; color: var(--muted); }}
    .identity {{ display: inline-flex; width: fit-content; margin-top: 12px; padding: 8px 12px; border: 1px solid var(--line); border-radius: 999px; background: var(--soft); color: var(--muted); font-size: 14px; }}
    @media (max-width: 840px) {{
      main {{ width: min(100% - 20px, 680px); padding-top: 18px; }}
      .quote-header, .body-grid {{ grid-template-columns: 1fr; }}
      h1 {{ max-width: 12ch; }}
    }}
  </style>
</head>
<body>
  <main>
    {f'<div class="banner">{_e(banner)}</div>' if banner else ''}
    <section class="card">
      <div class="quote-header">
        <div>
          <div class="eyebrow">{_e(document_type_label)} para {_e(workspace.get('customer_name') or 'cliente')}</div>
          <h1>{_e(title)}</h1>
          <p class="muted">Revisa los detalles, responde con una decisión y el agente se encargará del seguimiento.</p>
          <span class="identity">Acceso enviado a {_e(customer_identity)}</span>
        </div>
        <div class="summary">
          <div class="pill"><strong>Estado</strong>{_e(workspace.get('status'))}</div>
          <div class="pill"><strong>Documento</strong>{_e(workspace.get('document_id'))}</div>
          <div class="pill"><strong>Total</strong>{_money(total, currency) if total is not None else 'Según selección'}</div>
        </div>
      </div>
      <div class="body-grid">
        <div>
          <h2 class="section-title">Detalle</h2>
          <div class="table-wrap">
            <table>
              <thead><tr><th>Concepto</th><th>Cant.</th><th>Precio</th><th>Total</th></tr></thead>
              <tbody>{_items_html(items, currency)}</tbody>
            </table>
          </div>
        </div>
        <aside class="actions">
          {action_buttons}
          <div class="comment-box">
            <h2 class="section-title">Comentarios</h2>
            {_events_html(events)}
            <form method="post" action="/w/{_e(public_token)}/comment" class="decision-card">
              <label for="comment">Mensaje para el agente</label>
              <textarea id="comment" name="comment" placeholder="Escribe una pregunta o ajuste solicitado"></textarea>
              <button class="button secondary" type="submit">Enviar comentario</button>
            </form>
          </div>
        </aside>
      </div>
    </section>
  </main>
</body>
</html>"""


def comment_workspace(public_token: str, comment: str, actor_ref: str | None = None) -> dict[str, Any]:
    workspace = _get_workspace(public_token)
    clean_comment = (comment or "").strip()
    if not clean_comment:
        raise HTTPException(status_code=400, detail="Comment is required")
    actor = _customer_actor_ref(workspace) or actor_ref
    _set_workspace_status(workspace["workspace_id"], "commented")
    event = _record_event(workspace["workspace_id"], "commented", actor_ref=actor, comment=clean_comment)
    notification = _notify_agent_workspace_interaction(workspace, "commented", comment=clean_comment)
    return {"ok": True, "event": event, "notification": notification}


def reject_workspace(public_token: str, comment: str | None = None, actor_ref: str | None = None) -> dict[str, Any]:
    workspace = _get_workspace(public_token)
    actor = _customer_actor_ref(workspace) or actor_ref
    _set_workspace_status(workspace["workspace_id"], "rejected")
    event = _record_event(workspace["workspace_id"], "rejected", actor_ref=actor, comment=comment)
    notification = _notify_agent_workspace_interaction(workspace, "rejected", comment=comment)
    return {"ok": True, "event": event, "notification": notification}


def approve_workspace(public_token: str, actor_ref: str | None = None, signature: str | None = None) -> dict[str, Any]:
    workspace = _get_workspace(public_token)
    actor = _customer_actor_ref(workspace) or actor_ref
    signature_value = _customer_signature(workspace, signature)
    metadata: dict[str, Any] = {"signature": signature_value} if signature_value else {}
    result: dict[str, Any] = {"ok": True}

    if workspace.get("document_type") == "quote":
        order_payload = json.loads(sales_tool._handle_order_create({
            "quote_id": workspace["document_id"],
            "metadata": {"source": "customer_workspace", "workspace_id": workspace["workspace_id"]},
        }))
        if not order_payload.get("ok"):
            raise HTTPException(status_code=500, detail=order_payload.get("error") or "Order conversion failed")
        order_id = order_payload.get("order", {}).get("order_id")
        invoice_payload = json.loads(sales_tool._handle_invoice_create({
            "order_id": order_id,
            "metadata": {"source": "customer_workspace", "workspace_id": workspace["workspace_id"]},
        }))
        if not invoice_payload.get("ok"):
            raise HTTPException(status_code=500, detail=invoice_payload.get("error") or "Invoice conversion failed")
        invoice_id = invoice_payload.get("invoice", {}).get("invoice_id")
        metadata.update({"order_id": order_id, "invoice_id": invoice_id})
        result.update({"order_id": order_id, "invoice_id": invoice_id})
    elif workspace.get("document_type") == "invoice":
        metadata.update({"invoice_id": workspace["document_id"]})
    else:
        metadata.update({"document_id": workspace["document_id"]})

    _set_workspace_status(workspace["workspace_id"], "approved")
    event = _record_event(
        workspace["workspace_id"],
        "approved",
        actor_ref=actor,
        comment=signature_value,
        metadata=metadata,
    )
    notification = _notify_agent_workspace_interaction(workspace, "approved", metadata=metadata)
    result.update({"event": event, "notification": notification})
    return result


async def _form_text(request: Request, field: str) -> str | None:
    form = await request.form()
    value = form.get(field)
    return str(value).strip() if value is not None and str(value).strip() else None


def _request_lang(request: Request) -> str:
    return request.query_params.get("lang", "es")


@router.get("/w", response_class=HTMLResponse)
@router.get("/w/", response_class=HTMLResponse)
async def workspace_placeholder(request: Request) -> HTMLResponse:
    return HTMLResponse(render_agent_workspace_placeholder_html(_request_lang(request)))


@router.get("/w/{public_token}", response_class=HTMLResponse)
async def workspace_page(public_token: str, request: Request) -> HTMLResponse:
    try:
        return HTMLResponse(render_workspace_html(public_token))
    except HTTPException as exc:
        if exc.status_code in {404, 410}:
            return HTMLResponse(render_agent_workspace_placeholder_html(_request_lang(request)))
        raise


@router.post("/w/{public_token}/comment", response_class=HTMLResponse)
async def workspace_comment(public_token: str, request: Request) -> HTMLResponse:
    comment = await _form_text(request, "comment")
    actor_ref = await _form_text(request, "actor_ref")
    comment_workspace(public_token, comment or "", actor_ref=actor_ref)
    return HTMLResponse(render_workspace_html(public_token, banner="Comentario enviado. El agente recibió tu mensaje."))


@router.post("/w/{public_token}/approve", response_class=HTMLResponse)
async def workspace_approve(public_token: str, request: Request) -> HTMLResponse:
    actor_ref = await _form_text(request, "actor_ref")
    signature = await _form_text(request, "signature")
    approve_workspace(public_token, actor_ref=actor_ref, signature=signature)
    return HTMLResponse(render_workspace_html(public_token, banner="Aprobado. El agente continuará con la orden, factura y pago."))


@router.post("/w/{public_token}/reject", response_class=HTMLResponse)
async def workspace_reject(public_token: str, request: Request) -> HTMLResponse:
    comment = await _form_text(request, "comment")
    actor_ref = await _form_text(request, "actor_ref")
    reject_workspace(public_token, comment=comment, actor_ref=actor_ref)
    return HTMLResponse(render_workspace_html(public_token, banner="Rechazado. El agente recibió la respuesta."))
