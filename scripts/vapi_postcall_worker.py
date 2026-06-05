#!/usr/bin/env python3
"""Trusted Vapi post-call supervisor for SitioUno voice leads.

The public delivery sandbox only records Vapi webhooks/tool calls as append-only
JSONL audit events. This trusted worker consumes those events after the call
ends, groups tool calls by call_id, and performs the business work Sophie
committed to: CRM upsert, demo document generation, email delivery, and
interaction logging.

It is intentionally safe to run every minute: processed call ids are persisted
in a state file and are skipped on subsequent runs.
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_EVENT_DIR = Path(os.environ.get("EVENT_DIR", "/home/jean/zeus-runtime/delivery-sandbox/events"))
DEFAULT_EVENTS_PATH = Path(os.environ.get("VAPI_EVENTS_PATH", str(DEFAULT_EVENT_DIR / "events.jsonl")))
DEFAULT_STATE_PATH = Path(os.environ.get("VAPI_POSTCALL_STATE_PATH", str(DEFAULT_EVENT_DIR / "vapi_postcall_worker_state.json")))
DEFAULT_OUTPUT_DIR = Path(os.environ.get("VAPI_POSTCALL_OUTPUT_DIR", "/home/jean/generated/sitiouno_voice_tests"))
SENDGRID_API_BASE = "https://api.sendgrid.com"
FINAL_EVENT_TYPES = {"vapi_call_ended", "vapi_end_of_call_report"}


@dataclass
class PostCallJob:
    call_id: str
    lead: dict[str, Any]
    summary: dict[str, Any]
    final_event: dict[str, Any]
    deliverable_kind: str
    should_send_email: bool
    to_email: str | None = None


def _load_repo_runtime_env() -> dict[str, str]:
    try:
        from hermes_cli import agent_core_sql as sql  # type: ignore
        env = sql.runtime_env()
        return {str(k): str(v) for k, v in (env or {}).items() if v is not None}
    except Exception:
        return {}


def merged_env(explicit: dict[str, str] | None = None) -> dict[str, str]:
    return {**os.environ, **_load_repo_runtime_env(), **(explicit or {})}


def read_events(path: Path = DEFAULT_EVENTS_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            events.append(item)
    return events


def load_state(path: Path = DEFAULT_STATE_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"processed_call_ids": [], "processed_event_ids": [], "runs": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"processed_call_ids": [], "processed_event_ids": [], "runs": []}
    if not isinstance(data, dict):
        return {"processed_call_ids": [], "processed_event_ids": [], "runs": []}
    data.setdefault("processed_call_ids", [])
    data.setdefault("processed_event_ids", [])
    data.setdefault("runs", [])
    return data


def save_state(state: dict[str, Any], path: Path = DEFAULT_STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _event_metadata(event: dict[str, Any]) -> dict[str, Any]:
    raw = event.get("metadata")
    return raw if isinstance(raw, dict) else {}


def _event_call_id(event: dict[str, Any]) -> str | None:
    metadata = _event_metadata(event)
    raw_message = metadata.get("message")
    message: dict[str, Any] = raw_message if isinstance(raw_message, dict) else {}
    raw_call = message.get("call")
    call: dict[str, Any] = raw_call if isinstance(raw_call, dict) else {}
    values = [event.get("actor_ref"), metadata.get("call_id"), call.get("id")]
    for value in values:
        if value:
            return str(value)
    return None


def _tool_name(event: dict[str, Any]) -> str | None:
    name = _event_metadata(event).get("tool_name")
    return str(name) if name else None


def _tool_args(event: dict[str, Any]) -> dict[str, Any]:
    args = _event_metadata(event).get("arguments")
    return args if isinstance(args, dict) else {}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "si", "sí", "required", "requerido"}


def normalize_email(value: Any) -> str | None:
    raw = str(value or "").strip().lower()
    if not raw:
        return None
    raw = raw.replace(" arroba ", "@").replace(" at ", "@")
    raw = raw.replace(" punto ", ".").replace(" dot ", ".")
    raw = raw.replace(" ", "")
    match = re.search(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", raw)
    return match.group(0) if match else None


def normalize_phone(value: Any) -> str:
    digits = re.sub(r"\D+", "", str(value or ""))
    if len(digits) == 10:
        return "+1" + digits
    if len(digits) == 11 and digits.startswith("1"):
        return "+" + digits
    return str(value or "").strip()


def infer_deliverable_kind(lead: dict[str, Any], summary: dict[str, Any]) -> str:
    haystack = " ".join(str(v or "") for v in [
        lead.get("need"), lead.get("notes"), summary.get("summary"), summary.get("next_steps"), summary.get("outcome"),
    ]).lower()
    admin_accounting_terms = [
        "factura", "facturas", "invoice", "invoices", "cuenta de cobro", "cuentas de cobro",
        "contable", "contabilidad", "accounting", "bookkeeping", "tributario", "tributaria", "tax",
        "calendario tributario", "agenda", "recordatorio", "recordatorios", "administrativo", "administrativa",
    ]
    if any(word in haystack for word in admin_accounting_terms):
        return "admin_accounting_demo"
    if any(word in haystack for word in ["taller", "mecánico", "mecanico", "auto", "vehículo", "vehiculo"]):
        if any(word in haystack for word in ["cotización", "cotizacion", "quote"]):
            return "mechanic_quote_demo"
    if any(word in haystack for word in ["brochure", "folleto", "pdf", "presentación", "presentacion"]):
        return "agent_capabilities_pdf"
    if any(word in haystack for word in ["cotización", "cotizacion", "quote"]):
        return "generic_quote_demo"
    return "lead_followup_summary"


def _is_actionable_summary(summary: dict[str, Any]) -> bool:
    next_steps = str(summary.get("next_steps") or summary.get("summary") or "").lower()
    return _truthy(summary.get("follow_up_required")) or any(
        word in next_steps for word in ["enviar", "prepar", "correo", "whatsapp", "cotización", "cotizacion", "brochure", "pdf", "demo"]
    )


def build_pending_jobs(events: list[dict[str, Any]], processed_call_ids: set[str]) -> list[PostCallJob]:
    grouped: dict[str, dict[str, Any]] = {}
    for event in events:
        call_id = _event_call_id(event)
        if not call_id or call_id in processed_call_ids:
            continue
        bucket = grouped.setdefault(call_id, {"lead": None, "summary": None, "final": None})
        event_type = str(event.get("event_type") or "")
        if event_type == "vapi_tool_call":
            name = _tool_name(event)
            if name == "capture_voice_lead":
                bucket["lead"] = _tool_args(event)
            elif name == "send_call_summary":
                bucket["summary"] = _tool_args(event)
        elif event_type in FINAL_EVENT_TYPES:
            bucket["final"] = event

    jobs: list[PostCallJob] = []
    for call_id, bucket in grouped.items():
        lead = bucket.get("lead") or {}
        summary = bucket.get("summary") or {}
        final = bucket.get("final")
        if not final or not summary or not _is_actionable_summary(summary):
            continue
        to_email = normalize_email(lead.get("email") or summary.get("email"))
        deliverable_kind = infer_deliverable_kind(lead, summary)
        should_send_email = bool(to_email and any(k in deliverable_kind for k in ["demo", "pdf", "quote"]))
        jobs.append(PostCallJob(
            call_id=call_id,
            lead=lead,
            summary=summary,
            final_event=final,
            deliverable_kind=deliverable_kind,
            should_send_email=should_send_email,
            to_email=to_email,
        ))
    return jobs


def _slug(value: str, fallback: str = "lead") -> str:
    text = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return text or fallback


def generate_mechanic_quote_demo_pdf(job: PostCallJob, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import HRFlowable, Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    output_dir.mkdir(parents=True, exist_ok=True)
    name = str(job.lead.get("name") or job.summary.get("caller_name") or "cliente").strip() or "cliente"
    pdf_path = output_dir / f"cotizacion_demo_taller_mecanico_{_slug(name)}_{_slug(job.call_id)[:16]}.pdf"
    logo = Path("/home/jean/.hermes/reference-assets/sitiouno/brand/derived/sitiouno-logo-blue-on-white-1600x320.png")
    styles = getSampleStyleSheet()
    blue = colors.HexColor("#0B4FA8")
    dark = colors.HexColor("#172033")
    muted = colors.HexColor("#5C6470")
    light = colors.HexColor("#F4F7FB")
    styles.add(ParagraphStyle(name="TitleSU", parent=styles["Title"], textColor=blue, fontSize=22, leading=26, spaceAfter=8))
    styles.add(ParagraphStyle(name="H2SU", parent=styles["Heading2"], textColor=dark, fontSize=13, leading=16, spaceBefore=12, spaceAfter=6))
    styles.add(ParagraphStyle(name="BodySU", parent=styles["BodyText"], textColor=dark, fontSize=9.5, leading=13))
    styles.add(ParagraphStyle(name="SmallSU", parent=styles["BodyText"], textColor=muted, fontSize=8.5, leading=11))
    styles.add(ParagraphStyle(name="White", parent=styles["BodyText"], textColor=colors.white, fontSize=10, leading=13))

    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, rightMargin=0.55 * inch, leftMargin=0.55 * inch, topMargin=0.45 * inch, bottomMargin=0.45 * inch)
    story: list[Any] = []
    logo_cell: Any = Image(str(logo), width=2.1 * inch, height=0.42 * inch) if logo.exists() else Paragraph("SitioUno", styles["TitleSU"])
    header = Table([[logo_cell, Paragraph("<b>Demostración comercial</b><br/>Agente de cotizaciones para taller mecánico", styles["SmallSU"])]], colWidths=[3.1 * inch, 3.3 * inch])
    header.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (1, 0), (1, 0), "RIGHT"), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story.append(header)
    story.append(HRFlowable(width="100%", color=blue, thickness=1.2))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Cotización de ejemplo personalizada", styles["TitleSU"]))
    story.append(Paragraph(f"Cliente de demostración: <b>{name}</b> &nbsp;&nbsp; | &nbsp;&nbsp; Canal: llamada + WhatsApp/correo", styles["SmallSU"]))
    story.append(Spacer(1, 8))
    intro = Table([[Paragraph("<b>Escenario simulado:</b><br/>Un cliente de un taller mecánico envía por WhatsApp fotos/audio indicando rechinido al frenar en un Jetta 2018. El agente de SitioUno atiende, organiza la información, prepara una cotización preliminar y deja seguimiento listo para cierre.", styles["White"])]], colWidths=[6.4 * inch])
    intro.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), blue), ("LEFTPADDING", (0, 0), (-1, -1), 14), ("RIGHTPADDING", (0, 0), (-1, -1), 14), ("TOPPADDING", (0, 0), (-1, -1), 12), ("BOTTOMPADDING", (0, 0), (-1, -1), 12)]))
    story.append(intro)
    story.append(Spacer(1, 10))
    story.append(Paragraph("1. Información capturada por el agente", styles["H2SU"]))
    info = [["Dato", "Valor"], ["Vehículo", "Volkswagen Jetta 2018"], ["Síntoma reportado", "Rechinido al frenar, audio/foto recibidos por WhatsApp"], ["Servicio sugerido", "Diagnóstico de sistema de frenos + cambio de pastillas delanteras"], ["Tiempo estimado", "2 a 3 horas"], ["Estado", "Cotización preliminar lista para confirmar con el cliente"]]
    table = Table(info, colWidths=[1.8 * inch, 4.6 * inch])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), dark), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#DDE3EC")), ("BACKGROUND", (0, 1), (-1, -1), light), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    story.append(table)
    story.append(Paragraph("2. Cotización preliminar generada", styles["H2SU"]))
    items = [["Concepto", "Cantidad", "Precio unitario", "Total"], ["Diagnóstico de frenos", "1", "$25.00", "$25.00"], ["Pastillas delanteras estándar", "1 juego", "$95.00", "$95.00"], ["Mano de obra instalación", "1", "$65.00", "$65.00"], ["Rectificación/limpieza preventiva", "1", "$35.00", "$35.00"], ["Seguimiento post-servicio por WhatsApp", "Incluido", "$0.00", "$0.00"]]
    quote = Table(items, colWidths=[2.9 * inch, 1.0 * inch, 1.25 * inch, 1.25 * inch])
    quote.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), blue), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#DDE3EC")), ("ALIGN", (1, 1), (-1, -1), "RIGHT"), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light]), ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    story.append(quote)
    story.append(Spacer(1, 6))
    total = Table([["Subtotal", "$220.00"], ["Impuestos / ajustes", "Según política del taller"], ["Total estimado", "$220.00"]], colWidths=[4.8 * inch, 1.6 * inch])
    total.setStyle(TableStyle([("ALIGN", (1, 0), (1, -1), "RIGHT"), ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"), ("TEXTCOLOR", (0, -1), (-1, -1), blue), ("LINEABOVE", (0, -1), (-1, -1), 1, blue)]))
    story.append(total)
    story.append(Paragraph("3. Mensaje listo para enviar al cliente final", styles["H2SU"]))
    story.append(Paragraph("Hola, gracias por enviarnos la información de tu Jetta 2018. Según el síntoma reportado, te recomendamos revisar el sistema de frenos y realizar cambio de pastillas delanteras si el diagnóstico lo confirma. El estimado preliminar es de <b>$220.00</b> y el servicio toma aproximadamente <b>2 a 3 horas</b>. Podemos reservarte un espacio hoy o mañana. ¿Qué horario te conviene?", styles["BodySU"]))
    story.append(Paragraph("4. Capacidades demostradas del agente SitioUno", styles["H2SU"]))
    caps = [["Atención omnicanal", "Recibe llamadas, WhatsApp, fotos, audios y texto sin perder contexto."], ["CRM automático", "Registra lead, vehículo, necesidad, cotización, estado y próximos pasos."], ["Cotizaciones/documentos", "Genera propuestas, facturas, recibos o PDFs listos para enviar."], ["Seguimiento comercial", "Si el cliente no responde, agenda recordatorios y recupera oportunidades."], ["Escalamiento inteligente", "Si falta una validación técnica o precio especial, lo deja listo para aprobación."]]
    caps_table = Table(caps, colWidths=[1.55 * inch, 4.85 * inch])
    caps_table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#DDE3EC")), ("BACKGROUND", (0, 0), (0, -1), light), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    story.append(caps_table)
    story.append(Spacer(1, 10))
    story.append(Paragraph("Nota: Este documento es una demostración de capacidades preparada por SitioUno. Los precios son referenciales y se ajustan a la lista real, reglas y aprobaciones del negocio.", styles["SmallSU"]))
    doc.build(story)
    return pdf_path


def generate_generic_demo_pdf(job: PostCallJob, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    # Reuse the mechanic demo until more vertical-specific templates are added.
    return generate_mechanic_quote_demo_pdf(job, output_dir)


def generate_admin_accounting_demo_pdf(job: PostCallJob, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import HRFlowable, Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    output_dir.mkdir(parents=True, exist_ok=True)
    name = str(job.lead.get("name") or job.summary.get("caller_name") or "cliente").strip() or "cliente"
    company = str(job.lead.get("company") or job.lead.get("organization") or job.summary.get("company") or "la empresa").strip() or "la empresa"
    email = normalize_email(job.lead.get("email") or job.summary.get("email")) or "correo confirmado"
    pdf_path = output_dir / f"demo_formatos_admin_contable_{_slug(company)}_{_slug(name)}_{_slug(job.call_id)[:16]}.pdf"
    logo = Path("/home/jean/.hermes/reference-assets/sitiouno/brand/derived/sitiouno-logo-blue-on-white-1600x320.png")
    styles = getSampleStyleSheet()
    blue = colors.HexColor("#0B4FA8")
    dark = colors.HexColor("#172033")
    muted = colors.HexColor("#5C6470")
    light = colors.HexColor("#F4F7FB")
    styles.add(ParagraphStyle(name="TitleSU", parent=styles["Title"], textColor=blue, fontSize=20, leading=24, spaceAfter=8))
    styles.add(ParagraphStyle(name="H2SU", parent=styles["Heading2"], textColor=dark, fontSize=13, leading=16, spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="BodySU", parent=styles["BodyText"], textColor=dark, fontSize=9.3, leading=12.5))
    styles.add(ParagraphStyle(name="SmallSU", parent=styles["BodyText"], textColor=muted, fontSize=8.4, leading=11))
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, rightMargin=0.55 * inch, leftMargin=0.55 * inch, topMargin=0.45 * inch, bottomMargin=0.45 * inch)
    story: list[Any] = []
    logo_cell: Any = Image(str(logo), width=2.1 * inch, height=0.42 * inch) if logo.exists() else Paragraph("SitioUno", styles["TitleSU"])
    header = Table([[logo_cell, Paragraph("<b>Demostración comercial</b><br/>Agente administrativo-contable", styles["SmallSU"])]], colWidths=[3.1 * inch, 3.3 * inch])
    header.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (1, 0), (1, 0), "RIGHT"), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story.extend([header, HRFlowable(width="100%", color=blue, thickness=1.2), Spacer(1, 10)])
    story.append(Paragraph(f"Demostración para {company}", styles["TitleSU"]))
    story.append(Paragraph(f"Formatos administrativos y contables con agente SitioUno", styles["H2SU"]))
    story.append(Paragraph(f"Contacto: {name} · Empresa: {company} · Correo: {email}", styles["SmallSU"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Objetivo de la demo:", styles["H2SU"]))
    story.append(Paragraph("Mostrar cómo Sophie/Zeus puede construir formatos de cotización, factura, cuenta de cobro, agenda de tareas, recordatorios y calendario tributario por cliente, listos para adaptarse a las reglas reales del negocio.", styles["BodySU"]))
    sections = [
        ("1. Formato base de cotización", [["Campo", "Ejemplo inicial"], ["Cliente", "Cliente ABC LLC"], ["Servicio", "Asesoría contable mensual + preparación de reportes"], ["Alcance", "Revisión de documentos, clasificación, conciliación y reporte mensual"], ["Condiciones", "50% al iniciar / saldo contra entrega"], ["Validez", "15 días"]]),
        ("Detalle económico de ejemplo", [["Concepto", "Cantidad", "Precio", "Total"], ["Setup administrativo inicial", "1", "$250.00", "$250.00"], ["Gestión contable mensual", "1", "$450.00", "$450.00"], ["Calendario tributario por cliente", "1", "$120.00", "$120.00"], ["Total estimado", "", "", "$820.00"]]),
        ("2. Formato base de factura / cuenta de cobro", [["Campo", "Contenido"], ["Emisor", company], ["Cliente", "Cliente final"], ["Concepto", "Servicios administrativos/contables del mes"], ["Soportes", "Documentos adjuntos, aprobaciones y notas"], ["Estado", "Borrador, enviada, vencida, pagada"]]),
        ("3. Agenda, recordatorios y calendario tributario", [["Automatización", "Cómo la usa el agente"], ["Agenda", "Organiza llamadas, entregas de documentos y vencimientos"], ["Recordatorios", "Avisa a clientes cuando falta soporte o aprobación"], ["Calendario tributario", "Mantiene obligaciones por cliente, fecha y estado"], ["Seguimiento", "Registra en CRM cada pendiente y próxima acción"]]),
    ]
    for title, rows in sections:
        story.append(Paragraph(title, styles["H2SU"]))
        table = Table(rows, colWidths=[1.9 * inch] + ([4.5 * inch] if len(rows[0]) == 2 else [1.2 * inch, 1.2 * inch, 1.4 * inch]))
        table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), blue), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#DDE3EC")), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light]), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
        story.append(table)
    story.append(Spacer(1, 8))
    story.append(Paragraph("Próximo paso: convertir estos modelos en plantillas reales con datos, precios, reglas de aprobación, branding y flujos de envío propios de la operación.", styles["SmallSU"]))
    doc.build(story)
    return pdf_path


def _sendgrid_request(path: str, body: dict[str, Any], env: dict[str, str] | None = None) -> dict[str, Any]:
    active_env = merged_env(env)
    key = active_env.get("SENDGRID_API_KEY") or ""
    if not key:
        return {"ok": False, "configured": False, "adapter": "sendgrid", "status": "unavailable", "error": "SENDGRID_API_KEY missing"}
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{SENDGRID_API_BASE}{path}",
        data=data,
        method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "hermes-agent-vapi-postcall/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return {"ok": 200 <= resp.status < 300, "configured": True, "adapter": "sendgrid", "status": resp.status, "x_message_id": resp.headers.get("X-Message-Id")}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return {"ok": False, "configured": True, "adapter": "sendgrid", "status": exc.code, "error": raw[:1000]}
    except Exception as exc:
        return {"ok": False, "configured": True, "adapter": "sendgrid", "status": "error", "error": str(exc)}


def send_email_with_attachment(*, to_email: str, to_name: str | None, subject: str, text: str, html: str, attachment_path: Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    active_env = merged_env(env)
    from_email = active_env.get("SENDGRID_FROM_EMAIL") or active_env.get("NOTIFICATION_FROM_EMAIL") or ""
    from_name = active_env.get("SENDGRID_FROM_NAME") or active_env.get("NOTIFICATION_FROM_NAME") or "Sophie de SitioUno"
    if not from_email:
        return {"ok": False, "configured": False, "adapter": "sendgrid", "status": "unavailable", "error": "SENDGRID_FROM_EMAIL missing"}
    to = {"email": to_email}
    if to_name:
        to["name"] = to_name
    body = {
        "personalizations": [{"to": [to]}],
        "from": {"email": from_email, "name": from_name},
        "subject": subject,
        "content": [{"type": "text/plain", "value": text}, {"type": "text/html", "value": html}],
        "attachments": [{
            "content": base64.b64encode(attachment_path.read_bytes()).decode("ascii"),
            "type": "application/pdf",
            "filename": attachment_path.name,
            "disposition": "attachment",
        }],
    }
    return _sendgrid_request("/v3/mail/send", body, env=active_env)


def _crm_call(handler_name: str, args: dict[str, Any]) -> dict[str, Any]:
    from tools import crm_tool  # type: ignore
    handler = getattr(crm_tool, f"_handle_{handler_name}")
    return json.loads(handler(args))


def upsert_crm_records(job: PostCallJob, pdf_path: Path | None, email_result: dict[str, Any] | None) -> dict[str, Any]:
    name = str(job.lead.get("name") or job.summary.get("caller_name") or "Lead Vapi").strip()
    email = normalize_email(job.lead.get("email"))
    phone = normalize_phone(job.lead.get("phone"))
    base = _slug(email or phone or name)
    contact_id = f"contact-{base}"
    opp_id = f"opp-{base}-voice-demo"
    quote_id = f"quote-{base}-voice-demo"
    metadata = {
        "business_id": "sitiouno",
        "source_channel": "vapi",
        "service_type": "agent_sales_demo",
        "call_id": job.call_id,
        "labels": ["voice-lead", "postcall-worker", job.deliverable_kind],
        "document_path": str(pdf_path) if pdf_path else None,
        "sendgrid_status": (email_result or {}).get("status"),
        "sendgrid_message_id": (email_result or {}).get("x_message_id"),
        "notes": job.lead.get("notes") or job.summary.get("summary"),
    }
    contact = _crm_call("contact_upsert", {"contact_id": contact_id, "full_name": name, "email": email, "phone": phone, "status": "active", "source": "vapi_voice", "metadata": metadata})
    opportunity = _crm_call("opportunity_upsert", {"opportunity_id": opp_id, "contact_id": contact_id, "title": f"Demo agente SitioUno - {job.lead.get('need') or job.deliverable_kind}", "stage": "proposal", "status": "open", "currency": "USD", "metadata": metadata})
    quote = None
    if job.deliverable_kind in {"mechanic_quote_demo", "generic_quote_demo", "admin_accounting_demo"}:
        items = (
            [{"description": "Setup administrativo inicial", "quantity": 1, "unit_price": 250}, {"description": "Gestión contable mensual", "quantity": 1, "unit_price": 450}, {"description": "Calendario tributario por cliente", "quantity": 1, "unit_price": 120}]
            if job.deliverable_kind == "admin_accounting_demo"
            else [{"description": "Diagnóstico de frenos", "quantity": 1, "unit_price": 25}, {"description": "Pastillas delanteras estándar", "quantity": 1, "unit_price": 95}, {"description": "Mano de obra instalación", "quantity": 1, "unit_price": 65}, {"description": "Rectificación/limpieza preventiva", "quantity": 1, "unit_price": 35}]
        )
        quote = _crm_call("quote_create", {"quote_id": quote_id, "contact_id": contact_id, "opportunity_id": opp_id, "title": "Demo administrativo-contable - agente SitioUno" if job.deliverable_kind == "admin_accounting_demo" else "Cotización demo - agente SitioUno", "status": "sent" if (email_result or {}).get("ok") else "draft", "currency": "USD", "items": items, "metadata": metadata})
    interaction = _crm_call("interaction_record", {"contact_id": contact_id, "opportunity_id": opp_id, "channel": "voice/email", "direction": "outbound", "actor": "Sophie de SitioUno / Zeus post-call worker", "summary": f"Post-call worker procesó llamada {job.call_id}: {job.summary.get('summary') or job.lead.get('need')}. Email status: {(email_result or {}).get('status')}; documento: {pdf_path}", "metadata": metadata})
    return {"ok": True, "contact": contact, "opportunity": opportunity, "quote": quote, "interaction": interaction, "contact_id": contact_id, "opportunity_id": opp_id, "quote_id": quote_id if quote else None}


def _email_body(job: PostCallJob) -> tuple[str, str, str]:
    name = str(job.lead.get("name") or job.summary.get("caller_name") or "").strip() or "cliente"
    if job.deliverable_kind == "admin_accounting_demo":
        company = str(job.lead.get("company") or job.lead.get("organization") or "tu empresa").strip() or "tu empresa"
        subject = "Demo SitioUno: formatos administrativos y contables"
        text = textwrap.dedent(f"""\
        Hola {name},

        Soy Sophie de SitioUno. Como acordamos en la llamada, te comparto una demostración inicial para {company} con formatos administrativos y contables.

        Incluye:
        - formato base de cotización,
        - formato base de factura / cuenta de cobro,
        - agenda de tareas y entregas,
        - recordatorios a clientes,
        - calendario tributario por cliente.

        Este es un modelo inicial; en una implementación real Zeus/SitioUno lo ajusta a tus reglas, precios, aprobaciones, branding y flujo de trabajo.

        Saludos,
        Sophie de SitioUno
        Supervisión: Zeus
        """)
        return subject, text, "<br>".join(text.splitlines())
    subject = "Demostración SitioUno: cotización personalizada para taller mecánico"
    text = textwrap.dedent(f"""\
    Hola {name},

    Soy Sophie de SitioUno. Como acordamos en la llamada, te comparto una demostración de cómo un agente de SitioUno puede preparar una cotización personalizada para un taller mecánico.

    Incluye:
    - información capturada del cliente por llamada/WhatsApp,
    - cotización preliminar,
    - mensaje listo para enviar al cliente final,
    - capacidades del agente para CRM, documentos y seguimiento.

    Este es un ejemplo referencial; en una implementación real se ajusta a tus precios, reglas, servicios y flujo de aprobación.

    Saludos,
    Sophie de SitioUno
    Supervisión: Zeus
    """)
    html = "<br>".join(text.splitlines())
    return subject, text, html


def process_job(job: PostCallJob, *, state: dict[str, Any], output_dir: Path = DEFAULT_OUTPUT_DIR, env: dict[str, str] | None = None) -> dict[str, Any]:
    pdf_path: Path | None = None
    email_result: dict[str, Any] | None = None
    if job.deliverable_kind in {"mechanic_quote_demo", "generic_quote_demo", "agent_capabilities_pdf", "admin_accounting_demo"}:
        if job.deliverable_kind == "mechanic_quote_demo":
            pdf_path = generate_mechanic_quote_demo_pdf(job, output_dir)
        elif job.deliverable_kind == "admin_accounting_demo":
            pdf_path = generate_admin_accounting_demo_pdf(job, output_dir)
        else:
            pdf_path = generate_generic_demo_pdf(job, output_dir)
    if job.should_send_email and job.to_email and pdf_path:
        subject, text, html = _email_body(job)
        email_result = send_email_with_attachment(
            to_email=job.to_email,
            to_name=str(job.lead.get("name") or job.summary.get("caller_name") or "").strip() or None,
            subject=subject,
            text=text,
            html=html,
            attachment_path=pdf_path,
            env=env,
        )
    crm_result = upsert_crm_records(job, pdf_path, email_result)
    ok = bool(crm_result.get("ok")) and (not job.should_send_email or bool((email_result or {}).get("ok")))
    if ok:
        processed_calls = set(map(str, state.get("processed_call_ids", [])))
        processed_calls.add(job.call_id)
        state["processed_call_ids"] = sorted(processed_calls)
        processed_events = set(map(str, state.get("processed_event_ids", [])))
        for event in [job.final_event]:
            if event.get("event_id"):
                processed_events.add(str(event["event_id"]))
        state["processed_event_ids"] = sorted(processed_events)
    state.setdefault("runs", []).append({
        "ts": int(time.time()),
        "call_id": job.call_id,
        "ok": ok,
        "deliverable_kind": job.deliverable_kind,
        "to_email": job.to_email,
        "email_status": (email_result or {}).get("status"),
        "email_message_id": (email_result or {}).get("x_message_id"),
        "pdf_path": str(pdf_path) if pdf_path else None,
    })
    # Keep state compact.
    state["runs"] = state.get("runs", [])[-100:]
    return {"ok": ok, "call_id": job.call_id, "pdf_path": str(pdf_path) if pdf_path else None, "email": email_result, "crm": crm_result}


def run_once(events_path: Path = DEFAULT_EVENTS_PATH, state_path: Path = DEFAULT_STATE_PATH, output_dir: Path = DEFAULT_OUTPUT_DIR, env: dict[str, str] | None = None) -> list[dict[str, Any]]:
    events = read_events(events_path)
    state = load_state(state_path)
    processed = set(map(str, state.get("processed_call_ids", [])))
    jobs = build_pending_jobs(events, processed)
    results: list[dict[str, Any]] = []
    for job in jobs:
        results.append(process_job(job, state=state, output_dir=output_dir, env=env))
    if results:
        save_state(state, state_path)
    return results


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    events_path = DEFAULT_EVENTS_PATH
    state_path = DEFAULT_STATE_PATH
    output_dir = DEFAULT_OUTPUT_DIR
    if "--events" in argv:
        events_path = Path(argv[argv.index("--events") + 1])
    if "--state" in argv:
        state_path = Path(argv[argv.index("--state") + 1])
    if "--output-dir" in argv:
        output_dir = Path(argv[argv.index("--output-dir") + 1])
    results = run_once(events_path=events_path, state_path=state_path, output_dir=output_dir)
    if results:
        print(json.dumps({"processed": len(results), "results": results}, ensure_ascii=False, indent=2))
    return 0 if all(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
