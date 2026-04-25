"""
Alert Enrichment — LLM Prompt Templates
=======================================
Templates for turning raw alert clusters into Telegram-ready briefings.
Uses MiniMax LLM (via Hermes internal). For production, swap the
completion_call to your preferred provider.
"""

from dataclasses import dataclass
from typing import Optional

# ────────────────────────────────────────────────────────────────
# TELEGRAM FORMATTING HELPERS
# ────────────────────────────────────────────────────────────────

SEVERITY_EMOJI = {
    "critical": "🔴",
    "high": "🟠",
    "average": "🟡",
    "warning": "🔵",
    "info": "⚪",
}

SEVERITY_COLOR = {
    "critical": "🔴 CRITICAL",
    "high": "🟠 HIGH",
    "average": "🟡 AVERAGE",
    "warning": "🔵 WARNING",
    "info": "⚪ INFO",
}


@dataclass
class EnrichmentContext:
    """Context passed to the LLM for a cluster of alerts."""
    cluster_id: str
    device_name: str
    alert_types: list[str]
    site: str
    device_role: str
    is_core: bool
    is_critical: bool
    alerts: list[dict]          # raw alert records
    connected_devices: list[dict]
    cables: list[dict]
    metrics: dict
    redundancy_status: Optional[str]
    affected_components: list[str]
    nb_tags: list[str]


# ────────────────────────────────────────────────────────────────
# PROMPT BUILDERS
# ────────────────────────────────────────────────────────────────

def build_cluster_briefing_prompt(ctx: EnrichmentContext) -> str:
    """
    Generate a Telegram-ready briefing from a cluster of alerts on the same device.
    """
    device_line = f"📍 *{ctx.device_name}* ({ctx.device_role}) — {ctx.site}"
    if ctx.is_core:
        device_line += " | ⚠️ *CORE DEVICE*"
    if ctx.is_critical:
        device_line += " | 🏢 *CUSTOMER CRITICAL*"

    alert_lines = []
    for i, alert in enumerate(ctx.alerts, 1):
        sev = SEVERITY_EMOJI.get(alert.get("severity", ""), "⚪")
        alert_lines.append(f"  {i}. {sev} *{alert.get('alert_type', 'unknown').upper()}* — {alert.get('message', alert.get('description', 'No description'))}")

    alert_section = "\n".join(alert_lines) if alert_lines else "  Single alert"

    connected = []
    for d in ctx.connected_devices[:4]:
        connected.append(f"  • *{d.get('device_name', '?')}* ({d.get('device_role', '?')}) — via {d.get('interface', '?')}")

    cables = []
    for c in ctx.cables[:3]:
        cables.append(f"  • Cable `{c.get('id', '?')}` ({c.get('label', 'no label')}) — {c.get('type', '?')} {c.get('status', '')}")

    metrics_lines = []
    if ctx.metrics:
        m = ctx.metrics
        if "cpu" in m:
            metrics_lines.append(f"  • CPU: {m['cpu']}%")
        if "memory" in m:
            metrics_lines.append(f"  • Memory: {m['memory']}%")
        if "temperature" in m:
            metrics_lines.append(f"  • Temp: {m['temperature']}°C")
        if "ps0" in m:
            metrics_lines.append(f"  • PS0: {m['ps0']} | PS1: {m.get('ps1', '?')}")

    redundancy = f"\n🔄 Redundancy: *{ctx.redundancy_status or 'UNKNOWN'}*" if ctx.redundancy_status else ""

    prompt = f"""You are a NOC engineer writing a concise, actionable Telegram alert briefing.

## DEVICE CONTEXT
{device_line}

## ACTIVE ALERTS ({len(ctx.alerts)})
{alert_section}
{redundancy}

## CONNECTIVITY
{chr(10).join(connected) if connected else "  No connected devices found in NetBox"}

## CABLING
{chr(10).join(cables) if cables else "  No cables in NetBox"}

## LIVE METRICS
{chr(10).join(metrics_lines) if metrics_lines else "  No metrics available"}

## IMPACT ANALYSIS
Based on the above, provide a 2-3 sentence impact summary:
- What is affected (services, customers, redundancy)?
- What is the urgency for the on-call engineer?
- Any immediate actions required?

## RECOMMENDED ACTIONS
List 1-3 specific next steps for the on-call engineer. Be concrete:
- Which command to run first
- What to check in NetBox
- Whether to escalate immediately

Format the output as clean Telegram markdown. Use emojis for severity. Bold critical info. Keep it under 500 words.
"""
    return prompt


def build_site_briefing_prompt(ctx: EnrichmentContext) -> str:
    """
    Generate a site-wide digest when multiple devices have alerts.
    """
    prompt = f"""You are a NOC engineer writing a site-wide incident digest for Telegram.

## SITE: {ctx.site}

## DEVICE: {ctx.device_name} ({ctx.device_role})
{"⚠️ CORE DEVICE" if ctx.is_core else ""} {"🏢 CUSTOMER CRITICAL" if ctx.is_critical else ""}

## ALERT TYPES
{', '.join(ctx.alert_types)}

## AFFECTED COMPONENTS
{', '.join(ctx.affected_components) if ctx.affected_components else 'Multiple subsystems'}

## NUMBER OF ALERTS: {len(ctx.alerts)}

## CONNECTED DEVICES AT RISK
{chr(10).join([f"  • {d.get('device_name', '?')} ({d.get('device_role', '?')})" for d in ctx.connected_devices[:6]]) if ctx.connected_devices else "  None identified"}

## LIVE METRICS
{chr(10).join([f"  • {k.upper()}: {v}" for k, v in ctx.metrics.items()]) if ctx.metrics else "  No metrics available"}

## SITE IMPACT SUMMARY
2-3 sentences on the wider impact to the site.

## RECOMMENDED ACTIONS
1-3 concrete steps.

Format as Telegram markdown, use emojis, keep under 400 words.
"""
    return prompt


# ────────────────────────────────────────────────────────────────
# COMPLETION CALL (stub — integrate with your LLM provider)
# ────────────────────────────────────────────────────────────────

def complete(prompt: str, model: str = "MiniMax-M2.7", max_tokens: int = 600) -> str:
    """
    Call MiniMax LLM via the Anthropic-compatible endpoint.
    Uses the same API pattern as alert_enricher.py.
    """
    import os, urllib.request, json

    api_key = os.environ.get("MINIMAX_API_KEY", "")
    if not api_key:
        return _fallback_summary(prompt)

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        req = urllib.request.Request(
            "https://api.minimax.io/anthropic/v1/messages",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read())
            return result.get("content", [{}])[0].get("text", "")
    except Exception as e:
        return _fallback_summary(prompt)


def _fallback_summary(prompt: str) -> str:
    """Fallback when LLM is not available — extracts key info from prompt."""
    lines = prompt.strip().split("\n")
    device = next((l for l in lines if "DEVICE CONTEXT" in l or l.startswith("📍")), "Device unknown")
    severity_block = next((l for l in lines if "ACTIVE ALERTS" in l), "")
    alert_count = len([l for l in lines if l.strip().startswith(tuple("123456789"))])

    return f"""📋 *Alert Cluster Summary*

{device}

⚠️ *Manual review required — LLM enrichment not yet configured.*

Review the raw alert data in the cluster and take appropriate action.
"""
