"""Core ShinkaEvolve-OSINT Hermes plugin implementation."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

from . import bridge, providers
from . import policy

DOMAIN_ALIASES: dict[str, tuple[str, ...]] = {
    "national_security": ("国家安全", "経済安全", "サプライチェーン", "national_security"),
    "cyber_defense": ("サイバー", "cyber", "能動的防御"),
    "ukraine": ("ウクライナ", "ukraine", "キーウ", "kyiv", "ゼレンスキー"),
    "ai_defense": ("AI", "LAWS", "ai_defense", "人工知能"),
    "cognitive_warfare": ("認知戦", "偽情報", "cognitive"),
    "japan_russia": ("日ロ", "ロシア", "russia", "ウクライナ侵略"),
    "taiwan": ("台湾", "南西諸島", "taiwan", "taiwan_contingency"),
    "north_korea": ("北朝鮮", "朝鮮", "north_korea", "dprk", "dprk_chongryon", "総連"),
    "us_japan_alliance": ("日米同盟", "同盟", "alliance"),
    "constitution_defense": ("憲法", "反撃能力", "専守防衛", "constitutional_defense"),
    "space_security": ("宇宙", "衛星", "space"),
    "middle_east": ("中東", "ホルムズ", "イラン", "紅海", "middle_east", "iran", "middle_east_sealane"),
}

STATUS_SCHEMA = {
    "name": "shinka_osint_status",
    "description": "Show ShinkaEvolve-OSINT plugin readiness and example availability.",
    "parameters": {"type": "object", "properties": {}},
}

LIST_SCENARIOS_SCHEMA = {
    "name": "shinka_osint_list_scenarios",
    "description": "List MILSPEC OSINT scenarios for world affairs and security analysis.",
    "parameters": {
        "type": "object",
        "properties": {
            "example": {
                "type": "string",
                "description": "Shinka example directory (default: milspec_security_jp).",
            },
            "domain": {
                "type": "string",
                "description": "Optional domain filter (e.g. middle_east, taiwan, cyber_defense).",
            },
        },
    },
}

ANALYZE_SCHEMA = {
    "name": "shinka_osint_analyze",
    "description": (
        "Run one MILSPEC OSINT scenario and return score, evidence blocks, and audit summary."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "scenario_id": {
                "type": "string",
                "description": "Scenario ID from shinka_osint_list_scenarios.",
            },
            "example": {"type": "string"},
            "source_mode": {
                "type": "string",
                "enum": ["mock", "real"],
                "description": "mock uses bundled government corpus; real uses live retrieval when configured.",
            },
        },
        "required": ["scenario_id"],
    },
}

BRIEFING_SCHEMA = {
    "name": "shinka_osint_briefing",
    "description": (
        "Generate a daily world-affairs / security briefing by running matched MILSPEC scenarios."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "Freeform topic (e.g. 中東情勢, 台湾有事, サイバー防衛).",
            },
            "domain": {
                "type": "string",
                "description": "Domain shortcut (middle_east, taiwan, cyber_defense, ...).",
            },
            "scenario_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Explicit scenario IDs; overrides topic/domain matching when set.",
            },
            "max_scenarios": {
                "type": "integer",
                "minimum": 1,
                "maximum": 8,
                "description": "Maximum scenarios to evaluate in one briefing.",
            },
            "example": {"type": "string"},
            "source_mode": {"type": "string", "enum": ["mock", "real"]},
            "save_report": {
                "type": "boolean",
                "description": "Save JSON briefing under ~/.hermes/shinka-osint/briefings/.",
            },
            "llm_summary": {
                "type": "boolean",
                "description": (
                    "Add Japanese executive summary via Hermes LLM "
                    "(openai-codex / NVIDIA / Nous / xAI — no google-generativeai)."
                ),
            },
        },
    },
}

VERIFY_SCHEMA = {
    "name": "shinka_osint_verify",
    "description": "Verify MILSPEC corpus allowlist consistency and audit-chain integrity.",
    "parameters": {
        "type": "object",
        "properties": {
            "example": {"type": "string"},
        },
    },
}

AUDIT_SCHEMA = {
    "name": "shinka_osint_audit",
    "description": "Read recent MILSPEC audit-log entries for an OSINT example.",
    "parameters": {
        "type": "object",
        "properties": {
            "example": {"type": "string"},
            "last_n": {"type": "integer", "minimum": 1, "maximum": 200},
        },
    },
}


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _example_name(args: dict[str, Any]) -> str:
    return (args.get("example") or bridge.resolve_default_example()).strip()


def _source_mode(args: dict[str, Any]) -> str:
    mode = (args.get("source_mode") or "mock").strip().lower()
    return mode if mode in {"mock", "real"} else "mock"


def _briefings_dir() -> Path:
    path = get_hermes_home() / "shinka-osint" / "briefings"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_domain(domain: str) -> str:
    key = (domain or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key in DOMAIN_ALIASES:
        return key
    for canonical, aliases in DOMAIN_ALIASES.items():
        if key and any(key in alias.lower() for alias in aliases):
            return canonical
    return key


def _scenario_records(example: str) -> list[dict[str, Any]]:
    payload = bridge.call_tool(
        "shinka_list_scenarios",
        {"example": example},
    )
    scenarios = payload.get("scenarios") if isinstance(payload, dict) else None
    if not isinstance(scenarios, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in scenarios:
        if not isinstance(item, dict):
            continue
        row = dict(item)
        row.setdefault("scenario_id", row.get("id"))
        if not row.get("query"):
            row["query"] = row.get("description") or row.get("title") or ""
        if not row.get("domain"):
            sid = str(row.get("scenario_id") or "").lower()
            if "ukraine" in sid or policy._scenario_matches_ukraine(row):
                row["domain"] = "ukraine"
            elif "taiwan" in sid:
                row["domain"] = "taiwan"
            elif "dprk" in sid or "north_korea" in sid:
                row["domain"] = "north_korea"
        rows.append(row)
    return rows


def _gather_scenarios(example: str, pol: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Collect scenarios from primary + supplemental examples, apply permitted policy."""
    pol = pol or policy.load_policy()
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    examples = (example, *(pol.get("supplemental_examples") or ()))
    for ex in examples:
        if not ex:
            continue
        try:
            raw_rows = _scenario_records(ex)
        except Exception:
            continue
        for raw in raw_rows:
            sid = str(raw.get("scenario_id") or "")
            if not sid or sid in seen:
                continue
            row = dict(raw)
            row["_example"] = ex
            if policy.is_scenario_permitted(row, pol):
                merged.append(row)
                seen.add(sid)
    return merged


def _match_scenarios(
    scenarios: list[dict[str, Any]],
    *,
    topic: str = "",
    domain: str = "",
    scenario_ids: list[str] | None = None,
    max_scenarios: int = 3,
) -> list[dict[str, Any]]:
    if scenario_ids:
        wanted = {sid.strip() for sid in scenario_ids if sid and sid.strip()}
        matched = [s for s in scenarios if s.get("scenario_id") in wanted]
        return matched[:max_scenarios]

    domain_key = _normalize_domain(domain)
    topic_l = (topic or "").strip().lower()
    if not domain_key and topic_l:
        inferred = _normalize_domain(topic_l)
        if inferred in DOMAIN_ALIASES:
            domain_key = inferred
    scored: list[tuple[int, dict[str, Any]]] = []

    for scenario in scenarios:
        score = 0
        sid = str(scenario.get("scenario_id") or "")
        query = str(scenario.get("query") or "")
        s_domain = str(scenario.get("domain") or "")
        security_topics = scenario.get("security_topics") or []

        if domain_key and s_domain == domain_key:
            score += 100
        if domain_key:
            for alias in DOMAIN_ALIASES.get(domain_key, ()):
                if alias.lower() in query.lower() or alias.lower() in sid.lower():
                    score += 40

        if topic_l:
            if topic_l in query.lower() or topic_l in sid.lower():
                score += 80
            for token in re.split(r"[\s,/、。]+", topic_l):
                if len(token) >= 2 and token in query.lower():
                    score += 20
            for tag in security_topics:
                if isinstance(tag, str) and topic_l in tag.lower():
                    score += 15

        if score > 0:
            scored.append((score, scenario))

    if not scored and scenarios:
        # Daily default: one representative scenario per major domain island.
        seen_domains: set[str] = set()
        fallback: list[dict[str, Any]] = []
        for scenario in scenarios:
            s_domain = str(scenario.get("domain") or "")
            if s_domain and s_domain not in seen_domains:
                seen_domains.add(s_domain)
                fallback.append(scenario)
            if len(fallback) >= max_scenarios:
                break
        return fallback[:max_scenarios]

    scored.sort(key=lambda item: (-item[0], str(item[1].get("scenario_id") or "")))
    return [scenario for _, scenario in scored[:max_scenarios]]


def status() -> dict[str, Any]:
    info = bridge.root_status()
    info["llm"] = providers.provider_status()
    info["available"] = bridge.check_available()
    if info["available"]:
        try:
            examples = bridge.call_tool("shinka_list_examples", {})
            info["examples"] = examples.get("examples", [])
            info["example_count"] = examples.get("count", 0)
        except Exception as exc:  # pragma: no cover - defensive
            info["examples_error"] = str(exc)
    return info


def list_scenarios(example: str, domain: str = "") -> dict[str, Any]:
    pol = policy.load_policy()
    scenarios = _gather_scenarios(example, pol)
    domain_key = _normalize_domain(domain)
    if domain_key:
        scenarios = [
            s
            for s in scenarios
            if s.get("domain") == domain_key
            or any(
                alias.lower() in str(s.get("query") or "").lower()
                for alias in DOMAIN_ALIASES.get(domain_key, ())
            )
        ]
    return {
        "example": example,
        "domain_filter": domain_key or None,
        "count": len(scenarios),
        "scenarios": scenarios,
        "policy": {
            "permitted_domains": list(pol.get("permitted_domains") or ()),
            "ensure_cyber_scenarios": pol.get("ensure_cyber_scenarios"),
            "ensure_ukraine_scenarios": pol.get("ensure_ukraine_scenarios"),
            "supplemental_examples": list(pol.get("supplemental_examples") or ()),
        },
    }


def analyze(
    scenario_id: str,
    *,
    example: str,
    source_mode: str = "mock",
) -> dict[str, Any]:
    pol = policy.load_policy()
    allowed = _gather_scenarios(example, pol)
    if not any(str(s.get("scenario_id") or "") == scenario_id for s in allowed):
        return {
            "success": False,
            "error": f"scenario_id {scenario_id!r} is not in the permitted Shinka OSINT policy",
            "scenario_id": scenario_id,
        }
    use_example = example
    for row in allowed:
        if str(row.get("scenario_id") or "") == scenario_id:
            use_example = str(row.get("_example") or example)
            break
    return bridge.call_tool(
        "shinka_evaluate",
        {
            "example": use_example,
            "scenario_id": scenario_id,
            "source_mode": source_mode,
        },
    )


def briefing(
    *,
    topic: str = "",
    domain: str = "",
    scenario_ids: list[str] | None = None,
    max_scenarios: int = 3,
    example: str,
    source_mode: str = "mock",
    save_report: bool = False,
    llm_summary: bool = False,
) -> dict[str, Any]:
    pol = policy.load_policy()
    cap = max(1, min(max_scenarios, 8))
    scenarios = _gather_scenarios(example, pol)
    selected = _match_scenarios(
        scenarios,
        topic=topic,
        domain=domain,
        scenario_ids=scenario_ids,
        max_scenarios=cap,
    )
    selected = policy.ensure_cyber_in_selection(selected, scenarios, max_scenarios=cap, policy=pol)
    if not selected:
        return {
            "success": False,
            "error": "No scenarios matched. Run shinka_osint_list_scenarios first.",
            "example": example,
            "topic": topic,
            "domain": domain,
        }

    runs: list[dict[str, Any]] = []
    total_score = 0.0
    for scenario in selected:
        sid = str(scenario.get("scenario_id") or "")
        ex = str(scenario.get("_example") or example)
        result = analyze(sid, example=ex, source_mode=source_mode)
        score_block = result.get("score") if isinstance(result, dict) else {}
        total = 0.0
        if isinstance(score_block, dict):
            total = float(score_block.get("total") or score_block.get("combined_score") or 0)
        runs.append(
            {
                "scenario_id": sid,
                "domain": scenario.get("domain"),
                "query": scenario.get("query"),
                "result": result,
                "total_score": total,
            }
        )
        total_score += total

    payload = {
        "success": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "example": example,
        "topic": topic or None,
        "domain": _normalize_domain(domain) or None,
        "source_mode": source_mode,
        "scenario_count": len(runs),
        "average_score": round(total_score / len(runs), 4) if runs else 0.0,
        "runs": runs,
        "llm": providers.provider_status(),
        "policy": {
            "permitted_domains": list(pol.get("permitted_domains") or ()),
            "ensure_cyber_scenarios": pol.get("ensure_cyber_scenarios"),
            "ensure_ukraine_scenarios": pol.get("ensure_ukraine_scenarios"),
        },
    }

    if llm_summary:
        payload["executive_summary"] = providers.synthesize_executive_summary(
            topic=topic or domain or "安全保障",
            briefing_payload=payload,
        )

    if save_report:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        slug = re.sub(r"[^\w\-]+", "_", (topic or domain or "daily")[:40]).strip("_") or "daily"
        out = _briefings_dir() / f"{stamp}_{slug}.json"
        out.write_text(_json(payload), encoding="utf-8")
        payload["saved_report"] = str(out)

    return payload


def verify(example: str) -> dict[str, Any]:
    corpus = bridge.call_tool("shinka_corpus_status", {"example": example})
    integrity = bridge.call_tool("shinka_verify_integrity", {"example": example})
    return {
        "example": example,
        "corpus": corpus,
        "integrity": integrity,
    }


def audit(example: str, last_n: int = 20) -> dict[str, Any]:
    return bridge.call_tool(
        "shinka_audit_log",
        {"example": example, "last_n": max(1, min(last_n, 200))},
    )


def check_available() -> bool:
    return bridge.check_available()


def handle_status(args: dict[str, Any], **_: Any) -> str:
    return _json(status())


def handle_list_scenarios(args: dict[str, Any], **_: Any) -> str:
    return _json(list_scenarios(_example_name(args), domain=args.get("domain") or ""))


def handle_analyze(args: dict[str, Any], **_: Any) -> str:
    scenario_id = (args.get("scenario_id") or "").strip()
    if not scenario_id:
        return _json({"success": False, "error": "scenario_id is required"})
    return _json(
        analyze(
            scenario_id,
            example=_example_name(args),
            source_mode=_source_mode(args),
        )
    )


def handle_briefing(args: dict[str, Any], **_: Any) -> str:
    scenario_ids = args.get("scenario_ids")
    if scenario_ids is not None and not isinstance(scenario_ids, list):
        scenario_ids = [str(scenario_ids)]
    return _json(
        briefing(
            topic=args.get("topic") or "",
            domain=args.get("domain") or "",
            scenario_ids=scenario_ids,
            max_scenarios=int(args.get("max_scenarios") or 3),
            example=_example_name(args),
            source_mode=_source_mode(args),
            save_report=bool(args.get("save_report")),
            llm_summary=bool(args.get("llm_summary")),
        )
    )


def handle_verify(args: dict[str, Any], **_: Any) -> str:
    return _json(verify(_example_name(args)))


def handle_audit(args: dict[str, Any], **_: Any) -> str:
    return _json(audit(_example_name(args), last_n=int(args.get("last_n") or 20)))


HELP = """shinka-osint commands:
  /shinka-osint status
  /shinka-osint scenarios [domain]
  /shinka-osint analyze <scenario_id> [--real]
  /shinka-osint briefing [topic...] [--domain DOMAIN] [--save]
  /shinka-osint verify
  /shinka-osint audit [last_n]
"""


def handle_slash(raw_args: str) -> str:
    argv = (raw_args or "").strip().split()
    if not argv or argv[0] in {"help", "-h", "--help"}:
        return HELP

    command = argv[0].lower()
    if command == "status":
        return _json(status())
    if command in {"scenarios", "list", "list-scenarios"}:
        domain = ""
        if len(argv) >= 2:
            domain = argv[1]
        return _json(list_scenarios(bridge.resolve_default_example(), domain=domain))
    if command == "analyze":
        if len(argv) < 2:
            return _json({"success": False, "error": "usage: analyze <scenario_id>"})
        return _json(
            analyze(
                argv[1],
                example=bridge.resolve_default_example(),
                source_mode="real" if "--real" in argv else "mock",
            )
        )
    if command == "briefing":
        topic_tokens = [tok for tok in argv[1:] if not tok.startswith("--")]
        flags = {tok.lstrip("-") for tok in argv[1:] if tok.startswith("--")}
        domain = ""
        for tok in argv[1:]:
            if tok.startswith("--domain="):
                domain = tok.split("=", 1)[1]
        return _json(
            briefing(
                topic=" ".join(topic_tokens),
                domain=domain,
                example=bridge.resolve_default_example(),
                source_mode="real" if "real" in flags else "mock",
                save_report="save" in flags,
            )
        )
    if command == "verify":
        return _json(verify(bridge.resolve_default_example()))
    if command == "audit":
        last_n = 20
        if len(argv) >= 2:
            try:
                last_n = int(argv[1])
            except ValueError:
                pass
        return _json(audit(bridge.resolve_default_example(), last_n=last_n))
    return HELP
