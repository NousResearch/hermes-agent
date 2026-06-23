"""Permitted-scenario policy for ShinkaEvolve-OSINT (incl. cyber_defense)."""

from __future__ import annotations

from typing import Any

DEFAULT_PERMITTED_DOMAINS: tuple[str, ...] = (
    "national_security",
    "cyber_defense",
    "ukraine",
    "ai_defense",
    "cognitive_warfare",
    "japan_russia",
    "taiwan",
    "taiwan_contingency",
    "north_korea",
    "dprk_chongryon",
    "us_japan_alliance",
    "constitution_defense",
    "constitutional_defense",
    "space_security",
    "middle_east",
    "middle_east_sealane",
)

CYBER_SCENARIO_IDS: frozenset[str] = frozenset(
    {
        "cyber_defense",
        "cyber_defense_posture",
        "active_cyber_defense",
        "cyber_workforce",
    }
)

# Ukraine-focused scenarios (milspec japan_russia island + intl_situation_jp).
UKRAINE_SCENARIO_IDS: frozenset[str] = frozenset(
    {
        "ukraine_2026",
        "japan_russia_military_overview",
        "nato_russia",
    }
)

UKRAINE_QUERY_MARKERS: tuple[str, ...] = (
    "ウクライナ",
    "ukraine",
    "ゼレンスキー",
    "zaporizhzhia",
)

DEFAULT_SUPPLEMENTAL_EXAMPLES: tuple[str, ...] = ()


def _read_plugin_section() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        plugins = cfg.get("plugins") if isinstance(cfg.get("plugins"), dict) else {}
        for key in ("entries",):
            bucket = plugins.get(key) if isinstance(plugins.get(key), dict) else {}
            section = bucket.get("shinka-osint") if isinstance(bucket.get("shinka-osint"), dict) else {}
            if section:
                return section
        section = plugins.get("shinka-osint") if isinstance(plugins.get("shinka-osint"), dict) else {}
        return section or {}
    except Exception:
        return {}


def load_policy() -> dict[str, Any]:
    section = _read_plugin_section()
    raw_domains = section.get("permitted_domains")
    if isinstance(raw_domains, list) and raw_domains:
        permitted_domains = tuple(str(d).strip() for d in raw_domains if str(d).strip())
    else:
        permitted_domains = DEFAULT_PERMITTED_DOMAINS

    raw_ids = section.get("permitted_scenario_ids")
    permitted_scenario_ids: frozenset[str] | None = None
    if isinstance(raw_ids, list) and raw_ids:
        permitted_scenario_ids = frozenset(str(i).strip() for i in raw_ids if str(i).strip())

    supplemental = section.get("supplemental_examples")
    if isinstance(supplemental, list) and supplemental:
        supplemental_examples = tuple(str(x).strip() for x in supplemental if str(x).strip())
    else:
        supplemental_examples = DEFAULT_SUPPLEMENTAL_EXAMPLES

    return {
        "permitted_domains": permitted_domains,
        "permitted_scenario_ids": permitted_scenario_ids,
        "ensure_cyber_scenarios": bool(section.get("ensure_cyber_scenarios", True)),
        "ensure_ukraine_scenarios": bool(section.get("ensure_ukraine_scenarios", True)),
        "supplemental_examples": supplemental_examples,
    }


def is_scenario_permitted(scenario: dict[str, Any], policy: dict[str, Any] | None = None) -> bool:
    policy = policy or load_policy()
    sid = str(scenario.get("scenario_id") or "")
    domain = str(scenario.get("domain") or "")

    explicit = policy.get("permitted_scenario_ids")
    if explicit:
        if sid in explicit:
            return True
        if policy.get("ensure_cyber_scenarios") and sid in CYBER_SCENARIO_IDS:
            return True
        if policy.get("ensure_ukraine_scenarios") and (sid in UKRAINE_SCENARIO_IDS or _scenario_matches_ukraine(scenario)):
            return True
        return False

    permitted_domains = policy.get("permitted_domains") or DEFAULT_PERMITTED_DOMAINS
    if domain in permitted_domains:
        return True
    if policy.get("ensure_cyber_scenarios") and sid in CYBER_SCENARIO_IDS:
        return True
    if policy.get("ensure_ukraine_scenarios") and sid in UKRAINE_SCENARIO_IDS:
        return True
    if policy.get("ensure_ukraine_scenarios") and domain == "ukraine":
        return True
    return False


def filter_permitted_scenarios(
    scenarios: list[dict[str, Any]],
    policy: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    policy = policy or load_policy()
    return [s for s in scenarios if is_scenario_permitted(s, policy)]


def _scenario_matches_ukraine(scenario: dict[str, Any]) -> bool:
    sid = str(scenario.get("scenario_id") or "").lower()
    if sid in UKRAINE_SCENARIO_IDS:
        return True
    if str(scenario.get("domain") or "") == "ukraine":
        return True
    blob = " ".join(
        str(scenario.get(key) or "")
        for key in ("query", "title", "description")
    ).lower()
    return any(marker.lower() in blob for marker in UKRAINE_QUERY_MARKERS)


def _pick_priority_scenario(
    pool: list[dict[str, Any]],
    *,
    domain: str,
    id_set: frozenset[str],
    matcher,
) -> dict[str, Any] | None:
    for row in pool:
        sid = str(row.get("scenario_id") or "")
        if sid in id_set or str(row.get("domain") or "") == domain:
            return row
    for row in pool:
        if matcher(row):
            return row
    return None


def ensure_priority_in_selection(
    selected: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    *,
    max_scenarios: int,
    policy: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    policy = policy or load_policy()
    merged = list(selected)
    priorities: tuple[tuple[str, str, frozenset[str], Any], ...] = (
        ("ensure_cyber_scenarios", "cyber_defense", CYBER_SCENARIO_IDS, lambda s: str(s.get("domain") or "") == "cyber_defense"),
        ("ensure_ukraine_scenarios", "ukraine", UKRAINE_SCENARIO_IDS, _scenario_matches_ukraine),
    )
    for flag, domain, id_set, matcher in priorities:
        if not policy.get(flag):
            continue
        if any(matcher(s) for s in merged):
            continue
        pick = _pick_priority_scenario(pool, domain=domain, id_set=id_set, matcher=matcher)
        if not pick:
            continue
        sid = str(pick.get("scenario_id") or "")
        if any(str(s.get("scenario_id") or "") == sid for s in merged):
            continue
        merged.insert(0, pick)
    return merged[:max_scenarios]


def ensure_cyber_in_selection(
    selected: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    *,
    max_scenarios: int,
    policy: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return ensure_priority_in_selection(
        selected, pool, max_scenarios=max_scenarios, policy=policy
    )


def save_policy_to_config(
    *,
    permitted_domains: list[str] | None = None,
    ensure_cyber: bool = True,
    ensure_ukraine: bool = True,
    supplemental_examples: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Persist Shinka permitted-scenario policy into ~/.hermes/config.yaml."""
    try:
        from hermes_cli.config import load_config, save_config
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    cfg = load_config()
    plugins = cfg.setdefault("plugins", {})
    entries = plugins.setdefault("entries", {})
    section = entries.setdefault("shinka-osint", {})

    section["permitted_domains"] = list(permitted_domains or DEFAULT_PERMITTED_DOMAINS)
    section["ensure_cyber_scenarios"] = bool(ensure_cyber)
    section["ensure_ukraine_scenarios"] = bool(ensure_ukraine)
    section["supplemental_examples"] = list(
        supplemental_examples if supplemental_examples is not None else DEFAULT_SUPPLEMENTAL_EXAMPLES
    )

    payload = {
        "success": True,
        "dry_run": dry_run,
        "config_key": "plugins.entries.shinka-osint",
        "policy": {
            "permitted_domains": section["permitted_domains"],
            "ensure_cyber_scenarios": section["ensure_cyber_scenarios"],
            "ensure_ukraine_scenarios": section["ensure_ukraine_scenarios"],
            "supplemental_examples": section["supplemental_examples"],
            "cyber_scenario_ids": sorted(CYBER_SCENARIO_IDS),
            "ukraine_scenario_ids": sorted(UKRAINE_SCENARIO_IDS),
        },
    }
    if not dry_run:
        save_config(cfg)
    return payload
