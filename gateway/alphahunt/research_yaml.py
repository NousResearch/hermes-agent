"""AlphaHunt project research YAML envelope helpers.

This module is intentionally pure: it validates and renders research payloads
without network access, secrets, API calls, or database writes.
"""

from __future__ import annotations

import argparse
import copy
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

KINDS = frozenset(
    {
        "stock",
        "etf",
        "protocol",
        "commodity_theme",
        "macro_event",
        "industry_theme",
        "market",
    }
)

KNOWN_ASSET_CLASSES = frozenset(
    {
        "stock",
        "equity",
        "etf",
        "crypto",
        "commodity",
        "commodity_theme",
        "macro",
        "macro_event",
        "industry",
        "industry_theme",
        "market",
        "prediction_market",
    }
)

REQUIRED_NOTE_FIELDS = (
    "thesis",
    "key_assumptions",
    "risk_triggers",
    "invalidation_conditions",
    "observables",
    "next_check_at",
    "source_references",
)

MARKET_ACTIONS = frozenset(
    {
        "ignore",
        "observe",
        "research",
        "manual_review",
        "no_participation",
    }
)

MARKET_FORBIDDEN_TERMS = (
    "bet",
    "wager",
    "execute",
    "stake",
    "kelly",
)


class _LiteralString(str):
    pass


class _ResearchDumper(yaml.SafeDumper):
    pass


def _literal_representer(dumper: yaml.SafeDumper, data: _LiteralString) -> yaml.nodes.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


_ResearchDumper.add_representer(_LiteralString, _literal_representer)


def build_research_envelope(data: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized copy of a Hermes research payload."""

    if not isinstance(data, dict):
        raise TypeError("research data must be a dict")

    envelope = copy.deepcopy(data)
    for list_key in ("tickers", "contract_addresses", "aliases"):
        if list_key in envelope and envelope[list_key] is None:
            envelope[list_key] = []
    if "note" in envelope and isinstance(envelope["note"], dict):
        note = envelope["note"]
        for list_key in (
            "key_assumptions",
            "risk_triggers",
            "invalidation_conditions",
            "observables",
            "bull_case",
            "bear_case",
            "source_references",
        ):
            if list_key in note and note[list_key] is None:
                note[list_key] = []
    return envelope


def validate_research_envelope(envelope: dict[str, Any]) -> list[str]:
    """Validate an AlphaHunt project research envelope.

    Returns a list of human-readable errors. An empty list means the envelope is
    valid for the Hermes-side contract.
    """

    errors: list[str] = []
    if not isinstance(envelope, dict):
        return ["envelope must be a mapping"]

    _require_nonempty_string(envelope, "subject", errors)

    kind = envelope.get("kind")
    if not isinstance(kind, str) or not kind.strip():
        errors.append("kind is required")
    elif kind not in KINDS:
        errors.append(f"kind must be one of {sorted(KINDS)}")

    _validate_asset_class(envelope.get("asset_class"), errors)
    _require_nonempty_string(envelope, "research_markdown", errors)
    if "source_references" in envelope:
        errors.append("source_references must be nested under note.source_references")

    note = envelope.get("note")
    if not isinstance(note, dict):
        errors.append("note is required")
        note = {}

    for field in REQUIRED_NOTE_FIELDS:
        if field not in note:
            errors.append(f"note.{field} is required")

    _require_nonempty_string(note, "thesis", errors, prefix="note.")
    _require_nonempty_list(note, "key_assumptions", errors, prefix="note.")
    _require_nonempty_list(note, "risk_triggers", errors, prefix="note.")
    _require_nonempty_list(note, "invalidation_conditions", errors, prefix="note.")
    _validate_observables(note.get("observables"), errors)
    _validate_next_check_at(note.get("next_check_at"), errors)
    _validate_confidence(note.get("confidence"), errors)
    _require_nonempty_list(note, "source_references", errors, prefix="note.")

    if kind in {"stock", "etf"}:
        _require_nonempty_list(envelope, "tickers", errors)
    if kind == "protocol":
        _require_nonempty_string(envelope, "chain", errors)
    if kind == "market":
        _validate_market_boundary(envelope, note, errors)

    return errors


def dump_research_yaml(envelope: dict[str, Any]) -> str:
    """Render a research envelope as stable YAML."""

    rendered = _prepare_literals(copy.deepcopy(envelope))
    return yaml.dump(
        rendered,
        Dumper=_ResearchDumper,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )


def load_research_yaml(text: str) -> dict[str, Any]:
    """Load a research envelope from YAML text."""

    loaded = yaml.safe_load(text)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("research YAML must contain a mapping")
    return loaded


def sample_research_envelope(kind: str) -> dict[str, Any]:
    """Return a built-in sample envelope for docs and smoke tests."""

    samples = {
        "protocol": {
            "subject": "Ethena protocol basis-yield research",
            "kind": "protocol",
            "asset_class": "DRAFT:crypto_protocol",
            "chain": "ethereum",
            "contract_addresses": ["0x0000000000000000000000000000000000000000"],
            "aliases": ["Ethena", "USDe", "sUSDe"],
            "research_markdown": (
                "# Ethena Protocol Research\n\n"
                "Ethena's opportunity depends on durable demand for dollar-denominated crypto yield, "
                "exchange liquidity, and transparent collateral/risk disclosures.\n"
            ),
            "note": _base_note(
                "Ethena is worth continued research if collateral transparency and exchange liquidity remain intact.",
                ["USDe supply remains liquid across major venues.", "Funding-rate income remains observable."],
                ["Sustained depeg pressure.", "Material decline in disclosed backing quality."],
                ["USDe trades below 0.98 for a full day.", "Collateral reporting becomes stale."],
                "USDe secondary-market peg",
                "Public exchange market data",
                "0.98",
                "down",
                source_references=["https://docs.ethena.fi/"],
            ),
        },
        "stock": {
            "subject": "Example Semiconductor Inc research",
            "kind": "stock",
            "asset_class": "stock",
            "tickers": ["EXSM"],
            "aliases": ["Example Semiconductor"],
            "research_markdown": (
                "# Example Semiconductor Research\n\n"
                "The company is a placeholder equity research sample for validating the Hermes "
                "AlphaHunt research envelope.\n"
            ),
            "note": _base_note(
                "The equity case depends on revenue growth translating into durable margin expansion.",
                ["Demand remains resilient.", "Inventory normalizes over the next two quarters."],
                ["Guidance cut.", "Gross margin compression."],
                ["Revenue growth turns negative.", "Management withdraws margin targets."],
                "Quarterly revenue growth",
                "Company filings",
                "positive year over year",
                "up",
                source_references=["https://www.sec.gov/edgar"],
            ),
        },
        "etf": {
            "subject": "Example broad market ETF research",
            "kind": "etf",
            "asset_class": "etf",
            "tickers": ["EXETF"],
            "aliases": ["Example Broad Market ETF"],
            "research_markdown": (
                "# Example ETF Research\n\n"
                "This ETF sample focuses on exposure, liquidity, tracking quality, and macro sensitivity.\n"
            ),
            "note": _base_note(
                "The ETF remains useful as a liquid proxy if spreads and tracking error stay contained.",
                ["Underlying basket remains liquid.", "Tracking error remains low."],
                ["Creation/redemption stress.", "Unexpected index methodology change."],
                ["Bid/ask spreads widen materially.", "Tracking error breaches historical range."],
                "Bid/ask spread",
                "Fund sponsor and exchange data",
                "near recent median",
                "down",
                source_references=["https://www.sec.gov/edgar"],
            ),
        },
        "commodity_theme": {
            "subject": "Copper electrification demand research",
            "kind": "commodity_theme",
            "asset_class": "commodity_theme",
            "aliases": ["copper", "electrification", "grid demand"],
            "research_markdown": (
                "# Copper Theme Research\n\n"
                "Copper demand is tied to grid investment, electric equipment, construction activity, "
                "and mine supply disruptions.\n"
            ),
            "note": _base_note(
                "Copper deserves monitoring while electrification demand offsets cyclical construction softness.",
                ["Grid capex remains elevated.", "Mine supply disruptions persist."],
                ["China construction demand weakens further.", "Inventories rebuild quickly."],
                ["Visible inventories rise for four consecutive weeks.", "Treatment charges normalize sharply."],
                "Exchange copper inventory",
                "LME/COMEX public inventory data",
                "four-week trend",
                "up",
                source_references=["https://www.lme.com/"],
            ),
        },
        "macro_event": {
            "subject": "FOMC policy path research",
            "kind": "macro_event",
            "asset_class": "macro_event",
            "aliases": ["FOMC", "Federal Reserve", "policy rates"],
            "research_markdown": (
                "# FOMC Macro Research\n\n"
                "The macro event hinges on inflation progress, labor-market cooling, and communication "
                "around the expected policy path.\n"
            ),
            "note": _base_note(
                "Policy-path expectations should be reassessed after the next inflation and labor releases.",
                ["Inflation continues to moderate.", "Labor conditions cool without a sharp break."],
                ["Inflation re-accelerates.", "Unemployment rises abruptly."],
                ["Core inflation surprises higher twice.", "FOMC guidance shifts hawkish."],
                "Core PCE inflation",
                "BEA release",
                "monthly change",
                "down",
                source_references=["https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"],
            ),
        },
        "market": {
            "subject": "World Cup champion prediction market research",
            "kind": "market",
            "asset_class": "prediction_market",
            "aliases": ["World Cup champion", "prediction market"],
            "market_meta": {
                "rules": "Market resolves according to the venue's published champion settlement rules.",
                "settlement": "Winning national team after the final is officially completed.",
                "deadline": "2026-07-19T23:59:00+00:00",
            },
            "research_markdown": (
                "# World Cup Market Research\n\n"
                "This output is reference-only. It is for observation and manual review, not automated "
                "participation.\n\n"
                "Boundary: no_participation, no_execution, 不自动下注, 不自动交易, 不保证收益.\n"
            ),
            "note": _base_note(
                "The market is suitable only for observation because sports outcomes are high-variance.",
                ["Venue rules remain stable.", "Team availability data is current."],
                ["Rule ambiguity.", "Major injury or roster uncertainty."],
                ["Settlement wording changes.", "Liquidity becomes too thin for reliable interpretation."],
                "Market-implied probability",
                "Public market page",
                "large move after official news",
                "up",
                action_suggestion="no_participation",
                source_references=["https://www.fifa.com/"],
            ),
        },
    }
    if kind not in samples:
        raise ValueError(f"unknown sample kind: {kind}")
    return build_research_envelope(samples[kind])


def _base_note(
    thesis: str,
    assumptions: list[str],
    risks: list[str],
    invalidations: list[str],
    observable_name: str,
    observable_source: str,
    observable_threshold: str,
    observable_direction: str,
    *,
    action_suggestion: str = "research",
    source_references: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "thesis": thesis,
        "key_assumptions": assumptions,
        "risk_triggers": risks,
        "invalidation_conditions": invalidations,
        "observables": [
            {
                "name": observable_name,
                "source": observable_source,
                "threshold": observable_threshold,
                "direction": observable_direction,
            }
        ],
        "next_check_at": "2026-06-18T00:00:00+00:00",
        "bull_case": ["Key assumptions continue to hold."],
        "bear_case": ["Risk triggers become observable."],
        "confidence": 0.6,
        "action_suggestion": action_suggestion,
        "source_references": source_references or ["https://example.com/research-source"],
    }


def _require_nonempty_string(mapping: dict[str, Any], key: str, errors: list[str], *, prefix: str = "") -> None:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{prefix}{key} is required")


def _require_nonempty_list(mapping: dict[str, Any], key: str, errors: list[str], *, prefix: str = "") -> None:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        errors.append(f"{prefix}{key} must contain at least one item")


def _validate_asset_class(value: Any, errors: list[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append("asset_class is required")
        return
    if value in KNOWN_ASSET_CLASSES or value.startswith("DRAFT:"):
        return
    errors.append("asset_class must be a known class or use DRAFT:<name>")


def _validate_observables(value: Any, errors: list[str]) -> None:
    if not isinstance(value, list) or not value:
        errors.append("note.observables must contain at least one item")
        return
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            errors.append(f"note.observables[{index}] must be a mapping")
            continue
        for field in ("name", "source", "threshold", "direction"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                errors.append(f"note.observables[{index}].{field} is required")
        direction = item.get("direction")
        if isinstance(direction, str) and direction not in {"up", "down"}:
            errors.append(f"note.observables[{index}].direction must be up or down")


def _validate_next_check_at(value: Any, errors: list[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append("note.next_check_at is required")
        return
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        errors.append("note.next_check_at must be parseable ISO8601")


def _validate_confidence(value: Any, errors: list[str]) -> None:
    if value is None:
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
        errors.append("note.confidence must be a number between 0 and 1")


def _validate_market_boundary(envelope: dict[str, Any], note: dict[str, Any], errors: list[str]) -> None:
    market_meta = envelope.get("market_meta")
    if not isinstance(market_meta, dict):
        errors.append("market_meta is required for market research")
    else:
        for field in ("rules", "settlement", "deadline"):
            if not isinstance(market_meta.get(field), str) or not market_meta[field].strip():
                errors.append(f"market_meta.{field} is required for market research")
        deadline = market_meta.get("deadline")
        if isinstance(deadline, str) and deadline.strip():
            try:
                datetime.fromisoformat(deadline.replace("Z", "+00:00"))
            except ValueError:
                errors.append("market_meta.deadline must be parseable ISO8601")

    action = note.get("action_suggestion")
    if action not in MARKET_ACTIONS:
        errors.append(f"market note.action_suggestion must be one of {sorted(MARKET_ACTIONS)}")

    haystack = "\n".join(_iter_strings(envelope)).lower()
    for term in MARKET_FORBIDDEN_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", haystack):
            errors.append(f"market research must not contain forbidden term: {term}")

    if "reference-only" not in haystack:
        errors.append("market research must state reference-only")
    if "no_participation" not in haystack:
        errors.append("market research must state no_participation")
    if "no_execution" not in haystack:
        errors.append("market research must state no_execution")


def _iter_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        strings: list[str] = []
        for item in value.values():
            strings.extend(_iter_strings(item))
        return strings
    if isinstance(value, list):
        strings = []
        for item in value:
            strings.extend(_iter_strings(item))
        return strings
    return []


def _prepare_literals(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _prepare_literals(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_prepare_literals(item) for item in value]
    if isinstance(value, str) and "\n" in value:
        return _LiteralString(value)
    return value


def _validate_file(path: Path) -> int:
    envelope = load_research_yaml(path.read_text(encoding="utf-8"))
    errors = validate_research_envelope(envelope)
    if errors:
        for error in errors:
            print(f"validation_error: {error}", file=sys.stderr)
        return 1
    print(f"valid: {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate or render AlphaHunt project research YAML.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate", type=Path, metavar="FILE", help="validate a research YAML file")
    group.add_argument("--render-sample", choices=sorted(["protocol", "stock", "etf", "commodity_theme", "macro_event", "market"]))
    args = parser.parse_args(argv)

    if args.validate:
        return _validate_file(args.validate)

    envelope = sample_research_envelope(args.render_sample)
    errors = validate_research_envelope(envelope)
    if errors:
        for error in errors:
            print(f"validation_error: {error}", file=sys.stderr)
        return 1
    print(dump_research_yaml(envelope), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
