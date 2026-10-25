"""Record-local provenance for proactive suggestions. Source text is never authority."""

from __future__ import annotations

from typing import Any
import hashlib
import json
import re


ACTIONS = {"review": "Review the cited records."}
_FIELDS = frozenset({"kind", "entity_key", "title", "summary", "why_now", "suggested_action", "evidence"})
_URL = re.compile(r"https?://[^\s<>\"']+")
_TIME = re.compile(r"\b(?:today|tomorrow|urgent|asap|tonight|\d{4}-\d{2}-\d{2})\b|오늘|내일|긴급", re.IGNORECASE)
_REQUEST = re.compile(r"\b(?:please|could you|can you|need you)\b|부탁|해\s?주세요", re.IGNORECASE)
_DEADLINE = re.compile(r"\b(?:deadline|due|by)\b|기한|마감", re.IGNORECASE)


def observed_content_digest(content: Any) -> str:
    """Version the complete observed record, never a model-selected quote or hash."""
    encoded = json.dumps(content, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _record_nodes(value: Any, pointer: str = ""):
    if isinstance(value, dict):
        if "id" in value or not any(isinstance(v, (dict, list)) for v in value.values()):
            yield pointer, value
        else:
            for key, child in value.items():
                escaped = key.replace("~", "~0").replace("/", "~1")
                yield from _record_nodes(child, f"{pointer}/{escaped}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _record_nodes(child, f"{pointer}/{index}")
    elif isinstance(value, str):
        yield pointer, value


def observed_records(observations: list[dict]) -> list[dict]:
    """Split responses into records with observed IDs or stable JSON-pointer identities."""
    records = []
    for observation in observations:
        for pointer, content in _record_nodes(observation["response"]):
            record_id = content.get("id") if isinstance(content, dict) else None
            excerpt = max(_strings(content), key=len, default="")[:4096]
            url = content.get("url") if isinstance(content, dict) else None
            records.append({"source_id": observation["source_id"],
                            "capability_id": observation["capability_id"],
                            "record_id": str(record_id) if record_id is not None else f"pointer:{pointer}",
                            "json_pointer": pointer, "content": content, "excerpt": excerpt,
                            "content_digest": observed_content_digest(content),
                            **({"url": url} if isinstance(url, str) and _URL.fullmatch(url) else {}),
                            "data_use": sorted(set(observation.get("data_use") or ()))})
    return records


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def canonical_citation(citation: Any, records: list[dict]) -> dict | None:
    if not isinstance(citation, dict):
        return None
    if not all(isinstance(citation.get(k), str) for k in ("source_id", "record_id", "json_pointer", "excerpt")):
        return None
    excerpt = citation["excerpt"]
    if not excerpt.strip() or len(excerpt) > 4096:
        return None
    matches = [r for r in records if all(r[k] == citation[k] for k in ("source_id", "record_id", "json_pointer"))
               and ("capability_id" not in citation or r["capability_id"] == citation["capability_id"])]
    if len(matches) != 1:
        return None
    record = matches[0]
    text = list(_strings(record["content"]))
    if not any(excerpt in value for value in text):
        return None
    url = citation.get("url")
    if url is not None and (not isinstance(url, str) or not _URL.fullmatch(url)
                            or not any(url in _URL.findall(value) for value in text)):
        return None
    return {**{k: record[k] for k in ("source_id", "capability_id", "record_id", "json_pointer", "data_use")},
            "content_digest": observed_content_digest(record["content"]),
            "excerpt": excerpt, **({"url": url} if url else {})}


def _valid_candidate(candidate: Any) -> bool:
    if not isinstance(candidate, dict) or set(candidate) != _FIELDS:
        return False
    if not all(isinstance(candidate[k], str) and candidate[k].strip() and len(candidate[k]) <= 4096
               for k in _FIELDS - {"evidence"}):
        return False
    return (candidate["kind"] in {"cross_source", "direct_request", "deadline"}
            and candidate["suggested_action"] in ACTIONS
            and isinstance(candidate["evidence"], list) and 1 <= len(candidate["evidence"]) <= 20)


def _supported_claim(candidate: dict, citations: list[dict], records: list[dict]) -> bool:
    allowed_urls = {url for record in records for value in _strings(record["content"]) for url in _URL.findall(value)}
    if any(url not in allowed_urls for k in _FIELDS - {"evidence"} for url in _URL.findall(candidate[k])):
        return False
    if candidate["kind"] == "cross_source":
        return len({c["source_id"] for c in citations}) >= 2
    # Single-source urgency must be visible in an exact quote, not just a model's label.
    why = candidate["why_now"]
    if not any(why in c["excerpt"] for c in citations) or not _TIME.search(why):
        return False
    predicate = _REQUEST if candidate["kind"] == "direct_request" else _DEADLINE
    return bool(predicate.search(why))


def validate_findings(candidates: Any, records: list[dict]) -> list[dict]:
    """Canonical observed citations only; malformed or uncertain candidates are silent."""
    if not isinstance(candidates, list) or len(candidates) > 20:
        return []
    findings = []
    for candidate in candidates:
        if not _valid_candidate(candidate):
            continue
        citations = [canonical_citation(c, records) for c in candidate["evidence"]]
        if any(c is None for c in citations) or not _supported_claim(candidate, citations, records):
            continue
        findings.append({**candidate, "evidence": citations,
                         "suggested_action": ACTIONS[candidate["suggested_action"]],
                         "data_use": sorted({r for e in citations for r in e["data_use"]})})
    return findings
