"""Grok authoring layer for Torben GTM content packages."""

from __future__ import annotations

import copy
import json
import os
import re
from datetime import datetime, timezone
from typing import Any

import requests

from tools.xai_http import hermes_xai_user_agent, resolve_xai_http_credentials


DEFAULT_GROK_MODEL = "grok-4.3"
DEFAULT_TIMEOUT_SECONDS = 180


def grok_drafting_enabled() -> bool:
    value = str(os.getenv("TORBEN_GTM_GROK_DRAFTING", "1")).strip().lower()
    return value not in {"0", "false", "no", "off"}


def enrich_package_with_grok(
    package_payload: dict[str, Any],
    *,
    model: str | None = None,
    timeout_seconds: int | None = None,
    enable_x_search: bool | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return *package_payload* with Grok-authored draft assets when available.

    This function may call xAI, but it never posts to X/LinkedIn and never
    mutates anything outside the returned package object.
    """

    payload = copy.deepcopy(package_payload)
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if not grok_drafting_enabled():
        payload["grok_authoring"] = {
            "status": "disabled",
            "reason": "TORBEN_GTM_GROK_DRAFTING disabled",
            "generated_at": _iso(now),
        }
        return payload

    selected_model = str(model or os.getenv("TORBEN_GTM_GROK_MODEL") or DEFAULT_GROK_MODEL).strip()
    timeout = _positive_int(timeout_seconds or os.getenv("TORBEN_GTM_GROK_TIMEOUT_SECONDS"), DEFAULT_TIMEOUT_SECONDS)
    x_search_enabled = _truthy_env("TORBEN_GTM_GROK_X_SEARCH", default=True) if enable_x_search is None else bool(enable_x_search)

    try:
        creds = resolve_xai_http_credentials()
        api_key = str(creds.get("api_key") or "").strip()
        if not api_key:
            raise RuntimeError("xAI OAuth/API credentials are unavailable")
        base_url = str(creds.get("base_url") or "https://api.x.ai/v1").strip().rstrip("/")
        request_payload = _responses_payload(
            package_payload=payload,
            model=selected_model,
            enable_x_search=x_search_enabled,
        )
        response = requests.post(
            f"{base_url}/responses",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": hermes_xai_user_agent(),
            },
            json=request_payload,
            timeout=timeout,
        )
        response.raise_for_status()
        response_payload = response.json()
        generated = _parse_json_object(_extract_response_text(response_payload))
        payload = _merge_grok_output(
            payload=payload,
            generated=generated,
            provider=str(creds.get("provider") or "xai"),
            model=selected_model,
            enable_x_search=x_search_enabled,
            response_payload=response_payload,
            now=now,
        )
    except Exception as exc:  # noqa: BLE001 - fallback must keep Signal useful.
        payload["grok_authoring"] = {
            "status": "failed",
            "provider": "xai-oauth",
            "model": selected_model,
            "x_search_enabled": x_search_enabled,
            "error_type": type(exc).__name__,
            "error": _clean_line(str(exc), 240),
            "fallback": "deterministic_from_magnus_radar",
            "generated_at": _iso(now),
        }
        payload["draft_source"] = "deterministic_from_magnus_radar"
    return payload


def _responses_payload(
    *,
    package_payload: dict[str, Any],
    model: str,
    enable_x_search: bool,
) -> dict[str, Any]:
    prompt = _authoring_prompt(package_payload)
    payload: dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": (
                    "You are Magnus, Eric Freeman's GTM/content operator. "
                    "Write sharp, useful, founder-led security/AI content. "
                    "Return only one valid JSON object. Do not use markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "store": False,
    }
    if enable_x_search:
        payload["tools"] = [{"type": "x_search"}]
    return payload


def _authoring_prompt(package_payload: dict[str, Any]) -> str:
    compact = {
        "package_handle": package_payload.get("package_handle"),
        "package_kind": package_payload.get("package_kind"),
        "sources": package_payload.get("sources") or [],
        "brief": package_payload.get("brief") or {},
        "optimization_lens": package_payload.get("optimization_lens") or {},
        "deterministic_drafts": package_payload.get("drafts") or {},
    }
    return (
        "Create an approval-ready GTM content package from this evidence. "
        "Use the supplied source links and titles. If X Search is available, use it only for current social context "
        "around the topic; do not invent facts from X. Keep public posting blocked. "
        "Voice: sharp, direct, specific, founder/operator, lightly crass when the weak idea deserves it, never a "
        "personal dunk. The output must be actionable from Signal, not a placeholder.\n\n"
        "Return JSON with exactly these top-level keys:\n"
        "- article: {title, dek, hook, sections:[{heading,draft}], close, source_links:[{title,url}]}\n"
        "- linkedin_post: {opener, body, source_links:[{title,url}]}\n"
        "- x_thread: {posts:[string], source_links:[{title,url}]}\n"
        "- x_single_post: {body, source_links:[{title,url}]}\n"
        "- visual_plan: {summary, image_prompt, alt_text, components:[string]}\n"
        "- grok_notes: {angle, x_search_used, confidence, risks:[string]}\n\n"
        "Constraints:\n"
        "- Every draft is approval-required; do not ask to post.\n"
        "- LinkedIn should be complete enough to edit and send.\n"
        "- X thread should be 5 to 8 posts, each concise.\n"
        "- Include source links where available.\n"
        "- Use optimization_lens as the X distribution pressure-test: improve hooks, replies, repost/quote value, "
        "profile-click intent, dwell, and follow intent while avoiding not-interested, block, mute, and report signals.\n"
        "- Do not claim exact X ranking weights or private production behavior; treat the public xai-org/x-algorithm repo as directional evidence.\n"
        "- If evidence is thin, say what is missing inside grok_notes.risks rather than fabricating.\n\n"
        f"Package JSON:\n{json.dumps(compact, ensure_ascii=False, sort_keys=True)}"
    )


def _merge_grok_output(
    *,
    payload: dict[str, Any],
    generated: dict[str, Any],
    provider: str,
    model: str,
    enable_x_search: bool,
    response_payload: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    draft_keys = ("article", "linkedin_post", "x_thread", "x_single_post", "visual_plan")
    drafts = payload.get("drafts") if isinstance(payload.get("drafts"), dict) else {}
    merged_drafts = copy.deepcopy(drafts)
    for key in draft_keys:
        value = generated.get(key)
        if not isinstance(value, dict):
            continue
        normalized = _normalize_generated_asset(key, value, fallback=merged_drafts.get(key))
        normalized["status"] = "approval_required"
        normalized["authoring_provider"] = provider
        normalized["authoring_model"] = model
        normalized["authoring_mode"] = "grok_with_x_search" if enable_x_search else "grok"
        if key == "visual_plan":
            payload["visual_plan"] = normalized
            merged_drafts["visual_plan"] = normalized
        else:
            merged_drafts[key] = normalized
    payload["drafts"] = merged_drafts
    if isinstance(generated.get("grok_notes"), dict):
        payload["grok_notes"] = generated["grok_notes"]
    payload["grok_authoring"] = {
        "status": "success",
        "provider": provider,
        "model": model,
        "x_search_enabled": enable_x_search,
        "x_search_citations": _extract_citations(response_payload),
        "generated_at": _iso(now),
    }
    payload["draft_source"] = "grok"
    payload["content_package_status"] = "approval_required"
    payload["public_actions_taken"] = 0
    payload["external_mutations"] = 0
    return payload


def _normalize_generated_asset(key: str, value: dict[str, Any], *, fallback: Any) -> dict[str, Any]:
    result = copy.deepcopy(value)
    if key == "x_thread":
        posts = result.get("posts")
        if not isinstance(posts, list) or not any(str(item or "").strip() for item in posts):
            fallback_posts = fallback.get("posts") if isinstance(fallback, dict) else []
            result["posts"] = fallback_posts if isinstance(fallback_posts, list) else []
    elif key == "article":
        sections = result.get("sections")
        if not isinstance(sections, list) or not sections:
            fallback_sections = fallback.get("sections") if isinstance(fallback, dict) else []
            result["sections"] = fallback_sections if isinstance(fallback_sections, list) else []
    elif key in {"linkedin_post", "x_single_post"}:
        if not str(result.get("body") or "").strip() and isinstance(fallback, dict):
            result["body"] = fallback.get("body", "")
    elif key == "visual_plan":
        components = result.get("components")
        if not isinstance(components, list) and isinstance(fallback, dict):
            result["components"] = fallback.get("components", [])
    return result


def _extract_response_text(payload: dict[str, Any]) -> str:
    output_text = str(payload.get("output_text") or "").strip()
    if output_text:
        return output_text
    parts: list[str] = []
    for item in payload.get("output", []) or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            if content.get("type") in {"output_text", "text"}:
                text = str(content.get("text") or "").strip()
                if text:
                    parts.append(text)
    return "\n\n".join(parts).strip()


def _extract_citations(payload: dict[str, Any]) -> list[dict[str, str]]:
    citations: list[dict[str, str]] = []
    seen = set()
    for item in payload.get("citations", []) or []:
        if isinstance(item, str):
            url = item.strip()
            title = ""
        elif isinstance(item, dict):
            url = str(item.get("url") or "").strip()
            title = str(item.get("title") or "").strip()
        else:
            continue
        if url and url not in seen:
            citations.append({"url": url, "title": title})
            seen.add(url)
    for item in payload.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            for annotation in content.get("annotations", []) or []:
                if not isinstance(annotation, dict) or annotation.get("type") != "url_citation":
                    continue
                url = str(annotation.get("url") or "").strip()
                if url and url not in seen:
                    citations.append({"url": url, "title": str(annotation.get("title") or "").strip()})
                    seen.add(url)
    return citations


def _parse_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if not text:
        raise RuntimeError("Grok response was empty")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise RuntimeError("Grok response did not contain a JSON object") from None
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise RuntimeError("Grok response JSON must be an object")
    return payload


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _truthy_env(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _clean_line(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
