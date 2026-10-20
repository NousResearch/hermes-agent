"""Preview Translator backend — mounted at /api/plugins/preview-translator/.

Translates webpage text-node batches through Hermes' auxiliary LLM client so
the user can pick any configured model. Runs inside the gateway process.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

BATCH_SIZE = 32
MAX_TEXTS = 400


class TranslateRequest(BaseModel):
    texts: list[str] = Field(default_factory=list)
    target_lang: str = "English"
    model: Optional[str] = None
    provider: Optional[str] = None


class TranslateResponse(BaseModel):
    translations: list[str]
    model_used: str


def _load_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly

        return load_config_readonly() or {}
    except Exception:
        return {}


def _model_entries() -> list[dict[str, str]]:
    """Build a picker list from the live config: auto + every named model."""
    cfg = _load_config()
    seen: set[str] = set()
    models: list[dict[str, str]] = [{"id": "auto", "name": "Auto (session / main model)"}]
    seen.add("auto")

    default = (cfg.get("model") or {}) if isinstance(cfg.get("model"), dict) else {}
    default_model = str(default.get("default") or default.get("model") or "").strip()
    default_provider = str(default.get("provider") or "").strip()
    if default_model:
        mid = f"{default_provider}/{default_model}" if default_provider else default_model
        if mid not in seen:
            models.append({"id": mid, "name": f"Main · {default_model}"})
            seen.add(mid)

    providers = cfg.get("providers") or {}
    if isinstance(providers, dict):
        for prov_name, prov_cfg in providers.items():
            if not isinstance(prov_cfg, dict):
                continue
            p_models = prov_cfg.get("models")
            names: list[str] = []
            if isinstance(p_models, list):
                names = [str(m) for m in p_models]
            elif isinstance(p_models, dict):
                names = [str(m) for m in p_models.keys()]
            default_m = str(prov_cfg.get("default_model") or prov_cfg.get("model") or "").strip()
            if default_m:
                names.insert(0, default_m)
            for m in names:
                m = m.strip()
                if not m:
                    continue
                mid = f"{prov_name}/{m}"
                if mid in seen or m in seen:
                    continue
                models.append({"id": mid, "name": f"{prov_name} / {m}"})
                seen.add(mid)

    custom = cfg.get("custom_providers") or []
    if isinstance(custom, list):
        for prov in custom:
            if not isinstance(prov, dict):
                continue
            name = str(prov.get("name") or "").strip()
            model = str(prov.get("model") or "").strip()
            if not name or not model:
                continue
            mid = f"{name}/{model}"
            if mid in seen:
                continue
            models.append({"id": mid, "name": f"{name} / {model}"})
            seen.add(mid)

    return models


@router.get("/models")
async def list_available_models():
    return {"models": _model_entries()}


def _parse_model(model_id: Optional[str], provider: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Return (provider, model) from a picker id like 'gemini/gemini-flash-latest'."""
    raw = (model_id or "").strip()
    if not raw or raw == "auto":
        return None, None
    if provider:
        return provider.strip(), raw.split("/", 1)[-1]
    if "/" in raw:
        prov, model = raw.split("/", 1)
        return prov.strip() or None, model.strip() or None
    return None, raw


def _extract_json_array(raw: str) -> list[Any]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for key in ("translations", "items", "result"):
                val = parsed.get(key)
                if isinstance(val, list):
                    return val
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[[\s\S]*\]", cleaned)
    if match:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, list):
            return parsed
    raise ValueError("model did not return a JSON array")


def _message_text(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except Exception as exc:
        raise ValueError(f"unexpected LLM response shape: {exc}") from exc
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(str(part["text"]))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content or "")


def _translate_unique(texts: list[str], target_lang: str, provider: Optional[str], model: Optional[str]) -> list[str]:
    from agent.auxiliary_client import call_llm

    system_prompt = (
        f"You are a professional webpage translator. Translate each string into {target_lang}.\n"
        "Rules:\n"
        f"1. Return a JSON array of EXACTLY {len(texts)} strings, same order as the input.\n"
        "2. Preserve numbers, dates, punctuation, brand names, and code-like tokens.\n"
        "3. Do not add explanations, markdown fences, or extra keys."
    )
    kwargs: dict[str, Any] = {
        "task": "translation",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(texts, ensure_ascii=False)},
        ],
        "max_tokens": 4096,
        "timeout": 90.0,
        "reasoning_config": {"enabled": False},
    }
    if provider:
        kwargs["provider"] = provider
    if model:
        kwargs["model"] = model

    response = call_llm(**kwargs)
    parsed = _extract_json_array(_message_text(response))
    if len(parsed) != len(texts):
        raise ValueError(f"expected {len(texts)} translations, got {len(parsed)}")
    return [str(item) for item in parsed]


def _translate_all(texts: list[str], target_lang: str, provider: Optional[str], model: Optional[str]) -> list[str]:
    """Dedup, batch, translate, then restore original order."""
    index_of: dict[str, int] = {}
    unique: list[str] = []
    mapping: list[int] = []
    for text in texts:
        if text not in index_of:
            index_of[text] = len(unique)
            unique.append(text)
        mapping.append(index_of[text])

    translated_unique: list[str] = []
    for start in range(0, len(unique), BATCH_SIZE):
        chunk = unique[start : start + BATCH_SIZE]
        translated_unique.extend(_translate_unique(chunk, target_lang, provider, model))

    return [translated_unique[i] for i in mapping]


@router.post("/translate", response_model=TranslateResponse)
async def translate_page_texts(req: TranslateRequest):
    if not req.texts:
        return TranslateResponse(translations=[], model_used=req.model or "auto")
    if len(req.texts) > MAX_TEXTS:
        raise HTTPException(status_code=400, detail=f"too many strings (max {MAX_TEXTS})")

    provider, model = _parse_model(req.model, req.provider)
    model_used = f"{provider}/{model}" if provider and model else (model or req.model or "auto")

    try:
        translations = await asyncio.to_thread(
            _translate_all, req.texts, req.target_lang, provider, model
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("preview-translator failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)[:500]) from exc

    return TranslateResponse(translations=translations, model_used=model_used)
