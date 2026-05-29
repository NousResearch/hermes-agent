"""AlphaHunt local-Qwen analysis stage handler."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import time
from typing import Any, Dict, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

STAGE_MODES = frozenset({"cleaner", "screener", "sentinel", "packager"})
QWEN_MODES = STAGE_MODES | frozenset({"fast_triage"})
QWEN_ANALYZER = "qwen_7b_local"
QWEN_MODEL = "qwen2.5:7b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_QWEN_TIMEOUT_SEC = 90
DEFAULT_QWEN_NUM_PREDICT = 1024
DEFAULT_QWEN_NUM_CTX = 4096

PROMPTS: Dict[str, str] = {
    "cleaner": (
        "你只负责把 raw_snapshot 清洗成 normalized_event。保留 source、asset_class、"
        "identity_key、action_window、关键字段与显式 data gap。禁止给投资结论，禁止补脑补字段。"
    ),
    "screener": (
        "你只负责把 normalized_event 转成 opportunity_candidate。必须拆成 base_fields、"
        "asset_specific_fields、decision_fields，不得越权输出 green/act_now。"
    ),
    "sentinel": (
        "你只负责阻断。检查 Web3、A股/ETF、港股打新、美债、黄金的 blocking rules。"
        "risk_veto.active=true 时必须阻断 green/act_now。"
    ),
    "packager": (
        "你是 Agent 4 打包员。输入是前三段产出（cleaner_output_v1、screener_output_v1、"
        "sentinel_output_v1）。你只做拼装：把三段事实层结论合成 hermes_decision_object_v2 "
        "解释层需要的 context_packet（normalized_event + opportunity_candidate? + risk_review）。"
        "禁止新增事实、禁止给 action_color/recommended_action、禁止覆盖 sentinel 的 blocking risk。"
        "输出必须严格符合 packager_output_v1。"
    ),
    "fast_triage": (
        "你是一过筛 (fast_triage) 员。输入是 raw_snapshot 或 normalized_event。你只判断该事件"
        "是否有进入 4-stage agent chain 的价值。输出严格 JSON。"
    ),
}

SCHEMA_VERSION_BY_MODE = {
    "cleaner": "cleaner_output_v1",
    "screener": "screener_output_v1",
    "sentinel": "sentinel_output_v1",
    "packager": "packager_output_v1",
    "fast_triage": "fast_triage_v1",
}

OUTPUT_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "cleaner": {
        "type": "object",
        "required": ["status"],
        "properties": {
            "status": {"enum": ["ok", "rejected", "error"]},
            "reason": {"type": "string"},
            "normalized_event": {
                "type": "object",
                "required": ["event_id", "asset_class", "event_type", "source", "normalized_fields"],
                "properties": {
                    "event_id": {"type": "string", "minLength": 1},
                    "asset_class": {"type": "string", "minLength": 1},
                    "event_type": {"type": "string", "minLength": 1},
                    "source": {"type": "string", "minLength": 1},
                    "normalized_fields": {"type": "object", "minProperties": 1},
                },
            },
            "data_gaps": {"type": "array"},
        },
        "allOf": [
            {
                "if": {"properties": {"status": {"const": "ok"}}},
                "then": {"required": ["normalized_event", "data_gaps"]},
            }
        ],
    },
    "screener": {
        "type": "object",
        "required": ["status"],
        "properties": {
            "status": {"enum": ["ok", "rejected", "error"]},
            "reason": {"type": "string"},
            "opportunity_candidate": {
                "type": "object",
                "required": ["opportunity_id", "asset_class", "base_fields", "asset_specific_fields", "decision_fields"],
                "properties": {
                    "opportunity_id": {"type": "string", "minLength": 1},
                    "asset_class": {"type": "string", "minLength": 1},
                    "base_fields": {"type": "object", "minProperties": 1},
                    "asset_specific_fields": {"type": "object", "minProperties": 1},
                    "decision_fields": {"type": "object", "minProperties": 1},
                },
            },
        },
        "allOf": [
            {
                "if": {"properties": {"status": {"const": "ok"}}},
                "then": {"required": ["opportunity_candidate"]},
            }
        ],
    },
    "sentinel": {
        "type": "object",
        "required": ["status"],
        "properties": {
            "status": {"enum": ["ok", "rejected", "error"]},
            "reason": {"type": "string"},
            "risks": {"type": "array"},
            "risk_veto": {
                "type": "object",
                "required": ["active"],
                "properties": {"active": {"type": "boolean"}, "reason": {"type": "string"}},
            },
            "blocking_rules": {"type": "array"},
        },
        "allOf": [
            {
                "if": {"properties": {"status": {"const": "ok"}}},
                "then": {"required": ["risks", "risk_veto"]},
            }
        ],
    },
    "packager": {
        "type": "object",
        "required": ["status"],
        "properties": {
            "status": {"enum": ["ok", "rejected", "error"]},
            "reason": {"type": "string"},
            "context_packet": {
                "type": "object",
                "required": ["normalized_event", "risk_review"],
                "properties": {
                    "normalized_event": {
                        "type": "object",
                        "required": ["event_id", "asset_class", "event_type", "source", "normalized_fields"],
                        "properties": {
                            "event_id": {"type": "string", "minLength": 1},
                            "asset_class": {"type": "string", "minLength": 1},
                            "event_type": {"type": "string", "minLength": 1},
                            "source": {"type": "string", "minLength": 1},
                            "normalized_fields": {"type": "object", "minProperties": 1},
                        },
                    },
                    "opportunity_candidate": {
                        "type": "object",
                        "required": [
                            "opportunity_id",
                            "asset_class",
                            "base_fields",
                            "asset_specific_fields",
                            "decision_fields",
                        ],
                        "properties": {
                            "opportunity_id": {"type": "string", "minLength": 1},
                            "asset_class": {"type": "string", "minLength": 1},
                            "base_fields": {"type": "object", "minProperties": 1},
                            "asset_specific_fields": {"type": "object", "minProperties": 1},
                            "decision_fields": {"type": "object", "minProperties": 1},
                        },
                    },
                    "risk_review": {
                        "type": "object",
                        "required": ["risks", "risk_veto"],
                        "properties": {
                            "risks": {"type": "array"},
                            "risk_veto": {
                                "type": "object",
                                "required": ["active"],
                                "properties": {"active": {"type": "boolean"}, "reason": {"type": "string"}},
                            },
                            "blocking_rules": {"type": "array"},
                        },
                    },
                },
            },
        },
        "allOf": [
            {
                "if": {"properties": {"status": {"const": "ok"}}},
                "then": {"required": ["context_packet"]},
            }
        ],
    },
    "fast_triage": {
        "type": "object",
        "required": ["status", "triage_decision", "reason", "matched_signals", "data_gap"],
        "properties": {
            "status": {"const": "ok"},
            "triage_decision": {"enum": ["advance", "reject", "needs_human"]},
            "reason": {"type": "string"},
            "matched_signals": {"type": "array", "items": {"type": "string"}},
            "data_gap": {"type": "array"},
        },
    },
}


def is_qwen_analysis_payload(payload: Dict[str, Any]) -> bool:
    mode = str(payload.get("analysis_mode") or "").strip().lower()
    if mode == "fast_triage":
        return True
    if mode not in STAGE_MODES:
        return False
    ctx = payload.get("context") if isinstance(payload.get("context"), dict) else {}
    routing = ctx.get("routing_policy") if isinstance(ctx.get("routing_policy"), dict) else {}
    engine = str(routing.get("preferred_engine") or "").strip().lower()
    return not engine or engine in {"qwen_local", "qwen_7b_local"} or engine.startswith("qwen")


def build_prompt(payload: Dict[str, Any], *, validation_error: str = "") -> str:
    mode = str(payload.get("analysis_mode") or "").strip().lower()
    example = expected_output_shape(mode)
    repair = ""
    if validation_error:
        repair = f"\n上一次输出未通过 schema 校验：{validation_error}\n只返回修复后的严格 JSON。"
    return (
        f"{PROMPTS[mode]}\n"
        "硬性要求：只输出一个 JSON object，不要 Markdown，不要解释文字，不要代码块。\n"
        "顶层必须包含 output；output 必须包含 stage、stage_schema_version、analyzer、model、stage_output。\n"
        f"stage 必须是 {mode}；analyzer 必须是 {QWEN_ANALYZER}；model 必须是 {QWEN_MODEL}。\n"
        f"stage_schema_version 必须是 {SCHEMA_VERSION_BY_MODE[mode]}。\n"
        f"期望形状示例：{json.dumps(example, ensure_ascii=False, separators=(',', ':'))}\n"
        f"{repair}\n"
        f"输入 payload：{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
    )


def expected_output_shape(mode: str) -> Dict[str, Any]:
    examples = {
        "cleaner": {
            "status": "ok",
            "normalized_event": {
                "event_id": "<event id>",
                "asset_class": "<asset class>",
                "event_type": "<event type>",
                "source": "<source>",
                "normalized_fields": {"identity_key": "<identity key>"},
            },
            "data_gaps": [],
        },
        "screener": {
            "status": "ok",
            "opportunity_candidate": {
                "opportunity_id": "<opportunity id>",
                "asset_class": "<asset class>",
                "base_fields": {"source_event_id": "<event id>"},
                "asset_specific_fields": {"instrument": "<instrument>"},
                "decision_fields": {"screening_decision": "<candidate|reject|needs_human>"},
            },
        },
        "sentinel": {
            "status": "ok",
            "risks": [],
            "risk_veto": {"active": False, "reason": "none"},
            "blocking_rules": [],
        },
        "packager": {
            "status": "ok",
            "context_packet": {
                "normalized_event": {
                    "event_id": "<event id>",
                    "asset_class": "<asset class>",
                    "event_type": "<event type>",
                    "source": "<source>",
                    "normalized_fields": {"identity_key": "<identity key>"},
                },
                "opportunity_candidate": {
                    "opportunity_id": "<opportunity id>",
                    "asset_class": "<asset class>",
                    "base_fields": {"source_event_id": "<event id>"},
                    "asset_specific_fields": {"instrument": "<instrument>"},
                    "decision_fields": {"screening_decision": "<candidate|reject|needs_human>"},
                },
                "risk_review": {
                    "risks": [],
                    "risk_veto": {"active": False, "reason": "none"},
                    "blocking_rules": [],
                },
            },
        },
        "fast_triage": {
            "status": "ok",
            "triage_decision": "advance",
            "reason": "<one sentence>",
            "matched_signals": [],
            "data_gap": [],
        },
    }
    return {
        "output": {
            "stage": mode,
            "stage_schema_version": SCHEMA_VERSION_BY_MODE[mode],
            "analyzer": QWEN_ANALYZER,
            "model": QWEN_MODEL,
            "stage_output": examples[mode],
        }
    }


def call_ollama(prompt: str, *, timeout: int = DEFAULT_QWEN_TIMEOUT_SEC) -> str:
    url = os.environ.get("HERMES_QWEN_OLLAMA_URL", DEFAULT_OLLAMA_URL).strip() or DEFAULT_OLLAMA_URL
    model = os.environ.get("HERMES_QWEN_MODEL", QWEN_MODEL).strip() or QWEN_MODEL
    num_predict = int(os.environ.get("HERMES_QWEN_NUM_PREDICT", str(DEFAULT_QWEN_NUM_PREDICT)) or DEFAULT_QWEN_NUM_PREDICT)
    num_ctx = int(os.environ.get("HERMES_QWEN_NUM_CTX", str(DEFAULT_QWEN_NUM_CTX)) or DEFAULT_QWEN_NUM_CTX)
    resp = requests.post(
        url,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"num_predict": num_predict, "num_ctx": num_ctx},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return str(data.get("response") or "")


def parse_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("model output must be a JSON object")
    return obj


def validate_output(mode: str, callback: Dict[str, Any]) -> Tuple[bool, str]:
    output = callback.get("output") if isinstance(callback.get("output"), dict) else None
    if output is None:
        return False, "missing output object"
    expected = {
        "stage": mode,
        "stage_schema_version": SCHEMA_VERSION_BY_MODE[mode],
        "analyzer": QWEN_ANALYZER,
        "model": QWEN_MODEL,
    }
    for key, value in expected.items():
        if output.get(key) != value:
            return False, f"output.{key} expected {value!r}"
    stage_output = output.get("stage_output")
    if not isinstance(stage_output, dict):
        return False, "output.stage_output must be object"
    try:
        from jsonschema import Draft202012Validator

        errors = sorted(Draft202012Validator(OUTPUT_SCHEMAS[mode]).iter_errors(stage_output), key=lambda e: e.path)
        if errors:
            return False, errors[0].message
    except ImportError:
        return _validate_stage_output_minimal(mode, stage_output)
    return True, ""


def _validate_stage_output_minimal(mode: str, stage_output: Dict[str, Any]) -> Tuple[bool, str]:
    status = stage_output.get("status")
    if mode == "fast_triage":
        if status != "ok":
            return False, "status must be ok"
        if stage_output.get("triage_decision") not in {"advance", "reject", "needs_human"}:
            return False, "invalid triage_decision"
        for key in ("reason", "matched_signals", "data_gap"):
            if key not in stage_output:
                return False, f"{key} is required"
        return True, ""
    if status not in {"ok", "rejected", "error"}:
        return False, "invalid status"
    if status != "ok":
        return True, ""
    required_by_mode = {
        "cleaner": ("normalized_event", "data_gaps"),
        "screener": ("opportunity_candidate",),
        "sentinel": ("risks", "risk_veto"),
        "packager": ("context_packet",),
    }
    for key in required_by_mode[mode]:
        if key not in stage_output:
            return False, f"{key} is required"
    return True, ""


def normalize_callback(payload: Dict[str, Any], model_obj: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(payload.get("analysis_mode") or "").strip().lower()
    analysis_id = str(payload.get("analysis_id") or payload.get("request_id") or "").strip()
    result = {
        "analysis_id": analysis_id,
        "output": model_obj["output"],
    }
    event_id = str(payload.get("event_id") or "").strip()
    if event_id:
        result["event_id"] = event_id
    return result


def error_callback(payload: Dict[str, Any], mode: str, reason: str) -> Dict[str, Any]:
    analysis_id = str(payload.get("analysis_id") or payload.get("request_id") or "").strip()
    result: Dict[str, Any] = {
        "analysis_id": analysis_id,
        "output": {
            "stage": mode,
            "stage_schema_version": SCHEMA_VERSION_BY_MODE[mode],
            "analyzer": QWEN_ANALYZER,
            "model": QWEN_MODEL,
            "stage_output": {"status": "error", "reason": reason},
        },
    }
    event_id = str(payload.get("event_id") or "").strip()
    if event_id:
        result["event_id"] = event_id
    return result


def run_qwen_stage(payload: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(payload.get("analysis_mode") or "").strip().lower()
    if mode not in QWEN_MODES:
        raise ValueError(f"unsupported qwen analysis_mode: {mode}")
    timeout = int(os.environ.get("HERMES_QWEN_TIMEOUT_SEC", str(DEFAULT_QWEN_TIMEOUT_SEC)) or DEFAULT_QWEN_TIMEOUT_SEC)
    validation_error = ""
    for attempt in range(2):
        try:
            raw = call_ollama(build_prompt(payload, validation_error=validation_error), timeout=timeout)
            model_obj = parse_json_object(raw)
            callback = normalize_callback(payload, model_obj)
        except requests.Timeout:
            return error_callback(payload, mode, "qwen_timeout")
        except Exception as exc:
            validation_error = f"invalid_json:{exc}"
            if attempt == 0:
                continue
            return error_callback(payload, mode, f"schema_invalid:{validation_error}")
        ok, err = validate_output(mode, callback)
        if ok:
            return callback
        validation_error = err
    return error_callback(payload, mode, f"schema_invalid:{validation_error}")


def callback_headers(body: bytes, callback_auth: str = "") -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("CENTRAL_CALLBACK_API_KEY", "").strip() or callback_auth.strip()
    if api_key:
        headers["X-API-Key"] = api_key
        headers["Authorization"] = f"Bearer {api_key}"
    secret = (
        os.environ.get("CENTRAL_CALLBACK_SECRET", "").strip()
        or os.environ.get("HERMES_CALLBACK_SECRET", "").strip()
        or os.environ.get("HERMES_DISPATCH_SECRET", "").strip()
    )
    if secret:
        ts = str(int(time.time()))
        msg = f"{ts}.".encode("utf-8") + body
        sig = hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()
        headers["X-Hermes-Timestamp"] = ts
        headers["X-Hermes-Signature"] = f"sha256={sig}"
    return headers


def post_callback(payload: Dict[str, Any], callback: Dict[str, Any], *, callback_url: str = "", callback_auth: str = "") -> None:
    url = (callback_url or payload.get("callback_url") or os.environ.get("CENTRAL_CALLBACK_URL") or "").strip()
    if not url:
        return
    auth = callback_auth or str(payload.get("callback_auth") or "")
    body = json.dumps(callback, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    resp = requests.post(url, data=body, headers=callback_headers(body, auth), timeout=15)
    resp.raise_for_status()
