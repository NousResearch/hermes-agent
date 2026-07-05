"""TaskContract Builder for Task Runtime.

Builds a TaskContract v1.0.0 dict from ResolvedIntent + context + skills.

The contract is the immutable input to the ExecutionPipeline. It does not
contain any I/O state; it is purely declarative metadata.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any


CONTRACT_SCHEMA = "1.0.0"
CONTRACT_VERSION = 1
SCHEMA_VERSION = 1


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(s: str) -> str:
    return "sha256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()


def build(
    resolved_intent,
    context: dict[str, Any],
    skills: list,
    execution_mode: str = "dry-run",
) -> dict[str, Any]:
    """Build a TaskContract v1.0.0 dict.

    Returns a plain dict (no I/O). Caller may serialize / hash / persist.
    """
    contract_id = f"tc-{uuid.uuid4().hex[:12]}"

    # Provider/model defaults are conservative: use the user-configured main model
    # if discoverable via env; otherwise MiniMax-M3 (the user's documented default).
    producer_provider = "minimax"
    producer_model = "MiniMax-M3"
    producer_api_mode = "chat_completions"

    reviewer_provider = "openai-codex"
    reviewer_model = "gpt-5.5"
    reviewer_enabled = True

    # In MVP, dry-run mode disables HTTP-touching components by setting
    # execution flags; this is the contract-level hint.
    producer_http_in_dry_run = False
    reviewer_http_in_dry_run = False
    if execution_mode == "dry-run":
        producer_http_in_dry_run = True   # documentation only
        reviewer_http_in_dry_run = True  # documentation only

    contract_no_id = {
        "task_contract_schema": CONTRACT_SCHEMA,
        "task_contract_version": CONTRACT_SCHEMA,
        "contract_id_field_placeholder": True,  # replaced after fingerprint
        "intent": {
            "raw_text": resolved_intent.raw_text,
            "source": resolved_intent.source,
            "source_id": resolved_intent.source_id,
            "task_type": resolved_intent.task_type,
            "intent_id": resolved_intent.intent_id,
        },
        "context": {
            "hermes_home": context.get("hermes_home"),
            "hermes_home_display": context.get("hermes_home_display"),
            "vault_root": context.get("vault_root"),
            "skills_dir": context.get("skills_dir"),
            "kanban_board": context.get("kanban_board"),
            "config_version": context.get("config_version"),
            "hermes_disable_self_improvement": context.get("hermes_disable_self_improvement", True),
            "knowledge_refs": context.get("knowledge_refs", []),
            "memory_keys": context.get("memory_keys", []),
        },
        "skills": [
            {
                "skill_name": s.skill_name,
                "description": s.description,
                "installed": s.installed,
                "skill_path": s.skill_path,
                "source": s.source,
            }
            for s in skills
        ],
        "producer": {
            "provider": producer_provider,
            "model": producer_model,
            "api_mode": producer_api_mode,
            "execution_mode": execution_mode,
            "http_call_in_mode": producer_http_in_dry_run if execution_mode == "dry-run" else True,
        },
        "normalizer": {
            "enabled": True,
            "version": "1.1.0",
            "shadow_mode": execution_mode in ("dry-run", "shadow"),
            "ruleset_path": "tests/contract_tests/normalizer_ruleset.v1.1.0.yaml",
            "config_path": "tests/contract_tests/normalizer_config.v1.1.0.yaml",
        },
        "reviewer": {
            "enabled": reviewer_enabled,
            "version": "1.0.0",
            "provider": reviewer_provider,
            "model": reviewer_model,
            "execution_mode": execution_mode,
            "http_call_in_mode": reviewer_http_in_dry_run if execution_mode == "dry-run" else True,
        },
        "acceptance_criteria": [],
        "stop_conditions": [],
        "execution_mode": execution_mode,
        "metadata": resolved_intent.metadata,
        "contract_version": CONTRACT_VERSION,
        "schema_version": SCHEMA_VERSION,
    }

    # Compute fingerprint on the contract body WITHOUT the contract_id and
    # without the placeholder field.
    fingerprint_body = {k: v for k, v in contract_no_id.items() if k not in ("contract_id_field_placeholder",)}
    contract_fingerprint = _sha256(_canonical_json(fingerprint_body))
    contract_no_id["contract_fingerprint"] = contract_fingerprint
    contract_no_id["contract_id"] = contract_id
    del contract_no_id["contract_id_field_placeholder"]

    return contract_no_id