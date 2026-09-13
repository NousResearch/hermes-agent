"""Strict Hermes/Yeoman A2A profile loading and validation.

The JSON files in the separate contract repository are the source of truth. The
Hermes plugin resolves that checkout from ``A2A_CONTRACTS_PATH`` or the local
Hermes data directory, so development can use a checkout while production can
point at a pinned package/checkout.
"""

from __future__ import annotations

import json
import logging
import os
import tomllib
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CONTRACT_VERSION = "1.0.1"
PROFILE_URI = "urn:hermes-yeoman:a2a-profile:v1"
JSON_MEDIA_TYPE = "application/json"

_COMMON = "common"
_SKILL_REQUESTS = {
    "conversation": "skills/conversation/request.schema.json",
    "whatsapp.send": "skills/whatsapp.send/request.schema.json",
    "media.voice.generate": "skills/media.voice.generate/request.schema.json",
    "search.web": "skills/search.web/request.schema.json",
    "research.deep": "skills/research.deep/request.schema.json",
}
_SKILL_RESPONSES = {
    "conversation": "skills/conversation/response.schema.json",
    "whatsapp.send": "skills/whatsapp.send/response.schema.json",
    "media.voice.generate": "skills/media.voice.generate/response.schema.json",
    "search.web": "skills/search.web/response.schema.json",
    "research.deep": "skills/research.deep/response.schema.json",
}


class ContractViolation(ValueError):
    """A payload violates the Hermes/Yeoman profile or a skill schema."""


def _configured_contract_path() -> str:
    value = os.getenv("A2A_CONTRACTS_PATH", "").strip()
    if value:
        return value
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        candidates = [
            cfg.get("a2a_contracts_path"),
            ((cfg.get("platforms") or {}).get("a2a") or {}).get("contracts_path"),
            ((cfg.get("gateway") or {}).get("platforms") or {}).get("a2a", {}).get("contracts_path")
            if isinstance(((cfg.get("gateway") or {}).get("platforms") or {}).get("a2a"), dict) else None,
        ]
        for candidate in candidates:
            if isinstance(candidate, str) and candidate:
                return str(candidate)
    except Exception:
        logger.debug("A2A: could not read contract path from config", exc_info=True)
    return ""


def _contract_roots() -> list[Path]:
    roots: list[Path] = []
    configured = _configured_contract_path()
    if configured:
        roots.append(Path(configured).expanduser())
    try:
        import a2a_contracts  # type: ignore
        package_file = getattr(a2a_contracts, "__file__", None)
        if isinstance(package_file, str):
            package_dir = Path(package_file).resolve().parent
            roots.extend((package_dir, package_dir.parents[1]))
    except Exception:
        pass
    # Production consumes the lock-pinned package above. This checkout is only
    # a backwards-compatible local fallback; an explicit configured path is the
    # supported development override.
    roots.append(Path.home() / ".hermes" / "a2a-contracts")
    return roots


def _assert_contract_version(root: Path) -> None:
    manifest = root / "pyproject.toml"
    if not manifest.is_file():
        return  # installed wheels carry their version in the package metadata
    try:
        version = str(tomllib.loads(manifest.read_text(encoding="utf-8")).get("project", {}).get("version", ""))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ContractViolation(f"cannot read contract version from {manifest}: {exc}") from exc
    if version != CONTRACT_VERSION:
        raise ContractViolation(f"unsupported Hermes/Yeoman contract version {version!r}; expected {CONTRACT_VERSION}")


def contract_root() -> Path:
    for root in _contract_roots():
        if (root / "schemas" / "common" / "invocation.schema.json").is_file():
            _assert_contract_version(root)
            return root
    raise ContractViolation(
        f"Hermes/Yeoman contract v{CONTRACT_VERSION} is not installed; set A2A_CONTRACTS_PATH to a pinned checkout"
    )


def _schema_path(relative_path: str) -> Path:
    path = contract_root() / "schemas" / relative_path
    if not path.is_file():
        raise ContractViolation(f"contract schema not found: {relative_path}")
    return path


def _load(relative_path: str) -> dict[str, Any]:
    try:
        return json.loads(_schema_path(relative_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractViolation(f"cannot load contract schema {relative_path}: {exc}") from exc


def _validator(relative_path: str):
    try:
        from jsonschema import Draft202012Validator, FormatChecker
        from referencing import Registry, Resource  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - dependency metadata catches this
        raise ContractViolation("Hermes A2A contract validation requires jsonschema") from exc
    schema = _load(relative_path)
    registry = Registry()
    for path in (contract_root() / "schemas").rglob("*.schema.json"):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            identifier = document.get("$id")
            if identifier:
                registry = registry.with_resource(identifier, Resource.from_contents(document))
        except Exception:
            # The requested schema will produce the actionable error below. A
            # malformed unrelated file must not make every request uninspectable.
            continue
    return Draft202012Validator(schema, registry=registry, format_checker=FormatChecker())


def validate(instance: Any, relative_path: str) -> None:
    validator = _validator(relative_path)
    errors = sorted(validator.iter_errors(instance), key=lambda error: list(error.path))
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.path) or "$"
        raise ContractViolation(f"{relative_path}: {location}: {error.message}")


def validate_invocation(invocation: Any) -> None:
    validate(invocation, f"{_COMMON}/invocation.schema.json")


def validate_skill_request(skill: str, input_data: Any) -> None:
    path = _SKILL_REQUESTS.get(skill)
    if not path:
        raise ContractViolation(f"unknown contract skill: {skill}")
    validate(input_data, path)


def validate_skill_response(skill: str, output: Any) -> None:
    path = _SKILL_RESPONSES.get(skill)
    if not path:
        raise ContractViolation(f"unknown contract skill: {skill}")
    validate(output, path)


def validate_result(result: Any) -> None:
    """Validate the common result envelope and its skill-specific output."""
    if not isinstance(result, dict):
        raise ContractViolation("profile result must be an object")
    validate(result, f"{_COMMON}/result.schema.json")
    skill = str(result.get("skill") or "")
    output = result.get("output")
    if output is not None:
        validate_skill_response(skill, output)


def extract_invocation(message: dict[str, Any]) -> dict[str, Any] | None:
    """Extract and validate the single authoritative JSON DataPart.

    ``None`` means the message is unstructured conversation text. A JSON data
    part is never silently downgraded to text.
    """
    parts = message.get("parts", []) if isinstance(message, dict) else []
    if not isinstance(parts, list):
        raise ContractViolation("A2A message.parts must be an array")
    json_parts = [part for part in parts if isinstance(part, dict) and part.get("mediaType") == JSON_MEDIA_TYPE]
    malformed_data_parts = [
        part for part in parts
        if isinstance(part, dict) and "data" in part and part.get("mediaType") != JSON_MEDIA_TYPE
    ]
    if malformed_data_parts:
        raise ContractViolation("structured DataPart must use mediaType application/json")
    if len(json_parts) > 1:
        raise ContractViolation("exactly one application/json DataPart is allowed")
    if not json_parts:
        return None
    if "data" not in json_parts[0]:
        raise ContractViolation("application/json DataPart must contain data")
    data = json_parts[0].get("data")
    if not isinstance(data, dict):
        raise ContractViolation("application/json DataPart.data must be an object")
    validate_invocation(data)
    validate_skill_request(str(data["skill"]), data["input"])
    return data


def json_data_part(data: dict[str, Any]) -> dict[str, Any]:
    return {"data": data, "mediaType": JSON_MEDIA_TYPE}


def _correlation(task_id: str, context_id: str, reference_task_ids: list[str] | None = None) -> dict[str, Any]:
    correlation = {"task_id": task_id, "context_id": context_id}
    if reference_task_ids:
        correlation["reference_task_ids"] = list(reference_task_ids)
    return correlation


def result_for_reply(skill: str, reply: str, *, task_id: str = "", context_id: str = "",
                     reference_task_ids: list[str] | None = None) -> dict[str, Any]:
    """Map the live Hermes reply into the v1 skill result shape.

    Hermes' live agent remains responsible for doing the actual work. This
    adapter supplies a schema-valid envelope without inventing citations or
    delivery claims.
    """
    if skill == "conversation":
        output = {"text": reply or ""}
    elif skill == "search.web":
        output = {"results": [], "answer": reply or ""}
    elif skill == "research.deep":
        output = {"report": reply or "", "sources": []}
    else:
        raise ContractViolation(f"Hermes cannot execute skill {skill}")
    validate_skill_response(skill, output)
    result = {"skill": skill, "status": "completed", "output": output}
    if task_id and context_id:
        result["correlation"] = _correlation(task_id, context_id, reference_task_ids)
    validate_result(result)
    return result


def result_for_error(skill: str, code: str, message: str, retryable: bool = False, *, task_id: str = "",
                     context_id: str = "", reference_task_ids: list[str] | None = None,
                     status: str = "failed") -> dict[str, Any]:
    error = {"code": code, "message": message, "retryable": retryable}
    result = {"skill": skill, "status": status, "error": error}
    if task_id and context_id:
        result["correlation"] = _correlation(task_id, context_id, reference_task_ids)
    validate_result(result)
    return result


def advertised_skills(*, include_voice: bool = False) -> list[dict[str, Any]]:
    skills = [
        {
            "id": "conversation",
            "name": "Conversation",
            "description": "Free-form conversation without implicit external side effects.",
            "tags": ["conversation"],
            "inputModes": ["application/json", "text/plain"],
            "outputModes": ["application/json", "text/plain"],
        },
        {
            "id": "search.web",
            "name": "Web search",
            "description": "Search the web with Hermes-configured providers.",
            "tags": ["search", "web"],
            "inputModes": ["application/json"],
            "outputModes": ["application/json", "text/markdown"],
        },
        {
            "id": "research.deep",
            "name": "Deep research",
            "description": "Run a potentially long-running research task.",
            "tags": ["research", "async"],
            "inputModes": ["application/json"],
            "outputModes": ["application/json", "text/markdown"],
        },
    ]
    if include_voice:
        skills.append({
            "id": "media.voice.generate",
            "name": "Voice generation",
            "description": "Generate a voice artifact when a real Hermes TTS capability is enabled.",
            "tags": ["voice", "audio"],
            "inputModes": ["application/json"],
            "outputModes": ["application/json", "audio/ogg"],
        })
    return skills


__all__ = [
    "CONTRACT_VERSION",
    "PROFILE_URI",
    "JSON_MEDIA_TYPE",
    "ContractViolation",
    "advertised_skills",
    "contract_root",
    "extract_invocation",
    "json_data_part",
    "result_for_error",
    "result_for_reply",
    "validate_result",
    "validate_invocation",
    "validate_skill_request",
    "validate_skill_response",
]
