from __future__ import annotations

import json
from typing import Any

from story_automator.core.runtime_policy import PolicyError, load_runtime_policy, parser_runtime_config, step_contract
from story_automator.core.utils import COMMAND_TIMEOUT_EXIT, extract_json_line, print_json, read_text, run_cmd, trim_lines


def parse_output_action(args: list[str]) -> int:
    if len(args) < 2:
        print('{"status":"error","reason":"output file not found or empty"}')
        return 1
    output_file, step = args[:2]
    state_file = ""
    idx = 2
    while idx < len(args):
        if args[idx] == "--state-file":
            if idx + 1 >= len(args) or not args[idx + 1].strip() or args[idx + 1].startswith("--"):
                print_json({"status": "error", "reason": "parse_contract_invalid"})
                return 1
            state_file = args[idx + 1]
            idx += 2
            continue
        idx += 1
    try:
        content = read_text(output_file)
    except FileNotFoundError:
        print('{"status":"error","reason":"output file not found or empty"}')
        return 1
    if not content.strip():
        print('{"status":"error","reason":"output file not found or empty"}')
        return 1
    lines = trim_lines(content)[:150]
    try:
        policy = load_runtime_policy(state_file=state_file)
        contract = step_contract(policy, step)
        parse_contract = _load_parse_contract(contract)
        parser_cfg = parser_runtime_config(policy)
    except (FileNotFoundError, json.JSONDecodeError, ValueError, PolicyError):
        print_json({"status": "error", "reason": "parse_contract_invalid"})
        return 1
    prompt = _build_parse_prompt(contract, parse_contract, "\n".join(lines))
    result = run_cmd(
        str(parser_cfg["provider"]),
        "-p",
        "--model",
        str(parser_cfg["model"]),
        prompt,
        env={"STORY_AUTOMATOR_CHILD": "true", "CLAUDECODE": ""},
        timeout=int(parser_cfg["timeoutSeconds"]),
    )
    if result.exit_code != 0:
        reason = "sub-agent call timed out" if result.exit_code == COMMAND_TIMEOUT_EXIT else "sub-agent call failed"
        print_json({"status": "error", "reason": reason})
        return 1
    json_line = extract_json_line(result.output)
    if not json_line:
        print_json({"status": "error", "reason": "sub-agent returned invalid json"})
        return 1
    try:
        payload = json.loads(json_line)
    except json.JSONDecodeError:
        print_json({"status": "error", "reason": "sub-agent returned invalid json"})
        return 1
    if not _has_required_keys(payload, parse_contract.get("requiredKeys") or []):
        print_json({"status": "error", "reason": "sub-agent returned invalid json"})
        return 1
    if not _matches_schema(payload, parse_contract.get("schema") or {}):
        print_json({"status": "error", "reason": "sub-agent returned invalid json"})
        return 1
    print(json.dumps(payload, separators=(",", ":")))
    return 0


def _load_parse_contract(contract: dict[str, object]) -> dict[str, object]:
    parse = contract.get("parse") or {}
    payload = json.loads(read_text(str(parse.get("schemaPath") or "")))
    if not isinstance(payload, dict):
        raise ValueError("invalid parse schema")
    required_keys = payload.get("requiredKeys")
    if not isinstance(required_keys, list):
        raise ValueError("invalid parse schema")
    if any(not isinstance(key, str) or not key.strip() for key in required_keys):
        raise ValueError("invalid parse schema")
    if not isinstance(payload.get("schema"), dict):
        raise ValueError("invalid parse schema")
    return payload


def _build_parse_prompt(contract: dict[str, object], parse_contract: dict[str, object], content: str) -> str:
    label = str(contract.get("label") or "session")
    schema = json.dumps(parse_contract.get("schema") or {}, separators=(",", ":"))
    return f"Analyze this {label} session output. Return JSON only:\n{schema}\n\nSession output:\n---\n{content}\n---"


def _has_required_keys(payload: object, required_keys: list[Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    return all(isinstance(key, str) and key in payload for key in required_keys)


def _matches_schema(payload: object, schema: object) -> bool:
    if isinstance(schema, dict):
        if not isinstance(payload, dict):
            return False
        for key, child_schema in schema.items():
            if key not in payload or not _matches_schema(payload[key], child_schema):
                return False
        return True
    if not isinstance(schema, str):
        return False
    rule = schema.strip()
    if rule == "integer":
        return isinstance(payload, int) and not isinstance(payload, bool)
    if rule == "true|false":
        return isinstance(payload, bool)
    if rule == "path or null":
        return payload is None or (isinstance(payload, str) and bool(payload.strip()))
    if "|" in rule and " " not in rule:
        return isinstance(payload, str) and payload in rule.split("|")
    return isinstance(payload, str) and bool(payload.strip())
