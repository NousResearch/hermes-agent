#!/usr/bin/env python3
"""Generate normalized JSON Schema and desktop contracts from Pydantic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from plugins.study_os.contract_models import study_contract_json_schema


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = ROOT / "plugins" / "study_os" / "contracts" / "study-contracts.schema.json"
TYPESCRIPT_PATH = (
    ROOT
    / "apps"
    / "desktop"
    / "src"
    / "lib"
    / "generated"
    / "study-contracts.ts"
)


def _ts_literal(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _ts_type(schema: dict[str, Any]) -> str:
    reference = schema.get("$ref")
    if isinstance(reference, str):
        return reference.rsplit("/", 1)[-1]
    if "const" in schema:
        return _ts_literal(schema["const"])
    enum = schema.get("enum")
    if isinstance(enum, list):
        return " | ".join(_ts_literal(item) for item in enum) or "never"
    alternatives = schema.get("oneOf") or schema.get("anyOf")
    if isinstance(alternatives, list):
        types = list(dict.fromkeys(_ts_type(item) for item in alternatives))
        return " | ".join(types)
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        return " | ".join(_ts_type({**schema, "type": item}) for item in schema_type)
    if schema_type == "string":
        return "string"
    if schema_type in {"integer", "number"}:
        return "number"
    if schema_type == "boolean":
        return "boolean"
    if schema_type == "null":
        return "null"
    if schema_type == "array":
        item_type = _ts_type(schema.get("items") or {})
        return f"Array<{item_type}>"
    if schema_type == "object" or "properties" in schema:
        return _inline_object(schema)
    return "unknown"


def _property_name(name: str) -> str:
    return name if name.replace("_", "").isalnum() else _ts_literal(name)


def _inline_object(schema: dict[str, Any], indent: str = "") -> str:
    required = set(schema.get("required") or [])
    lines = ["{"]
    for name, value in (schema.get("properties") or {}).items():
        optional = "" if name in required else "?"
        lines.append(
            f"{indent}  {_property_name(name)}{optional}: {_ts_type(value)}"
        )
    if schema.get("additionalProperties") is True:
        lines.append(f"{indent}  [key: string]: unknown")
    lines.append(f"{indent}}}")
    return "\n".join(lines)


def _render_named_type(name: str, schema: dict[str, Any]) -> str:
    if schema.get("type") == "object" or "properties" in schema:
        body = _inline_object(schema)
        return f"export interface {name} {body}"
    return f"export type {name} = {_ts_type(schema)}"


def render_typescript(schema: dict[str, Any]) -> str:
    definitions = schema["$defs"]
    rendered_types = "\n\n".join(
        _render_named_type(name, value)
        for name, value in sorted(definitions.items())
    )
    schema_json = json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True)
    runtime = r'''
export type StudySchemaResult<T> = { ok: true; data: T } | { ok: false; errors: string[] }

type JsonSchema = Record<string, unknown>

const STUDY_CONTRACT_SCHEMA = __SCHEMA__ as unknown as JsonSchema

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function childPath(path: string, key: string): string {
  return path ? `${path}.${key}` : key
}

function schemaRecord(value: unknown): JsonSchema {
  return isRecord(value) ? value : {}
}

function resolveSchema(schema: JsonSchema): JsonSchema {
  const reference = schema.$ref
  if (typeof reference !== 'string' || !reference.startsWith('#/$defs/')) {
    return schema
  }
  const definitions = schemaRecord(STUDY_CONTRACT_SCHEMA.$defs)
  return schemaRecord(definitions[reference.slice('#/$defs/'.length)])
}

function validDate(value: string): boolean {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) {
    return false
  }
  const parsed = new Date(`${value}T00:00:00Z`)
  return !Number.isNaN(parsed.valueOf()) && parsed.toISOString().slice(0, 10) === value
}

function validDateTime(value: string): boolean {
  return !Number.isNaN(new Date(value).valueOf())
}

function validateAlternatives(
  value: unknown,
  alternatives: unknown[],
  path: string,
  errors: string[]
): boolean {
  const failures: string[][] = []
  for (const alternative of alternatives) {
    const branchErrors: string[] = []
    validateNode(value, schemaRecord(alternative), path, branchErrors)
    if (branchErrors.length === 0) {
      return true
    }
    failures.push(branchErrors)
  }
  const closest = failures.sort((left, right) => left.length - right.length)[0] ?? [`${path || 'value'} is invalid`]
  errors.push(...closest)
  return false
}

function applyRules(value: unknown, schema: JsonSchema, path: string, errors: string[]): void {
  const rules = schema['x-study-rules']
  if (!Array.isArray(rules) || !isRecord(value)) {
    return
  }
  for (const rule of rules) {
    if (typeof rule !== 'string') {
      continue
    }
    if (rule.startsWith('unique:')) {
      const [, collection, key] = rule.split(':')
      const items = value[collection]
      if (!Array.isArray(items)) {
        continue
      }
      const seen = new Set<string>()
      items.forEach((item, index) => {
        if (!isRecord(item) || typeof item[key] !== 'string') {
          return
        }
        const itemValue = item[key]
        if (seen.has(itemValue)) {
          errors.push(`${childPath(path, collection)}[${index}].${key} must be unique`)
        }
        seen.add(itemValue)
      })
    } else if (rule === 'schedule-invariants') {
      validateScheduleInvariants(value, path, errors)
    }
  }
}

function validateScheduleInvariants(value: Record<string, unknown>, path: string, errors: string[]): void {
  const range = value.range
  if (isRecord(range) && typeof range.start === 'string' && typeof range.end === 'string' && range.end < range.start) {
    errors.push(`${childPath(path, 'range')}.end must be on or after range.start`)
  }
  if (Array.isArray(value.phases)) {
    value.phases.forEach((phase, index) => {
      if (isRecord(phase) && typeof phase.start === 'string' && typeof phase.end === 'string' && phase.end < phase.start) {
        errors.push(`${childPath(path, 'phases')}[${index}].end must be on or after start`)
      }
    })
  }
  if (!Array.isArray(value.events)) {
    return
  }
  const seen = new Set<string>()
  value.events.forEach((event, index) => {
    if (!isRecord(event)) {
      return
    }
    const eventPath = `${childPath(path, 'events')}[${index}]`
    if (typeof event.id === 'string') {
      if (seen.has(event.id)) {
        errors.push(`${eventPath}.id must be unique`)
      }
      seen.add(event.id)
    }
    if (typeof event.start !== 'string' || typeof event.end !== 'string') {
      return
    }
    const start = new Date(event.start)
    const end = new Date(event.end)
    if (Number.isNaN(start.valueOf()) || Number.isNaN(end.valueOf())) {
      return
    }
    if (end <= start) {
      errors.push(`${eventPath}.end must be after start`)
      return
    }
    const actualMinutes = Math.floor((end.getTime() - start.getTime()) / 60_000)
    if (actualMinutes > 720) {
      errors.push(
        `${eventPath} spans more than 720 minutes; use phases for long-term ranges and events only for concrete study sessions`
      )
    } else if (typeof event.duration_minutes === 'number' && actualMinutes !== event.duration_minutes) {
      errors.push(`${eventPath}.duration_minutes does not match start/end`)
    }
    if (
      isRecord(range) &&
      typeof range.start === 'string' &&
      typeof range.end === 'string' &&
      (event.start.slice(0, 10) < range.start ||
        event.start.slice(0, 10) > range.end ||
        event.end.slice(0, 10) < range.start ||
        event.end.slice(0, 10) > range.end)
    ) {
      errors.push(`${eventPath} must fall inside range`)
    }
  })
}

function validateNode(value: unknown, rawSchema: JsonSchema, path: string, errors: string[]): void {
  const schema = resolveSchema(rawSchema)
  const alternatives = schema.oneOf ?? schema.anyOf
  if (Array.isArray(alternatives)) {
    if (!validateAlternatives(value, alternatives, path, errors)) {
      return
    }
    applyRules(value, schema, path, errors)
    return
  }
  if ('const' in schema && value !== schema.const) {
    errors.push(`${path || 'value'} must equal ${String(schema.const)}`)
    return
  }
  if (Array.isArray(schema.enum) && !schema.enum.includes(value)) {
    errors.push(`${path || 'value'} is unsupported`)
    return
  }
  const type = schema.type
  if (type === 'object') {
    if (!isRecord(value)) {
      errors.push(`${path || 'value'} must be an object`)
      return
    }
    const required = Array.isArray(schema.required) ? schema.required : []
    for (const key of required) {
      if (typeof key === 'string' && !(key in value)) {
        errors.push(`${childPath(path, key)} is required`)
      }
    }
    const properties = schemaRecord(schema.properties)
    for (const [key, childSchema] of Object.entries(properties)) {
      if (key in value) {
        validateNode(value[key], schemaRecord(childSchema), childPath(path, key), errors)
      }
    }
  } else if (type === 'array') {
    if (!Array.isArray(value)) {
      errors.push(`${path || 'value'} must be an array`)
      return
    }
    if (typeof schema.minItems === 'number' && value.length < schema.minItems) {
      errors.push(`${path || 'value'} must be a non-empty array`)
    }
    value.forEach((item, index) => validateNode(item, schemaRecord(schema.items), `${path}[${index}]`, errors))
  } else if (type === 'string') {
    if (typeof value !== 'string') {
      errors.push(`${path || 'value'} must be a string`)
      return
    }
    if (typeof schema.pattern === 'string' && !new RegExp(schema.pattern).test(value)) {
      errors.push(`${path || 'value'} has an invalid format`)
    } else if (schema.format === 'date' && !validDate(value)) {
      errors.push(`${path || 'value'} must be a valid ISO date`)
    } else if (schema.format === 'date-time' && !validDateTime(value)) {
      errors.push(`${path || 'value'} must be a valid ISO datetime`)
    }
  } else if (type === 'integer' || type === 'number') {
    if (typeof value !== 'number' || !Number.isFinite(value) || (type === 'integer' && !Number.isInteger(value))) {
      errors.push(`${path || 'value'} must be ${type === 'integer' ? 'an integer' : 'a number'}`)
      return
    }
    if (typeof schema.minimum === 'number' && value < schema.minimum) {
      errors.push(`${path || 'value'} must be at least ${schema.minimum}`)
    }
    if (typeof schema.maximum === 'number' && value > schema.maximum) {
      errors.push(`${path || 'value'} must be at most ${schema.maximum}`)
    }
  } else if (type === 'boolean' && typeof value !== 'boolean') {
    errors.push(`${path || 'value'} must be a boolean`)
    return
  } else if (type === 'null' && value !== null) {
    errors.push(`${path || 'value'} must be null`)
    return
  }
  applyRules(value, schema, path, errors)
}

function validateContract<T>(input: unknown, definition: 'StudyProject' | 'StudySchedule'): StudySchemaResult<T> {
  const errors: string[] = []
  const definitions = schemaRecord(STUDY_CONTRACT_SCHEMA.$defs)
  validateNode(input, schemaRecord(definitions[definition]), '', errors)
  return errors.length > 0 ? { ok: false, errors: [...new Set(errors)] } : { ok: true, data: input as T }
}

export function validateStudyProject(input: unknown): StudySchemaResult<StudyProject> {
  return validateContract<StudyProject>(input, 'StudyProject')
}

export function validateStudySchedule(input: unknown): StudySchemaResult<StudySchedule> {
  return validateContract<StudySchedule>(input, 'StudySchedule')
}
'''.strip()
    runtime = runtime.replace("__SCHEMA__", schema_json)
    return (
        "// Generated by plugins/study_os/scripts/generate_contracts.py. Do not edit.\n\n"
        + rendered_types
        + "\n\n"
        + runtime
        + "\n"
    )


def generated_files() -> dict[Path, str]:
    schema = study_contract_json_schema()
    return {
        SCHEMA_PATH: json.dumps(
            schema,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        TYPESCRIPT_PATH: render_typescript(schema),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when generated files differ from the canonical models.",
    )
    args = parser.parse_args()
    stale: list[Path] = []
    for path, content in generated_files().items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                stale.append(path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    if stale:
        for path in stale:
            print(f"stale generated StudyOS contract: {path.relative_to(ROOT)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
