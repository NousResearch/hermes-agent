"""Fail-closed loading and validation of registered JSON Schemas."""

from __future__ import annotations

import json
from pathlib import Path

from .errors import AdapterError


class SchemaRegistry:
    def __init__(self, paths: dict[str, str | Path]):
        try:
            from jsonschema import Draft202012Validator, FormatChecker
            from referencing import Registry, Resource
        except ImportError as exc:
            raise AdapterError(
                "INTERNAL_ERROR", "registered JSON Schema validator unavailable"
            ) from exc
        self._validators = {}
        resources = []
        loaded = {}
        for name, path in paths.items():
            schema = json.loads(Path(path).read_text(encoding="utf-8"))
            Draft202012Validator.check_schema(schema)
            loaded[name] = schema
            if schema.get("$id"):
                resources.append((schema["$id"], Resource.from_contents(schema)))
            resources.append((Path(path).name, Resource.from_contents(schema)))
        registry = Registry().with_resources(resources)
        for name, schema in loaded.items():
            self._validators[name] = Draft202012Validator(
                schema,
                registry=registry,
                format_checker=FormatChecker(),
            )

    def validate(self, name: str, value: dict) -> None:
        validator = self._validators.get(name)
        if validator is None:
            raise AdapterError("INTERNAL_ERROR", f"unregistered schema: {name}")
        errors = sorted(validator.iter_errors(value), key=lambda error: list(error.path))
        if errors:
            path = ".".join(str(item) for item in errors[0].path) or "$"
            raise AdapterError(
                "INVALID_REQUEST",
                f"schema validation failed at {path}: {errors[0].message}",
            )
