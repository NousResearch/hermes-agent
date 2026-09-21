"""Fail-closed, config-only transition between governed local deployments."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml


class LocalTransitionError(ValueError):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _route(record: Any, spec: dict[str, Any], path: str, changes: list[str]) -> None:
    if not isinstance(record, dict):
        return
    old, new, models = spec["old"], spec["new"], spec["models"]
    if record.get("provider") != old["provider"]:
        return
    model = record.get("model")
    if model not in models:
        raise LocalTransitionError(f"{path}: active old local provider has unknown model {model!r}")
    record["provider"] = new["provider"]
    record["model"] = models[model]
    if "base_url" in record:
        if record["base_url"] != old["base_url"]:
            raise LocalTransitionError(f"{path}: old local route has unexpected base_url {record['base_url']!r}")
        record["base_url"] = new["base_url"]
    record["timeout"] = new["request_timeout_seconds"]
    changes.append(path)


def transform_config(doc: dict[str, Any], spec: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Transform only supported active route shapes; catalogue declarations are special-cased."""
    out = copy.deepcopy(doc)
    changed: list[str] = []
    model = out.get("model")
    if isinstance(model, dict):
        _route({"provider": model.get("provider"), "model": model.get("default")}, spec, "model", changed)
        if changed and changed[-1] == "model":
            model["provider"] = spec["new"]["provider"]
            model["default"] = spec["models"][model["default"]]

    for i, entry in enumerate(out.get("fallback_providers") or []):
        _route(entry, spec, f"fallback_providers[{i}]", changed)

    auxiliary = out.get("auxiliary")
    if isinstance(auxiliary, dict):
        for task, cfg in auxiliary.items():
            if not isinstance(cfg, dict):
                continue
            _route(cfg, spec, f"auxiliary.{task}", changed)
            for i, entry in enumerate(cfg.get("fallback_chain") or []):
                _route(entry, spec, f"auxiliary.{task}.fallback_chain[{i}]", changed)

    presets = ((out.get("moa") or {}).get("presets") or {}) if isinstance(out.get("moa"), dict) else {}
    if isinstance(presets, dict):
        for name, preset in presets.items():
            if not isinstance(preset, dict):
                continue
            _route(preset.get("aggregator"), spec, f"moa.presets.{name}.aggregator", changed)
            for i, entry in enumerate(preset.get("reference_models") or []):
                _route(entry, spec, f"moa.presets.{name}.reference_models[{i}]", changed)

    council = out.get("council")
    if isinstance(council, dict) and isinstance(council.get("chairman"), str):
        try:
            chairman = json.loads(council["chairman"])
        except json.JSONDecodeError as exc:
            raise LocalTransitionError(f"council.chairman: invalid embedded JSON: {exc}") from exc
        def walk_json(value: Any, path: str) -> None:
            if isinstance(value, dict):
                _route(value, spec, path, changed)
                for key, child in value.items():
                    if key not in {"provider", "model", "base_url"}:
                        walk_json(child, f"{path}.{key}")
            elif isinstance(value, list):
                for i, child in enumerate(value):
                    walk_json(child, f"{path}[{i}]")
        walk_json(chairman, "council.chairman")
        council["chairman"] = json.dumps(chairman, separators=(",", ":"))

    providers = out.setdefault("providers", {})
    if not isinstance(providers, dict):
        raise LocalTransitionError("providers must be a mapping")
    provider_cfg = providers.setdefault(spec["new"]["provider"], {})
    if not isinstance(provider_cfg, dict):
        raise LocalTransitionError(f"providers.{spec['new']['provider']} must be a mapping")
    if provider_cfg.get("request_timeout_seconds") != spec["new"]["request_timeout_seconds"]:
        provider_cfg["request_timeout_seconds"] = spec["new"]["request_timeout_seconds"]
        changed.append(f"providers.{spec['new']['provider']}.request_timeout_seconds")

    declarations = out.get("custom_providers")
    if isinstance(declarations, list):
        old_decl = next((x for x in declarations if isinstance(x, dict) and x.get("name") == spec["old"]["provider_name"]), None)
        new_decl = next((x for x in declarations if isinstance(x, dict) and x.get("name") == spec["new"]["provider_name"]), None)
        if old_decl is not None and new_decl is None:
            declarations.append({
                "name": spec["new"]["provider_name"], "base_url": spec["new"]["base_url"],
                "api_key": old_decl.get("api_key", "not-needed"), "model": spec["models"]["qwen3.8-27b"],
                "models": [spec["models"]["qwen3.8-27b"], spec["models"]["carwin-moe"]],
                "request_timeout_seconds": spec["new"]["request_timeout_seconds"],
            })
            changed.append("custom_providers[+]turbofit-local")
    return out, changed


def _checked_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    try:
        rel = path.relative_to(root)
        valid = rel.parts == ("config.yaml",) or (len(rel.parts) == 3 and rel.parts[0] == "profiles" and rel.parts[2] == "config.yaml")
        if not valid or path.resolve(strict=True) != path or not path.is_file():
            raise ValueError("noncanonical config path")
    except (OSError, ValueError) as exc:
        raise LocalTransitionError(f"unsafe config path: {path}") from exc
    return path


def config_paths(root: Path) -> list[Path]:
    root = root.resolve(strict=True)
    paths = [root / "config.yaml", *sorted((root / "profiles").glob("*/config.yaml"))]
    return [_checked_path(root, p) for p in paths if p.exists() or p.is_symlink()]


def build_local_plan(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    root = root.resolve(strict=True)
    entries = []
    for path in config_paths(root):
        raw = path.read_bytes()
        after, changes = transform_config(yaml.safe_load(raw) or {}, spec)
        rendered = yaml.safe_dump(after, sort_keys=False, allow_unicode=True).encode()
        entries.append({"path": str(path), "input_sha256": sha256(raw), "after_sha256": sha256(rendered), "changes": changes})
    if not entries:
        raise LocalTransitionError("no configuration files selected")
    return {"mode": "local_transition", "target_root": str(root), "entries": entries}


def _plan_paths(plan):
    if plan.get("mode") != "local_transition" or not plan.get("entries"):
        raise LocalTransitionError("invalid or empty transition plan")
    root = Path(plan["target_root"]).resolve(strict=True)
    paths = [_checked_path(root, entry["path"]) for entry in plan["entries"]]
    if len(set(paths)) != len(paths) or set(paths) != set(config_paths(root)):
        raise LocalTransitionError("duplicate or incomplete configuration selection")
    return root, paths


@contextmanager
def _transition_lock(root):
    # Serialises cooperating transition tools only. Other config writers must
    # be quiescent during cutover: a hash check + rename is NOT universal CAS.
    import fcntl
    fd = os.open(root / ".turbofit-local-transition.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LocalTransitionError("another transition holds the configuration lock") from exc
        yield
    finally:
        os.close(fd)


def _write_checked(path, payload, expected):
    info = path.stat()
    fd, name = tempfile.mkstemp(prefix=".turbofit-route-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as out:
            if os.geteuid() == 0:
                os.fchown(out.fileno(), info.st_uid, info.st_gid)
            elif info.st_uid != os.geteuid():
                raise LocalTransitionError(f"cannot preserve file owner: {path}")
            os.fchmod(out.fileno(), info.st_mode & 0o777)
            out.write(payload)
            out.flush()
            os.fsync(out.fileno())
        if path.is_symlink() or sha256(path.read_bytes()) != expected:
            raise LocalTransitionError(f"concurrent edit: {path}")
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_DIRECTORY | os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        if path.read_bytes() != payload:
            raise LocalTransitionError(f"post-write conflict: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def apply_local_plan(plan: dict[str, Any], spec: dict[str, Any], backup_root: Path) -> None:
    root, paths = _plan_paths(plan)
    with _transition_lock(root):
        prepared = []
        for entry, path in zip(plan["entries"], paths):
            raw = path.read_bytes()
            if sha256(raw) != entry["input_sha256"]:
                raise LocalTransitionError(f"stale input: {path}")
            after, _ = transform_config(yaml.safe_load(raw) or {}, spec)
            rendered = yaml.safe_dump(after, sort_keys=False, allow_unicode=True).encode()
            if sha256(rendered) != entry["after_sha256"]:
                raise LocalTransitionError(f"after hash mismatch: {path}")
            prepared.append((entry, path, raw, rendered))
        backup_root.mkdir(mode=0o700, parents=True, exist_ok=False)
        # All byte-exact backups exist before the first configuration write.
        for entry, path, raw, rendered in prepared:
            backup = backup_root / path.relative_to(root)
            backup.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(raw); handle.flush(); os.fsync(handle.fileno())
        written = []
        try:
            for entry, path, raw, rendered in prepared:
                if raw != rendered:
                    _write_checked(path, rendered, entry["input_sha256"])
                    written.append((entry, path, raw))
        except (OSError, LocalTransitionError) as exc:
            conflicts = []
            for entry, path, raw in reversed(written):
                try:
                    _write_checked(path, raw, entry["after_sha256"])
                except (OSError, LocalTransitionError):
                    conflicts.append(str(path))
            raise LocalTransitionError(f"apply failed; compensation conflicts={conflicts}; {exc}") from exc


def rollback_local_plan(plan: dict[str, Any], backup_root: Path) -> None:
    root, paths = _plan_paths(plan)
    with _transition_lock(root):
        prepared = []
        for entry, path in zip(plan["entries"], paths):
            backup = backup_root / path.relative_to(root)
            if backup.resolve(strict=True) != backup.absolute():
                raise LocalTransitionError(f"unsafe backup path: {backup}")
            raw = backup.read_bytes()
            if sha256(raw) != entry["input_sha256"]:
                raise LocalTransitionError(f"backup hash mismatch: {backup}")
            current = sha256(path.read_bytes())
            if current not in (entry["input_sha256"], entry["after_sha256"]):
                raise LocalTransitionError(f"rollback conflict: {path}")
            prepared.append((entry, path, raw, current))
        for entry, path, raw, current in prepared:
            if current != entry["input_sha256"]:
                _write_checked(path, raw, current)
