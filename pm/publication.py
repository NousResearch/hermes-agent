"""Worker-local plugin selection and code publication. No application dependency imports.

Requests carry proposed data; discovery, snapshots and publication happen only
while the install lock is held. The stdlib boot journal owns crash recovery.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import threading
from pathlib import Path

from pm.environments import dependency_home_root, install_state_dir, runtime_facts_path
from pm.filesystem import durable_write_bytes, file_digest, read_bytes_or_none
from pm.workspace import enabled_plugin_dirs, _is_member_candidate


_METADATA_LOCK_HOLDER = threading.local()


def _metadata_records(data: bytes | None) -> dict:
    if data is None:
        return {}
    try:
        records = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Plugin install metadata changed while preparing the update; retry.") from exc
    if not isinstance(records, dict):
        raise ValueError("Plugin install metadata changed while preparing the update; retry.")
    return records


def candidate_members(extra_dirs=(), **selection):
    selected = enabled_plugin_dirs(**selection)
    for source in selected:
        validate_manifest(source)
    members = [source for source in selected if _is_member_candidate(source)]
    for directory in extra_dirs:
        directory = Path(directory)
        if directory not in members and _is_member_candidate(directory):
            members.append(directory)
    return members


def selection_snapshot() -> dict[Path, bytes | None]:
    from pm.plugins_state import dependency_homes
    return {home / "config.yaml": read_bytes_or_none(home / "config.yaml") for home in dependency_homes()}


def validate_manifest(source: Path) -> dict:
    from pm.plugin_declarations import read_python_declaration, manifest_version_error

    manifest = read_python_declaration(source).manifest
    reason = manifest_version_error(manifest, source.name)
    if reason:
        raise ValueError(reason)
    return manifest


class PluginSelection:
    def __init__(self, selection: dict):
        from hermes_yaml import roundtrip_yaml

        self.configs = selection_snapshot()
        self.home = Path(selection["home"]).resolve()
        if not self.home.is_relative_to(dependency_home_root().resolve()):
            raise ValueError("config path is outside Hermes state")
        self.path = self.home / "config.yaml"
        self.previous = read_bytes_or_none(self.path)
        expected = selection.get("expected_config")
        actual = hashlib.sha256(self.previous).hexdigest() if self.previous is not None else "missing"
        if expected is not None and expected != actual:
            raise ValueError("Plugin configuration changed since this selection was read; retry.")
        yaml = roundtrip_yaml()
        config = yaml.load(self.previous.decode("utf-8-sig")) if self.previous else {}
        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise ValueError(f"configuration must be a mapping: {self.path}")
        plugins = config.setdefault("plugins", {})
        if not isinstance(plugins, dict):
            raise ValueError(f"plugins must be a mapping in {self.path}")
        plugins["enabled"] = sorted(selection["enabled"])
        plugins["disabled"] = sorted(selection["disabled"])
        output = io.StringIO()
        yaml.dump(config, output)
        self.proposed = output.getvalue().encode("utf-8")
        self.members = candidate_members(selection.get("extra_dirs", ()), proposed_home=self.home,
                                         enabled=selection["enabled"], disabled=selection["disabled"])

    def publish(self, project: Path) -> None:
        if selection_snapshot() != self.configs or read_bytes_or_none(self.path) != self.previous:
            raise ValueError("plugin configuration changed while preparing publication; retry")
        row = {"config": str(self.path),
               "previous": base64.b64encode(self.previous).decode() if self.previous is not None else None,
               "facts_before": file_digest(runtime_facts_path(project)),
               "config_after": hashlib.sha256(self.proposed).hexdigest()}
        durable_write_bytes(install_state_dir(project) / "publication.json", json.dumps(row).encode())
        durable_write_bytes(self.path, self.proposed)


class StagedPlugin:
    def __init__(self, plugin: dict):
        from pm.store import tree_digest
        from pm.workspace import enabled_plugin_dirs, member_sources

        self.configs = selection_snapshot()
        self.target = Path(plugin["target"]).absolute()
        self.previous_target = Path(plugin.get("previous_target", self.target)).absolute()
        self.renaming = self.previous_target != self.target
        self.staged = Path(plugin["staged"]).resolve()
        if (not self.target.resolve().is_relative_to(dependency_home_root().resolve())
                or self.target.parent.name != "plugins" or self.target.is_symlink()
                or self.staged == self.target.resolve() or self.staged.is_relative_to(self.target.resolve())
                or self.target.resolve().is_relative_to(self.staged)):
            raise ValueError("plugin publication paths escape or overlap their home")
        if self.renaming and (self.previous_target.parent != self.target.parent
                              or self.previous_target.is_symlink() or not self.previous_target.is_dir()
                              or self.target.exists()):
            raise ValueError("plugin rename requires an installed source and an unused sibling destination")
        manifest = validate_manifest(self.staged)
        if manifest.get("name", self.target.name) != self.target.name:
            raise ValueError("The updated plugin changed its installed name; reinstall it explicitly.")
        self.staged_digest = tree_digest(self.staged)
        self.metadata = self.target.parent / ".install-metadata.json"
        previous = read_bytes_or_none(self.metadata)
        current = _metadata_records(previous)
        self.old_record = plugin["old_metadata"].get(self.target.name)
        if current.get(self.target.name) != self.old_record:
            raise ValueError("Plugin install metadata changed while preparing the update; retry.")
        self.previous_record = plugin["old_metadata"].get(self.previous_target.name)
        if self.renaming and (self.previous_record is None or self.old_record is not None
                              or current.get(self.previous_target.name) != self.previous_record):
            raise ValueError("Plugin install metadata changed while preparing the rename; retry.")
        self.new_record = plugin["new_metadata"].get(self.target.name)
        if self.new_record is None:
            raise ValueError("Plugin publication omitted its install metadata record.")
        self.target_digest = tree_digest(self.previous_target) if self.previous_target.exists() else None
        if self.target_digest != plugin["target_digest"]:
            raise ValueError("Plugin files changed while preparing the update; retry.")
        sources = member_sources(enabled_plugin_dirs(installing=self.previous_target))
        self.active = self.previous_target.resolve() in sources
        self.members = {}
        if self.active:
            sources.pop(self.previous_target.resolve(), None)
            sources[self.target.resolve()] = self.staged
            for source in sources.values():
                validate_manifest(source)
            self.members = {identity: source for identity, source in sources.items() if _is_member_candidate(source)}
        self.config_change = None
        if self.renaming:
            from hermes_yaml import roundtrip_yaml

            config_path = self.target.parent.parent / "config.yaml"
            previous = read_bytes_or_none(config_path)
            yaml = roundtrip_yaml()
            config = yaml.load(previous.decode("utf-8-sig")) if previous else {}
            config = config or {}
            changed = False
            for key in ("enabled", "disabled"):
                selected = (config.get("plugins") or {}).get(key, [])
                for index, name in enumerate(selected):
                    if name == self.previous_target.name:
                        selected[index] = self.target.name
                        changed = True
            memory = config.get("memory") or {}
            if memory.get("provider") == self.previous_target.name:
                memory["provider"] = self.target.name
                changed = True
            if changed:
                output = io.StringIO()
                yaml.dump(config, output)
                proposed = output.getvalue().encode("utf-8")
                self.config_change = (config_path, previous, proposed)

    def publish(self, project: Path) -> None:
        import os
        import uuid
        from hermes_cli.auth import _file_lock
        from pm.store import tree_digest

        if selection_snapshot() != self.configs:
            raise ValueError("Plugin enablement changed while preparing the update; retry.")
        if tree_digest(self.staged) != self.staged_digest:
            raise ValueError("Staged plugin files changed while preparing the update; retry.")
        lock = self.metadata.with_name(f"{self.metadata.name}.lock")
        with _file_lock(lock, _METADATA_LOCK_HOLDER, 10.0,
                        "Timed out waiting for the plugin install metadata lock"):
            previous = read_bytes_or_none(self.metadata)
            metadata = _metadata_records(previous)
            if metadata.get(self.target.name) != self.old_record:
                raise ValueError("Plugin install metadata changed while preparing the update; retry.")
            if self.renaming:
                if metadata.get(self.previous_target.name) != self.previous_record or self.target.exists():
                    raise ValueError("Plugin rename destination or metadata changed; retry.")
                metadata.pop(self.previous_target.name)
            metadata[self.target.name] = self.new_record
            proposed = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode()
            current = tree_digest(self.previous_target) if self.previous_target.exists() else None
            if current != self.target_digest:
                raise ValueError("Plugin files changed while preparing the update; retry.")
            backup = self.target.parent / f".previous-{uuid.uuid4().hex}"
            row = {
                "kind": "plugin", "target": str(self.target), "backup": str(backup), "metadata": str(self.metadata),
                "target_existed": self.target.exists(), "facts_before": file_digest(runtime_facts_path(project)),
                "metadata_before": base64.b64encode(previous).decode() if previous is not None else None,
                "metadata_after": base64.b64encode(proposed).decode(),
            }
            if self.renaming:
                row["previous_target"] = str(self.previous_target)
            if self.config_change is not None:
                config, previous_config, proposed_config = self.config_change
                if read_bytes_or_none(config) != previous_config:
                    raise ValueError("Plugin configuration changed while preparing the rename; retry.")
                row["config"] = str(config)
                row["config_before"] = base64.b64encode(previous_config).decode() if previous_config is not None else None
                row["config_after"] = hashlib.sha256(proposed_config).hexdigest()
            durable_write_bytes(install_state_dir(project) / "publication.json", json.dumps(row).encode())
            if self.previous_target.exists():
                os.replace(self.previous_target, backup)
            os.replace(self.staged, self.target)
            durable_write_bytes(self.metadata, proposed)
            if self.config_change is not None:
                durable_write_bytes(self.config_change[0], self.config_change[2])
