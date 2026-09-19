"""Worker-local plugin selection and code publication. No application dependency imports.

Requests carry proposed data; discovery, snapshots and publication happen only
while the install lock is held. The stdlib boot journal owns crash recovery.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path

from pm.environments import dependency_home_root, install_state_dir, runtime_facts_path
from hermes_cli.runtime_state import _atomic_bytes, _bytes, _digest
from pm.workspace import enabled_plugin_dirs, _is_member_candidate


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
    return {home / "config.yaml": _bytes(home / "config.yaml") for home in dependency_homes()}


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
        self.previous = _bytes(self.path)
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
        if selection_snapshot() != self.configs or _bytes(self.path) != self.previous:
            raise ValueError("plugin configuration changed while preparing publication; retry")
        row = {"config": str(self.path),
               "previous": base64.b64encode(self.previous).decode() if self.previous is not None else None,
               "facts_before": _digest(runtime_facts_path(project)),
               "config_after": hashlib.sha256(self.proposed).hexdigest()}
        _atomic_bytes(install_state_dir(project) / "publication.json", json.dumps(row).encode())
        _atomic_bytes(self.path, self.proposed)


class StagedPlugin:
    def __init__(self, plugin: dict):
        from pm.store import tree_digest
        from pm.workspace import enabled_plugin_dirs, member_sources

        self.configs = selection_snapshot()
        self.target = Path(plugin["target"]).absolute()
        self.staged = Path(plugin["staged"]).resolve()
        if (not self.target.resolve().is_relative_to(dependency_home_root().resolve())
                or self.target.parent.name != "plugins" or self.target.is_symlink()
                or self.staged == self.target.resolve() or self.staged.is_relative_to(self.target.resolve())
                or self.target.resolve().is_relative_to(self.staged)):
            raise ValueError("plugin publication paths escape or overlap their home")
        manifest = validate_manifest(self.staged)
        if manifest.get("name", self.target.name) != self.target.name:
            raise ValueError("The updated plugin changed its installed name; reinstall it explicitly.")
        self.staged_digest = tree_digest(self.staged)
        self.metadata = self.target.parent / ".install-metadata.json"
        self.previous = _bytes(self.metadata)
        current = json.loads(self.previous) if self.previous is not None else {}
        if current != plugin["old_metadata"]:
            raise ValueError("Plugin install metadata changed while preparing the update; retry.")
        self.target_digest = tree_digest(self.target) if self.target.exists() else None
        if self.target_digest != plugin["target_digest"]:
            raise ValueError("Plugin files changed while preparing the update; retry.")
        self.proposed = (json.dumps(plugin["new_metadata"], indent=2, sort_keys=True) + "\n").encode()
        sources = member_sources(enabled_plugin_dirs(installing=self.target))
        self.active = self.target.resolve() in sources
        self.members = {}
        if self.active:
            sources[self.target.resolve()] = self.staged
            for source in sources.values():
                validate_manifest(source)
            self.members = {identity: source for identity, source in sources.items() if _is_member_candidate(source)}

    def publish(self, project: Path) -> None:
        import os
        import uuid
        from pm.store import tree_digest

        if selection_snapshot() != self.configs:
            raise ValueError("Plugin enablement changed while preparing the update; retry.")
        if tree_digest(self.staged) != self.staged_digest:
            raise ValueError("Staged plugin files changed while preparing the update; retry.")
        if _bytes(self.metadata) != self.previous:
            raise ValueError("Plugin install metadata changed while preparing the update; retry.")
        current = tree_digest(self.target) if self.target.exists() else None
        if current != self.target_digest:
            raise ValueError("Plugin files changed while preparing the update; retry.")
        backup = self.target.parent / f".previous-{uuid.uuid4().hex}"
        row = {
            "kind": "plugin", "target": str(self.target), "backup": str(backup), "metadata": str(self.metadata),
            "target_existed": self.target.exists(), "facts_before": _digest(runtime_facts_path(project)),
            "metadata_before": base64.b64encode(self.previous).decode() if self.previous is not None else None,
            "metadata_after": base64.b64encode(self.proposed).decode(),
        }
        _atomic_bytes(install_state_dir(project) / "publication.json", json.dumps(row).encode())
        if self.target.exists():
            os.replace(self.target, backup)
        os.replace(self.staged, self.target)
        _atomic_bytes(self.metadata, self.proposed)
