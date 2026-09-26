"""Keep an update alive when an enabled plugin no longer fits the new core.

Admission refuses a plugin that does not fit, because the user is choosing and can
choose again. An update has nobody to ask and must never fail because of a plugin:
core moved (a newer Python, a bumped pin, a newer manifest contract) under a plugin
that was admitted against the old core. Such a plugin is disabled in every home that
enables it, the reason reaches the operator and the receipt, and the update continues
with the rest. Only a core that cannot build on its own still fails.

Disabling needs evidence about the plugin itself: its requires-python against the pinned
interpreter, its manifest contract, a resolver proof, or its build failing. A fetch or
tooling failure could be the moment, so the plugin gets one retry before it is disabled.
requires_hermes is judged against a version identity that can lag (a checkout without its
release tags), so a misfit there only sits out: config is untouched, boot skips it the same
way, and it rejoins when the verdict flips. A secondary profile whose config cannot be read
sits out until its config is fixed.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

from pm.environment import BuildFailure, ResolutionConflict
from pm.environments import install_state_dir, runtime_facts_path
from pm.filesystem import durable_write_bytes, file_digest, read_bytes_or_none
from pm.package import InstallError

Entry = tuple[Path, str, Path]  # (home plugins dir, selection key, plugin dir)


def _interpreter_version() -> str:
    """The Python the generation is built for; a plugin's requires-python is judged against it."""
    from pm._uv import _toolchain

    tools = _toolchain(explicit=True)
    if tools is None:
        raise InstallError("venv", "PM's pinned toolchain is unavailable")
    probe = subprocess.run([str(tools[1]), "-I", "-c", "import platform; print(platform.python_version())"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace",
                           check=True, timeout=60)
    return probe.stdout.strip()


def static_verdicts(entries: list[Entry], python_version: str) -> tuple[dict[Path, str], dict[Path, str]]:
    """``(disable, sit out)`` reasons found without a resolver, keyed by resolved dir."""
    from hermes_cli.plugins_manifest import requires_hermes_error
    from pm.plugin_declarations import manifest_version_error, read_python_declaration

    reasons: dict[Path, str] = {}
    waiting: dict[Path, str] = {}
    for _plugins_dir, _name, plugin_dir in entries:
        key = plugin_dir.resolve()
        if key in reasons or key in waiting:
            continue
        try:
            declaration = read_python_declaration(plugin_dir)
        except (OSError, ValueError, TypeError) as exc:
            reasons[key] = f"its dependency declaration is invalid: {exc}"
            continue
        # Mirrors enabled_member_dirs, so the recorded stamp is the one boot expects.
        hermes = requires_hermes_error(declaration.manifest)
        if hermes:
            waiting[key] = hermes
            continue
        manifest = manifest_version_error(declaration.manifest, plugin_dir.name)
        reason = (manifest.removeprefix(f"Plugin '{plugin_dir.name}' ") if manifest
                  else declaration.python_error(python_version))
        if reason:
            reasons[key] = reason
    return reasons, waiting


class PluginEviction:
    """Config edits disabling the plugins in *reasons*; published like a plugin selection."""

    def __init__(self, entries: list[Entry], reasons: dict[Path, str]):
        from hermes_yaml import roundtrip_yaml
        from pm.publication import selection_snapshot

        self.configs = selection_snapshot()
        by_home: dict[Path, list[str]] = {}
        for plugins_dir, name, plugin_dir in entries:
            if plugin_dir.resolve() in reasons:
                by_home.setdefault(plugins_dir.parent, []).append(name)
        self.edits: list[tuple[Path, bytes | None, bytes]] = []
        for home, names in by_home.items():
            path = home / "config.yaml"
            previous = read_bytes_or_none(path)
            yaml = roundtrip_yaml()
            config = (yaml.load(previous.decode("utf-8-sig")) if previous else None) or {}
            # read_home_selection already proved plugins/memory are mappings and the lists are lists.
            plugins = config.get("plugins")
            if plugins is None:
                plugins = config["plugins"] = {}
            disabled = plugins.get("disabled")
            if disabled is None:
                disabled = plugins["disabled"] = []
            memory = config.get("memory")
            for name in names:
                if name not in disabled:
                    disabled.append(name)
                # plugins.disabled does not veto memory.provider; the provider joins the union on its own.
                if isinstance(memory, dict) and str(memory.get("provider") or "").strip() == name:
                    memory["provider"] = ""
            output = io.StringIO()
            yaml.dump(config, output)
            self.edits.append((path, previous, output.getvalue().encode("utf-8")))

    def publish(self, project: Path) -> None:
        from pm.publication import selection_snapshot

        if selection_snapshot() != self.configs:
            raise ValueError("plugin configuration changed while preparing publication; retry")
        row = {"configs": [{"config": str(path),
                            "previous": base64.b64encode(previous).decode() if previous is not None else None,
                            "config_after": hashlib.sha256(proposed).hexdigest()}
                           for path, previous, proposed in self.edits],
               "facts_before": file_digest(runtime_facts_path(project))}
        durable_write_bytes(install_state_dir(project) / "publication.json", json.dumps(row).encode())
        for path, _previous, proposed in self.edits:
            durable_write_bytes(path, proposed)


def _trial(package, enabled, explicit: bool, plugin_dirs: list[Path]) -> str | None:
    """Why the last of *plugin_dirs* cannot join the build, or None when it builds."""
    cause = ""
    # A fetch or tooling failure can be the moment rather than the plugin: one more try.
    for _attempt in range(2):
        try:
            package.apply(enabled, explicit=explicit, plugin_dirs=plugin_dirs, skip_invalid_secondary=True)
            return None
        except (ResolutionConflict, BuildFailure) as exc:
            return f"the dependency environment no longer builds with it: {exc.cause[-400:]}"
        except InstallError as exc:
            cause = exc.cause
    return f"its dependencies could not be prepared, twice: {cause[-400:]}"


def sync_evicting(package, facts, fact: dict, *, extras, shipped, frozen, explicit: bool) -> None:
    """Build the discovered selection, disabling whatever plugin keeps it from building.

    Static misfits go first (no resolver needed). If the rest still fails, core alone is
    built to prove the plugins are the cause, then members are re-added in config order
    and each one that breaks the build is disabled too.
    """
    from pm import receipt
    from pm.install import _commit_selection, _runtime_state_matches, _target_selection
    from pm.plugins_state import dependency_homes, read_home_selection
    from pm.workspace import _is_member_candidate, enabled_plugin_entries

    notices: list[str] = []
    for home in dependency_homes()[1:]:
        try:
            read_home_selection(home)
        except ValueError as exc:
            notices.append(f"Skipped the plugins of profile {home}: {exc}; they rejoin once its config.yaml is fixed")
    entries = enabled_plugin_entries(skip_invalid_secondary=True)
    reasons, waiting = static_verdicts(entries, _interpreter_version())

    def members() -> list[Path]:
        return list(dict.fromkeys(plugin_dir for _plugins_dir, _name, plugin_dir in entries
                                  if plugin_dir.resolve() not in reasons and plugin_dir.resolve() not in waiting
                                  and _is_member_candidate(plugin_dir)))

    def commit() -> None:
        enabled, stamp, inputs = _target_selection(package, fact, extras=extras, inputs={"plugin_dirs": members()},
                                                   repair=False, shipped=shipped, frozen=frozen)
        receipt.record_feature_list(enabled)
        _commit_selection(package, facts, PluginEviction(entries, reasons) if reasons else None,
                          enabled=enabled, stamp=stamp, inputs=inputs,
                          current=_runtime_state_matches(fact, stamp), repair=False, explicit=explicit,
                          skip_invalid_secondary=True)

    kept = members()
    try:
        commit()
    except InstallError as failure:
        if not kept:
            raise
        enabled = _target_selection(package, fact, extras=extras, inputs={"plugin_dirs": []},
                                    repair=False, shipped=shipped, frozen=frozen)[0]
        try:
            package.apply(enabled, explicit=explicit, plugin_dirs=[], skip_invalid_secondary=True)
        except InstallError:
            raise failure from None
        fitting: list[Path] = []
        for member in kept:
            reason = _trial(package, enabled, explicit, [*fitting, member])
            if reason:
                reasons[member.resolve()] = reason
            else:
                fitting.append(member)
        commit()
    for plugins_dir, name, plugin_dir in entries:
        key = plugin_dir.resolve()
        if key in reasons:
            notices.append(f"Disabled plugin '{name}' in {plugins_dir.parent}: {reasons[key]}")
        elif key in waiting:
            notices.append(f"Left plugin '{name}' in {plugins_dir.parent} out of this update: {waiting[key]}; "
                           "it stays enabled and rejoins once Hermes reports a version it accepts")
    for message in notices:
        print(f"⚠ {message}", file=sys.stderr, flush=True)
        receipt.record_warning(message)
