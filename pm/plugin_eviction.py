"""Keep an update alive when an enabled plugin no longer fits the new core.

Admission refuses a plugin that does not fit, because the user is choosing and can
choose again. An update has nobody to ask and must never fail because of a plugin:
core moved (a newer Python, a bumped pin, a newer manifest contract) under a plugin
that was admitted against the old core. Such a plugin is disabled in every home that
enables it, the reason reaches the operator and the receipt, and the update continues
with the rest. Only a core that cannot build on its own still fails. The edit is recorded
in the home's config. A resolver or build verdict carries a fingerprint of what was tried
(core's dependency inputs plus the plugin's) and is retried once either side changed; a fetch
failure is retried by the next update. A plugin that fits again gets the edit undone; a user who re-enables, disables or removes it takes the decision back.
Retries are tried only on top of the enabled plugins that build, so a recorded plugin can never
cost an enabled one its place.

Disabling needs evidence about the plugin itself: its requires-python against the pinned
interpreter, its manifest contract, a resolver proof, or its build failing. A fetch or
tooling failure could be the moment, so the plugin gets one retry before it is disabled.
requires_hermes and a plugin's own requirement on hermes-agent are judged against a version
identity that can lag (a checkout without its release tags), so a misfit there only sits out:
config is untouched, boot skips it the same way, and it rejoins when the verdict flips. A secondary profile whose config cannot be read
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

from pm.environment import BuildFailure, ResolutionConflict  # noqa: TID251
from pm.environments import install_state_dir, runtime_facts_path
from pm.filesystem import durable_write_bytes, file_digest, native, read_bytes_or_none
from pm.package import InstallError
from pm.plugins_state import RECORD

Entry = tuple[Path, str, Path]  # (home plugins dir, selection key, plugin dir)


def _interpreter_version() -> str:
    """The Python the generation is built for; a plugin's requires-python is judged against it."""
    from pm._uv import _toolchain  # noqa: TID251

    tools = _toolchain(explicit=True)
    if tools is None:
        raise InstallError("venv", "PM's pinned toolchain is unavailable")
    probe = subprocess.run([native(tools[1]), "-I", "-c", "import platform; print(platform.python_version())"],
                           capture_output=True, text=True, check=True, timeout=60)
    return probe.stdout.strip()


def static_verdicts(entries: list[Entry], python_version: str,
                    extras: list[str]) -> tuple[dict[Path, str], dict[Path, str]]:
    """``(disable, sit out)`` reasons found without a resolver, keyed by resolved dir."""
    from pm.plugin_declarations import manifest_version_error, read_python_declaration
    from pm.workspace import hermes_fit_error

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
        # Mirrors build_member_dirs, so the recorded stamp is the one boot expects.
        hermes = hermes_fit_error(declaration, extras)
        if hermes:
            waiting[key] = hermes
            continue
        manifest = manifest_version_error(declaration.manifest, plugin_dir.name)
        reason = (manifest.removeprefix(f"Plugin '{plugin_dir.name}' ") if manifest
                  else declaration.python_error(python_version))
        if reason:
            reasons[key] = reason
    return reasons, waiting


def _discard(names: list, spellings: set[str]) -> None:
    """Remove *spellings* from a round-tripped list in place, so order and comments survive."""
    for index in reversed(range(len(names))):
        if names[index] in spellings:
            del names[index]


def _list(plugins, key: str) -> list:
    if plugins.get(key) is None:
        plugins[key] = []
    return plugins[key]


def _tree_names(plugins_dir: Path, plugin_dir: Path, name: str) -> tuple[set[str], set[str]]:
    """``(identity, own)``: every entry that names *plugin_dir* (``hermes_cli.plugin_identity``, the loader's
    rule) and those that name no other tree discovery loads from *plugins_dir*, so dropping *own* from
    plugins.enabled turns off this tree alone. A tree outside the home (bundled) keeps its spellings."""
    from hermes_cli.plugin_identity import installed_trees
    from pm.plugins_state import plugin_spellings

    trees = installed_trees(plugins_dir)
    identity = trees.get(plugin_dir)
    if identity is None:
        spellings = plugin_spellings(plugin_dir, name)
        return spellings, spellings
    others = {entry for tree, names in trees.items() if tree != plugin_dir for entry in names}
    return set(identity), set(identity) - others


def _disable(plugins, plugins_dir: Path, name: str, plugin_dir: Path, reason: str, tried: str | None = None) -> None:
    """Park *name* (the tree's registry key) in plugins.disabled, drop the entries that enable this tree alone
    from plugins.enabled and record what to undo."""
    identity, own = _tree_names(plugins_dir, plugin_dir, name)
    disabled, enabled = _list(plugins, "disabled"), plugins.get("enabled")
    if not isinstance(plugins.get(RECORD), dict):
        plugins[RECORD] = {}
    earlier = plugins[RECORD].get(name)
    if name in disabled and isinstance(earlier, dict):
        was_enabled = bool(earlier.get("enabled"))  # a retry that still fails keeps the first record
    else:
        was_enabled = isinstance(enabled, list) and bool(identity & set(enabled))
    if name not in disabled:
        disabled.append(name)
    if isinstance(enabled, list):
        _discard(enabled, own)
    plugins[RECORD][name] = {"enabled": was_enabled, "reason": " ".join(reason.split())[:300]}
    if tried:
        plugins[RECORD][name]["tried"] = tried


def _rejoin(plugins, plugins_dir: Path, name: str, plugin_dir: Path) -> None:
    """Undo an earlier update's _disable now that the plugin fits again."""
    record = plugins.get(RECORD) if isinstance(plugins.get(RECORD), dict) else {}
    earlier = record.get(name)
    if not isinstance(earlier, dict):
        return  # the user settled it since this update read the config: their choice stands
    identity, own = _tree_names(plugins_dir, plugin_dir, name)
    _discard(_list(plugins, "disabled"), own | {name})
    if earlier.get("enabled") and not identity & set(_list(plugins, "enabled")):
        plugins["enabled"].append(name)


def settle_record(plugins, settled=()) -> None:
    """A user's own selection takes the decision back: forget the record of every plugin they just
    enabled, disabled or removed (*settled*), so no later update turns it back on."""
    record = plugins.get(RECORD)
    for name in settled if isinstance(record, dict) else ():
        record.pop(name, None)
    _prune_record(plugins)


def _prune_record(plugins) -> None:
    """Forget records whose plugin left plugins.disabled (the user re-enabled or removed it)."""
    record = plugins.get(RECORD)
    if not isinstance(record, dict):
        plugins.pop(RECORD, None)
        return
    disabled = set(plugins.get("disabled") or ())
    for name in [name for name in record if name not in disabled]:
        del record[name]
    if not record:
        del plugins[RECORD]


def recorded_evictions() -> dict[Entry, dict]:
    """Plugins an earlier update disabled that their home still has disabled, with their record.
    A provider-only record (never in plugins.enabled) is retried only while memory.provider still
    names it, so a rejoin never installs a provider the user has since switched away from."""
    from pm.plugins_state import dependency_homes, plugin_spellings, read_home_selection
    from pm.workspace import selected_plugin_dir

    entries: dict[Entry, dict] = {}
    for home in dependency_homes():
        try:
            selection = read_home_selection(home) or {}
        except ValueError:
            continue  # sync_evicting already reports an unreadable profile
        plugins = selection.get("plugins") or {}
        provider = str((selection.get("memory") or {}).get("provider") or "").strip()
        record, disabled = plugins.get(RECORD), plugins.get("disabled") or []
        for name in record if isinstance(record, dict) else ():
            if not isinstance(name, str) or name not in disabled:
                continue
            try:
                plugin_dir = selected_plugin_dir(home / "plugins", name)
            except InstallError:
                continue  # a hand-edited key that names no plugin tree
            earlier = record[name] if isinstance(record[name], dict) else {}
            if plugin_dir is not None and (earlier.get("enabled") or provider in plugin_spellings(plugin_dir, name)):
                entries[(home / "plugins", name, plugin_dir)] = earlier
    return entries


class PluginEviction:
    """One update's config edits: disable the plugins in *reasons*, re-enable *rejoined* ones.

    A disabled plugin leaves plugins.enabled under every spelling and is recorded under
    plugins.disabled_by_update (was it enabled, and why), so a later update can retry it and undo
    exactly this edit. memory.provider is left alone: plugins.disabled keeps a parked provider out
    of the environment and the loader, and re-enabling the plugin brings the provider back with it.
    Published like a plugin selection.
    """

    def __init__(self, entries: list[Entry], reasons: dict[Path, str], rejoined: list[Entry] = (), *,
                 tried: dict[Path, str] | None = None, configs=None):
        from pm.publication import selection_snapshot

        # The snapshot the update decided from: a user edit since then refuses publication.
        self.configs = selection_snapshot() if configs is None else configs
        self.tried = tried = tried or {}
        from hermes_cli.plugin_identity import registry_key

        by_home: dict[Path, list[tuple[str, Path, str | None]]] = {}
        for plugins_dir, name, plugin_dir in entries:
            if plugin_dir.resolve() in reasons:
                # By the tree's own registry key: the entry that enabled it may name a working copy too.
                key = registry_key(plugins_dir, plugin_dir) or name
                if all(change[1] != plugin_dir for change in by_home.get(plugins_dir.parent, ())):
                    by_home.setdefault(plugins_dir.parent, []).append((key, plugin_dir, reasons[plugin_dir.resolve()]))
        for plugins_dir, name, plugin_dir in rejoined:
            by_home.setdefault(plugins_dir.parent, []).append((name, plugin_dir, None))
        self.changes = by_home
        self.edits: list[tuple[Path, bytes | None, bytes]] = [self._render(home) for home in by_home]

    def _render(self, home: Path) -> tuple[Path, bytes | None, bytes]:
        """*home*'s config.yaml as it is now, with this update's edits to its ``plugins`` section."""
        from hermes_yaml import roundtrip_yaml

        path = home / "config.yaml"
        previous = read_bytes_or_none(path)  # config-reader: ok — pm publication: byte-level compare-and-swap rewrite of the local file
        yaml = roundtrip_yaml()
        config = (yaml.load(previous.decode("utf-8-sig")) if previous else None) or {}
        # read_home_selection already proved plugins is a mapping and the lists are lists.
        plugins = config.get("plugins")
        if plugins is None:
            plugins = config["plugins"] = {}
        for name, plugin_dir, reason in self.changes[home]:
            if reason is None:
                _rejoin(plugins, home / "plugins", name, plugin_dir)
            else:
                _disable(plugins, home / "plugins", name, plugin_dir, reason, self.tried.get(plugin_dir.resolve()))
        _prune_record(plugins)
        output = io.StringIO()
        yaml.dump(config, output)
        return path, previous, output.getvalue().encode("utf-8")

    def publish(self, project: Path) -> None:
        from pm.publication import SelectionChanged, selection_snapshot

        if selection_snapshot() != self.configs:
            raise SelectionChanged("plugin configuration changed while preparing publication; retry")
        # Keys outside the plugin selection may have moved since: re-applied to the bytes as they are.
        self.edits = [self._render(home) for home in self.changes]
        row = {"configs": [{"config": str(path),
                            "previous": base64.b64encode(previous).decode() if previous is not None else None,
                            "config_after": hashlib.sha256(proposed).hexdigest()}
                           for path, previous, proposed in self.edits],
               "facts_before": file_digest(runtime_facts_path(project))}
        durable_write_bytes(install_state_dir(project) / "publication.json", json.dumps(row).encode())
        for path, _previous, proposed in self.edits:
            durable_write_bytes(path, proposed)


def _trial(package, enabled, explicit: bool, plugin_dirs: list[Path]) -> tuple[str, bool] | None:
    """Why the last of *plugin_dirs* cannot join the build, and whether that is a verdict on the
    plugin (a resolver proof or its build failing) rather than the moment; None when it builds."""
    cause = ""
    # A fetch or tooling failure can be the moment rather than the plugin: one more try.
    for _attempt in range(2):
        try:
            package.apply(enabled, explicit=explicit, plugin_dirs=plugin_dirs, skip_invalid_secondary=True)
            return None
        except (ResolutionConflict, BuildFailure) as exc:
            return f"the dependency environment no longer builds with it: {exc.cause[-400:]}", True
        except InstallError as exc:
            cause = exc.cause
    return f"its dependencies could not be prepared, twice: {cause[-400:]}", False


def sync_evicting(package, facts, fact: dict, *, extras, shipped, frozen, explicit: bool) -> None:
    """Build the discovered selection, disabling whatever plugin keeps it from building.

    Static misfits go first (no resolver needed). If the rest still fails, core alone is
    built to prove the plugins are the cause, then members are re-added in config order
    and each one that breaks the build is disabled too. Plugins an earlier update disabled
    are retried once their fingerprint changed, on top of the enabled plugins that build,
    and re-enabled when they fit.
    """
    from pm import receipt
    from pm.install import _target_selection
    from pm.plugins_state import dependency_homes, read_home_selection
    from pm.publication import selection_snapshot
    from pm.workspace import enabled_plugin_entries

    configs = selection_snapshot()
    notices: list[str] = []
    for home in dependency_homes()[1:]:
        try:
            read_home_selection(home)
        except ValueError as exc:
            notices.append(f"Skipped the plugins of profile {home}: {exc}; they rejoin once its config.yaml is fixed")
    entries = enabled_plugin_entries(skip_invalid_secondary=True)
    recorded = recorded_evictions()
    selection = dict(extras=extras, shipped=shipped, frozen=frozen)
    # Core's extras with no plugin: what every build of this update selects, so the static
    # verdicts judge plugin extras under the same selection boot's currency check reads.
    base = _target_selection(package, fact, inputs={"plugin_dirs": []}, repair=False, **selection)[0]
    reasons, waiting = static_verdicts([*entries, *recorded], _interpreter_version(), base)
    update = _EvictingUpdate(package, facts, fact, selection, base=base, explicit=explicit, configs=configs,
                             entries=entries, recorded=recorded, reasons=reasons, waiting=waiting)
    update.run()
    notices += _outcome_notices(entries, recorded, reasons, waiting, update.joined)
    for message in notices:
        print(f"⚠ {message}", file=sys.stderr, flush=True)
        receipt.record_warning(message)


class _EvictingUpdate:
    """One update's build: the enabled plugins first, then the recorded ones that are due."""

    def __init__(self, package, facts, fact: dict, selection: dict, *, base: list[str], explicit: bool, configs,
                 entries: list[Entry], recorded: dict[Entry, dict], reasons: dict[Path, str],
                 waiting: dict[Path, str]):
        from pm.plugin_declarations import read_python_declaration
        from pm.workspace import select_workspace_members

        self.package, self.facts, self.fact, self.selection = package, facts, fact, selection
        self.explicit, self.configs, self.entries, self.reasons = explicit, configs, entries, reasons
        self.retrying = list(recorded)
        self.tried: dict[Path, str] = {}
        self.verdicts: list[Path] = []  # disabled on a verdict about the plugin: fingerprinted at commit
        self.base = base

        def fits(plugin_dir: Path) -> bool:
            return plugin_dir.resolve() not in reasons and plugin_dir.resolve() not in waiting

        # Keep every fitting package-declaring plugin through retries so core-satisfied
        # constraints stay in any shared solve triggered by a retried plugin.
        self.kept = list(dict.fromkeys(
            plugin_dir for _plugins_dir, _name, plugin_dir in entries
            if fits(plugin_dir) and read_python_declaration(plugin_dir).installs_packages
        ))

        def _needs_trial(plugin_dir: Path) -> bool:
            if not read_python_declaration(plugin_dir).installs_packages:
                return False
            return bool(select_workspace_members([*self.kept, plugin_dir]))

        # A plugin with nothing to build rejoins on its static verdict; one with dependencies is
        # due only when core's inputs, its own or the enabled plugins beside it changed since the
        # attempt that failed.
        self.joined = [entry for entry in recorded if fits(entry[2]) and not _needs_trial(entry[2])]
        self._static_joined_packages = list(dict.fromkeys(
            entry[2] for entry in self.joined if read_python_declaration(entry[2]).installs_packages
        ))
        baseline = list(dict.fromkeys([*self.kept, *self._static_joined_packages]))
        # Those whose last failure was no verdict on the plugin (a download failure, a static
        # misfit since fixed) go first: they fitted beside the rest before.
        self.due = sorted((entry for entry, record in recorded.items()
                           if fits(entry[2]) and _needs_trial(entry[2])
                           and record.get("tried") != self.fingerprint(entry[2], baseline)),
                          key=lambda entry: "tried" in recorded[entry])

    def _effective_members(self, candidates: list[Path]) -> list[Path]:
        from pm.workspace import select_workspace_members

        return select_workspace_members(list(dict.fromkeys(candidates)))

    def _base_is_current(self) -> bool:
        from pm.install import _runtime_state_matches, _target_selection

        _, stamp, _ = _target_selection(self.package, self.fact, inputs={"plugin_dirs": []},
                                        repair=False, **self.selection)
        return _runtime_state_matches(self.fact, stamp)

    def fingerprint(self, plugin_dir: Path, beside: list[Path]) -> str:
        """What a retry of *plugin_dir* would build: core, the plugin and the enabled plugins."""
        return self.package.expected_stamp(self.base, plugin_dirs=list(dict.fromkeys([*beside, plugin_dir])))

    def commit(self, candidates: list[Path]) -> None:
        from pm import receipt
        from pm.install import _commit_selection, _runtime_state_matches, _target_selection

        surviving = list(dict.fromkeys(candidates))
        members = self._effective_members(surviving)
        # Fingerprinted against the package-declaring plugins this update leaves enabled (not the
        # collapsed workspace list), so the next update compares against the same baseline and
        # still retries if a core-satisfied enabled plugin's constraints change.
        self.tried = {member.resolve(): self.fingerprint(member, surviving) for member in self.verdicts}
        enabled, stamp, inputs = _target_selection(self.package, self.fact, inputs={"plugin_dirs": members},
                                                   repair=False, **self.selection)
        receipt.record_feature_list(enabled)
        edit = None
        if self.reasons or self.joined:
            edit = PluginEviction([*self.entries, *self.retrying], self.reasons, self.joined,
                                  tried=self.tried, configs=self.configs)
        _commit_selection(self.package, self.facts, edit, enabled=enabled, stamp=stamp, inputs=inputs,
                          current=_runtime_state_matches(self.fact, stamp), repair=False, explicit=self.explicit,
                          skip_invalid_secondary=True)

    def run(self) -> None:
        due = [entry[2] for entry in self.due]
        baseline = list(dict.fromkeys([*self.kept, *self._static_joined_packages]))
        if due:
            try:
                self.joined += self.due
                self.commit([*baseline, *due])
                return
            except InstallError:
                self.joined = self.joined[:-len(self.due)]
        try:
            if not due:
                self.commit(baseline)
                return
            base_members = self._effective_members(baseline)
            if base_members or not self._base_is_current():
                self.package.apply(self.base, explicit=self.explicit, plugin_dirs=base_members,
                                   skip_invalid_secondary=True)
        except InstallError as failure:
            self.evict_enabled(failure)
            return
        fitting = list(baseline)
        for entry in self.due:
            if self.try_member(fitting, entry[2]):
                self.joined.append(entry)
        self.commit(fitting)

    def try_member(self, fitting: list[Path], member: Path) -> bool:
        """Add *member* to *fitting* when it builds on top; else record why. Only a verdict on the
        plugin is fingerprinted: a fetch failure is the moment, so the next update retries it."""
        members = self._effective_members([*fitting, member])
        if not members:
            fitting.append(member)
            return True
        verdict = _trial(self.package, self.base, self.explicit, members)
        if verdict is None:
            fitting.append(member)
            return True
        self.reasons[member.resolve()] = verdict[0]
        if verdict[1]:
            self.verdicts.append(member)
        return False

    def evict_enabled(self, failure: InstallError) -> None:
        """Core alone must build; then each enabled member that breaks the build is disabled.
        Nothing rejoins in such an update: a retried plugin must never take an enabled one's place."""
        if not self._effective_members(self.kept):
            raise failure
        try:
            self.package.apply(self.base, explicit=self.explicit, plugin_dirs=[], skip_invalid_secondary=True)
        except InstallError:
            raise failure from None
        self.joined = [entry for entry in self.joined if entry not in self.due]
        fitting: list[Path] = []
        for member in self.kept:
            self.try_member(fitting, member)
        self.commit(list(dict.fromkeys([*fitting, *self._static_joined_packages])))


_RETRY = "a later update retries it"


def _outcome_notices(entries: list[Entry], recorded: dict[Entry, dict], reasons: dict[Path, str],
                     waiting: dict[Path, str], joined: list[Entry]) -> list[str]:
    notices = []
    for plugins_dir, name, plugin_dir in entries:
        key = plugin_dir.resolve()
        if key in reasons:
            notices.append(f"Disabled plugin '{name}' in {plugins_dir.parent}: {reasons[key]}; {_RETRY} and turns it "
                           f"back on once it fits, or run `hermes plugins enable {name}` sooner")
        elif key in waiting:
            notices.append(f"Left plugin '{name}' in {plugins_dir.parent} out of this update: {waiting[key]}; "
                           "it stays enabled and rejoins once Hermes reports a version it accepts")
    for entry, record in recorded.items():
        plugins_dir, name, plugin_dir = entry
        if entry in joined:
            notices.append(f"Turned plugin '{name}' in {plugins_dir.parent} back on: it fits this Hermes again")
            continue
        why = reasons.get(plugin_dir.resolve()) or waiting.get(plugin_dir.resolve()) or record.get("reason") or "unknown"
        notices.append(f"Plugin '{name}' in {plugins_dir.parent} is still disabled: {why}; {_RETRY}")
    return notices
