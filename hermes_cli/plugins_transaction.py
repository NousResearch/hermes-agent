"""Publish plugin code and its dependency selection through one recoverable handoff."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import shutil
import uuid


def recover_plugin_publication(project: Path, row: dict, journal: Path) -> None:
    from hermes_cli.fs_utils import rmtree_force
    from hermes_cli.runtime_paths import dependency_home_root, runtime_facts_path
    from hermes_cli.runtime_state import _atomic_bytes, _bytes, _digest

    target, backup, metadata = (Path(row[key]) for key in ("target", "backup", "metadata"))
    home = dependency_home_root().resolve()
    if (not target.resolve().is_relative_to(home) or target.parent.name != "plugins"
            or backup.parent != target.parent or not backup.name.startswith(".previous-")
            or metadata != target.parent / ".install-metadata.json"):
        raise ValueError("plugin publication paths escape their home")
    committed = row.get("committed") or _digest(runtime_facts_path(project)) != row["facts_before"]
    if committed:
        if backup.exists():
            rmtree_force(backup)
    else:
        old = base64.b64decode(row["metadata_before"], validate=True) if row["metadata_before"] is not None else None
        current = _bytes(metadata)
        new = base64.b64decode(row["metadata_after"], validate=True)
        if current not in (old, new):
            raise ValueError("plugin metadata changed after publication; preserve it for manual recovery")
        if backup.exists():
            if target.exists():
                rmtree_force(target)
            os.replace(backup, target)
        elif not row["target_existed"] and target.exists():
            rmtree_force(target)
        if old is None:
            metadata.unlink(missing_ok=True)
        else:
            _atomic_bytes(metadata, old)
    journal.unlink()


class PluginPublication:
    def __init__(self, project: Path, staged: Path, target: Path, metadata: dict):
        from hermes_cli.runtime_paths import install_state_dir, runtime_facts_path
        from hermes_cli.runtime_state import _atomic_bytes, _bytes, _digest

        self.project = project
        self.journal = install_state_dir(project) / "publication.json"
        backup = target.parent / f".previous-{uuid.uuid4().hex}"
        metadata_path = target.parent / ".install-metadata.json"
        previous = _bytes(metadata_path)
        proposed = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8")
        self.row = {
            "kind": "plugin", "target": str(target), "backup": str(backup), "metadata": str(metadata_path),
            "target_existed": target.exists(), "facts_before": _digest(runtime_facts_path(project)),
            "metadata_before": base64.b64encode(previous).decode() if previous is not None else None,
            "metadata_after": base64.b64encode(proposed).decode(),
        }
        _atomic_bytes(self.journal, json.dumps(self.row).encode())
        try:
            if target.exists():
                os.replace(target, backup)
            os.replace(staged, target)
            _atomic_bytes(metadata_path, proposed)
        except BaseException:
            self()
            raise

    def __call__(self) -> None:
        recover_plugin_publication(self.project, self.row, self.journal)

    def finish(self) -> None:
        from hermes_cli.runtime_state import _atomic_bytes

        # A code-only update has no new environment fact to mark its commit.
        self.row["committed"] = True
        _atomic_bytes(self.journal, json.dumps(self.row).encode())
        recover_plugin_publication(self.project, self.row, self.journal)


def publish_plugin(staged: Path, target: Path, old_metadata: dict, new_metadata: dict) -> None:
    from hermes_cli import plugins_cmd
    from hermes_cli.runtime_state import recover_publication, runtime_lock
    from pm import paths
    from pm.client import sync_venv
    from pm.workspace import _is_member_candidate, enabled_plugin_dirs, member_sources

    project = paths.repo_root()

    def candidate_members():
        if plugins_cmd._read_install_metadata() != old_metadata:
            raise plugins_cmd.PluginOperationError("Plugin install metadata changed while preparing the update; retry.")
        sources = member_sources(enabled_plugin_dirs())
        active = target.resolve() in sources
        if active:
            sources[target.resolve()] = staged
        for source in sources.values():
            plugins_cmd._check_manifest_version(plugins_cmd._read_manifest_for_install(source), source.name)
        return {identity: source for identity, source in sources.items() if _is_member_candidate(source)}

    # Validate all active dependencies. The installed identity remains stable while
    # PM snapshots the staged inputs into the candidate generation's workspace.
    active = target.resolve() in member_sources(enabled_plugin_dirs())
    if active:
        sync_venv(explicit=True, plugin_dirs=candidate_members,
                  before_publish=lambda: PluginPublication(project, staged, target, new_metadata))
    else:
        with runtime_lock(project):
            recover_publication(project)
            if target.resolve() in member_sources(enabled_plugin_dirs()):
                raise plugins_cmd.PluginOperationError("Plugin enablement changed while preparing the install; retry.")
            if plugins_cmd._read_install_metadata() != old_metadata:
                raise plugins_cmd.PluginOperationError("Plugin install metadata changed while preparing the install; retry.")
            PluginPublication(project, staged, target, new_metadata).finish()


def update_plugin(target: Path, *, catalog_entry=None) -> str:
    """Prepare a catalog re-pin or custom Git pull without changing the live tree."""
    import tempfile

    from hermes_cli import plugins_cmd as pc
    from hermes_cli.plugins_cmd_catalog import raise_if_removed
    from pm.store import tree_digest

    target = target.resolve()
    metadata = pc._read_install_metadata()
    record = dict(metadata.get(target.name, {}))
    if catalog_entry is None and record.get("pinned"):
        raise pc.PluginOperationError(f"Plugin '{target.name}' is pinned; reinstall with an explicit --ref to change it.")
    source = catalog_entry.install_identifier if catalog_entry else str(record.get("source") or "")
    if not source or (catalog_entry is None and not (target / ".git").is_dir()):
        raise pc.PluginOperationError(f"Plugin '{target.name}' has no owned Git checkout; reinstall from its source.")
    feed_revision = None
    if catalog_entry is None:
        from hermes_cli.plugins_provenance import Provenance, ProvenanceClass
        from hermes_cli.plugins_updates import check_local_provenance, default_fetch, parse_feed_yml

        checked = check_local_provenance(Provenance(target.name, ProvenanceClass.GIT, target, record))
        if checked.needs_fixing:
            raise pc.PluginOperationError(checked.needs_fixing)
        if record.get("update_url"):
            feed = parse_feed_yml(default_fetch(record["update_url"]))
            proposed = (feed.get("artifacts") or {}).get("git", "")
            if pc._EXACT_COMMIT_RE.fullmatch(proposed):
                feed_revision = proposed.lower()
            elif proposed not in (source, source.split("#", 1)[0]):
                raise pc.PluginOperationError("Update feed must select a commit or the recorded Git source.")
            if feed.get("min_hermes"):
                pc._check_manifest_version({"requires_hermes": feed["min_hermes"]}, target.name)
    raise_if_removed(target.name, source.split("#", 1)[0])
    before = tree_digest(target)
    with tempfile.TemporaryDirectory(prefix=".update-", dir=target.parent) as directory:
        staged = Path(directory) / "plugin"
        try:
            if catalog_entry:
                git_url, subdir = pc._resolve_git_url(source)
                revision = pc._clone_plugin_repo(staged, git_url, catalog_entry.sha)
                staged = pc._resolve_subdir_within(staged, subdir) if subdir else staged
                # Catalog re-pins cannot discard edits to a currently installed tree.
                if (target / ".git").is_dir():
                    git = pc._resolve_git_executable()
                    changed = pc._git_or_raise(git, target, "status", "--porcelain", "--untracked-files=normal",
                                               failure_prefix="Could not inspect plugin edits: ")
                    if changed.stdout.strip():
                        raise pc.PluginOperationError("Catalog plugin has local changes; save them before updating.")
                output = f"Updated to catalog pin {revision}"
                record.update(catalog_name=catalog_entry.name, catalog_tier=catalog_entry.tier,
                              pinned=True, source=pc._canonical_source(git_url, subdir))
            else:
                # Copy Git metadata and local changes. Autostash only ever touches the copy.
                shutil.copytree(target, staged, symlinks=True, ignore=shutil.ignore_patterns("__pycache__"))
                if feed_revision:
                    git = pc._resolve_git_executable()
                    status = pc._git_or_raise(git, target, "status", "--porcelain", failure_prefix="Could not inspect plugin edits: ")
                    if status.stdout.strip():
                        raise pc.PluginOperationError("Pinned feed update has local changes; save them before updating.")
                    pc._checkout_exact_revision(staged, git, feed_revision)
                    output = f"Updated to feed commit {feed_revision}"
                else:
                    ok, output = pc._git_pull_plugin_dir(staged)
                    if not ok:
                        raise pc.PluginOperationError(output)
                revision = pc._git_head_revision(staged, pc._resolve_git_executable())
            manifest = pc._read_manifest_for_install(staged)
            if manifest.get("name", target.name) != target.name:
                raise pc.PluginOperationError("The updated plugin changed its installed name; reinstall it explicitly.")
            pc._check_manifest_version(manifest, target.name)
            pc._scan_plugin_tree(staged, source, force=False)
            pc._copy_example_files(staged, pc._console())
            if tree_digest(target) != before:
                raise pc.PluginOperationError("Plugin files changed while preparing the update; retry.")
            record["revision"] = revision
            if tree_digest(staged) == before:
                return output
            publish_plugin(staged, target, metadata, {**metadata, target.name: record})
            return output
        except pc.PluginOperationError:
            raise
        except Exception as exc:
            raise pc.PluginOperationError(f"Plugin '{target.name}' update was not published: {exc}") from exc
