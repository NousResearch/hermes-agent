"""Publish plugin code and its dependency selection through one recoverable handoff."""
from __future__ import annotations

from pathlib import Path
import shutil


def recover_plugin_publication(project: Path, row: dict, journal: Path) -> None:
    """Recover a shipped caller's row through the stdlib boot-journal owner."""
    from hermes_cli.runtime_state import _recover_plugin_publication

    _recover_plugin_publication(project, row, journal)


def publish_plugin(staged: Path, target: Path, old_metadata: dict, new_metadata: dict,
                   *, target_digest: str | None = None, require_consent: bool = False) -> None:
    from pm.client import sync_venv
    from pm.store import tree_digest

    if require_consent:
        from hermes_cli import plugins_cmd
        from pm.workspace import enabled_plugin_dirs

        if target.resolve() in enabled_plugin_dirs(installing=target):
            consented, reason = plugins_cmd._install_plugin_python_deps(
                plugins_cmd._read_manifest_for_install(staged), staged, plugins_cmd._console())
            if not consented:
                raise plugins_cmd.PluginOperationError(
                    f"Reinstall declined: {reason}. The installed plugin and active environment are unchanged.")

    sync_venv(explicit=True, staged_plugin={
        "staged": str(staged.resolve()), "target": str(target.absolute()),
        "old_metadata": old_metadata, "new_metadata": new_metadata,
        "target_digest": target_digest if target_digest is not None else (tree_digest(target) if target.exists() else None),
    })


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
            publish_plugin(staged, target, metadata, {**metadata, target.name: record}, target_digest=before)
            return output
        except pc.PluginOperationError:
            raise
        except Exception as exc:
            raise pc.PluginOperationError(f"Plugin '{target.name}' update was not published: {exc}") from exc
