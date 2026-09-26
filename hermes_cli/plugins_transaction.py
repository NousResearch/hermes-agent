"""Publish plugin code and its dependency selection through one recoverable handoff."""
from __future__ import annotations

from pathlib import Path
import shutil


def recover_plugin_publication(project: Path, row: dict, journal: Path) -> None:
    """Recover a shipped caller's row through the stdlib boot-journal owner."""
    from hermes_cli.runtime_state import _recover_plugin_publication

    _recover_plugin_publication(project, row, journal)


def publish_plugin(staged: Path, target: Path, old_metadata: dict, new_metadata: dict,
                   *, target_digest: str | None = None, require_consent: bool = False,
                   previous_target: Path | None = None) -> None:
    from pm.client import sync_venv
    from pm.plugin_inputs import StagedUpdate
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

    sync_venv(explicit=True, plugins=StagedUpdate({
        "staged": str(staged.resolve()), "target": str(target.absolute()),
        "old_metadata": old_metadata, "new_metadata": new_metadata,
        "target_digest": target_digest if target_digest is not None else (tree_digest(target) if target.exists() else None),
        **({"previous_target": str(previous_target.absolute())} if previous_target is not None else {}),
    }))


def _refresh_declared_dependencies(target: Path, staged: Path, manifest: dict, *, interactive: bool) -> None:
    """Dependency changes carried by an update clear the same gates as an install.

    New Python requirements install packages into the shared environment: install and
    reinstall prompt for that, so an update prompts too, and an unattended one (dashboard,
    ``plugins.auto_apply`` from the gateway) is REFUSED rather than consented on the user's
    behalf. Publication replaces the whole tree, so a Node sidecar the user accepted earlier is
    rebuilt in the staged copy when its package.json/lock moved (custom pulls copy a stale
    node_modules; catalog re-pins clone without one).
    """
    from hermes_cli import plugins_cmd as pc
    from hermes_cli.runtime_state import _bytes
    from pm.plugin_declarations import read_python_declaration
    from pm.workspace import enabled_plugin_dirs, install_node_sidecar

    if (target / "node_modules").is_dir() and (staged / "package.json").is_file() and (
            not (staged / "node_modules").is_dir()
            or any(_bytes(target / name) != _bytes(staged / name) for name in ("package.json", "package-lock.json"))):
        reason = install_node_sidecar(staged, explicit=True)
        if reason:
            raise pc.PluginOperationError(
                f"Node dependencies could not be refreshed: {reason}. "
                "The installed plugin and active environment are unchanged.")
    if target.resolve() not in enabled_plugin_dirs(installing=target):
        return  # a disabled plugin's deps are admitted (and consented) by `hermes plugins enable`
    before, after = read_python_declaration(target), read_python_declaration(staged)
    added = tuple(spec for spec in after.install_requirements if spec not in before.install_requirements)
    if not added and not (after.is_member and not before.is_member):
        return
    if not interactive:
        raise pc.PluginOperationError(
            f"The update declares new Python dependencies ({', '.join(added) or 'in its pyproject.toml'}); "
            f"run `hermes plugins update {target.name}` in a terminal to review them.")
    consented, reason = pc._consent_python_deps(manifest.get("name", target.name), added, pc._console())
    if not consented:
        raise pc.PluginOperationError(
            f"Update declined: {reason}. The installed plugin and active environment are unchanged.")


def update_plugin(
    target: Path,
    *,
    catalog_entry=None,
    interactive: bool = False,
    preserved_files: Path | None = None,
) -> str:
    """Prepare a catalog re-pin or custom Git pull without changing the live tree.

    *interactive*: a terminal user is present to consent to newly declared dependencies;
    the dashboard and the gateway's auto-apply pass False and get a refusal instead."""
    import tempfile

    from hermes_cli import plugins_cmd as pc
    from hermes_cli.plugins_cmd_catalog import refuse_if_installed_removed
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
    refuse_if_installed_removed(target.name, target)
    before = tree_digest(target)
    with tempfile.TemporaryDirectory(prefix=".update-", dir=target.parent) as directory:
        staged = Path(directory) / "plugin"
        try:
            if catalog_entry:
                git_url, subdir = pc._resolve_git_url(source)
                revision = pc._clone_plugin_repo(staged, git_url, catalog_entry.sha)
                staged = pc._resolve_subdir_within(staged, subdir) if subdir else staged
                output = f"Updated to catalog pin {revision}"
                catalog_record = {
                    "name": catalog_entry.name,
                    "repo": catalog_entry.repo,
                    "tier": catalog_entry.tier,
                    "pin": catalog_entry.sha,
                    "sha": revision,
                }
                record.update(
                    catalog=catalog_record,
                    pinned=True,
                    source=pc._canonical_source(git_url, subdir),
                )
                record.pop("catalog_name", None)
                record.pop("catalog_tier", None)
                from hermes_cli.plugins_cmd_catalog import write_catalog_sidecar_record
                write_catalog_sidecar_record(staged, catalog_record, revision)
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
            if preserved_files is not None and preserved_files.exists():
                shutil.copytree(preserved_files, staged, dirs_exist_ok=True)
            manifest = pc._read_manifest_for_install(staged)
            installed_name = str(manifest.get("name") or target.name)
            if catalog_entry is None and installed_name != target.name:
                raise pc.PluginOperationError("The updated plugin changed its installed name; reinstall it explicitly.")
            try:
                new_target = pc._sanitize_plugin_name(installed_name, target.parent)
            except ValueError as exc:
                raise pc.PluginOperationError(str(exc)) from exc
            if new_target != target and new_target.exists():
                raise pc.PluginOperationError(
                    f"The updated plugin renamed itself to '{installed_name}', but that plugin already exists.")
            pc._check_manifest_version(manifest, installed_name)
            pc._scan_plugin_tree(staged, source, force=False)
            pc._copy_example_files(staged, pc._console())
            _refresh_declared_dependencies(target, staged, manifest, interactive=interactive)
            if tree_digest(target) != before:
                raise pc.PluginOperationError("Plugin files changed while preparing the update; retry.")
            record["revision"] = revision
            if new_target == target and tree_digest(staged) == before:
                return output
            publish_plugin(
                staged,
                new_target,
                metadata,
                {**metadata, installed_name: record},
                target_digest=before,
                previous_target=target if new_target != target else None,
            )
            return output
        except pc.PluginOperationError:
            raise
        except Exception as exc:
            raise pc.PluginOperationError(f"Plugin '{target.name}' update was not published: {exc}") from exc
