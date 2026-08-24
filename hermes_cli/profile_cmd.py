"""``hermes profile`` command — one handler per action, dispatched by ``PROFILE_ACTIONS``.

Imports from ``hermes_cli.profiles`` stay lazy (inside each handler) so tests can monkeypatch
the module attributes.
"""

from __future__ import annotations

from pathlib import Path
import os
import sys


def _die(msg: str, code: int = 1, *, err: bool = False) -> None:
    print(msg, file=sys.stderr if err else sys.stdout)
    sys.exit(code)


def _confirm(prompt: str) -> bool:
    """y/N prompt; EOF / Ctrl-C count as "no"."""
    try:
        answer = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = ""
    return answer in {"y", "yes"}


def _is_active(p, active: str) -> bool:
    return p.name == active or (active == "default" and p.is_default)


def _env_file_has_key(env_path: Path, key: str) -> bool:
    """True when *key* is assigned in *env_path*. Read as utf-8-sig: a Notepad-edited .env can
    carry a BOM that would hide the first key behind U+FEFF. A mis-encoded file (UnicodeDecodeError
    is a ValueError, not OSError) must not abort the install preview — skip the pre-check."""
    if not env_path.is_file():
        return False
    try:
        # .env is written as UTF-8 everywhere in the codebase, but a Notepad-edited file can carry a BOM —
        # read as utf-8-sig so the first key isn't hidden behind U+FEFF (#62617).
        for raw in env_path.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and line.split("=", 1)[0].strip() == key:
                return True
    except (OSError, UnicodeDecodeError):
        pass
    return False


def _render_distribution_plan(plan) -> None:
    """Print a human-readable summary of a pending distribution install."""
    from hermes_cli.profile_distribution import MANIFEST_FILENAME
    mf = plan.manifest
    print(f"\nDistribution: {mf.name} v{mf.version}")
    if mf.description:
        print(f"  {mf.description}")
    if mf.author:
        print(f"  Author:   {mf.author}")
    if mf.hermes_requires:
        print(f"  Requires: Hermes {mf.hermes_requires}")
    print(f"  Source:   {plan.provenance}")
    print(f"  Target:   {plan.target_dir}")
    if plan.existing:
        # Updating an existing distribution (dist-owned overwritten, config preserved, user
        # data untouched) vs overwriting a hand-built plain profile (same mechanics, but the
        # user didn't sign up for it).
        if (plan.target_dir / MANIFEST_FILENAME).is_file():
            print("  (profile exists — will overwrite distribution-owned files only)")
        else:
            print(
                "  ⚠ Profile exists but is NOT a distribution.  Installing here will\n"
                "    overwrite its SOUL.md, skills/, cron/, and mcp.json.\n"
                "    Your memories, sessions, auth.json, and .env will be preserved,\n"
                "    but any hand-edits to distribution-owned files will be lost."
            )
    if mf.env_requires:
        print("\n  Env vars:")
        for er in mf.env_requires:
            tag = "required" if er.required else "optional"
            # Shell environment OR the target profile's .env — don't nag about set keys.
            already = os.environ.get(er.name) is not None or (
                plan.target_dir.is_dir() and _env_file_has_key(plan.target_dir / ".env", er.name)
            )
            status = "✓ set" if already else ("needs setting" if er.required else "—")
            line = f"    • {er.name} ({tag}, {status})"
            if er.description:
                line += f" — {er.description}"
            print(line)
    if plan.has_cron:
        print(
            "\n  ⚠ This distribution ships cron jobs.  They will NOT run "
            "automatically — review and enable manually."
        )


def _profile_status(args):
    """Bare ``hermes profile`` — show current profile status."""
    from hermes_constants import display_hermes_home
    from hermes_cli.profiles import format_profile_label, get_active_profile_name, list_profiles
    profile_name = get_active_profile_name()
    dhh = display_hermes_home()
    current = next((p for p in list_profiles() if _is_active(p, profile_name)), None)
    label = format_profile_label(profile_name, current.display_name if current else "")
    print(f"\nActive profile: {label}")
    print(f"Path:           {dhh}")
    if current is not None:
        p = current
        if p.model:
            print(f"Model:          {p.model}" + (f" ({p.provider})" if p.provider else ""))
        print(f"Gateway:        {'running' if p.gateway_running else 'stopped'}")
        print(f"Skills:         {p.skill_count} installed")
        if p.alias_path:
            print(f"Alias:          {p.alias_name or p.name} → hermes -p {p.name}")
    print()


def _profile_list(args):
    from hermes_cli.profiles import format_profile_label, get_active_profile_name, list_profiles
    profiles = list_profiles()
    active = get_active_profile_name()
    if not profiles:
        print("No profiles found.")
        return
    print(f"\n {'Profile':<16} {'Model':<28} {'Gateway':<12} {'Alias':<12} {'Distribution'}")
    print(f" {'─' * 15}    {'─' * 27}    {'─' * 11}    {'─' * 11}    {'─' * 20}")
    for p in profiles:
        marker = " ◆" if _is_active(p, active) else "  "
        name = format_profile_label(p.name, p.display_name)
        model = (p.model or "—")[:26]
        gw = "running" if p.gateway_running else "stopped"
        alias = (p.alias_name or p.name) if p.alias_path and not p.is_default else "—"
        dist = f"{p.distribution_name}@{p.distribution_version or '?'}"[:30] if p.distribution_name else "—"
        print(f"{marker}{name:<15} {model:<28} {gw:<12} {alias:<12} {dist}")
    print()


def _profile_use(args):
    from hermes_cli.profiles import set_active_profile
    name = args.profile_name
    try:
        set_active_profile(name)
        print("Switched to: default (~/.hermes)" if name == "default" else f"Switched to: {name}")
    except (ValueError, FileNotFoundError) as e:
        _die(f"Error: {e}")


def _profile_create(args):
    from hermes_cli.profiles import (
        _get_wrapper_dir, _is_wrapper_dir_in_path, check_alias_collision, create_profile,
        create_wrapper_script, get_active_profile_name, seed_profile_skills,
    )
    name = args.profile_name
    clone = getattr(args, "clone", False)
    clone_all = getattr(args, "clone_all", False)
    no_alias = getattr(args, "no_alias", False)
    no_skills = getattr(args, "no_skills", False)
    clone_from = getattr(args, "clone_from", None)
    clone_config = clone or clone_from is not None
    cloned = clone_config or clone_all
    try:
        profile_dir = create_profile(
            name=name, clone_from=clone_from, clone_all=clone_all, clone_config=clone_config,
            no_alias=no_alias, no_skills=no_skills, description=getattr(args, "description", None),
        )
    except (ValueError, FileExistsError, FileNotFoundError) as e:
        _die(f"Error: {e}")
    print(f"\nProfile '{name}' created at {profile_dir}")
    if cloned:
        source_label = clone_from or get_active_profile_name()
        if clone_all:
            print(f"Full copy from {source_label} (excluding session history, backups, and snapshots).")
        else:
            print(f"Cloned config, .env, SOUL.md, and skills from {source_label}.")
        # Auto-clone Honcho config for the new profile (only with clone operations)
        try:
            from plugins.memory.honcho.cli import clone_honcho_for_profile
            if clone_honcho_for_profile(name):
                print(f"Honcho config cloned (peer: {name})")
        except Exception:
            pass  # Honcho plugin not installed or not configured
    else:
        # Fresh profiles only: clones already carry the source's (user-curated) skills.
        result = seed_profile_skills(profile_dir)
        if result and result.get("skipped_opt_out"):
            print("No bundled skills seeded (--no-skills). Delete .no-bundled-skills in the profile to opt back in.")
        elif result:
            print(f"{len(result.get('copied', []))} bundled skills synced.")
        else:
            print(f"⚠ Skills could not be seeded. Run `{name} update` to retry.")
    if not no_alias:
        collision = check_alias_collision(name)
        if collision:
            print(f"\n⚠ Cannot create alias '{name}' — {collision}")
            print(f"  Choose a custom alias:  hermes profile alias {name} --name <custom>")
            print(f"  Or access via flag:     hermes -p {name} chat")
        else:
            wrapper_path = create_wrapper_script(name)
            if wrapper_path:
                print(f"Wrapper created: {wrapper_path}")
                if not _is_wrapper_dir_in_path():
                    print(f"\n⚠ {_get_wrapper_dir()} is not in your PATH.")
                    print("  Add to your shell config (~/.bashrc or ~/.zshrc):")
                    print('    export PATH="$HOME/.local/bin:$PATH"')
    try:
        profile_dir_display = "~/" + profile_dir.relative_to(Path.home()).as_posix()
    except ValueError:
        profile_dir_display = str(profile_dir)
    print("\nNext steps:")
    print(f"  {name} setup              Configure API keys and model")
    print(f"  {name} chat               Start chatting")
    print(f"  {name} gateway start      Start the messaging gateway")
    if clone or clone_all:
        print(f"\n  Edit {profile_dir_display}/.env for different API keys")
        print(f"  Edit {profile_dir_display}/SOUL.md for different personality")
    else:
        print(f"\n  ⚠ This profile has no API keys yet. Run '{name} setup' first,")
        print("    or it will inherit keys from your shell environment.")
        print(f"  Edit {profile_dir_display}/SOUL.md to customize personality")
    print()


def _profile_delete(args):
    from hermes_cli.profiles import delete_profile
    try:
        delete_profile(args.profile_name, yes=getattr(args, "yes", False))
    except (ValueError, FileNotFoundError) as e:
        _die(f"Error: {e}")


def _describe_target_dir(name: str) -> Path:
    """Profile dir for ``describe``: ``default`` maps to the CURRENT home (get_hermes_home),
    everything else to its named directory."""
    from hermes_cli import profiles as _profiles_mod
    if _profiles_mod.normalize_profile_name(name) == "default":
        from hermes_constants import get_hermes_home as _hh
        return Path(_hh())
    return _profiles_mod.get_profile_dir(name)


def _profile_describe(args):
    from hermes_cli import profiles as _profiles_mod
    all_flag = bool(getattr(args, "all_missing", False))
    auto_flag = bool(getattr(args, "auto", False))
    overwrite_flag = bool(getattr(args, "overwrite", False))
    text_value = getattr(args, "text", None)
    name = getattr(args, "profile_name", None)
    if all_flag and not auto_flag:
        _die("profile describe: --all requires --auto", 2, err=True)
    if all_flag and (text_value or name):
        _die("profile describe: --all is mutually exclusive with a profile name / --text", 2, err=True)
    if not all_flag and not name:
        _die("profile describe: profile name is required (or --all --auto)", 2, err=True)
    if text_value and auto_flag:
        _die("profile describe: --text is mutually exclusive with --auto", 2, err=True)

    # Show current description if no operation requested.
    if name and not text_value and not auto_flag:
        try:
            profile_dir = _describe_target_dir(name)
        except Exception as exc:
            _die(f"Error: {exc}", err=True)
        if not profile_dir.is_dir():
            _die(f"Error: profile '{name}' not found", err=True)
        meta = _profiles_mod.read_profile_meta(profile_dir)
        desc = meta.get("description") or ""
        if not desc:
            print(f"(no description set for '{name}')")
        else:
            tag = "[auto] " if meta.get("description_auto") else ""
            print(f"{tag}{desc}")
        sys.exit(0)

    # --text path: just write the user-authored description.
    if text_value:
        try:
            _profiles_mod.write_profile_meta(_describe_target_dir(name), description=text_value, description_auto=False)
            print(f"Description updated for '{name}'.")
        except Exception as exc:
            _die(f"Error: {exc}", err=True)
        sys.exit(0)

    # --auto path: invoke the LLM describer.
    from hermes_cli import profile_describer as _pd
    if all_flag:
        targets = _pd.list_describable_profiles(missing_only=True)
        if not targets:
            _die("All profiles already have descriptions.", 0)
    else:
        targets = [name]
    ok_count = 0
    for tgt in targets:
        outcome = _pd.describe_profile(tgt, overwrite=overwrite_flag)
        if outcome.ok:
            ok_count += 1
            print(f"Described '{outcome.profile_name}': {outcome.description}")
        else:
            print(f"profile describe {outcome.profile_name}: {outcome.reason}", file=sys.stderr)
    sys.exit(0 if (ok_count > 0 if all_flag else ok_count == 1) else 1)


def _profile_show(args):
    name = args.profile_name
    from hermes_cli.profiles import (
        get_profile_dir, profile_exists, _read_config_model, _check_gateway_running,
        _served_by_running_multiplexer, _count_skills, _read_distribution_meta, _wrapper_path,
        find_alias_for_profile, format_profile_label, read_profile_meta,
    )
    if not profile_exists(name):
        _die(f"Error: Profile '{name}' does not exist.")
    profile_dir = get_profile_dir(name)
    model, provider = _read_config_model(profile_dir)
    gw = _check_gateway_running(profile_dir) or _served_by_running_multiplexer(name)
    dist_name, dist_version, dist_source = _read_distribution_meta(profile_dir)
    alias_name = find_alias_for_profile(name)
    display = read_profile_meta(profile_dir).get("display_name", "")
    print(f"\nProfile: {format_profile_label(name, display)}")
    print(f"Path:    {profile_dir}")
    if model:
        print(f"Model:   {model}" + (f" ({provider})" if provider else ""))
    print(f"Gateway: {'running' if gw else 'stopped'}")
    print(f"Skills:  {_count_skills(profile_dir)}")
    print(f".env:    {'exists' if (profile_dir / '.env').exists() else 'not configured'}")
    print(f"SOUL.md: {'exists' if (profile_dir / 'SOUL.md').exists() else 'not configured'}")
    if dist_name:
        print(f"Distribution: {dist_name}@{dist_version or '?'}")
        if dist_source:
            print(f"Installed from: {dist_source}")
        print(f"  (run `hermes profile info {name}` for full manifest)")
    if alias_name:
        print(f"Alias:   {alias_name} → hermes -p {name}  ({_wrapper_path(alias_name)})")
    print()


def _profile_alias(args):
    from hermes_cli.profiles import (
        _get_wrapper_dir, _is_wrapper_dir_in_path, check_alias_collision, create_wrapper_script,
        profile_exists, remove_wrapper_script, validate_alias_name,
    )
    name = args.profile_name
    remove = getattr(args, "remove", False)
    custom_name = getattr(args, "alias_name", None)
    if not profile_exists(name):
        _die(f"Error: Profile '{name}' does not exist.")
    alias_name = custom_name or name
    try:
        validate_alias_name(alias_name)
    except ValueError as exc:
        _die(f"Error: {exc}")
    if remove:
        if remove_wrapper_script(alias_name):
            print(f"✓ Removed alias '{alias_name}'")
        else:
            print(f"No alias '{alias_name}' found to remove.")
        return
    collision = check_alias_collision(alias_name)
    if collision:
        _die(f"Error: {collision}")
    wrapper_path = create_wrapper_script(alias_name, target=name if custom_name else None)
    if wrapper_path:
        print(f"✓ Alias created: {wrapper_path}")
        if not _is_wrapper_dir_in_path():
            print(f"⚠ {_get_wrapper_dir()} is not in your PATH.")


def _profile_rename(args):
    from hermes_cli.profiles import normalize_profile_name, rename_profile
    try:
        new_dir = rename_profile(args.old_name, args.new_name)
        if normalize_profile_name(args.old_name) != "default":
            print(f"\nProfile renamed: {args.old_name} → {args.new_name}")
            print(f"Path: {new_dir}\n")
    except (ValueError, FileExistsError, FileNotFoundError) as e:
        _die(f"Error: {e}")


def _profile_export(args):
    from hermes_cli.profiles import export_profile, get_profile_export_path
    name = args.profile_name
    try:
        output = args.output or str(get_profile_export_path(name))
        result_path = export_profile(name, output)
        print(f"✓ Exported '{name}' to {result_path}")
    except (ValueError, FileNotFoundError, OSError) as e:
        _die(f"Error: {e}")


def _profile_import(args):
    from hermes_cli.profiles import check_alias_collision, create_wrapper_script, import_profile
    try:
        profile_dir = import_profile(args.archive, name=getattr(args, "import_name", None))
        name = profile_dir.name
        print(f"✓ Imported profile '{name}' at {profile_dir}")
        if not check_alias_collision(name):
            wrapper_path = create_wrapper_script(name)
            if wrapper_path:
                print(f"  Wrapper created: {wrapper_path}")
        print()
    except (ValueError, FileExistsError, FileNotFoundError) as e:
        _die(f"Error: {e}")


def _profile_install(args):
    import tempfile
    from hermes_cli.profile_distribution import DistributionError, install_distribution, plan_install
    try:
        # Preview: stage into a scratch dir, show the manifest, then do the real install.
        # The double-stage avoids any side-effects if the user declines.
        with tempfile.TemporaryDirectory(prefix="hermes_dist_preview_") as tmp:
            plan = plan_install(args.source, Path(tmp), override_name=getattr(args, "install_name", None))
            _render_distribution_plan(plan)
            if not getattr(args, "yes", False) and not _confirm("\nProceed with install? [y/N] "):
                print("Install cancelled.")
                return
        plan = install_distribution(
            args.source, name=getattr(args, "install_name", None), force=getattr(args, "force", False),
            create_alias=getattr(args, "alias", False),
        )
        print(f"\n✓ Installed '{plan.manifest.name}' v{plan.manifest.version}")
        print(f"  Profile path: {plan.target_dir}")
        if plan.manifest.env_requires:
            print(
                f"  Next: copy .env.EXAMPLE to .env and fill in required keys:\n"
                f"    {plan.target_dir}/.env.EXAMPLE"
            )
        if plan.has_cron:
            print(
                "  Cron jobs were included but are NOT scheduled automatically.\n"
                f"  Review them with:  hermes -p {plan.manifest.name} cron list"
            )
        print(f"\n  Use with:      hermes -p {plan.manifest.name} chat")
    except (DistributionError, ValueError) as e:
        _die(f"Error: {e}")


def _profile_update(args):
    from hermes_cli.profile_distribution import DistributionError, read_manifest, update_distribution
    from hermes_cli.profiles import get_profile_dir, normalize_profile_name
    try:
        canon = normalize_profile_name(args.profile_name)
        current = read_manifest(get_profile_dir(canon))
        if current is None:
            _die(
                f"Error: Profile '{canon}' is not a distribution (no distribution.yaml). "
                "Only profiles installed via `hermes profile install` can be updated."
            )
        force_config = getattr(args, "force_config", False)
        if not getattr(args, "yes", False):
            print(f"\nUpdate '{canon}' from: {current.source or '(no source)'}")
            print(f"  Currently at version {current.version}")
            if force_config:
                print("  --force-config set: config.yaml WILL be overwritten.")
            else:
                print("  config.yaml will be preserved (pass --force-config to overwrite).")
            print("  User data (memories, sessions, auth, .env) will NOT be touched.")
            if not _confirm("\nProceed? [y/N] "):
                print("Update cancelled.")
                return
        plan = update_distribution(canon, force_config=force_config)
        print(f"\n✓ Updated '{plan.manifest.name}' → v{plan.manifest.version}")
        if plan.has_cron:
            print(f"  Cron files were refreshed.  Review with:  hermes -p {plan.manifest.name} cron list")
    except (DistributionError, ValueError) as e:
        _die(f"Error: {e}")


_INFO_FIELDS = (
    ("description", "Description:  "),
    ("author", "Author:       "),
    ("license", "License:      "),
    ("hermes_requires", "Requires:     Hermes "),
    ("source", "Source:       "),
    ("installed_at", "Installed:    "),
)


def _profile_info(args):
    from hermes_cli.profile_distribution import describe_distribution, DistributionError
    try:
        data = describe_distribution(args.profile_name)
    except (DistributionError, ValueError) as e:
        _die(f"Error: {e}")
    if not data:
        print(f"Profile '{args.profile_name}' is not a distribution (no distribution.yaml).")
        return
    print(f"\nDistribution: {data.get('name')}")
    print(f"Version:      {data.get('version', '?')}")
    for key, label in _INFO_FIELDS:
        if data.get(key):
            print(f"{label}{data[key]}")
    env_reqs = data.get("env_requires") or []
    if env_reqs:
        print("\nEnvironment variables:")
        for er in env_reqs:
            tag = "required" if er.get("required", True) else "optional"
            line = f"  {er['name']} ({tag})"
            if er.get("description"):
                line += f" — {er['description']}"
            print(line)
            if er.get("default") is not None:
                print(f"      default: {er['default']}")
    print()


# Order mirrors the original if/elif chain; None = bare ``hermes profile``.
PROFILE_ACTIONS = {
    None: _profile_status,
    'list': _profile_list,
    'use': _profile_use,
    'create': _profile_create,
    'delete': _profile_delete,
    'describe': _profile_describe,
    'show': _profile_show,
    'alias': _profile_alias,
    'rename': _profile_rename,
    'export': _profile_export,
    'import': _profile_import,
    'install': _profile_install,
    'update': _profile_update,
    'info': _profile_info,
}


def cmd_profile(args):
    """Profile management — create, delete, list, switch, alias."""
    handler = PROFILE_ACTIONS.get(getattr(args, "profile_action", None))
    if handler is not None:
        return handler(args)
