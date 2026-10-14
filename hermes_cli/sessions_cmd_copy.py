"""Cross-profile single-session copy CLI. Source opens observationally."""

from contextlib import ExitStack
import sqlite3


def cmd_copy(args):
    from hermes_constants import get_hermes_home
    from hermes_cli.profiles import get_profile_dir, normalize_profile_name, profile_exists
    from hermes_state import SessionDB

    try:
        target_profile = normalize_profile_name(args.to_profile)
        source_profile = normalize_profile_name(args.from_profile) if args.from_profile else None
        for label, profile in (("Source", source_profile), ("Target", target_profile)):
            if profile and not profile_exists(profile):
                raise ValueError(f"{label} profile '{profile}' does not exist.")
        source_home = get_profile_dir(source_profile) if source_profile else get_hermes_home()
        target_home = get_profile_dir(target_profile)
        source_label = source_profile or (
            "default" if source_home.resolve() == get_profile_dir("default").resolve()
            else source_home.name if source_home.parent.name == "profiles" else "current")
        if not (source_home / "state.db").exists():
            raise ValueError(f"Session '{args.session_id}' not found in profile '{source_label}'.")
        with ExitStack() as stack:
            source_db = SessionDB(db_path=source_home / "state.db", read_only=True)
            stack.callback(source_db.close)
            session_id = source_db.resolve_session_id(args.session_id)
            if not session_id:
                raise ValueError(f"Session '{args.session_id}' not found or ambiguous in profile '{source_label}'.")
            with source_db._read_ctx() as conn:
                omitted = len(source_db._expand_compression_lineage(conn, [session_id])) - 1
            target_db = SessionDB(db_path=target_home / "state.db")
            stack.callback(target_db.close)
            copied_id = source_db.copy_session_to(session_id, target_db, new_session_id=args.new_session_id)
        suffix = f" as '{copied_id}'" if copied_id != session_id else ""
        print(f"Copied session '{session_id}' from profile '{source_label}' to profile '{target_profile}'{suffix}.")
        if omitted:
            print(f"Warning: omitted {omitted} compression continuation session(s); only this session row was copied.")
        print("Profile memory, config, skills, credentials, cron, and gateway routing were not copied.")
        return 0
    except (ValueError, OSError, RuntimeError, sqlite3.Error) as exc:
        print(f"Error: {exc}")
        return 1
