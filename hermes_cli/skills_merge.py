"""Manual review UI for bundled-skill update conflicts."""
import json
import sys

from tools.skills_sync_merge import refresh_pending, resolve_skill, state_dir


def do_resolve(name, accept=None, refresh=False):
    from tools.skills_sync import _hermes_home

    if refresh:
        refresh_pending(name)
    pending = state_dir(_hermes_home(), name) / "pending"
    if not pending.exists():
        print(f"No pending update for {name}.")
        return True
    data = json.loads((pending / "pending.json").read_text(encoding="utf-8-sig"))
    print(f"Pending update for {name}: {pending}")
    print("Conflicts: " + ", ".join(data["conflicts"]))
    print("Compare base/, local/, upstream/; edit merged/ to combine both versions.")
    if accept is None and sys.stdin.isatty() and sys.stdout.isatty():
        from hermes_cli.curses_ui import curses_single_select

        choices = ["merged", "local", "upstream"]
        choice = curses_single_select(
            "Accept this reviewed skill update?",
            ["Accept edited merged/", "Keep local (acknowledge this upstream version)",
             "Accept upstream (discard local customizations)"],
            default_index=3, cancel_label="Leave pending",
        )
        accept = choices[choice] if choice is not None else None
    if accept is None:
        print(f"After review: hermes skills resolve {name} --accept merged|local|upstream")
        return True
    try:
        resolve_skill(name, accept)
    except (OSError, ValueError) as exc:
        print(f"Could not resolve {name}: {exc}")
        return False
    print(f"Accepted {accept} for {name}; future updates retain the new upstream base.")
    return True
