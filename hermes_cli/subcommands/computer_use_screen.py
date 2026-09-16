"""``hermes computer-use screen`` — the Bot Desktop screen a profile's ``computer_use`` drives on a
headless Linux gateway host, viewable from Hermes Desktop. ``status`` / ``start`` / ``stop`` /
``install`` mirror the Desktop pane's controls for ops shells and cloud images."""

from __future__ import annotations

import getpass
import json
import sys


def _screen_status(args) -> int:
    from tools.bot_desktop import lease, runtime
    st = runtime.status()
    if bool(getattr(args, "json", False)):
        print(json.dumps({**st.as_dict(), "lease": lease.public_view(lease.get())}, indent=2, sort_keys=True))
        return 0 if st.running else 1
    if not st.supported:
        print("Bot Desktop screens run on Linux gateway hosts only (this host keeps its real display).")
        return 1
    if not st.installed:
        print("Bot Desktop: packages missing → " + ", ".join(st.missing))
        print("  Install: " + (st.install_command or "hermes computer-use screen install"))
        return 1
    if st.running:
        holder = lease.public_view(lease.get())
        who = f"human (viewer {holder['viewer_hash']})" if holder["holder"] == lease.HUMAN else "agent"
        print(f"Bot Desktop [{st.profile}]: running on DISPLAY {st.display} ({st.geometry}), pid {st.pid}")
        print(f"  control: {who}   rfb socket: {st.socket}")
        print("  View it: Hermes Desktop → Bots → this bot → Screen")
        return 0
    if st.blocker:
        print(f"Bot Desktop [{st.profile}]: installed, not running. {st.blocker}")
        return 1
    print(f"Bot Desktop [{st.profile}]: installed, not running. Start: hermes computer-use screen start")
    return 1


def _screen_start(args) -> int:
    from tools.bot_desktop import runtime
    try:
        st = runtime.start()
    except RuntimeError as exc:
        print(f"Bot Desktop: {exc}")
        return 1
    print(f"Bot Desktop [{st.profile}]: running on DISPLAY {st.display} ({st.geometry})")
    return 0


def _screen_stop(args) -> int:
    from tools.bot_desktop import runtime
    if runtime.is_supported_host():
        # Same door as display.stop: a human mid-takeover is never yanked by a runbook or a stray
        # `screen stop`; the decision is taken under the lease lock so a takeover cannot race it.
        from tools.bot_desktop import lease
        if lease.release(unless_human=not bool(getattr(args, "force", False))).holder == lease.HUMAN:
            print("Bot Desktop: a human holds this screen; re-run with --force to take it down anyway.")
            return 1
    print("Bot Desktop: stopped" if runtime.stop() else "Bot Desktop: was not running")
    return 0


def _screen_install(args) -> int:
    from tools.bot_desktop import install, runtime
    if not runtime.is_supported_host():
        print("Bot Desktop screens run on Linux gateway hosts only.")
        return 1
    if not runtime.missing_binaries():
        print("Bot Desktop: packages already installed.")
        return 0
    cmd = runtime.install_command()
    if cmd is None:
        print("Bot Desktop: no supported package manager (apt/dnf/pacman) found. Install TigerVNC (Xvnc) and "
              "the Xfce core (xfwm4, xfce4-panel, xfdesktop, xfce4-settings) by hand.")
        return 1
    print(f"Bot Desktop: installing → {cmd}")
    if not bool(getattr(args, "yes", False)) and sys.stdin.isatty():
        answer = input("Proceed? [Y/n] ").strip().lower()
        if answer not in ("", "y", "yes"):
            return 1
    # Same runner as the Desktop pane's Install button: one per-profile slot, list-form spawn, sudo password
    # via stdin (-S) and never on the command line.
    try:
        rc = install.install_packages(ask_password=lambda: getpass.getpass("[sudo] password: "), on_line=print)
    except install.InstallBusy as exc:
        print(f"Bot Desktop: {exc}")
        return 1
    if rc == install.NO_SUDO:
        print("Bot Desktop: this host has no sudo. Run as root on the host:\n  " + cmd.removeprefix("sudo "))
        return 1
    if rc != 0:
        print(f"Bot Desktop: installer exited {rc}")
        return rc or 1
    missing = runtime.missing_binaries()
    if missing:
        print("Bot Desktop: still missing " + ", ".join(missing))
        return 1
    print("Bot Desktop: ready. Start with `hermes computer-use screen start` or from Hermes Desktop.")
    return 0


SCREEN_ACTIONS = {"status": _screen_status, "start": _screen_start, "stop": _screen_stop, "install": _screen_install}


def build_screen_parser(computer_use_sub, add_json_flag) -> None:
    screen = computer_use_sub.add_parser(
        "screen", help="Bot Desktop: the headless screen this profile's computer_use drives (Linux)",
        description="On a headless Linux gateway host Hermes gives each profile its own Xfce screen\n"
            "(TigerVNC Xvnc on a private Unix socket). The agent's computer_use and headed\n"
            "browser act on it; Hermes Desktop shows it live and lets a human take over for\n"
            "logins, 2FA or CAPTCHAs, then hand control back.\n\n"
            "`install` adds the system packages (apt/dnf/pacman); `start`/`stop` manage this\n"
            "profile's screen; `status` shows display, control holder and socket.")
    sub = screen.add_subparsers(dest="computer_use_screen_action")
    st = sub.add_parser("status", help="Show whether this profile's screen is installed/running and who holds control")
    add_json_flag(st, "Emit the status payload as JSON.")
    sub.add_parser("start", help="Start this profile's screen")
    stop = sub.add_parser("stop", help="Stop this profile's screen (hands control back to the agent first)")
    stop.add_argument("--force", action="store_true", help="Stop even while a human holds control from Hermes Desktop")
    inst = sub.add_parser("install", help="Install TigerVNC + Xfce core via the host package manager")
    inst.add_argument("-y", "--yes", action="store_true", help="Do not ask before running the package manager")

    def _cmd(args):
        handler = SCREEN_ACTIONS.get(str(getattr(args, "computer_use_screen_action", None) or ""))
        if handler is not None:
            return handler(args)
        screen.print_help()
    screen.set_defaults(screen_func=_cmd)
