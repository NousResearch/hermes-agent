"""``hermes computer-use screen`` — the Bot Desktop screen a profile's ``computer_use`` drives on a
headless Linux gateway host, viewable from Hermes Desktop. ``status`` / ``start`` / ``stop`` /
``install`` mirror the Desktop pane's controls for ops shells and cloud images."""

from __future__ import annotations

import json
import subprocess
import sys


def _screen_status(args) -> int:
    from tools.bot_desktop import lease, runtime
    st = runtime.status()
    if bool(getattr(args, "json", False)):
        print(json.dumps({**st.as_dict(), "lease": lease.get().as_dict()}, indent=2, sort_keys=True))
        return 0 if st.running else 1
    if not st.supported:
        print("Bot Desktop screens run on Linux gateway hosts only (this host keeps its real display).")
        return 1
    if not st.installed:
        print("Bot Desktop: packages missing → " + ", ".join(st.missing))
        print("  Install: " + (st.install_command or "hermes computer-use screen install"))
        return 1
    if st.running:
        holder = lease.get()
        who = f"human ({holder.viewer_id})" if holder.holder == "human" else "agent"
        print(f"Bot Desktop [{st.profile}]: running on DISPLAY {st.display} ({st.geometry}), pid {st.pid}")
        print(f"  control: {who}   rfb socket: {st.socket}")
        print("  View it: Hermes Desktop → Bots → this bot → Screen")
        return 0
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
    print("Bot Desktop: stopped" if runtime.stop() else "Bot Desktop: was not running")
    return 0


def _screen_install(args) -> int:
    from tools.bot_desktop import runtime
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
    rc = subprocess.run(cmd, shell=True, stdin=None).returncode  # windows-footgun: ok — Linux-only, apt/dnf/pacman
    if rc != 0:
        print(f"Bot Desktop: installer exited {rc}")
        return rc
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
    sub.add_parser("stop", help="Stop this profile's screen (hands control back to the agent first)")
    inst = sub.add_parser("install", help="Install TigerVNC + Xfce core via the host package manager")
    inst.add_argument("-y", "--yes", action="store_true", help="Do not ask before running the package manager")

    def _cmd(args):
        handler = SCREEN_ACTIONS.get(str(getattr(args, "computer_use_screen_action", None) or ""))
        if handler is not None:
            return handler(args)
        screen.print_help()
    screen.set_defaults(screen_func=_cmd)
