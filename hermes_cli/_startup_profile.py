"""Dependency-free CLI profile selection shared by bootstrap and command dispatch.

Keep imports safe before PM activates the application's dependency generation.
"""

import os
import re
import sys
from pathlib import Path

from hermes_cli import _startup_fast


# Resolve --profile/-p or the sticky active_profile before PM reads HERMES_HOME.
# Keep the flag for any dependency re-exec, then consume it before argparse.
_PROFILE_NAME_RE = r"^[a-z0-9][a-z0-9_-]{0,63}$"  # mirrors hermes_cli.profiles._PROFILE_ID_RE
# Set only when -p/--profile was on argv. Sticky active_profile must not count:
# `hermes desktop` with no flag must not overwrite Desktop's stored profile.
_explicit_cli_profile: str | None = None
_profile_override_applied = False
_profile_flag_span: tuple[int, int] | None = None


def _inside_mcp_add_args(argv: list, index: int) -> bool:
    """True once argv reaches `hermes mcp add ... --args <command argv>`.

    ``mcp add --args`` is command-argv passthrough. Flags after that point
    belong to the child MCP command (for example Docker MCP Toolkit's
    ``--profile``), not to Hermes' own profile selector.
    """
    try:
        mcp_index = argv.index("mcp", 0, index)
        argv.index("add", mcp_index + 1, index)
    except ValueError:
        return False
    return True


def _looks_like_hermes_invocation() -> bool:
    """False when ``sys.argv`` belongs to a test runner rather than a ``hermes`` run.

    pytest's own ``-p no:xdist`` reaches ``_scan_profile_flag`` through ``sys.argv`` at import
    time; it must stay a silent skip, while a real ``hermes -p 'Work Bot'`` must fail loudly.
    """
    return "pytest" not in (sys.argv[0] or "")


def _exit_invalid_profile_name(value: str) -> None:
    from hermes_cli.profiles import _invalid_profile_name_error

    print(f"Error: {_invalid_profile_name_error(value)}", file=sys.stderr)
    print("Run `hermes profile list` to see your profiles.", file=sys.stderr)
    sys.exit(2)


def _looks_like_option_value(value: str) -> bool:
    """A ``-p`` value that clearly belongs to some other tool (pytest's ``-p no:xdist``, a
    third-party ``-p --flag``), never a mistyped profile name."""
    return value.startswith("-") or ":" in value


def _scan_profile_flag(argv: list) -> tuple:
    """Find -p/--profile/--profile= in argv -> (name, tokens_consumed, index).

    Historically the flag worked even after the subcommand (`hermes chat -p
    coder`), so scan broadly; stop at ``--`` and at the `mcp add --args`
    passthrough region. The value is normalised (strip + casefold, matching
    ``profiles.normalize_profile_name``) before validation so ``-p Work`` selects
    ``work``. A value that cannot be a profile name is rejected so
    resolve_profile_env never sys.exits on it; the rejection is explained (exit 2)
    only when the flag comes BEFORE the first subcommand token under a real
    ``hermes`` run — after a subcommand, ``-p`` may belong to that subcommand or a
    plugin (`hermes kanban ... -p 8080`), and option-looking values (``no:xdist``,
    ``--flag``) are always a silent skip.
    """
    from hermes_cli._parser import top_level_value_flag_sets

    value_flags, optional_value_flags = top_level_value_flag_sets()
    i = 0
    saw_subcommand = False
    while i < len(argv):
        arg = argv[i]
        if arg == "--" or (arg == "--args" and _inside_mcp_add_args(argv, i)):
            break
        if arg in {"--profile", "-p"} and i + 1 < len(argv):
            raw = argv[i + 1]
            value = raw.strip().casefold()
            if re.match(_PROFILE_NAME_RE, value):
                return value, 2, i
            if not saw_subcommand and not _looks_like_option_value(raw) and _looks_like_hermes_invocation():
                _exit_invalid_profile_name(raw)
            break
        if arg.startswith("--profile="):
            return arg.split("=", 1)[1].strip().casefold(), 1, i
        takes_value = "=" not in arg and i + 1 < len(argv) and (
            arg in value_flags
            or (arg in optional_value_flags and not argv[i + 1].startswith("-"))
        )
        if not takes_value and not arg.startswith("-"):
            saw_subcommand = True
        i += 2 if takes_value else 1
    return None, 0, None


def _resolve_sudo_user_profile_env(name: str) -> str | None:
    """Resolve `sudo hermes -p <name>` against the invoking user's home.

    This runs before argparse, so `--run-as-user` is not available yet. For
    sudo invocations the best signal is SUDO_USER: root is only doing the
    privileged install/start action; the profile store belongs to the user.
    """
    if name == "default":
        return None
    from hermes_constants import named_profile_is_live, sudo_invoker_default_home

    sudo_home = sudo_invoker_default_home()
    if sudo_home is None:
        return None
    candidate = sudo_home / "profiles" / name
    return str(candidate) if named_profile_is_live(candidate) else None


def _under_gateway_supervisor(argv: list) -> bool:
    """A supervisor-launched gateway child must NOT follow the sticky active_profile.

    Each supervised slot has a fixed profile identity: named slots pass
    ``-p <name>`` or pin HERMES_HOME to the profile dir; a bare invocation
    means "the root HERMES_HOME profile". If a supervised default-profile
    child read active_profile, switching the active profile (dashboard,
    ``hermes profile use``) would silently redirect the default gateway into
    that profile — adopting its credentials and double-polling a Telegram
    token already owned by that profile's own gateway (#74872).

    Markers (see gateway/restart.py ``is_gateway_supervisor_process``):
    HERMES_SUPERVISED_CHILD (systemd unit / launchd plist / Windows task),
    HERMES_S6_SUPERVISED_CHILD (legacy s6 container), INVOCATION_ID (systemd
    service children only — consulted ONLY for gateway commands because it is
    inherited by every descendant of a systemd-launched process, e.g.
    self-hosted CI runners), HERMES_GATEWAY_EXTERNAL_SUPERVISOR (explicit
    opt-in). XPC_SERVICE_NAME is deliberately NOT consulted: interactive macOS
    terminals set it too.
    """
    if os.environ.get("HERMES_SUPERVISED_CHILD") or os.environ.get("HERMES_S6_SUPERVISED_CHILD"):
        return True
    is_gateway_cmd = next((a for a in argv if not a.startswith("-")), None) == "gateway"
    if is_gateway_cmd and os.environ.get("INVOCATION_ID"):
        return True
    return os.environ.get(
        "HERMES_GATEWAY_EXTERNAL_SUPERVISOR", ""
    ).strip().lower() in {"1", "true", "yes", "on"}


def _s6_supervised_gateway_run(argv: list) -> bool:
    """A bare ``gateway run`` inside the s6 image names the ``gateway-default`` slot too.

    ``_maybe_redirect_run_to_s6_supervision`` turns it into a start of the supervised slot for the
    current profile, and it is the image's own CMD. Following the sticky ``active_profile`` there
    started that profile's named slot on every container boot: the one the boot reconciler just
    registered down, because a started named slot is a second gateway beside the multiplexer.
    ``--no-supervise`` keeps the foreground run, which follows ``active_profile`` as before (#22502).
    """
    words = [a for a in argv if not a.startswith("-")]
    if words[:2] != ["gateway", "run"] or "--no-supervise" in argv:
        return False
    if os.environ.get("HERMES_GATEWAY_NO_SUPERVISE", "").lower() in ("1", "true", "yes"):
        return False
    from hermes_cli.service_manager import _s6_running
    return _s6_running()


def explicit_cli_profile() -> str | None:
    """Profile named by a consumed ``-p``/``--profile`` flag, else None.

    Sticky ``active_profile`` is not explicit. Desktop launch must not overwrite
    its stored profile when the user omitted the flag.
    """
    return _explicit_cli_profile


def _apply_profile_override(*, preserve_argv: bool = False) -> None:
    """Select the launch home; retain flags until PM has finished any re-exec."""
    global _explicit_cli_profile, _profile_override_applied, _profile_flag_span
    _startup_fast.normalize_hermes_home_env()
    _profile_override_applied = True
    _profile_flag_span = None
    _explicit_cli_profile = None
    argv = sys.argv[1:]
    profile_name, consume, profile_index = _scan_profile_flag(argv)

    # HERMES_HOME already set with no explicit flag: trust it only when it
    # points at a specific profile dir ("profiles" as immediate parent). If it
    # points at the hermes root (systemd hardcodes HERMES_HOME=/root/.hermes)
    # we must still read active_profile — the user may have run
    # `hermes profile use` and the gateway should honour it (#22502).
    hermes_home_env = os.environ.get("HERMES_HOME", "")
    if profile_name is None and hermes_home_env and Path(hermes_home_env).parent.name == "profiles":
        return

    if (profile_name is None and not _under_gateway_supervisor(argv)
            and not _startup_fast.is_desktop_ssh_backend_argv(argv)
            and not _s6_supervised_gateway_run(argv)):
        try:
            from hermes_constants import get_default_hermes_root

            active_path = get_default_hermes_root() / "active_profile"
            if active_path.exists():
                name = active_path.read_text(encoding="utf-8-sig").strip()
                if name and name != "default":
                    profile_name = name  # consume stays 0: nothing to strip
        except (UnicodeDecodeError, OSError):
            pass  # corrupted file, skip

    if profile_name is None:
        return
    try:
        from hermes_cli.profiles import resolve_profile_env

        hermes_home = resolve_profile_env(profile_name)
    except FileNotFoundError as exc:
        hermes_home = _resolve_sudo_user_profile_env(profile_name)
        if not hermes_home:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        # A bug in profiles.py must NEVER prevent hermes from starting
        print(f"Warning: profile override failed ({exc}), using default", file=sys.stderr)
        return
    os.environ["HERMES_HOME"] = hermes_home
    if consume > 0:
        _explicit_cli_profile = profile_name
    if consume > 0 and profile_index is not None:
        _profile_flag_span = (profile_index + 1, consume)
        if not preserve_argv:
            _consume_profile_flag()


def _consume_profile_flag() -> None:
    global _profile_flag_span
    if _profile_flag_span is not None:
        start, consume = _profile_flag_span
        sys.argv = sys.argv[:start] + sys.argv[start + consume :]
        _profile_flag_span = None


def finish_profile_override() -> None:
    """Consume the bootstrap's flag once, retaining its explicit profile identity.

    The CLI can be loaded as both __main__ and hermes_cli.main in one process.
    Resolving again after consuming ``-p default`` would follow active_profile.
    Non-bootstrap/embedded callers still resolve on their first CLI import.
    """
    if not _profile_override_applied:
        _apply_profile_override()
    else:
        _consume_profile_flag()
