"""Gateway liveness for ``hermes doctor`` (#139549).

Linger, s6 and Windows autostart only prove a supervisor unit was installed.
When A2A is configured, the gateway is reachable only if ``gateway.pid`` names
a live process whose start time matches the file — a recycled pid must not
count — and a TCP connect to 127.0.0.1 succeeds on every configured listener.
``doctor --fix`` offers ``hermes gateway start`` and does not spawn it.
"""

from __future__ import annotations

import socket
from pathlib import Path

from hermes_cli.doctor_report import (
    Finding, _section, check_fail, check_info, check_ok, doctor_check,
)

# Same default the A2A adapter binds when the platform is enabled without a port
# (plugins/platforms/a2a/adapter.py). Kept here so doctor does not import the adapter.
_A2A_DEFAULT_PORT = 9900
_START_CMD = "hermes gateway start"
_CONNECT_TIMEOUT_S = 0.4
_BAD = object()


def _coerce_port(value: object) -> int | None:
    """TCP port, or None when *value* is not an integer in 1..65535."""
    if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        return None
    try:
        port = int(str(value).strip()) if isinstance(value, str) else int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return port if 1 <= port <= 65535 else None


def _as_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return None


def _explicit_flag(block: dict) -> bool | None:
    """``enabled`` on the platform block, else inside ``extra`` (either spelling)."""
    if "enabled" in block:
        return _as_bool(block.get("enabled"))
    extra = block.get("extra")
    if isinstance(extra, dict) and "enabled" in extra:
        return _as_bool(extra.get("enabled"))
    return None


def _raw_block_port(block: dict) -> object:
    """Port from ``extra`` (wins) or the block top level: int, ``_BAD``, or None."""
    extra = block.get("extra") if isinstance(block.get("extra"), dict) else {}
    for value in (extra.get("port"), block.get("port")):
        if value in (None, ""):
            continue
        port = _coerce_port(value)
        return port if port is not None else _BAD
    return None


def _name_listed(value: object) -> bool:
    if not isinstance(value, list):
        return False
    return any(str(item).strip().lower() == "a2a" for item in value)


def _a2a_blocks(gateway: dict, cfg: dict) -> list[dict]:
    """Platform blocks in merge order: later entries win (config_loader)."""
    blocks: list[dict] = []
    for source in (gateway.get("platforms"), cfg.get("platforms")):
        if isinstance(source, dict) and isinstance(source.get("a2a"), dict):
            blocks.append(source["a2a"])
    shorthand = gateway.get("a2a")
    if isinstance(shorthand, dict):
        blocks.append(shorthand)
    return blocks


def _fold_block(state: tuple[bool, bool, object], block: dict) -> tuple[bool, bool, object]:
    disabled, enabled, port = state
    flag = _explicit_flag(block)
    if flag is True:
        disabled, enabled = False, True
    elif flag is False:
        disabled, enabled = True, False
    found = _raw_block_port(block)
    if found is not None:
        port = found
    return disabled, enabled, port


def _yaml_state(cfg: dict) -> tuple[bool, bool, object]:
    """(explicitly_disabled, enabled, port-or-_BAD-or-None) for one home's config."""
    if not isinstance(cfg, dict):
        return False, False, None
    gateway = cfg.get("gateway") if isinstance(cfg.get("gateway"), dict) else {}
    state: tuple[bool, bool, object] = (
        False,
        _name_listed(gateway.get("platforms")) or _name_listed(cfg.get("platforms")),
        None,
    )
    for block in _a2a_blocks(gateway, cfg):
        state = _fold_block(state, block)
    return state


def _load_cfg(home: Path) -> dict:
    # Raw file: a liveness probe must not expand another profile's ``${VAR}``
    # through this process's env, and must not write a config backup.
    from hermes_cli.config import read_user_config_raw

    cfg = read_user_config_raw(home / "config.yaml")
    return cfg if isinstance(cfg, dict) else {}


def _env_port(home: Path) -> object:
    """``A2A_PORT`` from this home's ``.env`` only.

    Process env is the launch profile under multiplex, so a secondary must not
    inherit it. ``load_env_file`` is the shared tokenizer and does not touch
    the environment.
    """
    from agent.secret_scope import load_env_file

    raw = (load_env_file(home / ".env").get("A2A_PORT") or "").strip()
    if not raw:
        return None
    port = _coerce_port(raw)
    return port if port is not None else _BAD


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left == right


def _scope() -> tuple[Path, list[Path]]:
    """(home that owns gateway.pid, homes whose A2A ports that process serves).

    A named profile that opted out of the host multiplexer keeps its own pid
    file. Everyone else is served by the default profile's gateway, so a
    doctor run from a satellite still has to prove that process and every
    listener it should have open.
    """
    from hermes_constants import get_default_hermes_root, get_hermes_home, profile_name_for_home

    active = get_hermes_home()
    name = profile_name_for_home(active)
    if name not in (None, "default"):
        from hermes_cli.profiles import profile_is_standalone

        if profile_is_standalone(active):
            return active, [active]
    from hermes_cli.profiles import profiles_to_serve

    default = get_default_hermes_root()
    homes = [path for _name, path in profiles_to_serve(True)]
    if not any(_same_path(active, path) for path in homes):
        homes.insert(0, active)
    return default, homes


def _port_for_home(home: Path) -> object:
    """Listener port, None when this home did not configure A2A, or ``_BAD``."""
    disabled, enabled, raw_port = _yaml_state(_load_cfg(home))
    if disabled:
        return None
    env_port = _env_port(home)
    if env_port is not None:
        return env_port
    if not enabled:
        return None
    if raw_port is None:
        return _A2A_DEFAULT_PORT
    return raw_port


def _configured_listeners(homes: list[Path]) -> tuple[list[int], list[str]] | None:
    ports: list[int] = []
    errors: list[str] = []
    configured = False
    for home in homes:
        choice = _port_for_home(home)
        if choice is None:
            continue
        configured = True
        if choice is _BAD:
            errors.append(f"A2A port configured for {home.name} is not a TCP port")
        elif isinstance(choice, int) and choice not in ports:
            ports.append(choice)
    if not configured:
        return None
    return sorted(ports), errors


def _read_pid_record(pid_home: Path) -> dict:
    from gateway.status import _read_pid_record

    record = _read_pid_record(pid_home / "gateway.pid")
    return record if isinstance(record, dict) else {}


def _process_matches_pid_record(record: dict) -> bool:
    """Live process whose start time matches the pid file. Missing start time fails closed."""
    from gateway.status import (
        _pid_exists, _pid_from_record, get_process_start_time, start_time_fingerprints_match,
    )

    pid = _pid_from_record(record)
    recorded = record.get("start_time")
    if pid is None or recorded is None or not _pid_exists(pid):
        return False
    current = get_process_start_time(pid)
    if current is None:
        return False
    try:
        return start_time_fingerprints_match(recorded, current)
    except (TypeError, ValueError):
        return False


def _identity_problem(record: dict) -> str:
    from gateway.status import _pid_exists, _pid_from_record

    pid = _pid_from_record(record)
    if pid is None:
        return "gateway.pid is missing or has no pid"
    if record.get("start_time") is None:
        return f"pid {pid} is not enough; gateway.pid has no start_time"
    if not _pid_exists(pid):
        return f"pid {pid} is not running"
    return f"pid {pid} start_time does not match gateway.pid"


def _tcp_local_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=_CONNECT_TIMEOUT_S):
            return True
    except OSError:
        return False


def _liveness_problems() -> list[str] | None:
    """None when A2A is not configured. Empty when the gateway is actually up."""
    pid_home, homes = _scope()
    found = _configured_listeners(homes)
    if found is None:
        return None
    ports, problems = found
    record = _read_pid_record(pid_home)
    if not _process_matches_pid_record(record):
        problems.append(_identity_problem(record))
    for port in ports:
        if not _tcp_local_open(port):
            problems.append(f"nothing listening on 127.0.0.1:{port}")
    return problems


def _report(should_fix: bool, f: Finding, problems: list[str]) -> None:
    _section("Gateway Liveness")
    if not problems:
        check_ok("Gateway process", "(pid and start_time match gateway.pid)")
        check_ok("A2A listener", "(TCP connect to 127.0.0.1 succeeded)")
        return
    check_fail("Gateway configured but unreachable", f"({'; '.join(problems)})")
    if should_fix:
        check_info(f"Offered fix: {_START_CMD}")
        f.issues.append(_START_CMD)
        return
    f.issues.append(f"Gateway configured but unreachable — run '{_START_CMD}'")


@doctor_check()
def _check_gateway_liveness(should_fix: bool, f: Finding) -> None:
    """Fail closed when A2A is configured but the process or its port is down."""
    problems = _liveness_problems()
    if problems is None:
        return
    _report(should_fix, f, problems)
