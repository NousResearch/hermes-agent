"""CLI orchestration for the Telegram Mini App lifecycle."""

from __future__ import annotations

import copy
import hashlib
import ipaddress
import json
import os
import re
import socket
import stat
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable

from hermes_cli.config import (
    get_env_value,
    get_env_value_prefer_dotenv,
    get_hermes_home,
    read_raw_config,
    save_config,
)

from . import service


DEFAULT_PORT = 8787
FORBIDDEN_EPHEMERAL_HOST_SUFFIXES = (".trycloudflare.com",)
BOT_TOKEN_RE = re.compile(r"^\d+:[A-Za-z0-9_-]{30,}$")


class MiniAppSetupError(RuntimeError):
    """A safe, actionable Mini App setup failure."""


def _supports_foreground() -> bool:
    """Return whether the clean runner's POSIX exec boundary is available."""
    return os.name == "posix"


def validate_public_url(value: str) -> str:
    raw = (value or "").strip().rstrip("/")
    parsed = urllib.parse.urlsplit(raw)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise MiniAppSetupError("--public-url must be a stable HTTPS URL.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise MiniAppSetupError(
            "--public-url must not contain credentials, a query, or a fragment."
        )
    if parsed.path not in ("", "/"):
        raise MiniAppSetupError("--public-url must be an HTTPS origin without a path.")
    hostname = parsed.hostname.lower().rstrip(".")
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        raise MiniAppSetupError(
            "--public-url must use a public DNS hostname, not an IP literal."
        )
    if hostname.endswith(FORBIDDEN_EPHEMERAL_HOST_SUFFIXES):
        raise MiniAppSetupError(
            "Ephemeral quick-tunnel URLs are not supported; provide a stable HTTPS origin."
        )
    _validate_public_resolution(hostname, parsed.port or 443)
    return urllib.parse.urlunsplit(("https", parsed.netloc, "", "", ""))


def _validate_public_resolution(hostname: str, port: int) -> None:
    try:
        answers = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise MiniAppSetupError(
            f"Could not resolve --public-url hostname: {hostname}"
        ) from exc
    addresses = {answer[4][0].split("%", 1)[0] for answer in answers if answer[4]}
    if not addresses:
        raise MiniAppSetupError(f"Could not resolve --public-url hostname: {hostname}")
    unsafe = []
    for address in addresses:
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError:
            unsafe.append(address)
            continue
        if not parsed.is_global:
            unsafe.append(address)
    if unsafe:
        raise MiniAppSetupError(
            "--public-url resolves to a non-public address; private, loopback, link-local, "
            f"and reserved destinations are rejected ({', '.join(sorted(unsafe))})."
        )


def default_listen_port(hermes_home: Path) -> int:
    """Choose a deterministic port so named profiles do not all collide on 8787."""
    try:
        from hermes_constants import get_default_hermes_root

        if hermes_home.resolve() == get_default_hermes_root().resolve():
            return DEFAULT_PORT
    except (ImportError, OSError):
        pass
    digest = hashlib.sha256(str(hermes_home.resolve()).encode()).digest()
    return 8800 + int.from_bytes(digest[:2], "big") % 1000


def validate_owner_ids(values: Iterable[str]) -> list[str]:
    owners: list[str] = []
    for raw in values:
        for item in str(raw).split(","):
            owner = item.strip()
            if not owner:
                continue
            if owner == "*" or not owner.isascii() or not owner.isdecimal():
                raise MiniAppSetupError(
                    "Mini App owners must be explicit positive numeric Telegram user IDs; "
                    "wildcards, usernames, and groups are rejected."
                )
            if int(owner) <= 0:
                raise MiniAppSetupError(
                    "Mini App owners must be positive user IDs; Telegram group/chat IDs are rejected."
                )
            if owner not in owners:
                owners.append(owner)
    if not owners:
        raise MiniAppSetupError("At least one explicit numeric --owner is required.")
    return owners


def _select_owners_interactively() -> list[str]:
    configured = (get_env_value("TELEGRAM_ALLOWED_USERS") or "").strip()
    if not configured:
        raise MiniAppSetupError(
            "No owners were provided. Pass one or more `--owner <numeric-id>` values."
        )
    candidates = validate_owner_ids([configured])
    if not (hasattr(sys.stdin, "isatty") and sys.stdin.isatty()):
        raise MiniAppSetupError(
            "Owner selection is required in non-interactive mode; pass `--owner <numeric-id>`."
        )
    print("Configured Telegram users:")
    for index, owner in enumerate(candidates, 1):
        print(f"  {index}. {owner}")
    selected = input("Select owner numbers (comma-separated): ").strip()
    indexes = validate_owner_ids(selected.split(","))
    chosen: list[str] = []
    for raw_index in indexes:
        index = int(raw_index)
        if index < 1 or index > len(candidates):
            raise MiniAppSetupError(f"Owner selection {index} is out of range.")
        chosen.append(candidates[index - 1])
    return list(dict.fromkeys(chosen))


def _write_private(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    finally:
        temp.unlink(missing_ok=True)


def _mini_app_config(config: dict) -> dict:
    platforms = config.setdefault("platforms", {})
    if not isinstance(platforms, dict):
        platforms = {}
        config["platforms"] = platforms
    telegram = platforms.setdefault("telegram", {})
    if not isinstance(telegram, dict):
        telegram = {}
        platforms["telegram"] = telegram
    extra = telegram.setdefault("extra", {})
    if not isinstance(extra, dict):
        extra = {}
        telegram["extra"] = extra
    mini_app = extra.setdefault("mini_app", {})
    if not isinstance(mini_app, dict):
        mini_app = {}
        extra["mini_app"] = mini_app
    return mini_app


def _persist_behavior(
    *, enabled: bool, public_url: str | None = None, port: int = DEFAULT_PORT
) -> None:
    config = read_raw_config()
    mini_app = _mini_app_config(config)
    mini_app["enabled"] = enabled
    if public_url is not None:
        mini_app["public_url"] = public_url
    mini_app["listen_port"] = port
    save_config(
        config,
        preserve_keys={
            ("platforms", "telegram", "extra", "mini_app", "enabled"),
            ("platforms", "telegram", "extra", "mini_app", "public_url"),
            ("platforms", "telegram", "extra", "mini_app", "listen_port"),
        },
    )


def _remove_behavior() -> None:
    """Remove the generated lifecycle mirror without disturbing Telegram config."""
    config = read_raw_config()
    platforms = config.get("platforms")
    telegram = platforms.get("telegram") if isinstance(platforms, dict) else None
    extra = telegram.get("extra") if isinstance(telegram, dict) else None
    if not isinstance(extra, dict) or "mini_app" not in extra:
        return
    extra.pop("mini_app")
    if not extra:
        telegram.pop("extra", None)
    try:
        save_config(config)
    except (OSError, RuntimeError, ValueError) as exc:
        raise MiniAppSetupError(
            "Could not remove the Mini App lifecycle mirror; uninstall made no changes."
        ) from exc


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _direct_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


def _same_origin(expected: str, actual: str) -> bool:
    left = urllib.parse.urlsplit(expected)
    right = urllib.parse.urlsplit(actual)
    return (left.scheme.lower(), left.hostname, left.port or 443) == (
        right.scheme.lower(),
        right.hostname,
        right.port or 443,
    )


def _probe(public_url: str, *, attempts: int = 20, delay: float = 0.5) -> None:
    parsed = urllib.parse.urlsplit(public_url)
    _validate_public_resolution(parsed.hostname or "", parsed.port or 443)
    opener = _direct_opener()
    health_url = f"{public_url}/health"
    last_error = "no response"
    for _ in range(attempts):
        try:
            req = urllib.request.Request(
                health_url, headers={"Accept": "application/json"}
            )
            with opener.open(req, timeout=5) as response:
                if response.status == 200 and _same_origin(
                    public_url, response.geturl()
                ):
                    break
                last_error = f"unsafe response (HTTP {response.status})"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(getattr(exc, "reason", exc))
        time.sleep(delay)
    else:
        raise MiniAppSetupError(
            f"Mini App health probe failed at {health_url}: {last_error}"
        )

    protected_url = f"{public_url}/api/me"
    try:
        with opener.open(protected_url, timeout=5) as response:
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise MiniAppSetupError(f"Mini App authentication probe failed: {exc}") from exc
    if status not in (401, 403):
        raise MiniAppSetupError(
            f"Mini App refused setup because an unauthenticated API request returned HTTP {status}, not 401/403."
        )


def _telegram_call(token: str, method: str, payload: dict[str, object]) -> object:
    endpoint = f"https://api.telegram.org/bot{token}/{method}"
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with _direct_opener().open(request, timeout=10) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise MiniAppSetupError(
            f"Telegram rejected {method} (HTTP {exc.code})."
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise MiniAppSetupError(
            "Could not configure the Telegram Mini App menu button."
        ) from exc
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise MiniAppSetupError(f"Telegram did not accept {method}.")
    return result.get("result")


def _get_menu_button(token: str, *, chat_id: str | None = None) -> dict:
    payload: dict[str, object] = {}
    if chat_id is not None:
        payload["chat_id"] = int(chat_id)
    result = _telegram_call(token, "getChatMenuButton", payload)
    if not isinstance(result, dict) or not isinstance(result.get("type"), str):
        raise MiniAppSetupError("Telegram returned an invalid menu button.")
    return result


def _apply_menu_button(token: str, button: dict, *, chat_id: str | None = None) -> None:
    payload: dict[str, object] = {"menu_button": copy.deepcopy(button)}
    if chat_id is not None:
        payload["chat_id"] = int(chat_id)
    _telegram_call(token, "setChatMenuButton", payload)


def _set_menu_button(
    token: str, public_url: str, *, chat_id: str | None = None
) -> None:
    _apply_menu_button(
        token,
        {
            "type": "web_app",
            "text": "Open Hermes",
            "web_app": {"url": public_url},
        },
        chat_id=chat_id,
    )


def _optional_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def _restore_bytes(path: Path, content: bytes | None) -> None:
    if content is None:
        path.unlink(missing_ok=True)
    else:
        _write_private(path, content.decode("utf-8"))


def _read_state(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _read_dedicated_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def _menu_snapshot(token: str, owners: Iterable[str]) -> dict[str, dict]:
    snapshot = {"global": _get_menu_button(token)}
    for owner in owners:
        snapshot[str(owner)] = _get_menu_button(token, chat_id=str(owner))
    return snapshot


def _restore_menus(token: str, snapshot: dict[str, dict]) -> None:
    global_button = snapshot.get("global")
    if isinstance(global_button, dict):
        _apply_menu_button(token, global_button)
    for owner, button in snapshot.items():
        if owner != "global" and isinstance(button, dict):
            _apply_menu_button(token, button, chat_id=owner)


def _validated_uninstall_recovery(
    dedicated: dict[str, str], state: dict
) -> tuple[str, dict[str, dict]]:
    """Validate and narrow the recovery material before uninstall mutates state."""
    token = dedicated.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not BOT_TOKEN_RE.fullmatch(token):
        raise MiniAppSetupError(
            "Cannot uninstall safely because the dedicated Telegram bot token is missing or invalid; "
            "the service files were preserved for recovery."
        )
    state_owners_raw = state.get("owners")
    if not isinstance(state_owners_raw, list):
        raise MiniAppSetupError(
            "Cannot uninstall safely because the owner recovery list is missing; "
            "the service files were preserved for recovery."
        )
    try:
        dedicated_owners = validate_owner_ids([
            dedicated.get("TELEGRAM_MINI_APP_OWNER_IDS", "")
        ])
        state_owners = validate_owner_ids(state_owners_raw)
    except MiniAppSetupError as exc:
        raise MiniAppSetupError(
            "Cannot uninstall safely because the owner recovery list is invalid; "
            "the service files were preserved for recovery."
        ) from exc
    if set(dedicated_owners) != set(state_owners):
        raise MiniAppSetupError(
            "Cannot uninstall safely because the dedicated and saved owner lists disagree; "
            "the service files were preserved for recovery."
        )

    menu_backups = state.get("menu_backups")
    required = ["global", *state_owners]
    if not isinstance(menu_backups, dict) or any(
        not isinstance(menu_backups.get(key), dict)
        or not isinstance(menu_backups[key].get("type"), str)
        for key in required
    ):
        raise MiniAppSetupError(
            "Cannot uninstall safely because the Telegram menu backup is incomplete; "
            "the service files were preserved for recovery."
        )
    # Never replay unexpected chat IDs from a manually edited/corrupt state file.
    recovery = {key: copy.deepcopy(menu_backups[key]) for key in required}
    return token, recovery


def setup(
    *, public_url: str, owner_values: list[str], listen_port: int | None = None
) -> None:
    home = Path(get_hermes_home())
    if listen_port is None:
        listen_port = default_listen_port(home)
    if not 1 <= listen_port <= 65535:
        raise MiniAppSetupError("--listen-port must be between 1 and 65535.")
    supervised = service._platform() == "systemd"
    if supervised:
        service.require_install_support()
    elif not _supports_foreground():
        raise MiniAppSetupError(
            "Foreground Mini App mode currently requires macOS or another POSIX "
            "system with env(1); Windows is not supported in this release."
        )
    url = validate_public_url(public_url)
    owners = (
        validate_owner_ids(owner_values)
        if owner_values
        else _select_owners_interactively()
    )
    token = (get_env_value_prefer_dotenv("TELEGRAM_BOT_TOKEN") or "").strip()
    if not BOT_TOKEN_RE.fullmatch(token):
        raise MiniAppSetupError(
            "Telegram is not configured; set up the gateway bot first."
        )

    paths = service.paths_for(home)
    previous_config = read_raw_config()
    previous_env = _optional_bytes(paths.env)
    previous_credentials = _read_dedicated_env(paths.env)
    previous_token = previous_credentials.get("TELEGRAM_BOT_TOKEN", "")
    previous_state_bytes = _optional_bytes(paths.state)
    previous_state = _read_state(paths.state)
    previous_installed = supervised and service.systemd_unit_path(home).exists()
    previous_running = service.status(home)[0] if supervised else False
    old_owners = (
        validate_owner_ids(previous_state.get("owners", []))
        if previous_state.get("owners")
        else []
    )
    same_bot = bool(previous_token) and previous_token == token
    affected_owners = list(dict.fromkeys([*old_owners, *owners]))
    rollback_menus = _menu_snapshot(
        token, affected_owners if same_bot else owners
    )
    previous_bot_menus = (
        _menu_snapshot(previous_token, old_owners)
        if previous_token and not same_bot
        else None
    )
    stored_backups = previous_state.get("menu_backups")
    if not isinstance(stored_backups, dict):
        stored_backups = {}
    if previous_token and not same_bot:
        required_backups = {"global", *old_owners}
        if not required_backups.issubset(stored_backups):
            raise MiniAppSetupError(
                "Cannot rotate the Telegram bot token because the previous menu "
                "backup is incomplete; restore the old bot menu before reconfiguring."
            )
    menu_backups = {
        "global": (
            stored_backups.get("global", rollback_menus["global"])
            if same_bot
            else rollback_menus["global"]
        ),
        **{
            owner: (
                stored_backups.get(owner, rollback_menus[owner])
                if same_bot
                else rollback_menus[owner]
            )
            for owner in owners
        },
    }
    try:
        _write_private(
            paths.env,
            f"TELEGRAM_BOT_TOKEN={token}\n"
            f"TELEGRAM_MINI_APP_OWNER_IDS={','.join(owners)}\n"
            f"TELEGRAM_MINI_APP_PUBLIC_URL={url}\n",
        )
        state = {
            "configured": True,
            "public_url": url,
            "owners": owners,
            "listen_port": listen_port,
            "menu_backups": menu_backups,
        }
        _write_private(paths.state, json.dumps(state, indent=2, sort_keys=True) + "\n")
        _persist_behavior(enabled=True, public_url=url, port=listen_port)

        if supervised:
            service.install(home)
            if previous_running:
                service.restart(home)
            else:
                service.start(home)
            _probe(url)
        _set_menu_button(token, url)
        for owner in owners:
            _set_menu_button(token, url, chat_id=owner)
        if same_bot:
            for removed_owner in set(old_owners) - set(owners):
                original = stored_backups.get(removed_owner)
                if isinstance(original, dict):
                    _apply_menu_button(token, original, chat_id=removed_owner)
        elif previous_token and stored_backups:
            _restore_menus(previous_token, stored_backups)
    except Exception:
        try:
            _restore_menus(token, rollback_menus)
        except Exception:
            pass
        if previous_token and previous_bot_menus is not None:
            try:
                _restore_menus(previous_token, previous_bot_menus)
            except Exception:
                pass
        if supervised:
            try:
                service.stop(home)
            except Exception:
                pass
            try:
                service.uninstall(home)
            except Exception:
                pass
        _restore_bytes(paths.env, previous_env)
        _restore_bytes(paths.state, previous_state_bytes)
        save_config(previous_config)
        if previous_installed and previous_env is not None:
            try:
                service.install(home)
                if previous_running:
                    service.start(home)
            except Exception:
                pass
        raise
    if supervised:
        print(f"✓ Telegram Mini App is running at {url}")
    else:
        print(f"✓ Telegram Mini App is configured for {url}")
        print(
            "  Native supervision is unavailable on this platform. "
            "Run `hermes gateway mini-app serve` in a dedicated foreground terminal; "
            "the command binds only to loopback and does not verify public ingress."
        )


def command(args) -> None:
    action = getattr(args, "mini_app_command", None)
    home = Path(get_hermes_home())
    try:
        if action == "setup":
            setup(
                public_url=args.public_url,
                owner_values=args.owner or [],
                listen_port=args.listen_port,
            )
        elif action == "serve":
            if not _supports_foreground():
                raise MiniAppSetupError(
                    "Foreground Mini App mode currently requires macOS or another "
                    "POSIX system with env(1); Windows is not supported in this release."
                )
            if service._platform() != "systemd":
                print(
                    "⚠ Foreground Mini App mode is unsandboxed on this platform; "
                    "keep it on loopback behind a trusted HTTPS reverse proxy.",
                    file=sys.stderr,
                )
            service.exec_clean_runner(home)
        elif action == "status":
            running, detail = service.status(home)
            marker = "✓" if running else "✗"
            print(f"{marker} Telegram Mini App: {detail}")
            paths = service.paths_for(home)
            if paths.state.exists():
                try:
                    state = json.loads(paths.state.read_text(encoding="utf-8"))
                    print(f"  URL: {state.get('public_url', 'not configured')}")
                except (OSError, json.JSONDecodeError):
                    pass
        elif action == "start":
            service.start(home)
            print("✓ Telegram Mini App started")
        elif action == "stop":
            service.stop(home)
            print("✓ Telegram Mini App stopped")
        elif action == "restart":
            service.restart(home)
            print("✓ Telegram Mini App restarted")
        elif action == "uninstall":
            paths = service.paths_for(home)
            dedicated = _read_dedicated_env(paths.env)
            state = _read_state(paths.state)
            token, menu_backups = _validated_uninstall_recovery(dedicated, state)
            previous_config = read_raw_config()
            _remove_behavior()
            try:
                _restore_menus(token, menu_backups)
            except Exception:
                try:
                    save_config(previous_config)
                except Exception:
                    pass
                raise
            service.uninstall(home)
            for path in (paths.env, paths.state, paths.stdout_log, paths.stderr_log):
                path.unlink(missing_ok=True)
            try:
                paths.root.rmdir()
            except OSError:
                pass
            print("✓ Telegram Mini App uninstalled")
        else:
            raise MiniAppSetupError(
                "Choose a Mini App command; run with --help for options."
            )
    except (MiniAppSetupError, service.MiniAppServiceError, RuntimeError) as exc:
        print(f"✗ {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
