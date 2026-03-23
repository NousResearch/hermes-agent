"""Dedicated CLI flows for Kasia setup and diagnostics."""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from gateway.kasia_config import DEFAULT_KASIA_BRIDGE_PORT, load_kasia_settings

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DEFAULT_KASIA_INDEXER_URL = "https://indexer.kasia.fyi"
DEFAULT_KASIA_NODE_WBORSH_URL = "wss://wrpc.kasia.fyi"
DEFAULT_KASIA_NETWORK = "mainnet"
DEFAULT_KASIA_FEE_POLICY = "auto"
DEFAULT_MAINNET_KNS_URL = "https://api.knsdomains.org/mainnet/api/v1"
DEFAULT_TESTNET_KNS_URL = "https://api.knsdomains.org/tn10/api/v1"
_TRUE_VALUES = {"true", "1", "yes"}


@dataclass(frozen=True, slots=True)
class KasiaCLIIO:
    get_env_value: Callable[[str], Optional[str]]
    save_env_value: Callable[[str, str], None]
    prompt: Callable[[str, Optional[str], bool], str]
    prompt_yes_no: Callable[[str, bool], bool]
    print_info: Callable[[str], None]
    print_success: Callable[[str], None]
    print_warning: Callable[[str], None]
    print_error: Callable[[str], None]


def _default_io() -> KasiaCLIIO:
    from hermes_cli.config import get_env_value, save_env_value
    from hermes_cli.setup import (
        print_error,
        print_info,
        print_success,
        print_warning,
        prompt,
        prompt_yes_no,
    )

    return KasiaCLIIO(
        get_env_value=get_env_value,
        save_env_value=save_env_value,
        prompt=prompt,
        prompt_yes_no=prompt_yes_no,
        print_info=print_info,
        print_success=print_success,
        print_warning=print_warning,
        print_error=print_error,
    )


def is_kasia_configured(get_env_value: Callable[[str], Optional[str]]) -> bool:
    """Return True when any Kasia configuration has been set."""
    return any(
        [
            get_env_value("KASIA_ENABLED"),
            get_env_value("KASIA_SEED_PHRASE"),
            get_env_value("KASIA_INDEXER_URL"),
            get_env_value("KASIA_INDEXER_URLS"),
            get_env_value("KASIA_NODE_WBORSH_URL"),
            get_env_value("KASIA_NODE_WBORSH_URLS"),
        ]
    )


def validate_kasia_seed_phrase(seed_phrase: str) -> tuple[bool, str | None]:
    """Validate Kasia/Kaspa mnemonic structure with a lightweight Node check."""
    normalized = " ".join(str(seed_phrase or "").strip().split())
    if not normalized:
        return False, "Kasia seed phrase cannot be empty."

    word_count = len(normalized.split(" "))
    if word_count not in {12, 24}:
        return False, "Kasia seed phrase should contain 12 or 24 words."

    env = dict(os.environ)
    env["KASIA_SEED_TO_VALIDATE"] = normalized
    validator = (
        "import { Mnemonic } from './scripts/kasia-bridge/lib/kaspa_sdk.js'; "
        "new Mnemonic(process.env.KASIA_SEED_TO_VALIDATE || '');"
    )

    try:
        subprocess.run(
            ["node", "--input-type=module", "-e", validator],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except FileNotFoundError:
        logger.warning("Skipping full Kasia seed validation because Node is unavailable")
    except subprocess.TimeoutExpired:
        logger.warning("Skipping full Kasia seed validation because Node timed out")
    except subprocess.CalledProcessError:
        return (
            False,
            "Kasia seed phrase is not a valid Kasia/Kaspa mnemonic. Please check the words and try again.",
        )

    return True, None


def _kasia_bridge_paths() -> tuple[Path, Path, Path]:
    """Return the Kasia bridge directory plus its key runtime paths."""
    bridge_dir = PROJECT_ROOT / "scripts" / "kasia-bridge"
    return bridge_dir, bridge_dir / "bridge.js", bridge_dir / "node_modules"


def _ensure_kasia_bridge_dependencies(io: KasiaCLIIO) -> bool:
    """Install Kasia bridge dependencies before setup validation uses them."""
    bridge_dir, bridge_script, node_modules_dir = _kasia_bridge_paths()
    if not bridge_script.exists():
        io.print_error(f"Kasia bridge script not found at {bridge_script}")
        return False

    if node_modules_dir.exists():
        return True

    io.print_info("Installing Kasia bridge dependencies...")
    try:
        result = subprocess.run(
            ["npm", "install", "--silent"],
            cwd=str(bridge_dir),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError:
        io.print_error("npm not found on PATH. Install Node.js/npm first.")
        return False
    except Exception as exc:
        io.print_error(f"Failed to install Kasia bridge dependencies: {exc}")
        return False

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip() or f"exit code {result.returncode}"
        io.print_error(f"Kasia bridge dependency install failed: {detail}")
        return False

    io.print_success("Kasia bridge dependencies installed")
    return True


def prompt_kasia_seed_phrase(
    *,
    get_env_value: Callable[[str], Optional[str]],
    prompt: Callable[[str, Optional[str], bool], str],
    print_info: Callable[[str], None],
    print_error: Callable[[str], None],
    validate_seed_phrase: Callable[[str], tuple[bool, str | None]] = validate_kasia_seed_phrase,
) -> str:
    """Prompt for a Kasia seed phrase, reusing any stored value when kept."""
    existing_seed = get_env_value("KASIA_SEED_PHRASE") or None
    print_info("🔒 Seed phrase input is hidden as you type.")
    if existing_seed:
        print_info("   Press Enter to keep the current stored seed phrase.")

    while True:
        seed_phrase = prompt(
            "Kasia seed phrase",
            default=existing_seed,
            password=True,
        )
        if not seed_phrase:
            return ""
        if existing_seed and seed_phrase == existing_seed:
            return seed_phrase

        is_valid, error = validate_seed_phrase(seed_phrase)
        if is_valid:
            return " ".join(seed_phrase.strip().split())

        print_error(error or "Invalid Kasia seed phrase.")


def _run_kasia_setup_prompts(
    io: KasiaCLIIO,
    *,
    prompt_seed_phrase: Callable[[], str],
) -> None:
    io.save_env_value("KASIA_ENABLED", "true")
    io.print_success("Kasia enabled")

    seed_phrase = prompt_seed_phrase()
    if seed_phrase:
        io.save_env_value("KASIA_SEED_PHRASE", seed_phrase)
        io.print_success("Kasia seed phrase saved")

    indexer_url = io.prompt(
        "Kasia indexer URL",
        default=io.get_env_value("KASIA_INDEXER_URL") or DEFAULT_KASIA_INDEXER_URL,
        password=False,
    )
    if indexer_url:
        normalized_indexer_url = indexer_url.rstrip("/")
        io.save_env_value("KASIA_INDEXER_URL", normalized_indexer_url)
        io.print_success(f"Kasia indexer URL saved: {normalized_indexer_url}")

    node_url = io.prompt(
        "Kaspa node URL",
        default=io.get_env_value("KASIA_NODE_WBORSH_URL") or DEFAULT_KASIA_NODE_WBORSH_URL,
        password=False,
    )
    if node_url:
        io.save_env_value("KASIA_NODE_WBORSH_URL", node_url)
        io.print_success(f"Kaspa node URL saved: {node_url}")

    network = io.prompt(
        "Kaspa network",
        default=io.get_env_value("KASIA_NETWORK") or DEFAULT_KASIA_NETWORK,
        password=False,
    )
    if network:
        io.save_env_value("KASIA_NETWORK", network)
        io.print_success(f"Kaspa network saved: {network}")

    fee_policy = io.prompt(
        "Kasia fee policy",
        default=DEFAULT_KASIA_FEE_POLICY,
        password=False,
    )
    if fee_policy:
        io.save_env_value("KASIA_FEE_POLICY", fee_policy)
        io.print_success(f"Kasia fee policy saved: {fee_policy}")

    io.print_info("🔒 Security: decide who can handshake and message Hermes over Kasia")
    allow_all = io.prompt_yes_no(
        "Allow all Kasia users to message Hermes?",
        False,
    )
    if allow_all:
        io.save_env_value("KASIA_ALLOW_ALL_USERS", "true")
        io.save_env_value("KASIA_ALLOWED_USERS", "")
        io.print_info("⚠️  Any Kasia address can now interact with Hermes.")
    else:
        io.save_env_value("KASIA_ALLOW_ALL_USERS", "false")
        allowed_users = io.prompt(
            "Allowed Kasia addresses (comma-separated, leave empty to set later)",
            default=io.get_env_value("KASIA_ALLOWED_USERS") or None,
            password=False,
        )
        if allowed_users:
            cleaned = ",".join(
                item.strip() for item in allowed_users.split(",") if item.strip()
            )
            io.save_env_value("KASIA_ALLOWED_USERS", cleaned)
            io.print_success("Kasia allowlist configured")
        else:
            io.print_warning(
                "No Kasia allowlist set yet. Add KASIA_ALLOWED_USERS later before opening access."
            )

    io.print_info("📬 Home Channel: where Hermes delivers cron job results and cross-platform messages.")
    io.print_info("   You can also set this later with /sethome in your Kasia chat.")
    home_channel = io.prompt(
        "Kasia home channel address (leave empty to set later)",
        default=io.get_env_value("KASIA_HOME_CHANNEL") or None,
        password=False,
    )
    if home_channel:
        io.save_env_value("KASIA_HOME_CHANNEL", home_channel.strip())
        io.print_success("Kasia home channel saved")


def kasia_summary_lines(
    get_env_value: Callable[[str], Optional[str]],
) -> list[str]:
    """Return a stable, operator-facing Kasia summary."""
    effective_kns_url, kns_overridden = resolve_kasia_kns_url(
        get_env_value("KASIA_NETWORK"),
        get_env_value("KASIA_KNS_URL"),
    )
    lines = [
        f"Indexer: {get_env_value('KASIA_INDEXER_URL') or '(not set)'}",
        f"Node: {get_env_value('KASIA_NODE_WBORSH_URL') or '(not set)'}",
        f"Network: {get_env_value('KASIA_NETWORK') or DEFAULT_KASIA_NETWORK}",
        f"Fee policy: {get_env_value('KASIA_FEE_POLICY') or DEFAULT_KASIA_FEE_POLICY}",
        f"KNS API: {effective_kns_url} ({'override' if kns_overridden else 'network default'})",
    ]
    return lines


def kasia_status_lines(
    kasia_settings,
    *,
    health: dict[str, Any] | None = None,
) -> list[str]:
    """Render operator-facing Kasia detail lines for Kasia diagnostics."""
    lines: list[str] = []
    effective_kns_url, kns_overridden = resolve_kasia_kns_url(
        kasia_settings.network,
        kasia_settings.kns_url,
    )
    lines.append(
        f"    KNS:        {effective_kns_url} ({'override' if kns_overridden else 'network default'})"
    )
    if kasia_settings.indexer_urls:
        lines.append(f"    Indexers:   {len(kasia_settings.indexer_urls)} configured")
    if kasia_settings.node_wborsh_urls:
        lines.append(f"    Nodes:      {len(kasia_settings.node_wborsh_urls)} configured")

    broadcast_channels = list(kasia_settings.allowed_broadcast_channels)
    if broadcast_channels:
        lines.append(
            "    Broadcasts: publish allowlist for "
            + ", ".join(f"#{channel}" for channel in broadcast_channels)
        )

    active_indexer = _active_health_url(health, "indexerPool", "indexerUrl")
    active_node = _active_health_url(health, "nodePool", "nodeUrl")
    if active_indexer:
        lines.append(f"    Active indexer: {active_indexer}")
    if active_node:
        lines.append(f"    Active node:    {active_node}")
    if (health or {}).get("indexerPool", {}).get("degraded"):
        lines.append("    Indexer pool:   degraded / failover active")
    if (health or {}).get("nodePool", {}).get("degraded"):
        lines.append("    Node pool:      degraded / failover active")
    return lines


def run_kasia_setup(
    io: Optional[KasiaCLIIO] = None,
    *,
    prompt_seed_phrase: Optional[Callable[[], str]] = None,
) -> bool:
    """Run the dedicated Kasia setup flow."""
    io = io or _default_io()
    if prompt_seed_phrase is None:
        prompt_seed_phrase = lambda: prompt_kasia_seed_phrase(
            get_env_value=io.get_env_value,
            prompt=io.prompt,
            print_info=io.print_info,
            print_error=io.print_error,
        )

    print()
    print("⚕ Kasia Setup")
    print("=" * 50)

    if is_kasia_configured(io.get_env_value):
        print()
        io.print_success("Kasia is already configured.")
        if not io.prompt_yes_no("Reconfigure Kasia?", False):
            return False

    print()
    io.print_info("Kasia lets Hermes send and receive messages on the Kaspa network.")
    io.print_info("Use a dedicated Kaspa wallet for Hermes, not your main wallet.")
    io.print_info("You will need a Kaspa seed phrase for Hermes.")
    io.print_info("Press Enter to use the recommended default shown in brackets.")
    print()

    if not _ensure_kasia_bridge_dependencies(io):
        return False

    _run_kasia_setup_prompts(io, prompt_seed_phrase=prompt_seed_phrase)

    print()
    io.print_success("Kasia configured!")
    for line in kasia_summary_lines(io.get_env_value):
        io.print_info(f"  {line}")
    return True


def fetch_kasia_bridge_health(bridge_port: Optional[int]) -> dict[str, Any] | None:
    """Best-effort bridge health fetch for the local Kasia bridge."""
    port = bridge_port or DEFAULT_KASIA_BRIDGE_PORT
    try:
        with urlopen(f"http://127.0.0.1:{port}/health", timeout=1.5) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def _request_kasia_bridge_json(
    path: str,
    *,
    bridge_port: Optional[int],
    method: str = "GET",
    payload: Optional[dict[str, Any]] = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Send a JSON request to the local Kasia bridge."""
    port = bridge_port or DEFAULT_KASIA_BRIDGE_PORT
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(
        f"http://127.0.0.1:{port}{path}",
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.reason
        try:
            error_body = json.loads(error.read().decode("utf-8"))
            detail = error_body.get("error") or detail
        except Exception:
            pass
        raise RuntimeError(str(detail or error)) from error
    except URLError as error:
        raise RuntimeError(str(error.reason or error)) from error


def complete_kasia_contact_approval(
    target: str,
    *,
    display_name: Optional[str] = None,
    bridge_port: Optional[int] = None,
) -> dict[str, Any]:
    """
    Best-effort live bridge follow-up after approving a Kasia contact.

    If the local bridge is reachable, prefer responding to an existing pending
    inbound handshake; otherwise fall back to initiating a fresh outbound one.
    """
    port = bridge_port or load_kasia_settings(env=os.environ).bridge_port
    if not fetch_kasia_bridge_health(port):
        return {"status": "bridge_unavailable", "bridge_port": port}

    try:
        response = _request_kasia_bridge_json(
            "/handshakes/respond",
            bridge_port=port,
            method="POST",
            payload={"chatId": target},
        )
        return {
            "status": "responded",
            "bridge_port": port,
            "bridge_status": response.get("status") or "sent",
            "result": response,
        }
    except Exception as respond_error:
        try:
            response = _request_kasia_bridge_json(
                "/handshakes/initiate",
                bridge_port=port,
                method="POST",
                payload={
                    "chatId": target,
                    "displayName": display_name,
                    "retry": False,
                },
            )
            bridge_status = str(response.get("status") or "").strip().lower()
            if bridge_status == "already_active":
                status = "already_active"
            elif bridge_status == "pending":
                status = "pending"
            else:
                status = "initiated"
            return {
                "status": status,
                "bridge_port": port,
                "bridge_status": bridge_status or "sent",
                "result": response,
            }
        except Exception as initiate_error:
            return {
                "status": "failed",
                "bridge_port": port,
                "respond_error": str(respond_error),
                "initiate_error": str(initiate_error),
            }


def _doctor_mark(ok: bool) -> str:
    return "✓" if ok else "✗"


def _format_sompi_balance(value: Any) -> Optional[str]:
    """Render a sompi integer value as a KAS string with 8 decimals."""
    try:
        sompi = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    sign = "-" if sompi < 0 else ""
    whole, fractional = divmod(abs(sompi), 100_000_000)
    return f"{sign}{whole}.{fractional:08d}"


def default_kasia_kns_url(network: Optional[str]) -> str:
    normalized_network = str(network or DEFAULT_KASIA_NETWORK).strip().lower()
    if normalized_network.startswith("mainnet"):
        return DEFAULT_MAINNET_KNS_URL
    return DEFAULT_TESTNET_KNS_URL


def resolve_kasia_kns_url(
    network: Optional[str],
    configured_kns_url: Optional[str],
) -> tuple[str, bool]:
    normalized_kns_url = str(configured_kns_url or "").strip()
    if normalized_kns_url:
        return normalized_kns_url, True
    return default_kasia_kns_url(network), False


def _node_runtime_status() -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except FileNotFoundError:
        return False, "node not found on PATH"
    except Exception as exc:
        return False, str(exc)

    version = (result.stdout or result.stderr or "").strip()
    if result.returncode == 0 and version:
        return True, version
    if version:
        return False, version
    return False, f"node exited with {result.returncode}"


def _doctor_line(label: str, ok: bool, detail: str) -> str:
    return f"  {_doctor_mark(ok)} {label:<14} {detail}"


def _active_health_url(health: dict[str, Any] | None, pool_key: str, direct_key: str) -> str:
    if not health:
        return ""
    pool = health.get(pool_key) or {}
    return str(pool.get("activeUrl") or health.get(direct_key) or "").strip()


def _wallet_funding_status_line(
    health: dict[str, Any] | None,
) -> tuple[bool, str] | None:
    """Build an operator-facing wallet funding summary from bridge health."""
    if not health:
        return None

    funding_state = str(health.get("walletFundingState") or "").strip().lower()
    on_chain = _format_sompi_balance(health.get("walletBalanceSompi"))
    spendable = _format_sompi_balance(health.get("availableMatureBalanceSompi"))
    recommended = _format_sompi_balance(health.get("recommendedMinBalanceSompi"))

    balance_parts = []
    if on_chain is not None:
        balance_parts.append(f"{on_chain} KAS on-chain")
    if spendable is not None:
        balance_parts.append(f"{spendable} KAS spendable")
    if recommended is not None:
        balance_parts.append(f"recommended >= {recommended} KAS")

    balance_text = ", ".join(balance_parts)
    detail = funding_state or "unknown"
    if balance_text:
        detail = f"{detail} ({balance_text})"

    return funding_state not in {"low", "unfunded"}, detail


def _kasia_paired_user_count() -> int:
    """Return the number of approved Kasia DM pairings stored on disk."""
    from hermes_cli.config import get_hermes_home

    approved_path = get_hermes_home() / "pairing" / "kasia-approved.json"
    if not approved_path.exists():
        return 0

    try:
        approved_users = json.loads(approved_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0

    if not isinstance(approved_users, dict):
        return 0

    return sum(1 for user_id in approved_users if str(user_id).strip())


def run_kasia_doctor() -> bool:
    """Run Kasia-specific diagnostics without affecting shared Hermes status."""
    settings = load_kasia_settings(env=os.environ)
    bridge_dir, bridge_script, node_modules_dir = _kasia_bridge_paths()
    node_ok, node_detail = _node_runtime_status()
    health = fetch_kasia_bridge_health(settings.bridge_port)

    issues = 0

    print()
    print("⚕ Kasia Doctor")
    print("=" * 50)

    print()
    print("Configuration")
    paired_user_count = _kasia_paired_user_count()
    config_lines = [
        ("Enabled", settings.enabled, "KASIA_ENABLED=true" if settings.enabled else "run `hermes kasia` to configure"),
        ("Seed phrase", bool(settings.seed_phrase), "configured" if settings.seed_phrase else "missing"),
        ("Indexer", bool(settings.indexer_url), settings.indexer_url or "missing"),
        ("Kaspa node", bool(settings.node_wborsh_url), settings.node_wborsh_url or "missing"),
        ("Fee policy", bool(settings.fee_policy), settings.fee_policy or DEFAULT_KASIA_FEE_POLICY),
        ("Home channel", bool(settings.home_channel), settings.home_channel or "not configured"),
    ]
    if settings.allow_all_users:
        access_ok = True
        access_detail = "allowing all Kasia users"
    else:
        access_parts: list[str] = []
        if settings.allowed_users:
            allowlist_count = len([item for item in settings.allowed_users.split(",") if item.strip()])
            access_parts.append(f"{allowlist_count} allowlisted address(es)")
        if paired_user_count:
            access_parts.append(f"{paired_user_count} paired user(s) approved")
        access_ok = bool(access_parts)
        access_detail = "; ".join(access_parts) if access_parts else "no allowlist or paired users configured"
    config_lines.append(("Access", access_ok, access_detail))

    for label, ok, detail in config_lines:
        print(_doctor_line(label, ok, detail))
        if label in {"Enabled", "Seed phrase", "Indexer", "Kaspa node"} and not ok:
            issues += 1

    for line in kasia_status_lines(settings, health=health):
        print(line)

    print()
    print("Bridge")
    bridge_lines = [
        ("Node.js", node_ok, node_detail),
        ("Bridge script", bridge_script.exists(), str(bridge_script) if bridge_script.exists() else f"missing: {bridge_script}"),
        (
            "Dependencies",
            node_modules_dir.exists(),
            "installed" if node_modules_dir.exists() else f"missing: {node_modules_dir}",
        ),
    ]
    for label, ok, detail in bridge_lines:
        print(_doctor_line(label, ok, detail))
        if not ok:
            issues += 1

    print()
    print("Runtime")
    if health:
        print(_doctor_line("Bridge health", True, f"reachable on 127.0.0.1:{settings.bridge_port}"))
        wallet_funding = _wallet_funding_status_line(health)
        if wallet_funding:
            funding_ok, funding_detail = wallet_funding
            print(_doctor_line("Wallet funding", funding_ok, funding_detail))
            if not funding_ok:
                issues += 1
        active_indexer = _active_health_url(health, "indexerPool", "indexerUrl")
        active_node = _active_health_url(health, "nodePool", "nodeUrl")
        if active_indexer:
            print(f"  • Active indexer: {active_indexer}")
        if active_node:
            print(f"  • Active node:    {active_node}")
        if (health.get("indexerPool") or {}).get("degraded"):
            print("  • Indexer pool is degraded / failover active")
        if (health.get("nodePool") or {}).get("degraded"):
            print("  • Node pool is degraded / failover active")
    else:
        print(
            _doctor_line(
                "Bridge health",
                False,
                f"not reachable on 127.0.0.1:{settings.bridge_port} (start the gateway to check live health)",
            )
        )
        issues += 1

    print()
    if issues:
        print("Kasia doctor found configuration or dependency issues.")
        print("Run `hermes kasia` to complete setup, then `hermes gateway status --deep` for gateway diagnostics.")
        return False

    print("Kasia configuration looks good.")
    print("Use `hermes gateway status --deep` for gateway/service diagnostics.")
    return True
