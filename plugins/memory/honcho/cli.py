"""``hermes honcho`` subcommands: setup wizard, status, peers, sessions, identity, migrate."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from hermes_constants import get_hermes_home
from plugins.memory.honcho.client import _first_parsed, _host_block, profile_host_key, resolve_active_host, resolve_config_path, HOST
from hermes_cli.config import cfg_get

RULE = "─" * 40
REASONING_LEVELS = ("minimal", "low", "medium", "high", "max")
_RETRY_HINT = "  Re-run 'hermes honcho setup' to retry, or choose an API key instead.\n"

# Settings a new profile host block inherits from the default block.
_INHERITED_KEYS = (
    "recallMode", "writeFrequency", "sessionStrategy", "contextTokens",
    "dialecticReasoningLevel", "dialecticDynamic", "dialecticMaxChars",
    "messageMaxChars", "dialecticMaxInputChars", "saveMessages", "observation",
)
# clone_honcho_for_profile also carries the operator's runtime-to-peer routing intent.
_CLONE_KEYS = _INHERITED_KEYS[:3] + ("sessionPeerPrefix",) + _INHERITED_KEYS[3:] + (
    "pinUserPeer", "userPeerAliases", "runtimePeerPrefix",
)
_IDENTITY_MAPPING_KEYS = ("pinPeerName", "pinUserPeer", "userPeerAliases", "runtimePeerPrefix")
# Setup-wizard answer -> identity-mapping shape ("2"/pooled answers are handled inline).
_SHAPE_CHOICES = {"1": "single", "me": "single", "just-me": "single", "3": "multi", "others": "multi",
                  "e": "raw", "edit": "raw", "raw": "raw"}
_MODES = {
    "hybrid": "auto-injected context + Honcho tools available (default)",
    "context": "auto-injected context only, Honcho tools hidden",
    "tools": "Honcho tools only, no auto-injected context",
}
_STRATEGIES = {
    "per-session": "each run starts clean, Honcho injects context automatically",
    "per-directory": "reuses session per dir, prior context auto-injected each run",
    "per-repo": "one session per git repository",
    "global": "single session across all directories",
}


# ── config access ──────────────────────────────────────────────────────────

_profile_override: str | None = None


def _host_key() -> str:
    """Active Honcho host key (``--target-profile`` override, else the active profile)."""
    if _profile_override:
        return HOST if _profile_override in {"default", "custom"} else profile_host_key(_profile_override)
    return resolve_active_host()


def _config_path() -> Path:
    """Active Honcho config path for reading (instance-local, default profile, or global)."""
    return resolve_config_path()


def _local_config_path() -> Path:
    """Instance-local write path; ~/.honcho/config.json is only a read fallback for cross-app interop."""
    return get_hermes_home() / "honcho.json"


def _read_config() -> dict:
    try:
        return json.loads(_config_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_config(cfg: dict, path: Path | None = None) -> None:
    path = path or _local_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    from utils import atomic_json_write
    atomic_json_write(path, cfg, mode=0o600)


def _label(host: str) -> str:
    return f"[{host}] " if host != "hermes" else ""


def _mask(key: str) -> str:
    return f"...{key[-8:]}" if len(key) > 8 else ("set" if key else "not set")


def _pref(block: dict, cfg: dict, key: str, default=None):
    """Host-block value, falling back to the root-level value (or ``default``)."""
    return block.get(key) or cfg.get(key, default)


def _active_block(cfg: dict) -> dict:
    return (cfg.get("hosts") or {}).get(_host_key(), {})


def _set_field(cfg: dict, key: str, value, echo: str) -> None:
    """Write one key on the active host block and echo the change."""
    host = _host_key()
    cfg.setdefault("hosts", {}).setdefault(host, {})[key] = value
    print(f"  {_label(host)}{echo}")


def _save(cfg: dict) -> None:
    _write_config(cfg)
    print(f"  Saved to {_config_path()}\n")


def _default_block_and_key(cfg: dict) -> tuple[dict, bool]:
    """(default host block, whether an API key is configured at root or env)."""
    return cfg_get(cfg, "hosts", HOST, default={}), bool(cfg.get("apiKey") or os.environ.get("HONCHO_API_KEY"))


def _resolve_api_key(cfg: dict) -> str:
    """API key with host -> root -> env fallback. A self-hosted ``baseUrl`` without a key
    yields ``"local"`` so credential guards accept it: the URL must be http/https (so
    ``baseUrl: true`` can't pass) or a schemeless host:port (legacy ``localhost:8000``;
    the SDK rejects those itself)."""
    key = _host_block(cfg, _host_key()).get("apiKey") or cfg.get("apiKey", "") or os.environ.get("HONCHO_API_KEY", "")
    if key:
        return key
    base_url = (cfg.get("baseUrl") or cfg.get("base_url") or os.environ.get("HONCHO_BASE_URL", "") or "").strip()
    if not base_url:
        return key
    from urllib.parse import urlparse
    try:
        parsed = urlparse(base_url)
    except (TypeError, ValueError):
        parsed = None
    if parsed and parsed.scheme in {"http", "https"} and parsed.netloc:
        return "local"
    lowered = base_url.lower()
    if lowered not in {"true", "false", "none", "null"} and any(c in base_url for c in ".:") and not base_url.isdigit():
        return "local"
    return key


def _prompt(label: str, default: str | None = None, secret: bool = False) -> str:
    sys.stdout.write(f"  {label}{f' [{default}]' if default else ''}: ")
    sys.stdout.flush()
    if secret and sys.stdin.isatty():
        from hermes_cli.secret_prompt import masked_secret_prompt
        val = masked_secret_prompt("")
    else:  # non-TTY (piped input, test runners) reads plaintext
        val = sys.stdin.readline().strip()
    return val or (default or "")


def _yes(answer: str) -> bool:
    return answer.strip().lower() in {"y", "yes"}


# ── Honcho connection ──────────────────────────────────────────────────────

def _connect(host: str | None, *, reset: bool = False):
    """(hcfg, client) for ``host``; lazy imports so tests can patch client.*."""
    from plugins.memory.honcho.client import HonchoClientConfig, get_honcho_client, reset_honcho_client
    if reset:
        reset_honcho_client()
    hcfg = HonchoClientConfig.from_global_config(host=host)
    return hcfg, get_honcho_client(hcfg)


def _session_manager(hcfg, client):
    """(manager, session_key) with the session ensured (get_or_create is idempotent)."""
    from plugins.memory.honcho.session import HonchoSessionManager
    mgr = HonchoSessionManager(honcho=client, config=hcfg)
    session_key = hcfg.resolve_session_name()
    mgr.get_or_create(session_key)
    return mgr, session_key


def _ensure_peer_exists(host_key: str | None = None) -> bool:
    """Create the AI (and user) peer in Honcho if missing. Idempotent; False on failure."""
    try:
        from plugins.memory.honcho.client import HonchoClientConfig, get_honcho_client
        hcfg = HonchoClientConfig.from_global_config(host=host_key)
        if not hcfg.enabled or not (hcfg.api_key or hcfg.base_url):
            return False
        client = get_honcho_client(hcfg)
        client.peer(hcfg.ai_peer)
        if hcfg.peer_name:
            client.peer(hcfg.peer_name)
        return True
    except Exception:
        return False


# ── profile sync ───────────────────────────────────────────────────────────

def _inherit_defaults(block: dict, default_block: dict, cfg: dict, keys: tuple[str, ...]) -> None:
    """Copy ``keys`` (and peerName) from the default host block into ``block`` where unset."""
    for key in keys:
        if (val := default_block.get(key)) is not None and key not in block:
            block[key] = val
    if (peer_name := _pref(default_block, cfg, "peerName")) and "peerName" not in block:
        block["peerName"] = peer_name


def clone_honcho_for_profile(profile_name: str) -> bool:
    """Create a host block for a new profile, cloned from the default host block
    (called during profile creation). False if Honcho isn't configured or the block exists."""
    cfg = _read_config()
    if not cfg:
        return False
    default_block, has_key = _default_block_and_key(cfg)
    new_host = profile_host_key(profile_name)
    if (not default_block and not has_key) or new_host in cfg.get("hosts", {}):
        return False

    new_block: dict = {}
    _inherit_defaults(new_block, default_block, cfg, _CLONE_KEYS)
    # Carry a legacy default-block pinPeerName forward under the canonical key.
    if "pinUserPeer" not in new_block and default_block.get("pinPeerName") is not None:
        new_block["pinUserPeer"] = default_block["pinPeerName"]
    # AI peer is profile-specific (bare profile name: Honcho peer IDs allow no dots);
    # workspace is shared so all profiles see the same context.
    new_block.update(aiPeer=profile_name, workspace=_pref(default_block, cfg, "workspace") or HOST,
                     enabled=default_block.get("enabled", True))
    cfg.setdefault("hosts", {})[new_host] = new_block
    _write_config(cfg)
    _ensure_peer_exists(new_host)  # eager so the peer exists before first message
    return True


def _sync_profiles(verbose: bool) -> int:
    """Clone host blocks for profiles lacking one; returns the count created."""
    say = print if verbose else (lambda *a: None)
    try:
        from hermes_cli.profiles import list_profiles
        profiles = list_profiles()
    except Exception as e:
        return say(f"  Could not list profiles: {e}\n") or 0
    cfg = _read_config()
    if not cfg:
        return say("  No Honcho config found. Run 'hermes honcho setup' first.\n") or 0
    default_block, has_key = _default_block_and_key(cfg)
    if not default_block and not has_key:
        return say("  Honcho not configured on default profile. Run 'hermes honcho setup' first.\n") or 0

    created = skipped = 0
    for p in (p for p in profiles if p.name != "default"):
        if clone_honcho_for_profile(p.name):
            say(f"  + {p.name} -> {profile_host_key(p.name)}")
            created += 1
        else:
            skipped += 1
    say(f"\n  {created} profile(s) synced." if created else "  All profiles already have Honcho config.")
    if skipped:
        say(f"  {skipped} profile(s) already configured (skipped).")
    say()
    return created


def cmd_sync(args) -> None:
    """Sync Honcho config to all existing profiles (inherits from the default block)."""
    _sync_profiles(verbose=True)


def sync_honcho_profiles_quiet() -> int:
    """Sync host blocks for all profiles from `hermes update`; no output, no exceptions."""
    return _sync_profiles(verbose=False)


def cmd_enable(args) -> None:
    """Enable Honcho for the active profile."""
    cfg = _read_config()
    host = _host_key()
    label = _label(host)
    block = cfg.setdefault("hosts", {}).setdefault(host, {})
    if block.get("enabled") is True:
        return print(f"  {label}Honcho is already enabled.\n")
    block["enabled"] = True

    if not block.get("aiPeer"):  # fresh profile block: clone settings from default
        default_block = cfg_get(cfg, "hosts", HOST, default={})
        _inherit_defaults(block, default_block, cfg, _INHERITED_KEYS)
        block.setdefault("aiPeer", host.split(".", 1)[1] if "." in host else host)
        block.setdefault("workspace", _pref(default_block, cfg, "workspace") or HOST)
    _write_config(cfg)
    print(f"  {label}Honcho enabled.")  # before the (possibly slow) peer-creation round trip
    peer_state = f"Peer '{block.get('aiPeer', host)}' ready." if _ensure_peer_exists(host) else "Peer creation deferred (no connection)."
    print(f"  {label}{peer_state}\n  Saved to {_config_path()}\n")


def cmd_disable(args) -> None:
    """Disable Honcho for the active profile."""
    cfg = _read_config()
    host = _host_key()
    block = cfg_get(cfg, "hosts", host, default={})
    if not block or block.get("enabled") is False:
        return print(f"  {_label(host)}Honcho is already disabled.\n")
    block["enabled"] = False
    print(f"  {_label(host)}Honcho disabled.")
    _save(cfg)


# ── identity mapping (setup wizard) ────────────────────────────────────────

def _resolve_effective_identity_mapping(cfg: dict, hermes_host: dict) -> tuple[bool, dict, str, bool, bool]:
    """``(pin, aliases, prefix, aliases_from_root, prefix_from_root)`` for the active host,
    mirroring ``from_global_config`` precedence (host over root; ``pinUserPeer`` beats
    ``pinPeerName``) so setup classifies the shape the gateway actually runs with.
    ``*_from_root`` lets writes skip inherited values."""
    pin_sources = (hermes_host.get("pinUserPeer"), hermes_host.get("pinPeerName"),
                   cfg.get("pinUserPeer"), cfg.get("pinPeerName"))
    pin = bool(next((v for v in pin_sources if v is not None), False))

    def _inherit(key):
        if key in hermes_host:
            return hermes_host.get(key), False
        val = cfg.get(key)
        return val, val is not None

    aliases_src, aliases_from_root = _inherit("userPeerAliases")
    prefix_src, prefix_from_root = _inherit("runtimePeerPrefix")
    aliases = aliases_src if isinstance(aliases_src, dict) else {}
    return pin, aliases, str(prefix_src or ""), aliases_from_root, prefix_from_root


def _scrub_identity_mapping(hermes_host: dict) -> None:
    """Drop every peer-mapping key so a stale alias/prefix/pin can't bleed into the new shape."""
    for key in _IDENTITY_MAPPING_KEYS:
        hermes_host.pop(key, None)


def _migrate_pin_key(block: dict) -> bool:
    """Rewrite legacy ``pinPeerName`` to canonical ``pinUserPeer`` in place (the
    resolver prefers the canonical key). Returns True if the block changed."""
    if "pinPeerName" not in block:
        return False
    block.setdefault("pinUserPeer", block.pop("pinPeerName"))
    return True


def _gateway_platforms() -> list[str] | None:
    """Connected gateway platforms, or None if undetectable (lazy + guarded: the memory
    plugin must not hard-depend on the gateway package)."""
    try:
        from gateway.config import load_gateway_config
        return [p.value for p in load_gateway_config().get_connected_platforms()]
    except Exception:
        return None


def _collect_operator_aliases(existing: dict, peer_target: str) -> dict:
    """Prompt for the operator's per-platform runtime IDs, aliasing each to ``peer_target``."""
    aliases = dict(existing)
    print(f"\n  Add runtime IDs that should alias to peer '{peer_target}'.\n"
          "  Leave blank to skip a platform.  Existing aliases are preserved.")
    for platform_label, alias_hint in (("Telegram UID", "e.g. 7654321"), ("Discord snowflake", "e.g. 491827364"),
                                       ("Slack user ID", "e.g. U04ABCDEF"), ("Matrix MXID", "e.g. @you:matrix.org")):
        entered = _prompt(f"  {platform_label} ({alias_hint})", default="").strip()
        if entered:
            aliases[entered] = peer_target
    return aliases


def _apply_runtime_prefix(hermes_host: dict, current_prefix: str, prefix_from_root: bool, label: str) -> None:
    """Write a host-level runtimePeerPrefix only when it diverges from an
    inherited root value; otherwise let the root cascade stand."""
    new_prefix = _prompt(label, default=current_prefix or "").strip()
    if new_prefix and not (prefix_from_root and new_prefix == current_prefix):
        hermes_host["runtimePeerPrefix"] = new_prefix


def _echo_identity_mapping(hermes_host: dict) -> None:
    print(f"  resolved →\n    pinUserPeer       = {bool(hermes_host.get('pinUserPeer'))}\n"
          f"    userPeerAliases   = {hermes_host.get('userPeerAliases') or '{}'}\n"
          f"    runtimePeerPrefix = {hermes_host.get('runtimePeerPrefix') or '(none)'}")


def _configure_raw_identity_mapping(hermes_host, current_pin, current_aliases, current_prefix,
                                    aliases_from_root, prefix_from_root) -> None:
    """Power-user escape hatch: set the three resolver knobs directly."""
    print("\n  Raw identity-mapping keys (resolver tries them top-down):")
    pin_in = _prompt("pinUserPeer — pin all gateway users to your peer? (true/false)",
                     default=str(bool(current_pin)).lower()).strip().lower()
    pin = pin_in in {"true", "t", "yes", "y", "1"}
    _scrub_identity_mapping(hermes_host)
    hermes_host["pinUserPeer"] = pin
    if pin:
        return
    aliases = dict(current_aliases) if isinstance(current_aliases, dict) and not aliases_from_root else {}
    print("  userPeerAliases — 'runtime_id=peer' pairs (blank line to finish):")
    while entry := _prompt("    alias", default="").strip():
        rid, _, peer = (p.strip() for p in entry.partition("="))
        if rid and peer:
            aliases[rid] = peer
    if aliases:
        hermes_host["userPeerAliases"] = aliases
    _apply_runtime_prefix(hermes_host, current_prefix, prefix_from_root,
                          "runtimePeerPrefix — namespace for unknown IDs (blank for none)")


def _setup_identity_mapping(cfg: dict, hermes_host: dict, current_peer: str) -> None:
    """Gateway identity mapping step. Only the gateway supplies a runtime user ID (CLI/TUI/
    desktop fall through to peerName), so the step is gated on gateway detection."""
    current_pin, current_aliases, current_prefix, aliases_from_root, prefix_from_root = (
        _resolve_effective_identity_mapping(cfg, hermes_host))
    current_shape = "single" if current_pin else "hybrid" if current_aliases else "multi"

    gw_platforms = _gateway_platforms()
    if gw_platforms:
        print(f"\n  Gateway platforms detected: {', '.join(gw_platforms)}")
    else:
        notice, question = (
            ("\n  Each gateway account (a Telegram user, a Discord user, ...)\n"
             "  resolves to a peer. Honcho builds one representation per peer.",
             "Running the Hermes gateway (Telegram/Discord/etc.)? (y/N)") if gw_platforms is None else
            ("\n  No gateway platforms connected — nothing to map.", "Configure anyway? (y/N)"))
        print(notice)
        if not _yes(_prompt(question, default="n")):
            return

    peer_target = hermes_host.get("peerName") or current_peer or "user"
    ai_peer_label = hermes_host.get("aiPeer") or cfg.get("aiPeer") or "hermes"
    # Fresh configs default to the personal shape; configured ones keep their detected shape.
    identity_configured = any(k in hermes_host or k in cfg for k in _IDENTITY_MAPPING_KEYS)
    default_choice = {"single": "1", "hybrid": "2", "multi": "3"}[current_shape] if identity_configured else "1"
    print("\n  This step covers the HUMAN mapping only. Each account using the\n"
          "  gateway resolves to a peer — the entity Honcho reasons about over\n"
          f"  time. This agent is already its own peer ('{ai_peer_label}'), and each\n"
          "  Hermes profile brings its own AI peer to the gateway.\n"
          "\n  How should accounts resolve?\n"
          "    [1] single peer — one person uses this agent; every account\n"
          f"        resolves to '{peer_target}'. The common personal setup.\n"
          "        Never for a gateway serving other people — their memory\n"
          "        would merge into yours\n"
          "    [2] your peer + one per other account — your accounts are\n"
          f"        aliased to '{peer_target}'; each other account gets its own\n"
          "        peer until you alias it. For a gateway you share\n"
          "    [3] one peer per account — no aliases; every account is its\n"
          "        own peer. For agents serving other people\n"
          "    [s] skip (leave untouched)   [e] edit raw keys\n"
          f"\n  Tip: alias your Telegram UID and your Discord ID to '{peer_target}' —\n"
          "  both accounts then resolve to one peer.")
    choice = _prompt("Choice", default=default_choice).strip().lower()

    if choice in {"2", "me+others", "both"}:
        pooled = _prompt("  Resolve all YOUR accounts to one peer? (Y/n)", default="y").strip().lower()
        shape = "hybrid" if pooled in {"y", "yes", ""} else "multi"
    else:
        shape = _SHAPE_CHOICES.get(choice, "skip")

    # Un-pinning without aliasing strands the pooled peerName history; steer toward pooling.
    if current_pin and shape == "multi":
        print(f"\n  ⚠ The peer '{peer_target}' already has a representation built\n"
              f"    from your messages. One peer per account means your accounts\n"
              f"    resolve to new peers with no history.")
        if _prompt(f"  Keep your accounts resolving to '{peer_target}' instead? (Y/n)", default="y").strip().lower() in {"y", "yes", ""}:
            shape = "hybrid"

    if shape == "skip":
        return print("  Identity mapping left untouched.")
    if shape == "raw":
        _configure_raw_identity_mapping(hermes_host, current_pin, current_aliases, current_prefix,
                                        aliases_from_root, prefix_from_root)
    else:
        # Preserve operator-curated host-level aliases across multi → multi re-runs. Root-sourced
        # aliases cascade naturally and are NOT copied down — an empty host map would mask a root baseline.
        prior_aliases = dict(current_aliases) if isinstance(current_aliases, dict) else {}
        if shape == "multi" and aliases_from_root:
            prior_aliases = {}
        _scrub_identity_mapping(hermes_host)  # each shape starts from a clean slate
        hermes_host["pinUserPeer"] = shape == "single"
        if shape == "single":
            print(f"  Every gateway account resolves to peer '{peer_target}'.")
        else:
            aliases = prior_aliases if shape == "multi" else _collect_operator_aliases(prior_aliases, peer_target)
            if aliases:
                hermes_host["userPeerAliases"] = aliases
            _apply_runtime_prefix(hermes_host, current_prefix, prefix_from_root,
                                  "Runtime peer prefix (e.g. 'telegram_', blank for none)" if shape == "multi" else
                                  "Runtime peer prefix for unknown users (e.g. 'telegram_', blank for none)")
            print("  Each gateway account resolves to its own peer." if shape == "multi" else
                  f"  Your accounts resolve to '{peer_target}'; each other account to its own peer.")
    _echo_identity_mapping(hermes_host)


# ── setup wizard ───────────────────────────────────────────────────────────

def _ensure_sdk_installed() -> bool:
    """Check honcho-ai is importable; offer to install if not. Returns True if ready."""
    try:
        import honcho  # noqa: F401
        return True
    except ImportError:
        pass
    print("  honcho-ai is not installed.")
    if not _yes(_prompt("Install it now? (honcho-ai==2.2.0)", default="y")):
        print("  Skipping install. Run: pip install 'honcho-ai==2.2.0'\n")
        return False
    print("  Installing honcho-ai...", flush=True)
    from tools.lazy_deps import install_specs  # env-aware: sealed hosted venvs redirect to the data volume
    result = install_specs(["honcho-ai==2.2.0"])
    if result.ok:
        print("  Installed.\n")
        return True
    print(f"  Cannot install: {result.reason}\n" if result.blocked else
          f"  Install failed:\n{(result.stderr or '').strip()}\n  Run manually: uv pip install 'honcho-ai==2.2.0'\n")
    return False


def _device_login_available() -> bool:
    """Whether the resolved host offers the RFC 8628 device grant. Fails closed."""
    try:
        from plugins.memory.honcho.oauth_flow import resolve_endpoints, supports_device_login
        return supports_device_login(resolve_endpoints())
    except Exception:
        return False


def _headless() -> tuple[bool, bool]:
    """(is_remote, can_open_browser) — degrades safely if hermes_cli internals move."""
    try:
        from hermes_cli.auth import _can_open_graphical_browser, _is_remote_session
        return _is_remote_session(), _can_open_graphical_browser()
    except Exception:
        return False, True


def _apply_grant_to_host(hermes_host: dict, cred) -> None:
    """Store an OAuth grant on the host block; the wizard's final save persists it."""
    hermes_host["apiKey"] = cred.access_token
    hermes_host["oauth"] = cred.oauth_block()
    if cred.consent_peer_name:  # default the peer prompt to the consent name
        hermes_host["peerName"] = cred.consent_peer_name
    print("  Authorized — token saved. Let's finish configuring.\n")


def _setup_local_auth(cfg: dict, hermes_host: dict) -> None:
    """Self-hosted Honcho may run with AUTH_USE_AUTH=true; clients then send a JWT signed with
    the server's AUTH_JWT_SECRET as the bearer token. It is stored under the host block (not
    top-level apiKey) so ``get_honcho_client`` treats it as an explicit local auth opt-in and
    cloud/hybrid switching is unaffected."""
    if new_url := _prompt("Base URL", default=cfg.get("baseUrl") or "http://localhost:8000"):
        cfg["baseUrl"] = new_url
    current_host_key = hermes_host.get("apiKey", "")
    print("\n  Local Honcho auth (JWT signed with the server's AUTH_JWT_SECRET).\n"
          f"  Leave blank if your server runs with AUTH_USE_AUTH=false. Current: {_mask(current_host_key)}")
    new_local_key = _prompt("Local JWT / bearer token (blank to skip / keep current)", secret=True)
    if new_local_key:
        hermes_host["apiKey"] = new_local_key
    elif current_host_key:
        print("  Keeping existing local JWT.")
    elif cfg.get("apiKey", ""):
        print("\n  Top-level API key present in config (kept for cloud/hybrid use).\n"
              "  Local connections will skip auth automatically until a local JWT is set above.")
    else:
        print("\n  No local JWT set. Local no-auth ready.")


def _setup_device_login(hermes_host: dict, write_path: Path, *, open_browser: bool) -> bool:
    """RFC 8628 device-code sign-in. Returns False if setup must abort."""
    from plugins.memory.honcho.oauth_flow import (
        AccessDenied, AuthorizationTimeout, DeviceCode, DeviceCodeExpired, DeviceFlowError, authorize_via_device_code,
    )

    def _show(device: DeviceCode) -> None:
        print(f"\n  To connect, on any device with a browser:\n\n    1. Open   {device.verification_uri}\n"
              f"    2. Enter  {device.user_code}\n\n  Or open directly:\n\n    {device.verification_uri_complete}\n")
        mins = max(1, device.expires_in // 60)
        print(f"  Waiting for approval (expires in {mins} min, Ctrl-C to cancel) ", end="", flush=True)

    print("\n  Requesting device code…")
    import webbrowser
    try:
        cred = authorize_via_device_code(
            config_path=write_path, source="hermes-cli", apply_config=False, display=_show,
            open_url=webbrowser.open if open_browser else None, on_poll=lambda: print(".", end="", flush=True),
        )
    except KeyboardInterrupt:
        print("\n  Cancelled. Re-run 'hermes honcho setup' to try again.\n")
    except (AuthorizationTimeout, DeviceCodeExpired):
        print("\n  Device code expired before approval.\n  Re-run 'hermes honcho setup' to get a new code.\n")
    except AccessDenied:
        print("\n  Sign-in was denied on the approval page.\n" + _RETRY_HINT)
    except Exception as e:
        print("\n  Too many device-code requests — wait a minute and re-run setup.\n"
              if isinstance(e, DeviceFlowError) and e.error == "http_429" else f"\n  Device sign-in failed: {e}\n" + _RETRY_HINT)
    else:
        print(" approved")
        _apply_grant_to_host(hermes_host, cred)
        return True
    return False


def _setup_browser_login(hermes_host: dict, write_path: Path) -> bool:
    """Loopback OAuth sign-in. Tokens merge into the in-memory cfg so the wizard's final save
    keeps them; settings stay wizard-owned (apply_config=False). Returns False on abort."""
    from plugins.memory.honcho.oauth_flow import authorize_via_loopback
    import webbrowser

    def _open(url: str) -> None:
        print(f"\n  Open this link to authorize (waiting up to 5 minutes):\n\n    {url}\n")
        webbrowser.open(url)

    print("\n  Starting browser sign-in…")
    try:
        cred = authorize_via_loopback(config_path=write_path, source="hermes-cli", apply_config=False, open_url=_open)
    except Exception as e:
        print(f"  OAuth sign-in failed: {e}\n" + _RETRY_HINT)
        return False
    _apply_grant_to_host(hermes_host, cred)
    return True


def _setup_cloud_auth(cfg: dict, hermes_host: dict, write_path: Path) -> bool:
    """Cloud auth: OAuth (browser), device code, or API key. Returns False on abort."""
    cfg.pop("baseUrl", None)  # cloud uses SDK default
    from plugins.memory.honcho.oauth import OAuthCredential
    existing_oauth = OAuthCredential.from_host_block(hermes_host)
    device_available = _device_login_available()
    is_remote, can_browse = _headless()

    print("\n  Auth method:")
    if existing_oauth is not None:
        print(f"    (currently connected via OAuth — client {existing_oauth.client_id})")
    print("    oauth  -- sign in via browser on this machine (recommended)")
    if device_available:
        print("    device -- device code: approve from a browser on another machine (SSH / headless)")
    print("    apikey -- paste an API key from https://app.honcho.dev")

    default_method = "oauth"
    if is_remote or not can_browse:
        default_method = "device" if device_available else "oauth"
        print("  (no usable local browser detected — device code recommended)" if device_available else
              "  (no usable local browser detected — browser sign-in may need an SSH tunnel to 127.0.0.1:8765)")
    method = _prompt("oauth, device, or apikey?" if device_available else "OAuth or API key?",
                     default=default_method).strip().lower()

    if device_available and method in {"device", "d"}:
        return _setup_device_login(hermes_host, write_path, open_browser=can_browse and not is_remote)
    if method in {"oauth", "o"}:
        return _setup_browser_login(hermes_host, write_path)
    print(f"\n  Current API key: {_mask(cfg.get('apiKey', ''))}")
    if new_key := _prompt("Honcho API key (leave blank to keep current)", secret=True):
        cfg["apiKey"] = new_key
    if cfg.get("apiKey"):
        return True
    print("\n  No API key configured. Get yours at https://app.honcho.dev\n"
          "  Run 'hermes honcho setup' again once you have a key.\n")
    return False


def _menu(header: str, *lines: str) -> None:
    print(f"\n  {header}:\n" + "\n".join(f"    {line}" for line in lines))


def _choice_step(hermes_host, key, current, label, valid, fallback=None) -> None:
    """Prompt for one of ``valid``; an invalid answer writes ``fallback`` (None = keep current)."""
    new = _prompt(label, default=current)
    if new in valid:
        hermes_host[key] = new
    elif fallback is not None:
        hermes_host[key] = fallback


def _setup_tuning(cfg: dict, hermes_host: dict) -> None:
    """Wizard steps 4-8: observation, write frequency, recall, budgets, reasoning, strategy."""
    _menu("Observation mode",
          "directional  -- all observations on, each AI peer builds its own view (default)",
          "unified      -- user observes self, AI observes others only")
    _choice_step(hermes_host, "observationMode", _pref(hermes_host, cfg, "observationMode", "directional"),
                 "Observation mode", {"unified", "directional"}, "directional")

    _menu("Write frequency",
          "async   -- background thread, no token cost (recommended)",
          "turn    -- sync write after every turn",
          "session -- batch write at session end only",
          "N       -- write every N turns (e.g. 5)")
    new_wf = _prompt("Write frequency", default=str(_pref(hermes_host, cfg, "writeFrequency", "async")))
    hermes_host["writeFrequency"] = _first_parsed([new_wf], int, new_wf if new_wf in {"async", "turn", "session"} else "async")

    _menu("Recall mode", *(f"{m:<7} -- {desc}" for m, desc in _MODES.items()))
    raw_recall = _pref(hermes_host, cfg, "recallMode", "hybrid")
    _choice_step(hermes_host, "recallMode", raw_recall if raw_recall in _MODES else "hybrid", "Recall mode", _MODES)

    current_ctx_tokens = _pref(hermes_host, cfg, "contextTokens")
    _menu("Context injection per turn (hybrid/context recall modes only)",
          "uncapped -- no limit (default)",
          "N        -- token limit per turn (e.g. 1200)")
    new_ctx_tokens = _prompt("Context tokens", default=str(current_ctx_tokens) if current_ctx_tokens else "uncapped").strip()
    if new_ctx_tokens.lower() in {"none", "uncapped", "no limit"}:
        hermes_host.pop("contextTokens", None)
    elif new_ctx_tokens and (val := _first_parsed([new_ctx_tokens], int, -1)) >= 0:  # non-numeric keeps current
        hermes_host["contextTokens"] = val

    _menu("Dialectic cadence",
          "How often Honcho rebuilds its user model (LLM call on Honcho backend).",
          "1 = every turn, 2 = every other turn, 3+ = sparser.",
          "Recommended: 1-5.")
    new_dialectic = _prompt("Dialectic cadence", default=str(_pref(hermes_host, cfg, "dialecticCadence") or "2"))
    if (val := _first_parsed([new_dialectic], int, None)) is None:
        hermes_host["dialecticCadence"] = 2
    elif val >= 1:
        hermes_host["dialecticCadence"] = val

    _menu("Dialectic reasoning level",
          "Depth Honcho uses when synthesizing user context on auto-injected calls.",
          "minimal  -- quick factual lookups",
          "low      -- straightforward questions (default)",
          "medium   -- multi-aspect synthesis",
          "high     -- complex behavioral patterns",
          "max      -- thorough audit-level analysis")
    _choice_step(hermes_host, "dialecticReasoningLevel", _pref(hermes_host, cfg, "dialecticReasoningLevel") or "low",
                 "Reasoning level", REASONING_LEVELS, "low")

    _menu("Session strategy", *(f"{s:<13} -- {desc}" for s, desc in _STRATEGIES.items()))
    _choice_step(hermes_host, "sessionStrategy", _pref(hermes_host, cfg, "sessionStrategy", "per-session"),
                 "Session strategy", _STRATEGIES)


def cmd_setup(args) -> None:
    """Interactive Honcho setup wizard."""
    cfg = _read_config()
    write_path, read_path = _local_config_path(), _config_path()
    print(f"\nHoncho memory setup\n{RULE}\n  Honcho gives Hermes persistent cross-session memory.\n  Config: {write_path}")
    if read_path != write_path and read_path.exists():
        print(f"  (seeding from existing config at {read_path})")
    print()
    if not _ensure_sdk_installed():
        return

    hermes_host = cfg.setdefault("hosts", {}).setdefault(_host_key(), {})
    _migrate_pin_key(cfg)  # canonicalize legacy pinPeerName before detection/writes
    _migrate_pin_key(hermes_host)

    # --- 1. Cloud or local? ---
    print("  Deployment:\n    cloud -- Honcho cloud (api.honcho.dev)\n    local -- self-hosted Honcho server")
    current_url = cfg.get("baseUrl") or cfg.get("base_url") or ""
    current_deploy = "local" if any(h in current_url for h in ("localhost", "127.0.0.1", "::1")) else "cloud"
    is_local = _prompt("Cloud or local?", default=current_deploy).lower() in {"local", "l"}
    cfg.pop("base_url", None)  # legacy snake_case key
    if is_local:
        _setup_local_auth(cfg, hermes_host)
    elif not _setup_cloud_auth(cfg, hermes_host, write_path):
        return

    # --- 3. Identity ---
    current_peer = hermes_host.get("peerName") or cfg.get("peerName", "")
    for key, label, default in (
        ("peerName", "Your name (user peer)", current_peer or os.getenv("USER", "user")),
        ("aiPeer", "AI peer name", _pref(hermes_host, cfg, "aiPeer", "hermes")),
        ("workspace", "Workspace ID", _pref(hermes_host, cfg, "workspace", "hermes")),
    ):
        if new := _prompt(label, default=default):
            hermes_host[key] = new

    _setup_identity_mapping(cfg, hermes_host, current_peer)
    print("\n  For a gateway with many users and agents, run\n"
          "  'hermes honcho peers map' to map accounts interactively.")

    _setup_tuning(cfg, hermes_host)
    hermes_host["enabled"] = True
    hermes_host.setdefault("saveMessages", True)
    _write_config(cfg)
    print(f"\n  Config written to {write_path}")

    try:  # auto-enable Honcho as memory provider in config.yaml
        from hermes_cli.config import load_config, save_config
        hermes_config = load_config()
        hermes_config.setdefault("memory", {})["provider"] = "honcho"
        save_config(hermes_config)
        print("  Memory provider set to 'honcho' in config.yaml")
    except Exception as e:
        print(f"  Could not auto-enable in config.yaml: {e}\n  Run: hermes config set memory.provider honcho")

    print("  Testing connection... ", end="", flush=True)
    try:
        hcfg, _client = _connect(_host_key(), reset=True)
        print("OK")
    except Exception as e:
        return print(f"FAILED\n  Error: {e}")

    print(f"""
  Honcho is ready.
  Session:   {hcfg.resolve_session_name()}
  Workspace: {hcfg.workspace_id}
  User:      {hcfg.peer_name}
  AI peer:   {hcfg.ai_peer}
  Observe:   {hcfg.observation_mode}
  Frequency: {hcfg.write_frequency}
  Recall:    {hcfg.recall_mode}
  Sessions:  {hcfg.session_strategy}

  Honcho tools available in chat:
    honcho_context   -- session context: summary, representation, card, messages
    honcho_search    -- semantic search over history
    honcho_profile   -- peer card, key facts
    honcho_reasoning -- ask Honcho a question, synthesized answer
    honcho_conclude  -- persist a user fact to memory

  Other commands:
    hermes honcho status     -- show full config
    hermes honcho mode       -- change recall/observation mode
    hermes honcho tokens     -- tune context and dialectic budgets
    hermes honcho peer       -- update peer names
    hermes honcho map <name> -- map this directory to a session name
""")


# ── status / peers ─────────────────────────────────────────────────────────

def _active_profile_name() -> str:
    """Active Hermes profile name (respects --target-profile override)."""
    if _profile_override:
        return _profile_override
    try:
        from hermes_cli.profiles import get_active_profile_name
        return get_active_profile_name()
    except Exception:
        return "default"


def _all_profile_host_configs() -> list[tuple[str, str, dict]]:
    """(profile_name, host_key, host_block) for every known profile, reading honcho.json once."""
    try:
        from hermes_cli.profiles import list_profiles
        profiles = list_profiles()
    except Exception:
        return [(_active_profile_name(), _host_key(), {})]
    cfg = _read_config()
    # _host_block (not hosts.get) keeps legacy dot-form keys ("hermes.work") readable.
    return [("default", HOST, cfg.get("hosts", {}).get(HOST, {}))] + [
        (p.name, profile_host_key(p.name), _host_block(cfg, profile_host_key(p.name)))
        for p in profiles if p.name != "default"
    ]


def cmd_status(args) -> None:
    """Show current Honcho config and connection status."""
    if getattr(args, "all", False):
        _cmd_status_all()
        return
    try:
        import honcho  # noqa: F401
    except ImportError:
        print("  honcho-ai is not installed. Run: hermes honcho setup\n")
        return

    cfg = _read_config()
    active_path = _config_path()
    write_path = _local_config_path()
    from plugins.memory.honcho.client import HonchoClientConfig, get_honcho_client
    not_found = f"  No Honcho config found at {active_path}\n  Run 'hermes honcho setup' to configure.\n"
    try:
        hcfg = HonchoClientConfig.from_global_config(host=_host_key())
    except Exception as e:
        return print(not_found if not cfg else f"  Config error: {e}\n")
    if not cfg and not (hcfg.api_key or hcfg.base_url):  # file missing and no env-var fallback either
        return print(not_found)

    # The OAuth access token is also stored under apiKey, so the auth line
    # distinguishes a refreshable grant from a static key explicitly.
    from plugins.memory.honcho.oauth import OAuthCredential
    raw = getattr(hcfg, "raw", None) or {}
    cred = OAuthCredential.from_host_block(raw.get("hosts", {}).get(hcfg.host) or {})
    profile = _active_profile_name()

    if cred is not None:
        import time as _time
        remaining = int(cred.expires_at - _time.time())
        token_state = f"valid {remaining // 60}m" if remaining > 0 else "expired — refreshes on next use"
        auth = f"OAuth ({cred.client_id}, token {token_state})"
    else:
        auth = f"API key ({_mask(hcfg.api_key or '')})"
    print(f"\nHoncho status{f' [{hcfg.host}]' if profile != 'default' else ''}\n" + RULE
          + (f"\n  Profile:        {profile}" if profile != "default" else ""))
    print(f"  Host:           {hcfg.host}\n  Enabled:        {hcfg.enabled}\n  Auth:           {auth}\n"
          f"  Workspace:      {hcfg.workspace_id}\n  Config:         {active_path}")
    global_path = Path.home() / ".honcho" / "config.json"
    if write_path != active_path:
        print(f"  Write to:       {write_path}  (profile-local)")
    if active_path == global_path:
        print("  Fallback:       (none — using global ~/.honcho/config.json)")
    elif global_path.exists():
        print(f"  Fallback:       {global_path}  (exists, cross-app interop)")

    dialectic_cadence = getattr(hcfg, "dialectic_cadence", None) or raw.get("dialecticCadence") or 1
    reasoning_cap = raw.get("reasoningLevelCap") or hcfg.reasoning_level_cap
    print(f"""  AI peer:        {hcfg.ai_peer}
  User peer:      {hcfg.peer_name or 'not set'}
  Session key:    {hcfg.resolve_session_name()}
  Session strat:  {hcfg.session_strategy}
  Recall mode:    {hcfg.recall_mode}
  Context budget: {hcfg.context_tokens or '(uncapped)'} tokens
  Dialectic cad:  every {dialectic_cadence} turn{'s' if dialectic_cadence != 1 else ''}
  Reasoning:      base={hcfg.dialectic_reasoning_level}, cap={reasoning_cap}, heuristic={'on' if hcfg.reasoning_heuristic else 'off'}
  Observation:    user(me={hcfg.user_observe_me},others={hcfg.user_observe_others}) ai(me={hcfg.ai_observe_me},others={hcfg.ai_observe_others})
  Write freq:     {hcfg.write_frequency}""")

    if not (hcfg.enabled and (hcfg.api_key or hcfg.base_url)):
        return print(f"\n  Not connected ({'disabled' if not hcfg.enabled else 'no API key or base URL'})\n")
    print("\n  Connection... ", end="", flush=True)
    try:
        _show_peer_cards(hcfg, get_honcho_client(hcfg))
        print("OK")
    except Exception as e:
        print(f"FAILED ({e})\n")


def _show_peer_cards(hcfg, client) -> None:
    """Fetch and display peer cards for the active profile."""
    try:
        mgr, session_key = _session_manager(hcfg, client)
        card = mgr.get_peer_card(session_key)
        if card:
            print(f"\n  User peer card ({len(card)} facts):\n" + "\n".join(f"    - {fact}" for fact in card[:10]))
            if len(card) > 10:
                print(f"    ... and {len(card) - 10} more")
        ai_text = mgr.get_ai_representation(session_key).get("representation", "")
        if ai_text:
            print(f"\n  AI peer representation:\n    {ai_text[:200] + ('...' if len(ai_text) > 200 else '')}")
        print("" if card or ai_text else "\n  No peer data yet (accumulates after first conversation)\n")
    except Exception as e:
        print(f"\n  Peer data unavailable: {e}\n")


def _cmd_status_all() -> None:
    """Show Honcho config overview across all profiles."""
    rows = _all_profile_host_configs()
    cfg = _read_config()
    active = _active_profile_name()
    print(f"\nHoncho profiles ({len(rows)})\n{'─' * 55}\n"
          f"  {'Profile':<14} {'Host':<22} {'Enabled':<9} {'Recall':<9} {'Write'}\n"
          f"  {'─' * 14} {'─' * 22} {'─' * 9} {'─' * 9} {'─' * 9}")
    for name, host, block in rows:
        enabled = block.get("enabled", cfg.get("enabled"))
        if enabled is None:
            enabled = _default_block_and_key(cfg)[1] if block else False
        marker = " *" if name == active else ""
        print(f"  {name + marker:<14} {host:<22} {'yes' if enabled else 'no':<9} "
              f"{_pref(block, cfg, 'recallMode', 'hybrid'):<9} {_pref(block, cfg, 'writeFrequency', 'async')}")
    print("\n  * active profile\n")


def _state_db_path() -> Path:
    """Return the state.db path for the targeted profile."""
    if _profile_override and _profile_override not in {"default", "custom"}:
        try:
            from hermes_cli.profiles import get_profile_dir
            return get_profile_dir(_profile_override) / "state.db"
        except Exception:
            pass
    return get_hermes_home() / "state.db"


def _seen_gateway_accounts(db_path: Path) -> list[dict]:
    """Return gateway accounts recorded in state.db, most recent first.

    The gateway stamps each session row with its routing peer (source,
    user_id, display_name, origin_json), so grouping rows by
    (source, user_id) enumerates every account the gateway has handled.
    Bot authors are skipped.
    """
    if not db_path.exists():
        return []
    import sqlite3
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return []
    # profile_name marks which profile a multiplexing gateway routed the
    # session to; older state.db files predate the column.
    base_query = """SELECT source, user_id,
                      MAX(COALESCE(display_name, '')),
                      MAX(COALESCE(origin_json, '')),
                      COUNT(*){profiles_col}
                 FROM sessions
                WHERE user_id IS NOT NULL AND user_id != ''
                GROUP BY source, user_id
                ORDER BY MAX(COALESCE(started_at, 0)) DESC"""
    try:
        try:
            rows = conn.execute(base_query.format(
                profiles_col=",\n                      GROUP_CONCAT(DISTINCT COALESCE(profile_name, 'default'))",
            )).fetchall()
        except sqlite3.OperationalError:
            rows = [r + (None,) for r in conn.execute(
                base_query.format(profiles_col=""),
            ).fetchall()]
    except sqlite3.Error:
        return []
    finally:
        conn.close()

    accounts = []
    for source, user_id, display_name, origin_json, count, profiles in rows:
        user_name = ""
        user_id_alt = ""
        is_bot = False
        if origin_json:
            try:
                origin = json.loads(origin_json)
                user_name = origin.get("user_name") or ""
                user_id_alt = origin.get("user_id_alt") or ""
                is_bot = bool(origin.get("is_bot"))
            except Exception:
                pass
        if is_bot:
            continue
        accounts.append({
            "platform": source or "?",
            "user_id": str(user_id),
            "user_id_alt": str(user_id_alt),
            "label": user_name or display_name or "",
            "sessions": count,
            "profiles": sorted(profiles.split(",")) if profiles else [],
        })
    return accounts


def _preview_peer_resolution(
    user_id: str, *, pin: bool, aliases: dict, prefix: str, peer_name: str,
    user_id_alt: str = "",
) -> str:
    """Mirror the runtime resolver ladder for display: pin → alias → prefix → raw."""
    import re

    def sanitize(s: str) -> str:
        return re.sub(r'[^a-zA-Z0-9_-]', '-', s)

    if pin and peer_name:
        return f"{sanitize(peer_name)} (pinned)"
    # The runtime resolver checks the primary ID first, then the alt ID
    # (Signal UUID, Feishu union_id) against the same alias map.
    for rid in (user_id, user_id_alt):
        alias = aliases.get(rid) if rid else None
        if isinstance(alias, str) and alias.strip():
            return sanitize(alias.strip())
    if prefix:
        return f"{sanitize(prefix + user_id)} (prefixed)"
    return sanitize(user_id)


# Workspaces holding thousands of peers (public bots) must not stall the CLI.
_PEERS_MAP_FETCH_CAP = 200


def _peers_map_client(workspace: str | None = None):
    """Return (client, config) for the active host, or (None, None) offline.

    ``workspace`` overrides the configured workspace for browse mode.
    """
    try:
        from dataclasses import replace
        from plugins.memory.honcho.client import HonchoClientConfig, get_honcho_client
        hcfg = HonchoClientConfig.from_global_config(host=_host_key())
        if not (hcfg.api_key or hcfg.base_url):
            return None, None
        if workspace and workspace != hcfg.workspace_id:
            hcfg = replace(hcfg, workspace_id=workspace)
        return get_honcho_client(hcfg), hcfg
    except Exception:
        return None, None


def _api_workspace_peers(client) -> list[dict] | None:
    """Fetch workspace peers (id + created date). None = API unavailable."""
    if client is None:
        return None
    try:
        peers: list[dict] = []
        page = 1
        while len(peers) < _PEERS_MAP_FETCH_CAP:
            batch = list(client.peers(page=page, size=50))
            if not batch:
                break
            for p in batch:
                peers.append({
                    "id": str(p.id),
                    "created": str(getattr(p, "created_at", "") or "")[:10],
                })
            if len(batch) < 50:
                break
            page += 1
        return peers
    except Exception:
        return None


def _api_workspaces(client) -> list[str] | None:
    """Workspace IDs this key can reach. None = API unavailable."""
    if client is None:
        return None
    try:
        return [str(w) for w in client.workspaces(size=50)]
    except Exception:
        return None


def _api_peer_detail(client, peer_id: str) -> str:
    """Short peek at a peer: its card, or a clear absence note."""
    try:
        card = client.peer(peer_id).get_card()
        if card:
            text = str(card).strip()
            return text[:400] + ("…" if len(text) > 400 else "")
        return "(no peer card yet)"
    except Exception as e:
        return f"(peer detail unavailable: {e})"


def _sanitize_peer_id(s: str) -> str:
    import re
    return re.sub(r'[^a-zA-Z0-9_-]', '-', s)


def _classify_workspace_peers(
    peer_ids: list[str], cfg: dict, accounts: list[dict],
    aliases: dict, prefix: str,
) -> dict[str, str]:
    """Label workspace peers from local config; 'unrecognized' when honest."""
    labels: dict[str, str] = {}
    active_host = _host_key()
    root_peer = cfg.get("peerName") or ""

    profile_rows = _all_profile_host_configs()
    hermes_hosts = {hostk for _, hostk, _ in profile_rows}
    for name, hostk, block in profile_rows:
        pn = block.get("peerName") or root_peer
        ai = block.get("aiPeer") or cfg.get("aiPeer") or hostk
        if pn:
            labels.setdefault(
                _sanitize_peer_id(pn),
                "your peer (peerName)" if hostk == active_host
                else f"peerName of profile {name}",
            )
        who = "this profile" if hostk == active_host else f"profile {name}"
        labels.setdefault(_sanitize_peer_id(ai), f"AI peer · {who}")

    # Host blocks that are not Hermes profiles: other apps sharing the config.
    for hostk, block in (cfg.get("hosts") or {}).items():
        if hostk in hermes_hosts or not isinstance(block, dict):
            continue
        for key, kind in (("peerName", "peer"), ("aiPeer", "AI peer")):
            val = block.get(key)
            if isinstance(val, str) and val.strip():
                labels.setdefault(
                    _sanitize_peer_id(val.strip()), f"{kind} of app '{hostk}'")

    for target in aliases.values():
        if isinstance(target, str) and target.strip():
            labels.setdefault(_sanitize_peer_id(target.strip()), "alias target")

    for acct in accounts:
        rid = acct["user_id"]
        candidates = [_sanitize_peer_id(rid)]
        if prefix:
            candidates.append(_sanitize_peer_id(prefix + rid))
        for candidate in candidates:
            labels.setdefault(
                candidate, f"runtime peer · {acct['platform']} {rid}")

    out = {}
    for pid in peer_ids:
        label = labels.get(pid)
        if label is None and pid.startswith("user-"):
            label = "fallback peer (pre-identity traffic)"
        out[pid] = label or "unrecognized"
    return out


def _sibling_resolutions(cfg: dict, acct: dict) -> dict[str, str]:
    """Resolved peer per profile for one account (profile name → peer)."""
    out = {}
    for name, _hostk, block in _all_profile_host_configs():
        pin, aliases, prefix, _, _ = _resolve_effective_identity_mapping(cfg, block)
        pn = block.get("peerName") or cfg.get("peerName") or ""
        out[name] = _preview_peer_resolution(
            acct["user_id"], pin=pin, aliases=aliases, prefix=prefix,
            peer_name=pn, user_id_alt=acct["user_id_alt"],
        )
    return out


def _resolution_base(resolved: str) -> str:
    """Strip display suffixes so the name can be checked against peer IDs."""
    return resolved.removesuffix(" (pinned)").removesuffix(" (prefixed)")


def _render_peers_map_view(
    workspace: str, ws_peers: list[dict] | None, labels: dict,
    accounts: list[dict], cfg: dict, *,
    pin: bool, working: dict, prefix: str, peer_name: str,
) -> None:
    peer_id_set = {p["id"] for p in (ws_peers or [])}
    if ws_peers is None:
        print(f"\nWorkspace '{workspace}' — peers unavailable (offline or not configured)")
        print("  Mapping still works; target peers are typed instead of picked.")
    else:
        print(f"\nWorkspace '{workspace}' — {len(ws_peers)} peers\n" + "─" * 62)
        if not ws_peers:
            print("  No peers here yet — peers appear after the first conversation.")
            print("  Wrong workspace? 'w' lists the workspaces this key can see.")
        for i, p in enumerate(ws_peers, 1):
            print(f"  p{i:<4} {p['id']:<30} {labels.get(p['id'], '')}")
        if len(ws_peers) >= _PEERS_MAP_FETCH_CAP:
            print(f"  … listing capped at {_PEERS_MAP_FETCH_CAP} peers.")
        recognized = any(
            v.startswith(("your peer", "AI peer")) for v in labels.values()
        )
        if ws_peers and not recognized:
            print(f"\n  None of these match your configured identity ('{peer_name or workspace}').")
            print("  Wrong workspace? 'w' lists the workspaces this key can see.")

    active_profile = _active_profile_name()
    print(f"\nGateway accounts seen on this machine ({len(accounts)})\n" + "─" * 62)
    if not accounts:
        print("  None recorded yet. Accounts appear here after the gateway")
        print("  handles a message from them. You can still map a runtime ID")
        print("  by typing it at the prompt below.")
        return
    print(f"  {'#':<4} {'Platform':<10} {'Runtime ID':<22} {'Name':<14} {'Resolves to'}")
    for idx, acct in enumerate(accounts, 1):
        resolved = _preview_peer_resolution(
            acct["user_id"], pin=pin, aliases=working,
            prefix=prefix, peer_name=peer_name,
            user_id_alt=acct["user_id_alt"],
        )
        marker = ""
        if ws_peers is not None:
            marker = " ✓" if _resolution_base(resolved) in peer_id_set else " ○ new"
        siblings = _sibling_resolutions(cfg, acct)
        mine = _resolution_base(resolved)
        diverging = {
            n: _resolution_base(v) for n, v in siblings.items()
            if n != active_profile and _resolution_base(v) != mine
        }
        div = ""
        if diverging:
            div = "  ≠ " + ", ".join(f"{n}→{v}" for n, v in sorted(diverging.items()))
        via = ""
        profiles = acct.get("profiles") or []
        if profiles and active_profile not in profiles:
            via = f"  (traffic → {', '.join(profiles)})"
        label = acct["label"][:13]
        print(
            f"  {idx:<4} {acct['platform']:<10} {acct['user_id']:<22} "
            f"{label:<14} {resolved}{marker}{div}{via}"
        )


def _workspaces_flow(client, current_ws: str, cfg: dict, host: str):
    """List reachable workspaces; browse one; optionally repoint the profile.

    Returns (workspace, client, ws_peers) after a confirmed switch, else None.
    """
    ws_list = _api_workspaces(client)
    if not ws_list:
        print("  Workspace list unavailable (offline or not configured).")
        return None
    print(f"\n  Workspaces this key can see ({len(ws_list)}):")
    for i, w in enumerate(ws_list, 1):
        marker = "  ← current" if w == current_ws else ""
        print(f"    {i:<4} {w}{marker}")
    print("\n  Tip: for a full workspace browser, install honcho-cli")
    print("  (uv tool install honcho-cli).")
    sel = _prompt("Browse a workspace (number, blank to go back)", default="").strip()
    if not (sel.isdigit() and 1 <= int(sel) <= len(ws_list)):
        return None
    target_ws = ws_list[int(sel) - 1]
    b_client, _ = _peers_map_client(workspace=target_ws)
    b_peers = _api_workspace_peers(b_client)
    if b_peers is None:
        print(f"  Could not list peers of '{target_ws}'.")
        return None
    print(f"\n  Workspace '{target_ws}' — {len(b_peers)} peers")
    for p in b_peers[:30]:
        print(f"    {p['id']}")
    if len(b_peers) > 30:
        print(f"    … and {len(b_peers) - 30} more")
    if target_ws == current_ws:
        return None
    switch = _prompt(
        f"Point this profile at '{target_ws}'? Existing memory stays in '{current_ws}'. (y/N)",
        default="n",
    ).strip().lower()
    if switch not in {"y", "yes"}:
        return None
    cfg.setdefault("hosts", {}).setdefault(host, {})["workspace"] = target_ws
    _write_config(cfg)
    print(f"  workspace → '{target_ws}' (written to host block [{host}])")
    return target_ws, b_client, b_peers


def _save_alias_map(cfg: dict, host: str, working: dict, aliases_from_root: bool) -> None:
    """Persist the edited alias map, asking for scope when it is shared."""
    profiles = _all_profile_host_configs()
    write_root = aliases_from_root
    if aliases_from_root and len(profiles) > 1:
        scope = _prompt(
            "Apply to all profiles (root) or only this profile? (all/this)",
            default="all",
        ).strip().lower()
        if scope in {"this", "t", "host", "only"}:
            write_root = False
            print(f"  This forks [{host}] from the shared root map — future root")
            print("  edits no longer reach this profile.")

    if write_root:
        if working:
            cfg["userPeerAliases"] = working
        else:
            cfg.pop("userPeerAliases", None)
        target_desc = "root config (shared by all profiles)"
        hermes_host = _host_block(cfg, host)
        active_ws = hermes_host.get("workspace") or cfg.get("workspace") or host
        other_ws: dict[str, list[str]] = {}
        for name, hostk, block in profiles:
            ws = block.get("workspace") or cfg.get("workspace") or hostk
            if ws != active_ws:
                other_ws.setdefault(ws, []).append(name)
        for ws, names in sorted(other_ws.items()):
            print(f"  ⚠ root aliases also apply in workspace '{ws}' (profile")
            print(f"    {', '.join(names)}) — picked peers may not exist there.")
    else:
        block = cfg.setdefault("hosts", {}).setdefault(host, {})
        if working:
            block["userPeerAliases"] = working
        else:
            block.pop("userPeerAliases", None)
        target_desc = f"host block [{host}]"

    _write_config(cfg)
    print(f"\n  userPeerAliases = {working if working else '{}'}")
    print(f"  written to {target_desc} in {_local_config_path()}\n")


def cmd_peers_map(args) -> None:
    """Interactively map gateway accounts to Honcho user peers."""
    cfg = _read_config()
    host = _host_key()
    hermes_host = _host_block(cfg, host)
    (
        pin, aliases, prefix, aliases_from_root, _prefix_from_root,
    ) = _resolve_effective_identity_mapping(cfg, hermes_host)
    peer_name = hermes_host.get("peerName") or cfg.get("peerName") or ""

    if pin:
        print("\n  pinUserPeer is on: every gateway account resolves to peer")
        print(f"  '{peer_name or '(peerName not set)'}' and aliases have no effect.")
        print("  Turn the pin off with 'hermes honcho setup' to use per-account peers.")
        cont = _prompt("Edit aliases anyway? (y/N)", default="n").strip().lower()
        if cont not in {"y", "yes"}:
            print("  Nothing changed.\n")
            return

    accounts = _seen_gateway_accounts(_state_db_path())
    working = dict(aliases) if isinstance(aliases, dict) else {}

    client, client_cfg = _peers_map_client()
    workspace = (
        getattr(client_cfg, "workspace_id", None)
        or hermes_host.get("workspace") or cfg.get("workspace") or host
    )
    ws_peers = _api_workspace_peers(client)
    labels = _classify_workspace_peers(
        [p["id"] for p in (ws_peers or [])], cfg, accounts, working, prefix,
    )

    _render_peers_map_view(
        workspace, ws_peers, labels, accounts, cfg,
        pin=pin, working=working, prefix=prefix, peer_name=peer_name,
    )

    print("\n  Map: account number or a runtime ID · pN inspects a peer ·")
    print("  w lists workspaces · blank finishes.")
    changed = False
    while True:
        sel = _prompt("Account (blank to finish)", default="").strip()
        if not sel:
            break
        low = sel.lower()

        if low == "w":
            switched = _workspaces_flow(client, workspace, cfg, host)
            if switched:
                workspace, client, ws_peers = switched
                labels = _classify_workspace_peers(
                    [p["id"] for p in ws_peers], cfg, accounts, working, prefix,
                )
                _render_peers_map_view(
                    workspace, ws_peers, labels, accounts, cfg,
                    pin=pin, working=working, prefix=prefix, peer_name=peer_name,
                )
            continue

        if low.startswith("p") and low[1:].isdigit() and ws_peers:
            n = int(low[1:])
            if 1 <= n <= len(ws_peers):
                pid = ws_peers[n - 1]["id"]
                print(f"\n  {pid} — {labels.get(pid, '')}")
                print(f"  {_api_peer_detail(client, pid)}\n")
            continue

        if sel.isdigit() and 1 <= int(sel) <= len(accounts):
            acct = accounts[int(sel) - 1]
            rid = acct["user_id"]
            label = f"{acct['platform']} {rid}" + (
                f" ({acct['label']})" if acct["label"] else ""
            )
            prev_resolved = _resolution_base(_preview_peer_resolution(
                rid, pin=pin, aliases=working, prefix=prefix,
                peer_name=peer_name, user_id_alt=acct["user_id_alt"],
            ))
        else:
            rid = sel
            label = sel
            prev_resolved = _resolution_base(_preview_peer_resolution(
                rid, pin=pin, aliases=working, prefix=prefix, peer_name=peer_name,
            ))

        current = working.get(rid, "")
        hint = " (pN from the peers table, a name, '-' clears)" if ws_peers else ""
        entered = _prompt(f"Peer for {label}{hint}", default=current).strip()
        if entered == "-":
            if rid in working:
                del working[rid]
                changed = True
                print(f"    cleared: {rid}")
            continue
        if entered.lower().startswith("p") and entered[1:].isdigit() and ws_peers:
            n = int(entered[1:])
            if 1 <= n <= len(ws_peers):
                entered = ws_peers[n - 1]["id"]
        if entered and entered != current:
            working[rid] = entered
            changed = True
            peer_id_set = {p["id"] for p in (ws_peers or [])}
            print(f"    {rid} → {entered} — future messages resolve to '{entered}'")
            if ws_peers is not None and _sanitize_peer_id(entered) not in peer_id_set:
                print(f"    '{entered}' is a new peer — created on first message.")
            if prev_resolved in peer_id_set and prev_resolved != _sanitize_peer_id(entered):
                print(f"    peer '{prev_resolved}' keeps its existing history.")

    if not changed:
        print("  Nothing changed.\n")
        return
    _save_alias_map(cfg, host, working, aliases_from_root)


def cmd_peers(args) -> None:
    """Show peer identities across all profiles."""
    if getattr(args, "peers_action", None) == "map":
        cmd_peers_map(args)
        return

    rows = _all_profile_host_configs()
    cfg = _read_config()
    print(f"\nHoncho peer identities ({len(rows)} profiles)\n{'─' * 50}\n"
          f"  {'Profile':<14} {'User peer':<16} {'AI peer'}\n  {'─' * 14} {'─' * 16} {'─' * 18}")
    for name, host, block in rows:
        user = _pref(block, cfg, "peerName") or "(not set)"
        print(f"  {name:<14} {user:<16} {_pref(block, cfg, 'aiPeer') or host}")
    print()


# ── sessions / map ─────────────────────────────────────────────────────────

def cmd_sessions(args) -> None:
    """List known directory → session name mappings."""
    sessions = _read_config().get("sessions", {})
    if not sessions:
        return print(f"  No session mappings configured.\n\n  Add one with: hermes honcho map <session-name>\n"
                     f"  Or edit {_config_path()} directly.\n")
    cwd = os.getcwd()
    print(f"\nHoncho session mappings ({len(sessions)})\n" + RULE)
    for path, name in sorted(sessions.items()):
        print(f"  {name:<30} {path}{' ←' if path == cwd else ''}")
    print()


def cmd_map(args) -> None:
    """Map current directory to a Honcho session name."""
    if not args.session_name:
        return cmd_sessions(args)
    session_name = args.session_name.strip()
    if not session_name:
        return print("  Session name cannot be empty.\n")
    import re
    if (sanitized := re.sub(r'[^a-zA-Z0-9_-]', '-', session_name).strip('-')) != session_name:
        print(f"  Session name sanitized to: {sanitized}")
        session_name = sanitized
    cwd = os.getcwd()
    cfg = _read_config()
    cfg.setdefault("sessions", {})[cwd] = session_name
    _write_config(cfg)
    print(f"  Mapped {cwd}\n     → {session_name}\n")


# ── peer / mode / strategy / tokens ────────────────────────────────────────

def _show_or_set_fields(args, fields: tuple, show) -> None:
    """Shared body of ``peer`` / ``tokens``: with no flags, ``show(block, cfg)`` prints the
    current values; otherwise each ``(attr, key, echo, valid)`` flag that is set is written
    (an invalid value aborts before saving)."""
    cfg = _read_config()
    values = {attr: getattr(args, attr, None) for attr, _key, _echo, _valid in fields}
    if all(v is None for v in values.values()):
        return show(_active_block(cfg), cfg)
    for attr, key, echo, valid in fields:
        if (value := values[attr]) is None:
            continue
        if valid is not None and value not in valid:  # only --reasoning carries a choice set
            return print(f"  Invalid reasoning level '{value}'. Options: {', '.join(valid)}")
        value = value.strip() if isinstance(value, str) else value
        _set_field(cfg, key, value, echo.format(value))
    _save(cfg)


def cmd_peer(args) -> None:
    """Show or update peer names and dialectic reasoning level."""
    def show(hermes, cfg):
        print(f"""
Honcho peers
{RULE}
  User peer:   {_pref(hermes, cfg, 'peerName') or '(not set)'}
    Your identity in Honcho. Messages you send build this peer's card.
  AI peer:     {_pref(hermes, cfg, 'aiPeer') or _host_key()}
    Hermes' identity in Honcho. Seed with 'hermes honcho identity <file>'.
    Dialectic calls ask this peer questions to warm session context.

  Dialectic reasoning:  {_pref(hermes, cfg, 'dialecticReasoningLevel') or 'low'}  ({', '.join(REASONING_LEVELS)})
  Dialectic cap:        {_pref(hermes, cfg, 'dialecticMaxChars') or 600} chars
""")
    _show_or_set_fields(args, (("user", "peerName", "User peer -> {}", None), ("ai", "aiPeer", "AI peer   -> {}", None),
                               ("reasoning", "dialecticReasoningLevel", "Dialectic reasoning level -> {}", REASONING_LEVELS)), show)


def _show_or_set_choice(args, *, attr: str, key: str, noun: str, title: str, choices: dict,
                        default: str, width: int) -> None:
    """Shared body of ``mode`` / ``strategy``: list choices, or set one on the active host."""
    cfg = _read_config()
    value = getattr(args, attr, None)
    if value is None:
        current = _pref(_active_block(cfg), cfg, key) or default
        print(f"\nHoncho {title}\n" + RULE)
        print("\n".join(f"  {m:<{width}}  {desc}{' <-' if m == current else ''}" for m, desc in choices.items()))
        return print(f"\n  Set with: hermes honcho {attr} [{'|'.join(choices)}]\n")
    if value not in choices:
        return print(f"  Invalid {noun} '{value}'. Options: {', '.join(choices)}\n")
    host = _host_key()
    cfg.setdefault("hosts", {}).setdefault(host, {})[key] = value
    _write_config(cfg)
    print(f"  {_label(host)}{title[0].upper() + title[1:]} -> {value}  ({choices[value]})\n")


def cmd_mode(args) -> None:
    """Show or set the recall mode."""
    _show_or_set_choice(args, attr="mode", key="recallMode", noun="mode", title="recall mode",
                        choices=_MODES, default="hybrid", width=10)


def cmd_strategy(args) -> None:
    """Show or set the session strategy."""
    _show_or_set_choice(args, attr="strategy", key="sessionStrategy", noun="strategy",
                        title="session strategy", choices=_STRATEGIES, default="per-session", width=15)


def cmd_tokens(args) -> None:
    """Show or set token budget settings."""
    def show(hermes, cfg):
        print(f"""
Honcho budgets
{RULE}

  Context     {_pref(hermes, cfg, 'contextTokens') or '(Honcho default)'} tokens
    Raw memory retrieval. Honcho returns stored facts/history about
    the user and session, injected directly into the system prompt.

  Dialectic   {_pref(hermes, cfg, 'dialecticMaxChars') or 600} chars, reasoning: {_pref(hermes, cfg, 'dialecticReasoningLevel') or 'low'}
    AI-to-AI inference. Hermes asks Honcho's AI peer a question
    (e.g. "what were we working on?") and Honcho runs its own model
    to synthesize an answer. Used for first-turn session continuity.
    Level controls how much reasoning Honcho spends on the answer.

  Set with: hermes honcho tokens [--context N] [--dialectic N]
""")
    _show_or_set_fields(args, (("context", "contextTokens", "context tokens -> {}", None),
                               ("dialectic", "dialecticMaxChars", "dialectic cap  -> {} chars", None)), show)


# ── identity / migrate ─────────────────────────────────────────────────────

def cmd_identity(args) -> None:
    """Seed AI peer identity or show both peer representations."""
    cfg = _read_config()
    if not _resolve_api_key(cfg):
        return print("  No API key configured. Run 'hermes honcho setup' first.\n")
    file_path = getattr(args, "file", None)
    try:
        hcfg, client = _connect(_host_key())
        mgr, session_key = _session_manager(hcfg, client)
    except Exception as e:
        return print(f"  Honcho connection failed: {e}\n")

    if getattr(args, "show", False):
        from plugins.memory.honcho.session import HonchoAuthError
        try:
            user_card = mgr.get_peer_card(session_key)
            ai_rep = mgr.get_ai_representation(session_key)
        except HonchoAuthError as e:
            return print(f"  Honcho authentication failed: {e}\n")
        print(f"\nUser peer ({hcfg.peer_name or 'not set'})\n" + RULE)
        print("\n".join(f"  {fact}" for fact in user_card) if user_card
              else "  No user peer card yet. Send a few messages to build one.")
        print(f"\nAI peer ({hcfg.ai_peer})\n" + RULE)
        print(ai_rep.get("representation") or ai_rep.get("card")
              or "  No representation built yet.\n  Run 'hermes honcho identity <file>' to seed one.")
        print()
        return

    if not file_path:
        print(f"""
Honcho identity management
{RULE}
  User peer: {hcfg.peer_name or 'not set'}
  AI peer:   {hcfg.ai_peer}

    hermes honcho identity --show        — show both peer representations
    hermes honcho identity <file>        — seed AI peer from SOUL.md or any .md/.txt
""")
        return

    p = Path(file_path).expanduser()
    if not p.exists():
        return print(f"  File not found: {p}\n")
    content = p.read_text(encoding="utf-8").strip()
    if not content:
        return print(f"  File is empty: {p}\n")
    if mgr.seed_ai_identity(session_key, content, source=p.name):
        print(f"  Seeded AI peer identity from {p.name} into session '{session_key}'\n"
              f"  Honcho will incorporate this into {hcfg.ai_peer}'s representation over time.\n")
    else:
        print("  Failed to seed identity. Check logs for details.\n")


def _find_memory_files(names: list[str]) -> list[Path]:
    """Existing files named ``names`` in cwd then ~/.openclaw, deduplicated."""
    candidates = (d / name for name in names for d in (Path(os.getcwd()), Path.home() / ".openclaw"))
    return list(dict.fromkeys(p for p in candidates if p.exists()))


def _migrate_upload(mgr, session_key: str, user_files: list[Path]) -> None:
    dirs_with_files = set(str(f.parent) for f in user_files)
    # List (not generator) so every directory is attempted even after one succeeds.
    if any([mgr.migrate_memory_files(session_key, d) for d in dirs_with_files]):
        print(f"  Uploaded user memory files from: {', '.join(dirs_with_files)}")
    else:
        print("  Nothing uploaded (files may already be migrated or empty).")


def _migrate_seed(mgr, session_key: str, agent_files: list[Path]) -> None:
    for f in agent_files:
        content = f.read_text(encoding="utf-8").strip()
        if content:
            ok = mgr.seed_ai_identity(session_key, content, source=f.name)
            print(f"    {f.name}: {'seeded' if ok else 'failed'}")


def _offer(question: str, action, files: list[Path]) -> None:
    """Ask, then run ``action(mgr, session_key, files)`` against a fresh client."""
    if not _yes(_prompt(question, default="y")):
        return
    try:
        action(*_session_manager(*_connect(None, reset=True)), files)
    except Exception as e:
        print(f"  Failed: {e}")


def cmd_migrate(args) -> None:
    """Step-by-step migration guide: OpenClaw native memory → Hermes + Honcho."""
    user_files = _find_memory_files(["USER.md", "MEMORY.md"])  # facts about the user
    agent_files = _find_memory_files(["SOUL.md", "IDENTITY.md", "AGENTS.md", "TOOLS.md", "BOOTSTRAP.md"])
    cfg = _read_config()
    has_key = bool(_resolve_api_key(cfg))

    print("\nHoncho migration: OpenClaw native memory → Hermes\n" + "─" * 50)
    print("""
  OpenClaw's native memory stores context in local markdown files
  (USER.md, MEMORY.md, SOUL.md, ...) and injects them via QMD search.
  Honcho replaces that with a cloud-backed, LLM-observable memory layer:
  context is retrieved semantically, injected automatically each turn,
  and enriched by a dialectic reasoning layer that builds over time.

Step 1  Create a Honcho account
""")
    if has_key:
        print(f"  Honcho API key already configured: {_mask(cfg['apiKey'])}\n  Skip to Step 2.")
    else:
        print("""  Honcho is a cloud memory service that gives Hermes persistent memory
  across sessions. You need an API key to use it.

  1. Get your API key at https://app.honcho.dev
  2. Run:  hermes honcho setup
     Paste the key when prompted.
""")
        if _yes(_prompt("  Run 'hermes honcho setup' now?", default="y")):
            cmd_setup(args)
            cfg = _read_config()
            has_key = bool(cfg.get("apiKey", ""))
        else:
            print("\n  Run 'hermes honcho setup' when ready, then re-run this walkthrough.")

    print("\nStep 2  Detected OpenClaw memory files\n")
    if user_files or agent_files:
        for files, label in ((user_files, "User memory"), (agent_files, "Agent identity")):
            if files:
                peer = "user" if files is user_files else "AI"
                print(f"  {label} ({len(files)} file(s)) — will go to Honcho {peer} peer:")
                print("\n".join(f"    {f}" for f in files))
    else:
        print("  No OpenClaw native memory files found in cwd or ~/.openclaw/.\n"
              "  If your files are elsewhere, copy them here before continuing,\n"
              "  or seed them manually:  hermes honcho identity <path/to/file>")

    print("""
Step 3  Migrate user memory files → Honcho user peer

  USER.md and MEMORY.md contain facts about you that the agent should
  remember across sessions. Honcho will store these under your user peer
  and inject relevant excerpts into the system prompt automatically.
""")
    if user_files:
        print(f"  Found: {', '.join(f.name for f in user_files)}")
        print("""
  These are picked up automatically the first time you run 'hermes'
  with Honcho configured and no prior session history.
  (Hermes calls migrate_memory_files() on first session init.)

  If you want to migrate them now without starting a session:""")
        print("    hermes honcho migrate  — this step handles it interactively\n" * len(user_files), end="")
        if has_key:
            _offer("  Upload user memory files to Honcho now?", _migrate_upload, user_files)
        else:
            print("  Run 'hermes honcho setup' first, then re-run this step.")
    else:
        print("  No user memory files detected. Nothing to migrate here.")

    print("""
Step 4  Seed AI identity files → Honcho AI peer

  SOUL.md, IDENTITY.md, AGENTS.md, TOOLS.md, BOOTSTRAP.md define the
  agent's character, capabilities, and behavioral rules. In OpenClaw
  these are injected via file search at prompt-build time.

  In Hermes, they are seeded once into Honcho's AI peer through the
  observation pipeline. Honcho builds a representation from them and
  from every subsequent assistant message (observe_me=True). Over time
  the representation reflects actual behavior, not just declaration.
""")
    if agent_files:
        print(f"  Found: {', '.join(f.name for f in agent_files)}")
        print()
        if has_key:
            _offer("  Seed AI identity from all detected files now?", _migrate_seed, agent_files)
        else:
            print("  Run 'hermes honcho setup' first, then seed manually:")
            print("\n".join(f"    hermes honcho identity {f}" for f in agent_files))
    else:
        print("  No agent identity files detected.\n  To seed manually:  hermes honcho identity <path/to/SOUL.md>")

    print("""
Step 5  What changes vs. OpenClaw native memory

  Storage
    OpenClaw: markdown files on disk, searched via QMD at prompt-build time.
    Hermes:   cloud-backed Honcho peers. Files can stay on disk as source
              of truth; Honcho holds the live representation.

  Context injection
    OpenClaw: file excerpts injected synchronously before each LLM call.
    Hermes:   Honcho context fetched async at turn end, injected next turn.
              First turn has no Honcho context; subsequent turns are loaded.

  Memory growth
    OpenClaw: you edit files manually to update memory.
    Hermes:   Honcho observes every message and updates representations
              automatically. Files become the seed, not the live store.

  Honcho tools (available to the agent during conversation)
    honcho_context   — session context: summary, representation, card, messages
    honcho_search        — semantic search over stored context
    honcho_profile       — fast peer card snapshot
    honcho_reasoning     — ask Honcho a question, synthesized answer
    honcho_conclude      — write a conclusion/fact back to memory

  Session naming
    OpenClaw: no persistent session concept — files are global.
    Hermes:   per-session by default — each run gets its own session
              Map a custom name:  hermes honcho map <session-name>

Step 6  Next steps
""")
    if not has_key:
        print("  1. hermes honcho setup              — configure API key (required)\n"
              "  2. hermes honcho migrate            — re-run this walkthrough")
    else:
        print("""  1. hermes honcho status             — verify Honcho connection
  2. hermes                           — start a session
     (user memory files auto-uploaded on first turn if not done above)
  3. hermes honcho identity --show    — verify AI peer representation
  4. hermes honcho tokens             — tune context and dialectic budgets
  5. hermes honcho mode               — view or change memory mode""")
    print()


# ── dispatch / argparse ────────────────────────────────────────────────────

# (subcommand, help, handler, ((arg, kwargs), ...)); order defines --help order.
_SUBCOMMANDS = (
    ("setup", "Initial Honcho setup (redirects to hermes memory setup)", None, ()),
    ("status", "Show current Honcho config and connection status", cmd_status, (
        ("--all", dict(action="store_true", help="Show config overview across all profiles")),
    )),
    ("peers", "Show peer identities across all profiles ('peers map' to map gateway accounts)", cmd_peers, (
        ("peers_action", dict(nargs="?", default=None, choices=("map",), metavar="map",
                              help="'map': interactively map gateway accounts to user peers")),
    )),
    ("sessions", "List known Honcho session mappings", cmd_sessions, ()),
    ("map", "Map current directory to a Honcho session name (no arg = list mappings)", cmd_map, (
        ("session_name", dict(nargs="?", default=None,
                              help="Session name to associate with this directory. Omit to list current mappings.")),
    )),
    ("peer", "Show or update peer names and dialectic reasoning level", cmd_peer, (
        ("--user", dict(metavar="NAME", help="Set user peer name")),
        ("--ai", dict(metavar="NAME", help="Set AI peer name")),
        ("--reasoning", dict(metavar="LEVEL", choices=REASONING_LEVELS,
                             help="Set default dialectic reasoning level (minimal/low/medium/high/max)")),
    )),
    ("mode", "Show or set recall mode (hybrid/context/tools)", cmd_mode, (
        ("mode", dict(nargs="?", metavar="MODE", choices=tuple(_MODES),
                      help="Recall mode to set (hybrid/context/tools). Omit to show current.")),
    )),
    ("strategy", "Show or set session strategy (per-session/per-directory/per-repo/global)", cmd_strategy, (
        ("strategy", dict(nargs="?", metavar="STRATEGY", choices=tuple(_STRATEGIES),
                          help="Session strategy to set. Omit to show current.")),
    )),
    ("tokens", "Show or set token budget for context and dialectic", cmd_tokens, (
        ("--context", dict(type=int, metavar="N", help="Max tokens Honcho returns from session.context() per turn")),
        ("--dialectic", dict(type=int, metavar="N", help="Max chars of dialectic result to inject into system prompt")),
    )),
    ("identity", "Seed or show the AI peer's Honcho identity representation", cmd_identity, (
        ("file", dict(nargs="?", default=None, help="Path to file to seed from (e.g. SOUL.md). Omit to show usage.")),
        ("--show", dict(action="store_true", help="Show current AI peer representation from Honcho")),
    )),
    ("migrate", "Step-by-step migration guide from openclaw-honcho to Hermes Honcho", cmd_migrate, ()),
    ("enable", "Enable Honcho for the active profile", cmd_enable, ()),
    ("disable", "Disable Honcho for the active profile", cmd_disable, ()),
    ("sync", "Sync Honcho config to all existing profiles", cmd_sync, ()),
)
_HANDLERS = {name: handler for name, _help, handler, _args in _SUBCOMMANDS if handler}


def honcho_command(args) -> None:
    """Route honcho subcommands."""
    global _profile_override
    _profile_override = getattr(args, "target_profile", None)
    sub = getattr(args, "honcho_command", None)
    if sub == "setup":  # honcho setup goes through the unified memory-provider path
        print("\n  Honcho is configured via the memory provider system.\n  Running 'hermes memory setup'...\n")
        from hermes_cli.memory_setup import cmd_setup_provider
        return cmd_setup_provider("honcho")
    handler = cmd_status if sub is None else _HANDLERS.get(sub)
    if handler is None:
        return print(f"  Unknown honcho command: {sub}\n"
                     "  Available: status, sessions, map, peer, mode, strategy, tokens, identity, migrate, enable, disable, sync\n")
    handler(args)


def register_cli(subparser) -> None:
    """Build the ``hermes honcho`` argparse subcommand tree on the ``hermes honcho`` parser."""
    subparser.add_argument("--target-profile", metavar="NAME", dest="target_profile",
                           help="Target a specific profile's Honcho config without switching")
    subs = subparser.add_subparsers(dest="honcho_command")
    for name, help_text, _handler, arguments in _SUBCOMMANDS:
        parser = subs.add_parser(name, help=help_text)
        for flag, kwargs in arguments:
            parser.add_argument(flag, **kwargs)
    subparser.set_defaults(func=honcho_command)
