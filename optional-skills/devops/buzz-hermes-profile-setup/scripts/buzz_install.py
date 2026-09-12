#!/usr/bin/env python3
"""buzz_install.py — connect a Hermes profile to a Buzz (Nostr) community.

Standard library only. bech32 (NIP-19) and BIP-340 Schnorr signing are implemented
inline, mirroring the dependency-free signer in Hermes' bundled Buzz adapter
(``plugins/platforms/buzz/nostr_auth.py``), so no ``pip install`` is needed.

Pipeline (each step is logged):
  1. Resolve inputs: BUZZ_INSTALL_* env vars -> interactive prompts -> defaults.
  2. Locate the ``buzz`` CLI (configured path, PATH, ~/bin) or build it from source.
  3. Reuse the profile's existing BUZZ_PRIVATE_KEY, or mint a fresh keypair
     (bech32 round-trip self-verified so a padding bug cannot slip through).
  4. If the key is not yet a relay member: owner mints an invite (NIP-98),
     accept the join policy when the community has one, agent claims (NIP-98).
  5. Set the agent's Buzz profile: name / avatar / about / NIP-05, presence, status.
  6. Write BUZZ_PRIVATE_KEY to the profile's .env (0600); never into config.yaml.
  7. ``hermes -p <profile> config set`` every Buzz key.
  8. Restart the profile's gateway and confirm ``gateway_state.json`` says connected.

``--dry-run`` resolves inputs and prints the plan without touching the network,
the filesystem, or the gateway.

Relay HTTP goes through ``curl``: the relay sits behind Cloudflare, which rejects
Python's ``urllib`` User-Agent (error 1010).
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import pathlib
import secrets
import shutil
import subprocess
import sys
import time
from typing import Optional

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
BUZZ_REPO = "https://github.com/block/buzz.git"
HOME = pathlib.Path.home()
PROFILES = HOME / ".hermes" / "profiles"
DEFAULT_CLI = HOME / "bin" / "buzz"
BUILD_DIR = HOME / ".buzz-build"
PRESENCE_VALUES = ("online", "away", "offline")
TRUE_WORDS = {"1", "true", "yes", "y", "on"}
FALSE_WORDS = {"0", "false", "no", "n", "off", ""}


def log(msg: str) -> None:
    print(msg, flush=True)


def warn(msg: str) -> None:
    print(f"WARN: {msg}", file=sys.stderr, flush=True)


def die(msg: str) -> None:
    print(f"\nFATAL: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def to_bool(value: str, *, name: str) -> bool:
    v = (value or "").strip().lower()
    if v in TRUE_WORDS:
        return True
    if v in FALSE_WORDS:
        return False
    die(f"{name} must be true/false, got {value!r}")
    return False  # unreachable; keeps type-checkers happy


# ---------------------------------------------------------------- bech32 (NIP-19)
CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
_GENERATORS = (0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3)


def _polymod(values: list[int]) -> int:
    chk = 1
    for v in values:
        top = chk >> 25
        chk = ((chk & 0x1FFFFFF) << 5) ^ v
        for i, g in enumerate(_GENERATORS):
            if (top >> i) & 1:
                chk ^= g
    return chk


def _hrp_expand(hrp: str) -> list[int]:
    return [ord(c) >> 5 for c in hrp] + [0] + [ord(c) & 31 for c in hrp]


def _convertbits(data, from_bits: int, to_bits: int, pad: bool) -> list[int]:
    acc = bits = 0
    out: list[int] = []
    maxv = (1 << to_bits) - 1
    for v in data:
        acc = (acc << from_bits) | v
        bits += from_bits
        while bits >= to_bits:
            bits -= to_bits
            out.append((acc >> bits) & maxv)
    if pad and bits:
        # Without this final padded group the last bit of the key is silently dropped
        # and the nsec decodes to a *different* private key.
        out.append((acc << (to_bits - bits)) & maxv)
    return out


def bech32_encode(hrp: str, data: bytes) -> str:
    vals = _convertbits(list(data), 8, 5, pad=True)
    pm = _polymod(_hrp_expand(hrp) + vals + [0] * 6) ^ 1
    vals += [(pm >> 5 * (5 - i)) & 31 for i in range(6)]
    return hrp + "1" + "".join(CHARSET[d] for d in vals)


def bech32_decode(expected_hrp: str, text: str) -> bytes:
    if text.lower() != text and text.upper() != text:
        raise ValueError("bech32 string mixes upper- and lowercase")
    text = text.lower()
    pos = text.rfind("1")
    if pos < 1 or pos + 7 > len(text):
        raise ValueError("invalid bech32 string")
    hrp, payload = text[:pos], text[pos + 1 :]
    if hrp != expected_hrp:
        raise ValueError(f"expected {expected_hrp}1..., got {hrp}1...")
    try:
        vals = [CHARSET.index(c) for c in payload]
    except ValueError as exc:
        raise ValueError("invalid character in bech32 string") from exc
    if _polymod(_hrp_expand(hrp) + vals) != 1:
        raise ValueError("bech32 checksum mismatch")
    out = bytes(_convertbits(vals[:-6], 5, 8, pad=False))
    if len(out) != 32:
        raise ValueError(f"{expected_hrp} must encode exactly 32 bytes")
    return out


# ---------------------------------------------------------------- secp256k1 / BIP-340
_P = 2**256 - 2**32 - 977
_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
_G = (
    0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
    0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8,
)
Point = Optional[tuple[int, int]]


def _point_add(a: Point, b: Point) -> Point:
    if a is None or b is None:
        return b if a is None else a
    x1, y1 = a
    x2, y2 = b
    if x1 == x2:
        if (y1 + y2) % _P == 0:
            return None
        lam = (3 * x1 * x1) * pow(2 * y1, _P - 2, _P)
    else:
        lam = (y2 - y1) * pow(x2 - x1, _P - 2, _P)
    lam %= _P
    x3 = (lam * lam - x1 - x2) % _P
    return x3, (lam * (x1 - x3) - y1) % _P


def _point_mul(k: int, point: Point = _G) -> Point:
    result: Point = None
    addend = point
    while k:
        if k & 1:
            result = _point_add(result, addend)
        addend = _point_add(addend, addend)
        k >>= 1
    return result


def _tagged_hash(tag: str, payload: bytes) -> bytes:
    th = hashlib.sha256(tag.encode()).digest()
    return hashlib.sha256(th + th + payload).digest()


def decode_private_key(value: str) -> bytes:
    """Accept an ``nsec1...`` or 64-hex private key and return 32 raw bytes."""
    raw = value.strip()
    if raw.lower().startswith("nsec1"):
        key = bech32_decode("nsec", raw)
    else:
        try:
            key = bytes.fromhex(raw)
        except ValueError as exc:
            raise ValueError("private key must be nsec or 64 hex chars") from exc
        if len(key) != 32:
            raise ValueError("private key must be 32 bytes")
    k = int.from_bytes(key, "big")
    if not 1 <= k < _N:
        raise ValueError("private key outside the secp256k1 range")
    return key


def pubkey_x(secret: bytes) -> bytes:
    pt = _point_mul(int.from_bytes(secret, "big"))
    assert pt is not None
    return pt[0].to_bytes(32, "big")


def schnorr_sign(msg32: bytes, secret: bytes, aux: Optional[bytes] = None) -> bytes:
    if len(msg32) != 32:
        raise ValueError("BIP-340 signs a 32-byte message")
    d = int.from_bytes(secret, "big")
    pt = _point_mul(d)
    assert pt is not None
    px = pt[0].to_bytes(32, "big")
    d = d if pt[1] % 2 == 0 else _N - d
    aux = aux if aux is not None else secrets.token_bytes(32)
    t = (d ^ int.from_bytes(_tagged_hash("BIP0340/aux", aux), "big")).to_bytes(32, "big")
    k = int.from_bytes(_tagged_hash("BIP0340/nonce", t + px + msg32), "big") % _N
    if k == 0:
        raise RuntimeError("BIP-340 produced a zero nonce")
    rpt = _point_mul(k)
    assert rpt is not None
    rx = rpt[0].to_bytes(32, "big")
    k = k if rpt[1] % 2 == 0 else _N - k
    e = int.from_bytes(_tagged_hash("BIP0340/challenge", rx + px + msg32), "big") % _N
    return rx + ((k + e * d) % _N).to_bytes(32, "big")


def generate_keypair() -> tuple[str, str, str]:
    """Return (nsec, npub, pubkey_hex) for a brand-new random identity."""
    sec = secrets.token_bytes(32)
    while not 1 <= int.from_bytes(sec, "big") < _N:  # pragma: no cover - astronomically rare
        sec = secrets.token_bytes(32)
    x = pubkey_x(sec)
    nsec, npub = bech32_encode("nsec", sec), bech32_encode("npub", x)
    if bech32_decode("nsec", nsec) != sec or bech32_decode("npub", npub) != x:
        die("keypair self-check failed (bech32 round-trip)")
    return nsec, npub, x.hex()


# ---------------------------------------------------------------- NIP-98 (kind 27235)
def sign_nip98(secret: bytes, url: str, body: bytes, method: str = "POST") -> str:
    """Return an ``Authorization: Nostr <b64>`` header value for one HTTP request."""
    pub_hex = pubkey_x(secret).hex()
    created_at = int(time.time())
    tags = [["u", url], ["method", method], ["payload", hashlib.sha256(body).hexdigest()]]
    serialized = json.dumps([0, pub_hex, created_at, 27235, tags, ""], separators=(",", ":"))
    event_id = hashlib.sha256(serialized.encode()).digest()
    event = {
        "id": event_id.hex(),
        "pubkey": pub_hex,
        "created_at": created_at,
        "kind": 27235,
        "tags": tags,
        "content": "",
        "sig": schnorr_sign(event_id, secret).hex(),
    }
    import base64

    return "Nostr " + base64.b64encode(json.dumps(event, separators=(",", ":")).encode()).decode()


# ---------------------------------------------------------------- process helpers
def curl(url: str, body: Optional[dict] = None, auth: Optional[str] = None) -> tuple[int, str, str]:
    if not shutil.which("curl"):
        die("curl not found on PATH (needed because the relay rejects urllib behind Cloudflare)")
    cmd = ["curl", "-sS", "-m", "30", "-H", f"User-Agent: {UA}"]
    if body is not None:
        cmd += ["-X", "POST", "-H", "Content-Type: application/json", "-d", json.dumps(body, separators=(",", ":"))]
    if auth:
        cmd += ["-H", f"Authorization: {auth}"]
    cmd.append(url)
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def sh(cmd: list[str], env=None, cwd=None, check: bool = True) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd)
    out = (p.stdout or "").strip()
    if check and p.returncode != 0:
        die(f"command failed ({p.returncode}): {' '.join(cmd)}\n{(p.stderr or out).strip()}")
    return p.returncode, out


def parse_json(text: str, what: str) -> dict:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        die(f"{what}: relay returned non-JSON: {text[:200]!r}")
    if isinstance(data, dict) and data.get("error"):
        die(f"{what} failed: {data.get('error')}: {data.get('message', '')}".rstrip(": "))
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------- input resolution
def _env(key: str) -> str:
    return os.environ.get(key, "").strip()


def resolve_inputs(interactive: Optional[bool] = None) -> dict:
    tty = sys.stdin.isatty() if interactive is None else interactive

    def ask(key: str, prompt: str, *, required: bool = False, default: str = "", secret: bool = False) -> str:
        v = _env(key)
        if v:
            return v
        if not tty:
            if required:
                die(f"missing required env var {key} (set it, or run from a terminal)")
            return default
        p = f"{prompt} [{default}]: " if default else f"{prompt}: "
        raw = (getpass.getpass(p) if secret else input(p)).strip()
        return raw or default

    cfg = {
        "profile": ask("BUZZ_INSTALL_PROFILE", "Hermes profile name", required=True),
        "relay": ask("BUZZ_INSTALL_RELAY", "Community relay URL (https://<c>.communities.buzz.xyz)", required=True),
        "owner_key": ask(
            "BUZZ_INSTALL_OWNER_NSEC",
            "Owner nsec/hex (mints the invite; used once, never saved; blank if already a member)",
            secret=True,
        ),
        "agent_name": ask("BUZZ_INSTALL_AGENT_NAME", "Agent display name"),
        "avatar": ask("BUZZ_INSTALL_AGENT_AVATAR", "Agent avatar URL (hosted image)"),
        "about": ask("BUZZ_INSTALL_AGENT_ABOUT", "Agent bio / about"),
        "nip05": ask("BUZZ_INSTALL_AGENT_NIP05", "NIP-05 id (optional)"),
        "presence": ask("BUZZ_INSTALL_PRESENCE", "Presence (online/away/offline)", default="online"),
        "status_text": ask("BUZZ_INSTALL_STATUS_TEXT", "Status line (optional)"),
        "status_emoji": ask("BUZZ_INSTALL_STATUS_EMOJI", "Status emoji (optional)"),
        "allow_all": ask("BUZZ_INSTALL_ALLOW_ALL", "Allow all community members to chat? (true/false)", default="true"),
        "allowed_users": ask("BUZZ_INSTALL_ALLOWED_USERS", "Allowed users (comma-sep npub/hex, whitelist mode)"),
        "channels": ask("BUZZ_INSTALL_CHANNELS", "Channels to watch (comma-sep UUIDs; empty = all)"),
        "home_channel": ask("BUZZ_INSTALL_HOME_CHANNEL", "Home channel UUID (cron/notify delivery)"),
        "require_mention": ask(
            "BUZZ_INSTALL_REQUIRE_MENTION", "Only reply when @-mentioned in channels? (true/false)", default="true"
        ),
        "poll_interval": ask("BUZZ_INSTALL_POLL_INTERVAL", "Poll interval (seconds)", default="4"),
        "cli_path": ask("BUZZ_INSTALL_CLI_PATH", "buzz CLI path", default=str(DEFAULT_CLI)),
        "soul": ask("BUZZ_INSTALL_SOUL", "Personality: path to a text file -> SOUL.md (optional)"),
        "rotate_key": ask("BUZZ_INSTALL_ROTATE_KEY", "Ignore existing key and mint a new identity? (true/false)", default="false"),
    }
    cfg["relay"] = cfg["relay"].rstrip("/")
    if not cfg["relay"].startswith("https://"):
        die("relay URL must start with https://")
    if cfg["presence"] not in PRESENCE_VALUES:
        die(f"presence must be one of {', '.join(PRESENCE_VALUES)}")
    try:
        float(cfg["poll_interval"])
    except ValueError:
        die("poll interval must be a number of seconds")
    cfg["agent_name"] = cfg["agent_name"] or cfg["profile"]
    cfg["allow_all"] = to_bool(cfg["allow_all"], name="BUZZ_INSTALL_ALLOW_ALL")
    cfg["require_mention"] = to_bool(cfg["require_mention"], name="BUZZ_INSTALL_REQUIRE_MENTION")
    cfg["rotate_key"] = to_bool(cfg["rotate_key"], name="BUZZ_INSTALL_ROTATE_KEY")
    if not cfg["allow_all"] and not cfg["allowed_users"]:
        warn("allow_all=false with an empty allowed_users list: nobody will be able to talk to the agent")
    return cfg


# ---------------------------------------------------------------- buzz CLI
def ensure_cli(cli_path: str) -> str:
    """Return a path to an executable ``buzz`` binary, building it if necessary."""
    candidate = pathlib.Path(cli_path).expanduser()
    if candidate.is_file() and os.access(candidate, os.X_OK):
        log(f"[cli] using {candidate}")
        return str(candidate)
    on_path = shutil.which("buzz")
    if on_path:
        log(f"[cli] using {on_path} (found on PATH)")
        return on_path
    for tool in ("git", "cargo"):
        if not shutil.which(tool):
            die(f"{tool} not found; install it or point BUZZ_INSTALL_CLI_PATH at a prebuilt buzz binary")
    log("[cli] building buzz-cli from source (first run only, ~1-2 min)...")
    if not (BUILD_DIR / "Cargo.toml").exists():
        sh(["git", "clone", "--depth", "1", BUZZ_REPO, str(BUILD_DIR)])
    sh(["cargo", "build", "--release", "-p", "buzz-cli"], cwd=str(BUILD_DIR))
    built = BUILD_DIR / "target" / "release" / "buzz"
    if not built.exists():
        die("cargo build finished but target/release/buzz is missing")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(built, candidate)
    candidate.chmod(0o755)
    log(f"[cli] built -> {candidate}")
    return str(candidate)


def buzz_env(cfg: dict, nsec: str) -> dict:
    env = os.environ.copy()
    env["BUZZ_RELAY_URL"] = cfg["relay"]
    env["BUZZ_PRIVATE_KEY"] = nsec
    return env


def is_member(cli: str, cfg: dict, nsec: str) -> bool:
    rc, _ = sh([cli, "channels", "list"], env=buzz_env(cfg, nsec), check=False)
    return rc == 0


# ---------------------------------------------------------------- membership
def mint_and_claim(cfg: dict, agent_secret: bytes) -> None:
    if not cfg["owner_key"]:
        die("this key is not a relay member yet; set BUZZ_INSTALL_OWNER_NSEC so the owner can mint an invite")
    try:
        owner_secret = decode_private_key(cfg["owner_key"])
    except ValueError as exc:
        die(f"owner key: {exc}")
    relay = cfg["relay"]

    url = f"{relay}/api/invites"
    body: dict = {}
    raw = json.dumps(body, separators=(",", ":")).encode()
    rc, out, err = curl(url, body, sign_nip98(owner_secret, url, raw))
    if rc:
        die(f"mint request failed: {err or out}")
    code = parse_json(out, "mint").get("code")
    if not code:
        die(f"mint returned no invite code: {out[:200]}")
    log("[membership] invite minted")

    rc, out, _ = curl(f"{relay}/api/join-policy")
    version = ""
    if rc == 0:
        try:
            version = (json.loads(out).get("policy") or {}).get("version", "")
        except (json.JSONDecodeError, AttributeError):
            version = ""

    claim_body: dict = {"code": code}
    if version:
        rc, out, err = curl(
            f"{relay}/api/invites/accept-policy",
            {"code": code, "policy_version": version, "age_confirmed": True},
        )
        if rc:
            die(f"accept-policy request failed: {err or out}")
        receipt = parse_json(out, "accept-policy").get("receipt")
        if not receipt:
            die(f"accept-policy returned no receipt: {out[:200]}")
        claim_body["policy_receipt"] = receipt
        log(f"[membership] join policy v{version} accepted")

    url = f"{relay}/api/invites/claim"
    raw = json.dumps(claim_body, separators=(",", ":")).encode()
    rc, out, err = curl(url, claim_body, sign_nip98(agent_secret, url, raw))
    if rc:
        die(f"claim request failed: {err or out}")
    result = parse_json(out, "claim")
    log(f"[membership] claimed -> status={result.get('status')} role={result.get('role')}")


# ---------------------------------------------------------------- buzz profile
def set_profile(cli: str, cfg: dict, nsec: str) -> None:
    env = buzz_env(cfg, nsec)
    args = [cli, "users", "set-profile", "--name", cfg["agent_name"]]
    if cfg["avatar"]:
        args += ["--avatar", cfg["avatar"]]
    if cfg["about"]:
        args += ["--about", cfg["about"]]
    if cfg["nip05"]:
        args += ["--nip05", cfg["nip05"]]
    sh(args, env=env)
    sh([cli, "users", "set-presence", "--status", cfg["presence"]], env=env)
    if cfg["status_text"]:
        args = [cli, "users", "set-status", "--text", cfg["status_text"]]
        if cfg["status_emoji"]:
            args += ["--emoji", cfg["status_emoji"]]
        sh(args, env=env)
    log(f"[profile] name={cfg['agent_name']!r} avatar={'yes' if cfg['avatar'] else 'no'} presence={cfg['presence']}")


# ---------------------------------------------------------------- hermes wiring
def profile_dir(profile: str) -> pathlib.Path:
    d = PROFILES / profile
    if not d.is_dir():
        die(f"Hermes profile {profile!r} not found at {d} (create it with `hermes profile create {profile}`)")
    return d


def read_existing_key(profile: str) -> str:
    env_path = PROFILES / profile / ".env"
    if not env_path.exists():
        return ""
    for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("BUZZ_PRIVATE_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def write_env(profile: str, nsec: str) -> pathlib.Path:
    env_path = profile_dir(profile) / ".env"
    lines = env_path.read_text(encoding="utf-8", errors="replace").splitlines() if env_path.exists() else []
    lines = [ln for ln in lines if not ln.startswith("BUZZ_PRIVATE_KEY=")]
    lines.append(f"BUZZ_PRIVATE_KEY={nsec}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        env_path.chmod(0o600)
    except OSError:  # pragma: no cover - non-POSIX filesystems
        pass
    log(f"[env] BUZZ_PRIVATE_KEY -> {env_path}")
    return env_path


def csv_json(csv: str) -> str:
    return json.dumps([x.strip() for x in (csv or "").split(",") if x.strip()])


def config_set(profile: str, key: str, value: str) -> None:
    rc, out = sh(["hermes", "-p", profile, "config", "set", key, value], check=False)
    if rc:
        warn(f"hermes config set {key} failed: {out or rc}")


def write_config(cfg: dict, cli: str) -> None:
    if not shutil.which("hermes"):
        die("hermes not found on PATH")
    p, pre = cfg["profile"], "gateway.platforms.buzz"
    settings = {
        f"{pre}.enabled": "true",
        f"{pre}.extra.relay_url": cfg["relay"],
        f"{pre}.extra.channels": csv_json(cfg["channels"]),
        f"{pre}.extra.home_channel": cfg["home_channel"],
        f"{pre}.extra.poll_interval": cfg["poll_interval"],
        f"{pre}.extra.cli_path": cli,
        f"{pre}.extra.allowed_users": csv_json(cfg["allowed_users"]),
        f"{pre}.extra.require_mention": str(cfg["require_mention"]).lower(),
        f"{pre}.extra.allow_all_users": str(cfg["allow_all"]).lower(),
        # Recommended defaults from the Buzz messaging docs: keep channels clean.
        "display.platforms.buzz.interim_assistant_messages": "false",
        "display.platforms.buzz.tool_progress": "off",
    }
    for key, value in settings.items():
        config_set(p, key, value)
    log(f"[config] {len(settings)} Buzz keys written for profile {p!r}")


def write_soul(cfg: dict) -> None:
    if not cfg["soul"]:
        return
    src = pathlib.Path(cfg["soul"]).expanduser()
    if not src.is_file():
        die(f"soul file not found: {src}")
    dst = profile_dir(cfg["profile"]) / "SOUL.md"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    log(f"[soul] {src} -> {dst}")


def restart_gateway(profile: str) -> None:
    rc, out = sh(["hermes", "-p", profile, "gateway", "restart"], check=False)
    if rc:
        warn(f"gateway restart returned {rc}: {out}")
    else:
        log("[gateway] restarted")


def verify(profile: str, attempts: int = 30, delay: float = 2.0) -> bool:
    state = PROFILES / profile / "gateway_state.json"
    for _ in range(attempts):
        if state.exists():
            try:
                data = json.loads(state.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
            buzz = data.get("platforms", data).get("buzz") if isinstance(data, dict) else None
            if isinstance(buzz, dict) and buzz.get("state") == "connected":
                log("[verify] gateway_state.json: buzz connected")
                return True
        time.sleep(delay)
    warn("buzz is not 'connected' in gateway_state.json yet; it may still be starting - re-check shortly")
    return False


# ---------------------------------------------------------------- main
def print_plan(cfg: dict, existing_key: str) -> None:
    mode = "community (any member)" if cfg["allow_all"] else f"whitelist ({cfg['allowed_users'] or 'EMPTY'})"
    identity = "reuse key from profile .env" if existing_key and not cfg["rotate_key"] else "mint a NEW keypair"
    log("---- PLAN (dry run, nothing changed) ----")
    log(f"profile:     {cfg['profile']}")
    log(f"relay:       {cfg['relay']}")
    log(f"identity:    {identity}")
    log(f"owner key:   {'provided' if cfg['owner_key'] else 'not provided (ok only if already a member)'}")
    log(f"agent name:  {cfg['agent_name']}")
    log(f"avatar:      {cfg['avatar'] or '-'}")
    log(f"access:      {mode}; require_mention={cfg['require_mention']}")
    log(f"channels:    {cfg['channels'] or 'all joined'}; home={cfg['home_channel'] or 'first watched'}")
    log(f"cli path:    {cfg['cli_path']}")
    log(f"soul:        {cfg['soul'] or '-'}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Connect a Hermes profile to a Buzz community.")
    parser.add_argument("--dry-run", action="store_true", help="resolve inputs and print the plan; change nothing")
    parser.add_argument("--non-interactive", action="store_true", help="never prompt; fail on missing required env vars")
    args = parser.parse_args(argv)

    log("== buzz-hermes-profile-setup ==")
    cfg = resolve_inputs(interactive=False if args.non_interactive else None)
    profile_dir(cfg["profile"])
    existing = read_existing_key(cfg["profile"])

    if args.dry_run:
        print_plan(cfg, existing)
        return 0

    cli = ensure_cli(cfg["cli_path"])

    if existing and not cfg["rotate_key"]:
        try:
            secret = decode_private_key(existing)
        except ValueError as exc:
            die(f"existing BUZZ_PRIVATE_KEY in profile .env is invalid ({exc}); set BUZZ_INSTALL_ROTATE_KEY=true")
        nsec, npub = bech32_encode("nsec", secret), bech32_encode("npub", pubkey_x(secret))
        log(f"[identity] reusing existing key {npub}")
    else:
        nsec, npub, _ = generate_keypair()
        secret = decode_private_key(nsec)
        log(f"[identity] new keypair {npub}")

    if is_member(cli, cfg, nsec):
        log("[membership] key is already a relay member; skipping invite")
    else:
        mint_and_claim(cfg, secret)

    set_profile(cli, cfg, nsec)
    write_env(cfg["profile"], nsec)
    write_config(cfg, cli)
    write_soul(cfg)
    restart_gateway(cfg["profile"])
    ok = verify(cfg["profile"])

    log("\n---- SUMMARY ----")
    log(f"profile:  {cfg['profile']}")
    log(f"npub:     {npub}")
    log(f"pubkey:   {pubkey_x(secret).hex()}")
    log(f"name:     {cfg['agent_name']}")
    log(f"relay:    {cfg['relay']}")
    log(f"member:   {'yes' if is_member(cli, cfg, nsec) else 'NO - see references/buzz-internals.md'}")
    log("\nNext: DM the agent from a different account than the owner key and confirm it replies.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
