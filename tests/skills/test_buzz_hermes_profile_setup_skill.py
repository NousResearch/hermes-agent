"""Tests for the buzz-hermes-profile-setup optional skill.

Covers:
  - SKILL.md frontmatter meets the authoring standards (<=60-char description, fields,
    human-first author credit).
  - scripts/buzz_install.py imports with the standard library only.
  - bech32 (NIP-19) and BIP-340 Schnorr against *published* test vectors
    (not sign-then-verify with the same code), plus a byte-exact cross-check
    against Hermes' in-tree ``plugins/platforms/buzz/nostr_auth.py``.
  - NIP-98 header construction and signature validity.
  - Input resolution, .env key handling, config writes, invite flow, gateway verification,
    and --dry-run, all with subprocess/network mocked out.
"""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import re
import stat
import sys
from pathlib import Path
from unittest import mock

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO / "optional-skills" / "devops" / "buzz-hermes-profile-setup"
SCRIPT = SKILL_DIR / "scripts" / "buzz_install.py"

# Published NIP-19 bech32 examples (nsec / npub) from
# https://github.com/nostr-protocol/nips/blob/master/19.md
NSEC_VECTOR = "nsec1vl029mgpspedva04g90vltkh6fvh240zqtv9k0t9af8935ke9laqsnlfe5"
NSEC_HEX = "67dea2ed018072d675f5415ecfaed7d2597555e202d85b3d65ea4e58d2d92ffa"
NPUB_VECTOR = "npub10elfcs4fr0l0r8af98jlmgdh9c8tcxjvz9qkw038js35mp4dma8qzvjptg"
NPUB_HEX = "7e7e9c42a91bfef19fa929e5fda1b72e0ebc1a4c1141673e2794234d86addf4e"

# Official BIP-340 Schnorr vectors 0, 1, 2 from
# https://github.com/bitcoin/bips/blob/master/bip-0340/test-vectors.csv
# (32-byte messages, valid keys, deterministic aux). Compared byte-for-byte
# against the BIP — a sign-then-verify loop on this same module would not
# catch a systematic deviation that relays would reject.
BIP340_VECTORS = [
    {
        "index": 0,
        "sk": bytes.fromhex("00" * 31 + "03"),
        "pk": "F9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9",
        "aux": bytes(32),
        "msg": bytes(32),
        "sig": (
            "E907831F80848D1069A5371B402410364BDF1C5F8307B0084C55F1CE2DCA8215"
            "25F66A4A85EA8B71E482A74F382D2CE5EBEEE8FDB2172F477DF4900D310536C0"
        ),
    },
    {
        "index": 1,
        "sk": bytes.fromhex("B7E151628AED2A6ABF7158809CF4F3C762E7160F38B4DA56A784D9045190CFEF"),
        "pk": "DFF1D77F2A671C5F36183726DB2341BE58FEAE1DA2DECED843240F7B502BA659",
        "aux": bytes.fromhex("00" * 31 + "01"),
        "msg": bytes.fromhex("243F6A8885A308D313198A2E03707344A4093822299F31D0082EFA98EC4E6C89"),
        "sig": (
            "6896BD60EEAE296DB48A229FF71DFE071BDE413E6D43F917DC8DCF8C78DE3341"
            "8906D11AC976ABCCB20B091292BFF4EA897EFCB639EA871CFA95F6DE339E4B0A"
        ),
    },
    {
        "index": 2,
        "sk": bytes.fromhex("C90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B14E5C9"),
        "pk": "DD308AFEC5777E13121FA72B9CC1B7CC0139715309B086C960E18FD969774EB8",
        "aux": bytes.fromhex("C87AA53824B4D7AE2EB035A2B5BBBCCC080E76CDC6D1692C4B0B62D798E6D906"),
        "msg": bytes.fromhex("7E2D58D8B3BCDF1ABADEC7829054F90DDA9805AAB56C77333024B9D0A508B75C"),
        "sig": (
            "5831AAEED7B44BB74E5EAB94BA9D4294C49BCF2A60728D8B4C200F50DD313C1B"
            "AB745879A5AD954A72C45A91C3A51D3C7ADEA98D82F8481E0E1E03674A6F3FB7"
        ),
    },
    {
        "index": 3,
        "sk": bytes.fromhex("0B432B2677937381AEF05BB02A66ECD012773062CF3FA2549E44F58ED2401710"),
        "pk": "25D1DFF95105F5253C4022F628A996AD3A0D95FBF21D468A1B33F8C160D8F517",
        "aux": bytes.fromhex("FF" * 32),
        "msg": bytes.fromhex("FF" * 32),
        "sig": (
            "7EB0509757E246F19449885651611CB965ECC1A187DD51B64FDA1EDC9637D5EC"
            "97582B9CB13DB3933705B32BA982AF5AF25FD78881EBB32771FC5922EFC66EA3"
        ),
    },
]


def _load_nostr_auth():
    """Hermes' production BIP-340 signer — an independent implementation."""
    path = REPO / "plugins" / "platforms" / "buzz" / "nostr_auth.py"
    spec = importlib.util.spec_from_file_location("hermes_nostr_auth", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def skill_source() -> str:
    return (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter(skill_source) -> dict:
    m = re.search(r"^---\n(.*?)\n---", skill_source, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("buzz_install", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def profiles(tmp_path, monkeypatch, mod):
    root = tmp_path / "profiles"
    (root / "agent").mkdir(parents=True)
    monkeypatch.setattr(mod, "PROFILES", root)
    return root


BASE_ENV = {
    "BUZZ_INSTALL_PROFILE": "agent",
    "BUZZ_INSTALL_RELAY": "https://team.communities.buzz.xyz/",
}


def _set_env(monkeypatch, **extra):
    for key in list(BASE_ENV) + [k for k in extra]:
        monkeypatch.delenv(key, raising=False)
    for key, value in {**BASE_ENV, **extra}.items():
        monkeypatch.setenv(key, value)


def _verify_schnorr(mod, msg: bytes, pub_x: bytes, sig: bytes) -> bool:
    """Minimal BIP-340 verifier built on the script's curve helpers."""
    p, n = mod._P, mod._N
    px = int.from_bytes(pub_x, "big")
    y_sq = (pow(px, 3, p) + 7) % p
    y = pow(y_sq, (p + 1) // 4, p)
    if pow(y, 2, p) != y_sq:
        return False
    if y % 2:
        y = p - y
    r = int.from_bytes(sig[:32], "big")
    s = int.from_bytes(sig[32:], "big")
    if r >= p or s >= n:
        return False
    e = int.from_bytes(mod._tagged_hash("BIP0340/challenge", sig[:32] + pub_x + msg), "big") % n
    sg = mod._point_mul(s)
    ep = mod._point_mul((n - e) % n, (px, y))
    rp = mod._point_add(sg, ep)
    return rp is not None and rp[1] % 2 == 0 and rp[0] == r


# ---------------------------------------------------------------- SKILL.md
def test_skill_layout_exists() -> None:
    assert (SKILL_DIR / "SKILL.md").is_file()
    assert SCRIPT.is_file()
    assert (SKILL_DIR / "references" / "buzz-internals.md").is_file()


def test_description_hardline(frontmatter) -> None:
    desc = frontmatter["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars: {desc!r}"
    assert desc.endswith(".")


def test_required_frontmatter_fields(frontmatter) -> None:
    for field in ("name", "description", "version", "author", "license", "platforms"):
        assert field in frontmatter, f"missing frontmatter field: {field}"
    assert frontmatter["name"] == SKILL_DIR.name
    assert frontmatter["metadata"]["hermes"]["tags"]


def test_author_credits_human_first(frontmatter) -> None:
    assert str(frontmatter["author"]).startswith("Marco Rodrigues (dadhalfdev)")


def test_skill_points_at_helper_script(skill_source) -> None:
    assert "scripts/buzz_install.py" in skill_source
    assert "--dry-run" in skill_source


# ---------------------------------------------------------------- crypto primitives
def test_script_is_stdlib_only(mod) -> None:
    src = SCRIPT.read_text(encoding="utf-8")
    assert "coincurve" not in src
    assert "import requests" not in src


def test_bech32_nip19_vectors(mod) -> None:
    assert mod.bech32_decode("nsec", NSEC_VECTOR).hex() == NSEC_HEX
    assert mod.bech32_encode("nsec", bytes.fromhex(NSEC_HEX)) == NSEC_VECTOR
    assert mod.bech32_decode("npub", NPUB_VECTOR).hex() == NPUB_HEX
    assert mod.bech32_encode("npub", bytes.fromhex(NPUB_HEX)) == NPUB_VECTOR


def test_bech32_rejects_bad_input(mod) -> None:
    with pytest.raises(ValueError):
        mod.bech32_decode("nsec", NSEC_VECTOR[:-1] + "q")  # checksum
    with pytest.raises(ValueError):
        mod.bech32_decode("nsec", NPUB_VECTOR)  # wrong hrp
    with pytest.raises(ValueError):
        mod.bech32_decode("nsec", NSEC_VECTOR[:10].upper() + NSEC_VECTOR[10:])  # mixed case


def test_decode_private_key_accepts_nsec_and_hex(mod) -> None:
    assert mod.decode_private_key(NSEC_VECTOR).hex() == NSEC_HEX
    assert mod.decode_private_key(NSEC_HEX).hex() == NSEC_HEX
    with pytest.raises(ValueError):
        mod.decode_private_key("00" * 32)  # zero is out of range
    with pytest.raises(ValueError):
        mod.decode_private_key("not-a-key")


@pytest.mark.parametrize("vec", BIP340_VECTORS, ids=lambda v: f"bip340-{v['index']}")
def test_schnorr_matches_official_bip340_vector(mod, vec) -> None:
    """Byte-exact against the BIP CSV — not against a verifier we wrote."""
    assert mod.pubkey_x(vec["sk"]).hex().upper() == vec["pk"]
    sig = mod.schnorr_sign(vec["msg"], vec["sk"], aux=vec["aux"])
    assert sig.hex().upper() == vec["sig"]


def test_signer_matches_in_tree_nostr_auth(mod) -> None:
    """Same bytes as Hermes' production signer (independent implementation)."""
    nostr_auth = _load_nostr_auth()
    for vec in BIP340_VECTORS:
        theirs = nostr_auth.schnorr_sign(
            vec["msg"], vec["sk"].hex(), auxiliary_randomness=vec["aux"]
        )
        ours = mod.schnorr_sign(vec["msg"], vec["sk"], aux=vec["aux"])
        assert ours == theirs
        assert nostr_auth.public_key_hex(vec["sk"].hex()).upper() == vec["pk"]
        assert theirs.hex().upper() == vec["sig"]


def test_generate_keypair_round_trips_and_matches_nostr_auth(mod) -> None:
    nostr_auth = _load_nostr_auth()
    nsec, npub, pub_hex = mod.generate_keypair()
    secret = mod.bech32_decode("nsec", nsec)
    assert mod.bech32_decode("npub", npub).hex() == pub_hex
    assert nostr_auth.public_key_hex(nsec) == pub_hex
    msg = hashlib.sha256(b"hello buzz").digest()
    aux = bytes(range(32))
    assert mod.schnorr_sign(msg, secret, aux=aux) == nostr_auth.schnorr_sign(
        msg, nsec, auxiliary_randomness=aux
    )


def test_nip98_header_structure_and_signature(mod) -> None:
    secret = bytes.fromhex(NSEC_HEX)
    url = "https://team.communities.buzz.xyz/api/invites"
    body = b"{}"
    header = mod.sign_nip98(secret, url, body)
    assert header.startswith("Nostr ")
    event = json.loads(base64.b64decode(header[len("Nostr ") :]))
    assert event["kind"] == 27235
    assert event["pubkey"] == mod.pubkey_x(secret).hex()
    assert ["u", url] in event["tags"]
    assert ["method", "POST"] in event["tags"]
    assert ["payload", hashlib.sha256(body).hexdigest()] in event["tags"]
    serialized = json.dumps(
        [0, event["pubkey"], event["created_at"], 27235, event["tags"], ""], separators=(",", ":")
    ).encode()
    assert event["id"] == hashlib.sha256(serialized).hexdigest()
    # Signature is a valid BIP-340 Schnorr over the event id (own-code verifier
    # is fine here — the signer itself is pinned to official vectors above).
    assert _verify_schnorr(mod, bytes.fromhex(event["id"]), bytes.fromhex(event["pubkey"]), bytes.fromhex(event["sig"]))


# ---------------------------------------------------------------- input resolution
def test_resolve_inputs_from_env_normalizes(mod, monkeypatch) -> None:
    _set_env(monkeypatch, BUZZ_INSTALL_ALLOW_ALL="no", BUZZ_INSTALL_ALLOWED_USERS="npub1a,npub1b")
    cfg = mod.resolve_inputs(interactive=False)
    assert cfg["relay"] == "https://team.communities.buzz.xyz"  # trailing slash stripped
    assert cfg["agent_name"] == "agent"  # defaults to profile
    assert cfg["allow_all"] is False
    assert cfg["require_mention"] is True
    assert cfg["rotate_key"] is False
    assert cfg["presence"] == "online"
    assert cfg["cli_path"].endswith("buzz")


def test_resolve_inputs_missing_required_is_fatal(mod, monkeypatch) -> None:
    for key in BASE_ENV:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(SystemExit):
        mod.resolve_inputs(interactive=False)


def test_resolve_inputs_rejects_bad_values(mod, monkeypatch) -> None:
    _set_env(monkeypatch, BUZZ_INSTALL_RELAY="http://insecure.example")
    with pytest.raises(SystemExit):
        mod.resolve_inputs(interactive=False)
    _set_env(monkeypatch, BUZZ_INSTALL_PRESENCE="busy")
    with pytest.raises(SystemExit):
        mod.resolve_inputs(interactive=False)
    _set_env(monkeypatch, BUZZ_INSTALL_ALLOW_ALL="maybe")
    with pytest.raises(SystemExit):
        mod.resolve_inputs(interactive=False)


def test_csv_json(mod) -> None:
    assert mod.csv_json("") == "[]"
    assert json.loads(mod.csv_json(" a, b ,,c ")) == ["a", "b", "c"]


# ---------------------------------------------------------------- profile .env
def test_write_env_replaces_existing_key_and_restricts_mode(mod, profiles) -> None:
    env_path = profiles / "agent" / ".env"
    env_path.write_text("OPENROUTER_API_KEY=x\nBUZZ_PRIVATE_KEY=old\n", encoding="utf-8")
    mod.write_env("agent", "nsec1new")
    lines = env_path.read_text(encoding="utf-8").splitlines()
    assert lines == ["OPENROUTER_API_KEY=x", "BUZZ_PRIVATE_KEY=nsec1new"]
    assert mod.read_existing_key("agent") == "nsec1new"
    if sys.platform != "win32":
        assert stat.S_IMODE(env_path.stat().st_mode) == 0o600


def test_write_env_creates_file_when_missing(mod, profiles) -> None:
    assert mod.read_existing_key("agent") == ""
    mod.write_env("agent", "nsec1fresh")
    assert mod.read_existing_key("agent") == "nsec1fresh"


def test_profile_dir_missing_is_fatal(mod, profiles) -> None:
    with pytest.raises(SystemExit):
        mod.profile_dir("ghost")


# ---------------------------------------------------------------- invite flow (mocked curl)
def _cfg(**overrides) -> dict:
    cfg = {
        "profile": "agent",
        "relay": "https://team.communities.buzz.xyz",
        "owner_key": NSEC_VECTOR,
    }
    cfg.update(overrides)
    return cfg


def test_mint_and_claim_with_join_policy(mod, monkeypatch) -> None:
    calls: list[tuple] = []
    responses = iter(
        [
            (0, json.dumps({"code": "v2.abc"}), ""),
            (0, json.dumps({"policy": {"version": "3"}}), ""),
            (0, json.dumps({"receipt": "r-1"}), ""),
            (0, json.dumps({"status": "joined", "role": "member"}), ""),
        ]
    )

    def fake_curl(url, body=None, auth=None):
        calls.append((url, body, auth))
        return next(responses)

    monkeypatch.setattr(mod, "curl", fake_curl)
    agent_secret = bytes.fromhex(NPUB_HEX)  # any in-range scalar works as a secret here
    mod.mint_and_claim(_cfg(), agent_secret)

    assert [c[0].rsplit("/", 1)[-1] for c in calls] == ["invites", "join-policy", "accept-policy", "claim"]
    # Mint is signed by the owner; claim is signed by the agent.
    owner_pub = mod.pubkey_x(bytes.fromhex(NSEC_HEX)).hex()
    agent_pub = mod.pubkey_x(agent_secret).hex()
    mint_event = json.loads(base64.b64decode(calls[0][2][len("Nostr ") :]))
    claim_event = json.loads(base64.b64decode(calls[3][2][len("Nostr ") :]))
    assert mint_event["pubkey"] == owner_pub
    assert claim_event["pubkey"] == agent_pub
    assert calls[2][1] == {"code": "v2.abc", "policy_version": "3", "age_confirmed": True}
    assert calls[3][1] == {"code": "v2.abc", "policy_receipt": "r-1"}


def test_mint_and_claim_without_policy(mod, monkeypatch) -> None:
    responses = iter(
        [
            (0, json.dumps({"code": "v2.xyz"}), ""),
            (0, "{}", ""),
            (0, json.dumps({"status": "joined", "role": "member"}), ""),
        ]
    )
    calls: list[tuple] = []

    def fake_curl(url, body=None, auth=None):
        calls.append((url, body, auth))
        return next(responses)

    monkeypatch.setattr(mod, "curl", fake_curl)
    mod.mint_and_claim(_cfg(), bytes.fromhex(NPUB_HEX))
    assert len(calls) == 3
    assert calls[2][1] == {"code": "v2.xyz"}


def test_mint_and_claim_requires_owner_key(mod, monkeypatch) -> None:
    monkeypatch.setattr(mod, "curl", mock.Mock(side_effect=AssertionError("network must not be touched")))
    with pytest.raises(SystemExit):
        mod.mint_and_claim(_cfg(owner_key=""), bytes.fromhex(NPUB_HEX))


def test_mint_error_payload_is_fatal(mod, monkeypatch) -> None:
    monkeypatch.setattr(mod, "curl", lambda *a, **k: (0, json.dumps({"error": "forbidden", "message": "nope"}), ""))
    with pytest.raises(SystemExit):
        mod.mint_and_claim(_cfg(), bytes.fromhex(NPUB_HEX))


# ---------------------------------------------------------------- hermes wiring
def test_write_config_sets_every_buzz_key(mod, monkeypatch) -> None:
    seen: dict[str, str] = {}

    def fake_sh(cmd, env=None, cwd=None, check=True):
        assert cmd[:5] == ["hermes", "-p", "agent", "config", "set"]
        seen[cmd[5]] = cmd[6]
        return 0, ""

    monkeypatch.setattr(mod, "sh", fake_sh)
    monkeypatch.setattr(mod.shutil, "which", lambda name: "/usr/bin/hermes" if name == "hermes" else None)
    cfg = {
        "profile": "agent",
        "relay": "https://team.communities.buzz.xyz",
        "channels": "c1,c2",
        "home_channel": "c1",
        "poll_interval": "4",
        "allowed_users": "",
        "require_mention": True,
        "allow_all": False,
    }
    mod.write_config(cfg, "/opt/bin/buzz")
    pre = "gateway.platforms.buzz"
    assert seen[f"{pre}.enabled"] == "true"
    assert seen[f"{pre}.extra.relay_url"] == cfg["relay"]
    assert json.loads(seen[f"{pre}.extra.channels"]) == ["c1", "c2"]
    assert seen[f"{pre}.extra.cli_path"] == "/opt/bin/buzz"
    assert seen[f"{pre}.extra.allow_all_users"] == "false"
    assert seen[f"{pre}.extra.require_mention"] == "true"
    assert seen["display.platforms.buzz.tool_progress"] == "off"
    assert f"{pre}.extra.credentials_file" not in seen


def test_ensure_cli_uses_existing_executable(mod, tmp_path, monkeypatch) -> None:
    binary = tmp_path / "buzz"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setattr(mod, "sh", mock.Mock(side_effect=AssertionError("must not build")))
    assert mod.ensure_cli(str(binary)) == str(binary)


def test_ensure_cli_falls_back_to_path(mod, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(mod.shutil, "which", lambda name: "/usr/local/bin/buzz" if name == "buzz" else None)
    monkeypatch.setattr(mod, "sh", mock.Mock(side_effect=AssertionError("must not build")))
    assert mod.ensure_cli(str(tmp_path / "missing")) == "/usr/local/bin/buzz"


def test_verify_reads_platform_state(mod, profiles) -> None:
    state = profiles / "agent" / "gateway_state.json"
    state.write_text(json.dumps({"platforms": {"buzz": {"state": "connected"}}}), encoding="utf-8")
    assert mod.verify("agent", attempts=1, delay=0) is True
    state.write_text(json.dumps({"platforms": {"buzz": {"state": "disconnected"}}}), encoding="utf-8")
    assert mod.verify("agent", attempts=1, delay=0) is False
    state.write_text("not json", encoding="utf-8")
    assert mod.verify("agent", attempts=1, delay=0) is False


# ---------------------------------------------------------------- dry run
def test_dry_run_touches_nothing(mod, profiles, monkeypatch, capsys) -> None:
    _set_env(monkeypatch)
    monkeypatch.setattr(mod, "curl", mock.Mock(side_effect=AssertionError("no network in dry run")))
    monkeypatch.setattr(mod, "sh", mock.Mock(side_effect=AssertionError("no subprocess in dry run")))
    assert mod.main(["--dry-run", "--non-interactive"]) == 0
    out = capsys.readouterr().out
    assert "PLAN" in out
    assert "mint a NEW keypair" in out
    assert not (profiles / "agent" / ".env").exists()


def test_dry_run_reports_key_reuse(mod, profiles, monkeypatch, capsys) -> None:
    _set_env(monkeypatch)
    (profiles / "agent" / ".env").write_text(f"BUZZ_PRIVATE_KEY={NSEC_VECTOR}\n", encoding="utf-8")
    assert mod.main(["--dry-run", "--non-interactive"]) == 0
    assert "reuse key from profile .env" in capsys.readouterr().out
