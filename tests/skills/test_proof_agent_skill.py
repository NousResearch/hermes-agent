"""Tests for the optional blockchain/proof-agent skill.

Two layers:

1. Static checks on SKILL.md frontmatter and the shipped payment helper
   (always run; stdlib only).
2. Functional checks that drive ``scripts/nano-pay.cjs`` as a subprocess
   against a mock Nano RPC node served from this process (skipped when a
   ``node`` >= 18 binary is unavailable). No live network calls: the mock
   node binds to 127.0.0.1 and NANO_RPC_URLS points the script at it.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "blockchain"
    / "proof-agent"
)
SKILL_MD = SKILL_DIR / "SKILL.md"
SCRIPT = SKILL_DIR / "scripts" / "nano-pay.cjs"
PACKAGE_JSON = SKILL_DIR / "scripts" / "package.json"

ZERO_HASH = "0" * 64
RAW_PER_XNO = 10**30


# ---------------------------------------------------------------------------
# Layer 1: static checks
# ---------------------------------------------------------------------------

def read_frontmatter() -> dict[str, str]:
    text = SKILL_MD.read_text(encoding="utf-8")
    match = re.match(r"\A---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, "SKILL.md must start with YAML frontmatter"
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        m = re.match(r"^(\w[\w-]*):\s*(.*)$", line)
        if m:
            fields[m.group(1)] = m.group(2).strip()
    return fields


def test_description_is_short_single_sentence():
    description = read_frontmatter()["description"]
    assert len(description) <= 60, len(description)
    assert description.endswith(".")


def test_frontmatter_required_fields():
    fields = read_frontmatter()
    assert fields["name"] == "proof-agent"
    assert "version" in fields
    assert "license" in fields
    assert "platforms" in fields


def test_skill_body_uses_modern_sections():
    body = SKILL_MD.read_text(encoding="utf-8")
    for section in (
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ):
        assert section in body, f"missing section: {section}"


def test_helper_script_ships_with_pinned_dependency():
    assert SCRIPT.is_file()
    manifest = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))
    pinned = manifest["dependencies"]["nanocurrency-web"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", pinned), pinned
    assert (SKILL_DIR / "scripts" / "package-lock.json").is_file()


def test_helper_script_never_prints_the_seed_outside_new():
    source = SCRIPT.read_text(encoding="utf-8")
    # `new` intentionally prints the freshly generated seed; the seed from
    # NANO_SEED must never be echoed back.
    assert "console.log(seed" not in source
    assert re.search(r"seed\s*=\s*process\.env\.NANO_SEED", source)


# ---------------------------------------------------------------------------
# Layer 2: functional checks against a mock Nano node
# ---------------------------------------------------------------------------

def node_available() -> bool:
    node = shutil.which("node")
    if not node:
        return False
    try:
        out = subprocess.run(
            [node, "--version"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
        return int(out.lstrip("v").split(".")[0]) >= 18
    except Exception:
        return False


def deps_installed() -> bool:
    return (SKILL_DIR / "scripts" / "node_modules" / "nanocurrency-web").is_dir()


functional = pytest.mark.skipif(
    not (node_available() and deps_installed()),
    reason="requires node >= 18 and `npm ci` in the skill's scripts/ dir",
)


class MockNanoNode:
    """In-memory Nano node: enough RPC surface for nano-pay.cjs.

    Tracks per-account balance/frontier and a receivable pool. `process`
    validates the block fields the way a real node's state machine would
    (balances, frontiers, link) minus signature/work verification.
    """

    def __init__(self):
        self.accounts: dict[str, dict] = {}
        self.receivable: dict[str, dict[str, str]] = {}
        self.work_requests: list[str] = []
        self.actions_seen: list[str] = []
        self.block_counter = 0

    def seed_pending(self, account: str, amount_raw: int):
        self.block_counter += 1
        send_hash = f"{self.block_counter:064X}"
        self.receivable.setdefault(account, {})[send_hash] = str(amount_raw)

    def handle(self, req: dict) -> dict:
        action = req.get("action")
        self.actions_seen.append(action)
        if action == "account_info":
            acct = self.accounts.get(req["account"])
            if not acct:
                return {"error": "Account not found"}
            return {
                "balance": acct["balance"],
                "frontier": acct["frontier"],
                "representative": acct["representative"],
            }
        if action == "receivable":
            blocks = self.receivable.get(req["account"], {})
            return {"blocks": {h: amt for h, amt in blocks.items()} or ""}
        if action == "work_generate":
            self.work_requests.append(req["hash"])
            return {"work": "cafebabecafebabe"}
        if action == "process":
            return self.process_block(req)
        return {"error": f"unsupported action {action}"}

    def process_block(self, req: dict) -> dict:
        blk = req["block"]
        if isinstance(blk, str):
            blk = json.loads(blk)
        account = blk["account"]
        subtype = req.get("subtype")
        prev = blk["previous"]
        state = self.accounts.get(account)
        if subtype in ("receive", "open"):
            if subtype == "open":
                assert prev == ZERO_HASH, "open block must have zero previous"
                assert state is None, "open on an existing account"
                old_balance = 0
            else:
                assert state is not None and prev == state["frontier"]
                old_balance = int(state["balance"])
            link = blk["link"].upper()
            pending = self.receivable.get(account, {})
            assert link in pending, "receive link must match a receivable hash"
            amount = int(pending.pop(link))
            assert int(blk["balance"]) == old_balance + amount
        elif subtype == "send":
            assert state is not None and prev == state["frontier"]
            new_balance = int(blk["balance"])
            old_balance = int(state["balance"])
            assert 0 <= new_balance < old_balance
            dest_pub = blk["link"].upper()
            self.block_counter += 1
            # Credit destination's receivable pool keyed by this block hash.
            dest = self._pub_to_account.get(dest_pub)
            if dest:
                self.receivable.setdefault(dest, {})[
                    f"{self.block_counter:064X}"
                ] = str(old_balance - new_balance)
        else:
            return {"error": f"bad subtype {subtype}"}
        self.block_counter += 1
        new_hash = f"{self.block_counter:064X}"
        self.accounts[account] = {
            "balance": blk["balance"],
            "frontier": new_hash,
            "representative": blk["representative"],
        }
        return {"hash": new_hash}

    # Mapping of hex public keys to nano_ addresses, registered by tests so
    # `send` can credit the destination account.
    _pub_to_account: dict[str, str] = {}

    def register(self, address: str, public_key: str):
        self._pub_to_account[public_key.upper()] = address


@pytest.fixture()
def mock_node():
    node = MockNanoNode()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length))
            body = json.dumps(node.handle(req)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    node.url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield node
    finally:
        server.shutdown()


def run_cli(args, seed="", rpc_urls=""):
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "NANO_SEED": seed,
        "NANO_RPC_URLS": rpc_urls,
    }
    return subprocess.run(
        ["node", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=SCRIPT.parent,
    )


@functional
def test_new_generates_distinct_wallets():
    a = json.loads(run_cli(["new"]).stdout)
    b = json.loads(run_cli(["new"]).stdout)
    assert a["address"].startswith("nano_")
    assert len(a["seed"]) == 64
    assert a["seed"] != b["seed"]


@functional
def test_address_is_deterministic_for_a_seed():
    w = json.loads(run_cli(["new"]).stdout)
    out = run_cli(["address"], seed=w["seed"])
    assert out.stdout.strip() == w["address"]


@functional
def test_wallet_commands_fail_cleanly_without_seed():
    out = run_cli(["address"])
    assert out.returncode == 1
    assert "NANO_SEED" in out.stderr


@functional
def test_balance_of_unopened_account_is_zero(mock_node):
    w = json.loads(run_cli(["new"]).stdout)
    out = json.loads(
        run_cli(["balance"], seed=w["seed"], rpc_urls=mock_node.url).stdout
    )
    assert out == {"address": w["address"], "balanceRaw": "0", "balanceXno": 0}


@functional
def test_fund_prints_nano_uri_with_exact_raw_amount(mock_node):
    w = json.loads(run_cli(["new"]).stdout)
    out = json.loads(
        run_cli(["fund", "0.01"], seed=w["seed"], rpc_urls=mock_node.url).stdout
    )
    assert out["uri"] == f"nano:{w['address']}?amount={RAW_PER_XNO // 100}"
    assert "owner" in out["message"]


@functional
def test_receive_opens_account_then_pockets_further_blocks(mock_node):
    w = json.loads(run_cli(["new"]).stdout)
    mock_node.seed_pending(w["address"], 2 * RAW_PER_XNO)
    mock_node.seed_pending(w["address"], 1 * RAW_PER_XNO)
    out = json.loads(
        run_cli(["receive"], seed=w["seed"], rpc_urls=mock_node.url).stdout
    )
    assert out["ok"] is True
    assert out["received"] == 2
    assert out["balanceRaw"] == str(3 * RAW_PER_XNO)
    # First receive is an open block: work is computed on the public key,
    # not on a frontier.
    assert mock_node.work_requests[0] not in (ZERO_HASH,)


@functional
def test_send_auto_receives_pending_first(mock_node):
    sender = json.loads(run_cli(["new"]).stdout)
    recipient = json.loads(run_cli(["new"]).stdout)
    mock_node.seed_pending(sender["address"], 5 * RAW_PER_XNO)
    out = json.loads(
        run_cli(
            ["send", recipient["address"], str(2 * RAW_PER_XNO)],
            seed=sender["seed"],
            rpc_urls=mock_node.url,
        ).stdout
    )
    assert out["ok"] is True and out["hash"]
    balance = json.loads(
        run_cli(["balance"], seed=sender["seed"], rpc_urls=mock_node.url).stdout
    )
    assert balance["balanceRaw"] == str(3 * RAW_PER_XNO)


@functional
def test_send_rejects_bad_recipient_and_overdraft(mock_node):
    w = json.loads(run_cli(["new"]).stdout)
    bad = run_cli(
        ["send", "not_an_address", "1"], seed=w["seed"], rpc_urls=mock_node.url
    )
    assert bad.returncode == 1
    assert "invalid recipient" in bad.stderr

    other = json.loads(run_cli(["new"]).stdout)
    mock_node.seed_pending(w["address"], RAW_PER_XNO)
    over = run_cli(
        ["send", other["address"], str(10 * RAW_PER_XNO)],
        seed=w["seed"],
        rpc_urls=mock_node.url,
    )
    assert over.returncode == 1
    assert "insufficient balance" in over.stderr


@functional
def test_rpc_failover_skips_dead_endpoint(mock_node):
    w = json.loads(run_cli(["new"]).stdout)
    urls = f"http://127.0.0.1:1/dead,{mock_node.url}"
    out = run_cli(["balance"], seed=w["seed"], rpc_urls=urls)
    assert out.returncode == 0
    assert json.loads(out.stdout)["balanceRaw"] == "0"
