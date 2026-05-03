#!/usr/bin/env python3
"""AgenticBets CLI — list markets, check odds, place bets, claim winnings.

Uses the AgenticBets Markets API for reads and Bankr Submit API for writes.

Usage: scripts/agenticbets.py <command> [args...]

Environment:
  BANKR_CONFIG  Path to Bankr config file (default: ~/.bankr/config.json)
"""

import json
import os
import sys
import urllib.request
import urllib.error

MARKETS_API = "https://agenticbets.dev/api/bankr/markets"
BANKR_API = "https://api.bankr.bot"
USDC_ADDRESS = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
USDC_DECIMALS = 6
BASE_CHAIN_ID = 8453

# V1 — existing markets; V2 — AGBETS
PREDICTION_V1 = "0xABADeb002247f2bd908Eeedb32918aEc304A0233"
PREDICTION_V2 = "0x2CD785Ba87e0841A8458141bc43d23a56a00557f"
AGBETS_TOKEN = "0x37d183FCf1DA460a64D21E754b3E6144C4e11BA3"


def load_bankr_key():
    """Read Bankr API key from config."""
    config_path = os.environ.get("BANKR_CONFIG", os.path.expanduser("~/.bankr/config.json"))
    if not os.path.exists(config_path):
        print(f"ERROR: Bankr config not found at {config_path}", file=sys.stderr)
        print("Run `bankr login email <your-email>` to set up.", file=sys.stderr)
        sys.exit(1)
    with open(config_path) as f:
        config = json.load(f)
    key = config.get("apiKey") or config.get("api_key")
    if not key:
        print(f"ERROR: No apiKey in {config_path}", file=sys.stderr)
        sys.exit(1)
    return key


def bankr_request(method, path, body=None):
    """Make an authenticated request to Bankr Wallet API."""
    api_key = load_bankr_key()
    url = f"{BANKR_API}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "X-API-Key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "agenticbets-bankr-skill/1.0",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body_text = e.read().decode() if e.fp else ""
        print(f"Bankr API error: {e.code} {e.reason} — {body_text}", file=sys.stderr)
        sys.exit(1)


def get_wallet_address():
    """Get the user's Bankr wallet address."""
    data = bankr_request("GET", "/wallet/me")
    return data.get("address") or data.get("walletAddress")


def submit_tx(to, data_hex, description="", value="0"):
    """Submit a transaction via Bankr Submit API."""
    tx = {
        "to": to,
        "chainId": BASE_CHAIN_ID,
        "data": data_hex,
        "value": value,
    }
    result = bankr_request("POST", "/wallet/submit", {
        "transaction": tx,
        "description": description,
        "waitForConfirmation": True,
    })
    tx_hash = result.get("transactionHash") or result.get("hash") or result.get("txHash")
    status = result.get("status", "unknown")
    if status == "reverted":
        print(f"  WARNING: Transaction reverted! Hash: {tx_hash}", file=sys.stderr)
    return tx_hash


def fetch_markets():
    """Fetch markets from AgenticBets API."""
    req = urllib.request.Request(
        MARKETS_API,
        headers={"Accept": "application/json", "User-Agent": "agenticbets-bankr-skill/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        return data["markets"]
    except Exception as e:
        print(f"ERROR: Failed to fetch markets — {e}", file=sys.stderr)
        sys.exit(1)


def find_market(symbol_or_address):
    """Find a market by symbol or token address."""
    markets = fetch_markets()
    q = symbol_or_address.lower()
    for m in markets:
        if m["symbol"].lower() == q or m["token"].lower() == q:
            return m
    return None


def get_prediction_contract(token_address):
    """Return the correct prediction contract for a token."""
    if token_address.lower() == AGBETS_TOKEN.lower():
        return PREDICTION_V2
    return PREDICTION_V1


def encode_approve(spender, amount_raw):
    """Encode ERC20 approve(address,uint256) calldata."""
    # approve(address,uint256) selector = 0x095ea7b3
    spender_padded = spender.lower().replace("0x", "").zfill(64)
    amount_hex = hex(amount_raw)[2:].zfill(64)
    return f"0x095ea7b3{spender_padded}{amount_hex}"


def encode_bet(token, amount_raw, position):
    """Encode bet(address,uint256,uint8) calldata."""
    # bet(address,uint256,uint8) selector = 0x37a02e62
    token_padded = token.lower().replace("0x", "").zfill(64)
    amount_hex = hex(amount_raw)[2:].zfill(64)
    position_hex = hex(position)[2:].zfill(64)
    return f"0x37a02e62{token_padded}{amount_hex}{position_hex}"


def encode_claim(token, epochs):
    """Encode claim(address,uint256[]) calldata."""
    # claim(address,uint256[]) selector = 0x45718278
    token_padded = token.lower().replace("0x", "").zfill(64)
    # ABI encode dynamic array: offset, length, elements
    offset_hex = hex(64)[2:].zfill(64)  # offset to array data (2 * 32 bytes)
    length_hex = hex(len(epochs))[2:].zfill(64)
    elements = "".join(hex(e)[2:].zfill(64) for e in epochs)
    return f"0x45718278{token_padded}{offset_hex}{length_hex}{elements}"


def encode_claimable(token, epoch, user):
    """Encode claimable(address,uint256,address) calldata."""
    # claimable(address,uint256,address) selector = 0xd3c035fc
    token_padded = token.lower().replace("0x", "").zfill(64)
    epoch_hex = hex(epoch)[2:].zfill(64)
    user_padded = user.lower().replace("0x", "").zfill(64)
    return f"0xd3c035fc{token_padded}{epoch_hex}{user_padded}"


# -- Commands ----------------------------------------------------------------

def cmd_list(status="open"):
    """List markets filtered by status."""
    markets = fetch_markets()
    if status != "all":
        markets = [m for m in markets if m["status"] == status]

    if not markets:
        print(f"No {status} markets found.")
        return

    print(f"{'Symbol':<10} {'Pool':>10} {'UP':>5} {'DOWN':>5} {'Status':<10} {'Time Left'}")
    print("-" * 60)
    for m in markets:
        bull = round(m["bullPct"])
        bear = 100 - bull
        time_left = f"{m['secondsToLock']}s" if m.get("secondsToLock") and m["secondsToLock"] > 0 else "-"
        print(f"${m['symbol']:<9} ${m['poolUsdc']:>9.2f} {bull:>4}% {bear:>4}% {m['status']:<10} {time_left}")


def cmd_odds(symbol):
    """Show odds for a market."""
    market = find_market(symbol)
    if not market:
        print(f"No market found for '{symbol}'")
        sys.exit(1)

    bull = round(market["bullPct"])
    bear = 100 - bull
    time_info = f"{market['secondsToLock']}s to lock" if market.get("secondsToLock") and market["secondsToLock"] > 0 else market["status"]

    print(f"${market['symbol']} — Epoch {market['epoch']}")
    print(f"  UP:   {bull}%")
    print(f"  DOWN: {bear}%")
    print(f"  Pool: ${market['poolUsdc']:.2f} USDC")
    print(f"  {time_info}")


def cmd_info(symbol):
    """Detailed market info."""
    market = find_market(symbol)
    if not market:
        print(f"No market found for '{symbol}'")
        sys.exit(1)

    print(json.dumps(market, indent=2))


def cmd_bet(symbol, direction, amount):
    """Place a bet."""
    market = find_market(symbol)
    if not market:
        print(f"No market found for '{symbol}'")
        sys.exit(1)

    token = market["token"]
    prediction = get_prediction_contract(token)
    position = 0 if direction.lower() in ("up", "bull") else 1
    direction_label = "UP" if position == 0 else "DOWN"
    amount_raw = int(float(amount) * (10 ** USDC_DECIMALS))

    print(f"Placing ${amount} {direction_label} bet on ${market['symbol']}...")

    # 1. Approve USDC for this bet amount
    print("  Approving USDC...")
    approve_data = encode_approve(prediction, amount_raw)
    approve_hash = submit_tx(
        USDC_ADDRESS,
        approve_data,
        description=f"Approve ${amount} USDC for AgenticBets {market['symbol']} bet",
    )
    print(f"  Approve tx: https://basescan.org/tx/{approve_hash}")

    # 2. Place bet
    print("  Submitting bet...")
    bet_data = encode_bet(token, amount_raw, position)
    bet_hash = submit_tx(
        prediction,
        bet_data,
        description=f"Place ${amount} {direction_label} bet on {market['symbol']} epoch {market['epoch']}",
    )
    print(f"  Bet tx: https://basescan.org/tx/{bet_hash}")
    print(f"  Done! ${amount} {direction_label} on ${market['symbol']} epoch {market['epoch']}")


def cmd_claim(symbol, *epoch_args):
    """Claim winnings."""
    market = find_market(symbol)
    if not market:
        print(f"No market found for '{symbol}'")
        sys.exit(1)

    epochs = [int(e) for e in epoch_args]
    token = market["token"]
    prediction = get_prediction_contract(token)

    print(f"Claiming ${market['symbol']} epochs {epochs}...")
    claim_data = encode_claim(token, epochs)
    tx_hash = submit_tx(
        prediction,
        claim_data,
        description=f"Claim {market['symbol']} winnings for epochs {epochs}",
    )
    print(f"  Claim tx: https://basescan.org/tx/{tx_hash}")
    print("  Done!")


def cmd_claimable(symbol, epoch):
    """Check if an epoch is claimable."""
    market = find_market(symbol)
    if not market:
        print(f"No market found for '{symbol}'")
        sys.exit(1)

    wallet = get_wallet_address()
    token = market["token"]

    print(f"Checking claimable: ${market['symbol']} epoch {epoch} for {wallet}")
    print("(Use the AgenticBets website to check — contract reads via Bankr API coming soon)")


# -- Entry point -------------------------------------------------------------

COMMANDS = {
    "list": lambda args: cmd_list(args[0] if args else "open"),
    "odds": lambda args: cmd_odds(args[0]) if args else print("Usage: odds <symbol>"),
    "info": lambda args: cmd_info(args[0]) if args else print("Usage: info <symbol>"),
    "bet": lambda args: cmd_bet(args[0], args[1], args[2]) if len(args) >= 3 else print("Usage: bet <symbol> <up|down> <amount>"),
    "claim": lambda args: cmd_claim(args[0], *args[1:]) if len(args) >= 2 else print("Usage: claim <symbol> <epoch> [epoch...]"),
    "claimable": lambda args: cmd_claimable(args[0], int(args[1])) if len(args) >= 2 else print("Usage: claimable <symbol> <epoch>"),
}

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("AgenticBets — Prediction markets on Base")
        print()
        print("Commands:")
        print("  list [status]              List markets (open/locked/settled/all)")
        print("  odds <symbol>              Show bull/bear odds")
        print("  info <symbol>              Detailed market info (JSON)")
        print("  bet <symbol> <up|down> <$> Place a USDC bet")
        print("  claim <symbol> <epoch...>  Claim settled winnings")
        print("  claimable <symbol> <epoch> Check if epoch is claimable")
        sys.exit(0)

    cmd = sys.argv[1]
    args = sys.argv[2:]
    COMMANDS[cmd](args)


if __name__ == "__main__":
    main()
