"""
Cosmos Degen Trader v4.1 - Whale Hunter + Tightened Loss Prevention

High-frequency Solana memecoin trader powered by the Cosmos swarm.
- DIRECT Pump.fun bonding curve buys (no Jupiter needed pre-graduation)
- PumpPortal local transaction API for speed-critical buys
- Pre-bonding curve sniping: buy seconds after launch, sell before/after graduation
- Pump.fun WebSocket for instant new launch detection + trade velocity tracking
- Wallet graph analysis to detect cabals and insider coordination
- Quantum-enhanced analysis: IBM Quantum QAOA, quantum random timing, FarsightProtocol
- Deep swarm integration: Grok (X sentiment), DeepSeek (TA), Gemini (multi-factor)
- X/Twitter sentinel: real-time cabal movement detection via Grok
- Copy trading: track most profitable wallets from GMGN/Birdeye, pre-buy on their moves
- Trading memory: learns from every trade via MemorySystem + KnowledgeGraph
- Alchemy RPC for fast on-chain reads/sends
- Jupiter v6 execution with priority fees
- DexScreener + GMGN + Birdeye for comprehensive market data

Requires: pip install solders solana aiohttp websockets
"""

import asyncio
import json
import time
import logging
import os
import base64
import hashlib
import random
import struct
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple, Set
from pathlib import Path
from datetime import datetime, timedelta

import aiohttp

logger = logging.getLogger("cosmos.trading")


# ============================================================
# COSMOS AUTH LOCK — Trader will NOT run without the swarm
# ============================================================
class _CosmosAuthLock:
    """Validates this trader is running inside the Cosmos ecosystem.

    Without a valid session token from the Cosmos server,
    the trading engine refuses to start. Prevents standalone use.
    """
    _SESSION_SALT = b"cosmos_swarm_v3"
    _valid_token: Optional[str] = None

    @classmethod
    def generate_session_token(cls) -> str:
        """Called by server.py at startup — generates a one-time session token
        derived from the Cosmos Nexus state + current process."""
        try:
            from Cosmos.core.nexus import Nexus
            nexus_exists = Nexus._instance is not None
        except ImportError:
            nexus_exists = False

        # Token = HMAC(salt, pid + boot_time + nexus_state)
        import hmac
        payload = f"{os.getpid()}:{time.monotonic():.0f}:{nexus_exists}:{id(cls)}".encode()
        token = hmac.new(cls._SESSION_SALT, payload, hashlib.sha256).hexdigest()[:32]
        cls._valid_token = token
        logger.debug(f"Auth lock: session token generated")
        return token

    @classmethod
    def validate(cls, token: Optional[str] = None) -> bool:
        """Validate the trader is authorized to run."""
        # Check 1: Must be running inside Cosmos process (Nexus importable)
        try:
            from Cosmos.core.nexus import Nexus
            from Cosmos.memory.memory_system import MemorySystem
        except ImportError:
            logger.error("AUTH LOCK: Cosmos core not found. Trader requires the full swarm.")
            return False

        # Check 2: Server must have generated a session token
        if cls._valid_token is None:
            # Allow if Nexus singleton exists (running inside full server)
            if Nexus._instance is not None:
                return True
            logger.error("AUTH LOCK: No session token. Start via Cosmos server, not standalone.")
            return False

        # Check 3: Token must match
        if token and token == cls._valid_token:
            return True

        # Check 4: Running in same process as server (token set in-process)
        if cls._valid_token is not None:
            return True

        logger.error("AUTH LOCK: Invalid session token. Trader locked to Cosmos swarm.")
        return False

    @classmethod
    def lock_check(cls):
        """Hard check — raises if not authorized."""
        if not cls.validate():
            raise RuntimeError(
                "Cosmos Degen Trader is locked to the Cosmos AI Swarm. "
                "It cannot run standalone. Deploy the full Cosmos system."
            )


# ============================================================
# CONSTANTS
# ============================================================
SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
LAMPORTS_PER_SOL = 1_000_000_000

# v3.8: Jupiter migrated from quote-api.jup.ag (DEAD) → lite-api.jup.ag → api.jup.ag
# Using lite-api (no key needed, still works) with api.jup.ag (key needed) as upgrade path
JUPITER_QUOTE_URL = "https://lite-api.jup.ag/swap/v1/quote"
JUPITER_SWAP_URL = "https://lite-api.jup.ag/swap/v1/swap"
JUPITER_PRICE_URL = "https://api.jup.ag/price/v2"
# Raydium Trade API fallback (no key needed)
RAYDIUM_QUOTE_URL = "https://transaction-v1.raydium.io/compute/swap-base-in"
RAYDIUM_SWAP_URL = "https://transaction-v1.raydium.io/transaction/swap-base-in"

DEXSCREENER_TOKENS = "https://api.dexscreener.com/latest/dex/tokens"
DEXSCREENER_BOOSTS = "https://api.dexscreener.com/token-boosts/latest/v1"
DEXSCREENER_PROFILES = "https://api.dexscreener.com/token-profiles/latest/v1"

PUMPFUN_WS_URL = "wss://pumpportal.fun/api/data"
PUMPPORTAL_LOCAL_API = "https://pumpportal.fun/api/trade-local"
RAYDIUM_AMM_PROGRAM = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"

# Multi-launchpad program IDs
BONK_LAUNCHLAB_PROGRAM = "LanMV9sAd7wArD4vJFi2qDdfnVhFxYSUg6eADduJ3uj"
BONK_PLATFORM_CONFIG = "FfYek5vEz23cMkWsdJwG2oa6EphsvXSHrGpdALN4g6W1"
BAGS_DBC_PROGRAM = "dbcij3LWUppWqq96dh6gJWwBifmcGfLSB5D4DuSMaqN"
BAGS_CREATOR_PROGRAM = "BAGSB9TpGrZxQbEsrEznv5jXXdwyP6AXerN8aVRiAmcv"
BAGS_API_URL = "https://public-api-v2.bags.fm/api/v1"

# Platform identifiers for multi-launchpad support
PLATFORM_PUMP = "pump"        # pump.fun
PLATFORM_BONK = "bonk"        # letsbonk.fun (via Raydium LaunchLab)
PLATFORM_BAGS = "bags"        # bags.fm (via Meteora DBC)

# Pump.fun Program Addresses (bonding curve direct trading)
PUMP_PROGRAM_ID = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
PUMP_GLOBAL_ACCOUNT = "4wTV1YmiEkRvAtNtsSGPtUrqRYQMe5SKy2uB4Jjaxnjf"
PUMP_FEE_RECIPIENTS = [
    "62qc2CNXwrYqQScmEdiZFFAnJR262PxWEuNQtxfafNgV",
    "7VtfL8fvgNfhz17qKRMjzQEXgbdpnHHHQRh54R9jP2RJ",
    "7hTckgnGnLQR6sdH7YkqFTAA7VwTfYFaZ6EhEsU3saCX",
    "9rPYyANsfQZw3DnDmKE3YCQF5E8oD89UXoHn9JFEhJUz",
    "AVmoTthdrX6tKt4nDjco2D775W2YK3sDhxPcMmzUAmTY",
    "CebN5WGQ4jvEPvsVU4EoHEpgzq1VV7AbicfhtW4xC9iM",
    "FWsW1xNtWscwNmKv6wVsU1iTzRN6wmmk3MjxRP5tT7hz",
    "G5UZAVbAf46s7cKWoyKu8kYTip9DGTpbLZ2qa9Aq69dP",
]
PUMP_BUY_DISCRIMINATOR = bytes([102, 6, 61, 18, 1, 218, 235, 234])
PUMP_SELL_DISCRIMINATOR = bytes([51, 230, 133, 164, 1, 127, 131, 173])
PUMP_GRADUATION_SOL = 85.0  # ~85 SOL triggers graduation to PumpSwap
PUMP_INITIAL_VIRTUAL_TOKEN = 1_073_000_000_000_000  # 6 decimals
PUMP_INITIAL_VIRTUAL_SOL = 30_000_000_000  # 30 SOL in lamports
PUMP_INITIAL_REAL_TOKEN = 793_100_000_000_000
TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
ASSOC_TOKEN_PROGRAM = "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"
SYSTEM_PROGRAM = "11111111111111111111111111111111"
RENT_SYSVAR = "SysvarRent111111111111111111111111111111111"

DEFAULT_RPC = "https://api.mainnet-beta.solana.com"

# Copy trading / smart money sources
GMGN_SMART_MONEY_URL = "https://gmgn.ai/defi/quotation/v1/rank/sol/swaps/24h"
BIRDEYE_BASE_URL = "https://public-api.birdeye.so"
PUMPFUN_API_URL = "https://frontend-api-v3.pump.fun"

WALLET_DIR = Path(__file__).parent / ".wallets"
STATE_FILE = Path(__file__).parent / ".trader_state.json"


# ============================================================
# DATA CLASSES
# ============================================================
@dataclass
class TokenInfo:
    address: str
    symbol: str
    name: str
    pair_address: str
    price_usd: float
    liquidity_usd: float
    volume_24h: float
    price_change_5m: float = 0.0
    price_change_1h: float = 0.0
    age_minutes: float = 0.0
    holders: int = 0
    score: float = 0.0
    fdv: float = 0.0
    buy_count_5m: int = 0
    sell_count_5m: int = 0
    # Enhanced fields
    cabal_score: float = 0.0       # 0=no cabal, 100=definite cabal
    rug_probability: float = 0.0   # 0-1 from quantum Monte Carlo
    swarm_sentiment: str = ""      # BUY/SKIP/STRONG_BUY
    source: str = "dexscreener"    # dexscreener/pumpfun/raydium/bonding_curve
    creator_wallet: str = ""
    top_holders_connected: bool = False
    # v3.5: Bonding curve fields
    platform: str = ""                   # pump/bonk/bags - which launchpad
    on_bonding_curve: bool = False       # still on bonding curve (any platform)
    curve_progress: float = 0.0          # 0-100% towards graduation
    curve_sol_raised: float = 0.0        # SOL raised so far
    buy_velocity_per_min: float = 0.0    # buys per minute (momentum)
    dev_bought_more: bool = False        # dev buying their own token = bullish
    initial_buy_sol: float = 0.0         # creator's initial buy size


@dataclass
class BondingCurveState:
    """Parsed state of a pump.fun bonding curve account."""
    virtual_token_reserves: int = 0
    virtual_sol_reserves: int = 0
    real_token_reserves: int = 0
    real_sol_reserves: int = 0
    token_total_supply: int = 0
    complete: bool = False

    @property
    def price_sol(self) -> float:
        """Current price per token in SOL."""
        if self.virtual_token_reserves == 0:
            return 0
        return (self.virtual_sol_reserves / 1e9) / (self.virtual_token_reserves / 1e6)

    @property
    def sol_raised(self) -> float:
        """Total SOL raised from bonding curve sales."""
        return self.real_sol_reserves / 1e9

    @property
    def progress_pct(self) -> float:
        """Percent progress towards graduation (0-100)."""
        if PUMP_GRADUATION_SOL <= 0:
            return 0
        return min(100.0, (self.sol_raised / PUMP_GRADUATION_SOL) * 100)

    @property
    def tokens_remaining(self) -> float:
        """Tokens still available on the curve."""
        return self.real_token_reserves / 1e6

    def calc_tokens_for_sol(self, sol_amount: float) -> int:
        """Calculate how many raw tokens you get for X SOL."""
        sol_lamports = int(sol_amount * 1e9)
        fee = int(sol_lamports * 0.01)  # 1% pump.fun fee
        sol_after_fee = sol_lamports - fee
        k = self.virtual_token_reserves * self.virtual_sol_reserves
        new_virtual_sol = self.virtual_sol_reserves + sol_after_fee
        if new_virtual_sol == 0:
            return 0
        new_virtual_tokens = k // new_virtual_sol
        tokens_out = self.virtual_token_reserves - new_virtual_tokens
        return max(0, tokens_out)

    def calc_sol_for_tokens(self, token_amount: int) -> int:
        """Calculate how much SOL lamports you get for selling X raw tokens."""
        k = self.virtual_token_reserves * self.virtual_sol_reserves
        new_virtual_tokens = self.virtual_token_reserves + token_amount
        if new_virtual_tokens == 0:
            return 0
        new_virtual_sol = k // new_virtual_tokens
        sol_out = self.virtual_sol_reserves - new_virtual_sol
        fee = int(sol_out * 0.01)
        return max(0, sol_out - fee)


@dataclass
class Position:
    token_address: str
    symbol: str
    entry_price: float
    amount_tokens: float
    amount_sol_spent: float
    entry_time: float
    take_profit_levels: List[float] = field(default_factory=lambda: [1.12, 1.20, 1.35])  # v4.1: even tighter scalp
    stop_loss: float = 0.80  # v4.1: tighter stop — cut at 20% loss, preserve capital (was 30%)
    partial_sells: int = 0
    source: str = ""  # what detected it
    entry_velocity: float = 0.0    # buys/min at time of entry
    peak_velocity: float = 0.0     # highest velocity seen since entry
    on_bonding_curve: bool = False  # v3.9: still on pump.fun bonding curve (no DEX pool yet)


@dataclass
class Trade:
    timestamp: float
    action: str
    token_address: str
    symbol: str
    amount_sol: float
    price_usd: float
    tx_signature: str
    pnl_sol: float = 0.0
    reason: str = ""


@dataclass
class WalletProfile:
    """Profile of a wallet we're tracking."""
    address: str
    funded_by: str = ""           # who sent this wallet its first SOL
    total_sol_received: float = 0
    token_buys: int = 0
    connected_wallets: Set[str] = field(default_factory=set)
    last_seen: float = 0
    is_whale: bool = False
    win_rate: float = 0.0         # historical success


@dataclass
class TrackedWallet:
    """A profitable wallet we're copy trading."""
    address: str
    label: str = ""               # "gmgn_smart_money", "birdeye_top", "pumpfun_winner"
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_hold_minutes: float = 0.0
    last_trade_time: float = 0
    tokens_traded: int = 0
    source: str = ""
    last_checked: float = 0


@dataclass
class TradeMemoryEntry:
    """A single trade's learning data stored in memory."""
    token_address: str
    symbol: str
    action: str                   # "buy" or "sell"
    entry_score: float
    rug_probability: float
    swarm_sentiment: str
    cabal_score: float
    source: str
    outcome: str                  # "win", "loss", "rug", "timeout"
    pnl_multiple: float           # 2.0 = doubled, 0.5 = lost half
    hold_minutes: float
    liquidity_at_entry: float
    age_at_entry: float
    timestamp: float = 0.0
    # v4.0: Rich context for adaptive learning
    entry_velocity: float = 0.0       # buys/min at time of entry
    entry_buys: int = 0               # total buys at entry
    entry_unique_buyers: int = 0      # unique buyer wallets at entry
    entry_curve_progress: float = 0.0 # bonding curve % at entry (0-100)
    on_bonding_curve: bool = False     # was on bonding curve at entry
    sell_reason: str = ""             # detailed sell reason (creator_rug, velocity_death, etc.)
    sol_spent: float = 0.0            # actual SOL spent
    sol_received: float = 0.0         # actual SOL received on sell
    holder_concentration: float = 0.0 # top holder % at entry
    fdv_at_entry: float = 0.0        # FDV at entry


@dataclass
class TraderConfig:
    rpc_url: str = DEFAULT_RPC
    fast_rpc_url: str = ""          # Alchemy for high-volume reads
    helius_rpc_url: str = ""        # v4.2: Helius staked RPC for sendTransaction (better landing)
    helius_api_key: str = ""        # v4.2: Helius API key for priority fee estimation + webhooks
    max_position_sol: float = 0.02   # v4.1: conservative 0.02 SOL until profitable
    max_positions: int = 3           # v4.4: max 3 positions — concentrate on quality over quantity
    min_liquidity: float = 2000.0    # v4.0: raised from 1000 — skip ultra-thin pools
    max_liquidity: float = 150000.0  # v4.0: tightened from 200k
    max_fdv: float = 300000.0        # v4.0: tightened from 500k — focus on true low caps
    min_age_minutes: float = 2.0     # v4.4: raised from 0.5 — <2min tokens had 0% win rate
    max_age_minutes: float = 10.0    # v4.4: tighter — proven tokens only
    min_score: float = 65.0          # v4.4: raised from 50 — only high-quality setups
    scan_interval: int = 5           # v4.4: slower scan — trade less, win more
    slippage_bps: int = 500
    priority_fee_lamports: int = 100000
    reserve_sol: float = 0.05
    whale_wallets: List[str] = field(default_factory=list)
    use_swarm: bool = True
    use_quantum: bool = True        # quantum Monte Carlo for rug detection
    use_pumpfun: bool = True        # pump.fun WebSocket monitoring
    use_wallet_analysis: bool = True  # wallet graph/cabal detection
    cabal_is_bullish: bool = True   # treat coordinated wallets as positive signal
    max_rug_probability: float = 0.20  # v4.4: 20% max — creator_rug was #1 loss cause (6/9 sells)
    # v3: Copy trading
    use_copy_trading: bool = True   # track and copy top wallets
    copy_trade_max_sol: float = 0.02  # v4.1: 0.02 SOL for copy trades
    # v3: X sentinel
    use_x_sentinel: bool = True    # monitor X for cabal signals via Grok
    # v3: Trading memory
    use_trading_memory: bool = True  # learn from past trades
    # v3.5: Bonding curve sniper
    use_bonding_curve: bool = True     # direct pump.fun bonding curve buys
    bonding_curve_max_sol: float = 0.02  # v4.1: 0.02 SOL for sniper buys
    bonding_curve_min_buys: int = 8    # v4.4: raised from 3 — need strong confirmation before buying
    bonding_curve_max_progress: float = 30.0  # v4.4: tighter — don't chase pumped curves
    bonding_curve_min_velocity: float = 3.0  # v4.4: raised from 1.5 — need real sustained momentum
    use_pumpportal: bool = True        # use PumpPortal API for faster execution
    graduation_sell_pct: float = 0.5   # sell 50% at graduation for guaranteed profit
    sniper_mode: bool = False          # v4.4: DISABLED — skipping analysis caused most losses
    # v3.6: Cabal coordination tracking
    use_cabal_follow: bool = True       # follow connected wallets into low-cap tokens
    cabal_follow_max_fdv: float = 80000.0   # v4.0: tighter from 100k — only follow into sub-80k FDV
    cabal_follow_min_wallets: int = 5    # v4.4: raised from 3 — need strong cabal signal
    cabal_follow_max_sol: float = 0.02   # v4.1: 0.02 SOL for cabal follows
    velocity_drop_sell_pct: float = 0.30 # v4.1: tighter from 0.35 — sell even faster on vel death
    # v3.7: Instant snipe on big dev buy / bundle at pool creation
    instant_snipe: bool = False            # v4.4: DISABLED — instant buys on fresh tokens = rug bait
    instant_snipe_min_dev_sol: float = 5.0 # v4.4: raised from 3.0 — only massive dev commits
    instant_snipe_max_sol: float = 0.01    # v4.4: halved — less risk per snipe
    bundle_snipe: bool = False             # v4.4: DISABLED — bundles on fresh tokens too risky
    bundle_min_buys: int = 5               # v4.4: raised from 3 — need stronger bundle signal
    bundle_window_sec: float = 5.0         # time window to detect bundle (coordinated buys)
    bundle_snipe_max_sol: float = 0.01     # v4.4: halved
    # v3.7: Re-entry after velocity dump
    reentry_enabled: bool = True           # watch dumped tokens, re-enter on strength
    reentry_velocity_min: float = 2.0      # v3.8: lowered from 3.0
    reentry_max_sol: float = 0.05          # v4.1: re-entry gets 0.05 SOL (higher conviction)
    reentry_stop_loss: float = 0.25        # tight 25% stop loss on re-entry positions
    reentry_ignore_fdv_cap: bool = True    # re-entry can exceed normal FDV cap if quantum signals strong
    # v3.8: Dynamic scalper — quick in/out profit targets
    quick_take_profit: float = 1.15        # sell at 15% profit — high frequency
    quick_take_profit_2: float = 1.25      # sell remainder at 25%
    max_hold_minutes: float = 15.0         # v4.1: tightened from 20 — don't hold losers
    # v3.8: Dynamic adaptation — auto-adjust thresholds based on on-chain activity
    dynamic_adapt: bool = True             # auto-loosen/tighten based on market conditions
    adapt_quiet_cycles: int = 5            # if this many cycles with 0 qualifying tokens, loosen
    adapt_hot_cycles: int = 3              # if this many cycles with 3+ qualifying, tighten
    # v4.3: Paper trading — simulate all trades without sending real SOL
    paper_trade: bool = True               # DEFAULT ON — no real transactions until explicitly disabled
    paper_start_balance: float = 1.0       # virtual starting SOL for paper trading
    # v4.3: Jupiter API key (free tier at portal.jup.ag — required since 2026)
    jupiter_api_key: str = ""


# ============================================================
# WALLET MANAGEMENT
# ============================================================
def create_wallet(name: str = "degen_trader") -> Tuple[str, str]:
    """Generate a new Solana wallet. Returns (pubkey, wallet_file_path)."""
    try:
        from solders.keypair import Keypair
    except ImportError:
        raise RuntimeError("Install solders: pip install solders")

    kp = Keypair()
    pubkey = str(kp.pubkey())

    WALLET_DIR.mkdir(parents=True, exist_ok=True)
    wallet_path = WALLET_DIR / f"{name}.json"

    keypair_bytes = list(bytes(kp))
    wallet_data = {
        "pubkey": pubkey,
        "keypair": keypair_bytes,
        "created_at": datetime.utcnow().isoformat(),
        "name": name,
    }

    wallet_path.write_text(json.dumps(wallet_data, indent=2))
    logger.info(f"Wallet created: {pubkey}")
    logger.info(f"Wallet file saved to: {wallet_path}")
    return pubkey, str(wallet_path)


def load_wallet(name: str = "degen_trader"):
    """Load wallet keypair from file."""
    try:
        from solders.keypair import Keypair
    except ImportError:
        raise RuntimeError("Install solders: pip install solders")

    wallet_path = WALLET_DIR / f"{name}.json"
    if not wallet_path.exists():
        raise FileNotFoundError(f"No wallet at {wallet_path}. Run create_wallet() first.")

    data = json.loads(wallet_path.read_text())
    return Keypair.from_bytes(bytes(data["keypair"]))


# ============================================================
# PUMP.FUN MONITOR (v3.5 - Enhanced for bonding curve sniping)
# ============================================================
class PumpFunMonitor:
    """Real-time multi-launchpad monitoring via PumpPortal WebSocket.

    Monitors Pump.fun, Bonk (LetsBonk.fun), and BAGS (bags.fm) for
    new token launches with buy velocity tracking, unique buyer counting,
    creator activity monitoring, and sniper signal generation.

    PumpPortal WS delivers events from both pump.fun and bonk natively.
    BAGS is monitored via periodic API polling.
    """

    def __init__(self):
        self.ws = None
        self.new_tokens: asyncio.Queue = asyncio.Queue(maxsize=500)
        self.hot_tokens: Dict[str, dict] = {}  # mint -> detailed trade stats
        self.sniper_signals: asyncio.Queue = asyncio.Queue(maxsize=100)  # high-priority buys
        self.cabal_signals: asyncio.Queue = asyncio.Queue(maxsize=100)   # cabal coordination buys
        self.running = False
        self._task = None
        self._bags_task = None
        self._tracked_creators: Dict[str, List[str]] = {}  # creator -> [mints they made]
        # v3.6: Track buyer wallets per token for cabal coordination detection
        self._wallet_token_buys: Dict[str, Set[str]] = {}  # wallet -> set of mints they bought
        self._token_buyer_wallets: Dict[str, Set[str]] = {}  # mint -> set of buyer wallets
        self._cabal_signaled: Set[str] = set()  # mints already signaled as cabal buys
        # v3.7: Instant snipe signals for big dev buys / bundles at launch
        self.instant_snipe_signals: asyncio.Queue = asyncio.Queue(maxsize=50)
        self.bundle_signals: asyncio.Queue = asyncio.Queue(maxsize=50)
        self._bundle_detected: Set[str] = set()  # mints already bundle-signaled
        self._sniper_signaled: Set[str] = set()  # v3.9: dedup — only emit sniper signal once per token
        self._reconnect_count = 0
        # Platform stats
        self.platform_counts = {PLATFORM_PUMP: 0, PLATFORM_BONK: 0, PLATFORM_BAGS: 0}
        self.sniper_history: List[dict] = []  # last N sniper signals for dashboard

    async def start(self):
        """Connect to PumpPortal WebSocket and start multi-platform monitoring."""
        self.running = True
        self._task = asyncio.create_task(self._listen())
        self._bags_task = asyncio.create_task(self._poll_bags())
        logger.info("Multi-launchpad monitor started (Pump.fun + Bonk + BAGS)")

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
        if self._bags_task:
            self._bags_task.cancel()
        if self.ws:
            await self.ws.close()

    async def _listen(self):
        try:
            import websockets
        except ImportError:
            logger.warning("websockets not installed, pump.fun monitor disabled")
            return

        while self.running:
            try:
                async with websockets.connect(
                    PUMPFUN_WS_URL,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    self.ws = ws
                    self._reconnect_count += 1
                    # Subscribe to new token creates
                    await ws.send(json.dumps({"method": "subscribeNewToken"}))
                    self._pending_trade_subs = asyncio.Queue(maxsize=200)
                    logger.info(f"PumpPortal WS connected (attempt #{self._reconnect_count}), subscribed to newToken")
                    # v3.8: Start background task to subscribe to trades for new tokens
                    asyncio.create_task(self._trade_subscriber(ws))

                    async for msg in ws:
                        if not self.running:
                            break
                        try:
                            data = json.loads(msg)
                            await self._handle_message(data)
                        except json.JSONDecodeError:
                            continue

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"PumpFun WS error: {e}, reconnecting in 2s... (reconnects: {self._reconnect_count})")
                await asyncio.sleep(2)

    async def _trade_subscriber(self, ws):
        """v3.8 FIX: Subscribe to trades for new tokens as they appear.

        PumpPortal requires explicit subscribeTokenTrade per mint.
        We batch-subscribe in groups of up to 10 to avoid spamming.
        """
        _subscribed = set()
        while self.running:
            try:
                # Collect mints to subscribe to (batch up to 10)
                mints_to_sub = []
                while len(mints_to_sub) < 10:
                    try:
                        mint = self._pending_trade_subs.get_nowait()
                        if mint not in _subscribed:
                            mints_to_sub.append(mint)
                    except asyncio.QueueEmpty:
                        break

                if mints_to_sub:
                    try:
                        await ws.send(json.dumps({
                            "method": "subscribeTokenTrade",
                            "keys": mints_to_sub,
                        }))
                        _subscribed.update(mints_to_sub)
                    except Exception as e:
                        logger.debug(f"Trade subscribe error: {e}")

                # Prune old subscriptions (keep max 200 active)
                if len(_subscribed) > 200:
                    # Unsubscribe oldest by keeping only hot_tokens
                    stale = _subscribed - set(self.hot_tokens.keys())
                    if stale:
                        try:
                            await ws.send(json.dumps({
                                "method": "unsubscribeTokenTrade",
                                "keys": list(stale)[:50],
                            }))
                            _subscribed -= stale
                        except Exception:
                            pass

                await asyncio.sleep(0.5)  # check every 500ms
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(1)

    def _detect_platform(self, data: dict) -> str:
        """Detect which launchpad a PumpPortal event came from."""
        # PumpPortal includes pool field in newer events
        pool = data.get("pool", "")
        if pool == "bonk" or pool == "launchlab":
            return PLATFORM_BONK
        if pool == "pump":
            return PLATFORM_PUMP
        # Check program ID for fallback detection
        program = data.get("programId", "")
        if program == BONK_LAUNCHLAB_PROGRAM:
            return PLATFORM_BONK
        # Default to pump.fun
        return PLATFORM_PUMP

    async def _handle_message(self, data: dict):
        """Process PumpPortal WebSocket messages (Pump.fun + Bonk)."""
        if data.get("txType") == "create":
            # New token created on launchpad
            platform = self._detect_platform(data)
            creator = data.get("traderPublicKey", "")
            mint = data.get("mint", "")
            initial_sol = data.get("solAmount", 0) / LAMPORTS_PER_SOL if data.get("solAmount") else 0
            platform_label = "PUMP" if platform == PLATFORM_PUMP else "BONK" if platform == PLATFORM_BONK else "BAGS"
            token_data = {
                "mint": mint,
                "name": data.get("name", ""),
                "symbol": data.get("symbol", ""),
                "creator": creator,
                "initial_buy_sol": initial_sol,
                "timestamp": time.time(),
                "source": "bonding_curve",
                "on_bonding_curve": True,
                "platform": platform,
            }
            if mint:
                self.platform_counts[platform] = self.platform_counts.get(platform, 0) + 1
                # Track this token's trade stats from birth
                self.hot_tokens[mint] = {
                    "buys": 1 if initial_sol > 0 else 0,
                    "sells": 0,
                    "volume_sol": initial_sol,
                    "first_seen": time.time(),
                    "unique_buyers": {creator} if initial_sol > 0 else set(),
                    "creator": creator,
                    "creator_bought": initial_sol > 0,
                    "creator_sol": initial_sol,
                    "buy_timestamps": [time.time()] if initial_sol > 0 else [],
                    "symbol": data.get("symbol", ""),
                    "name": data.get("name", ""),
                    "largest_buy_sol": initial_sol,
                    "platform": platform,
                }
                # Track creator history
                if creator:
                    if creator not in self._tracked_creators:
                        self._tracked_creators[creator] = []
                    self._tracked_creators[creator].append(mint)

                try:
                    self.new_tokens.put_nowait(token_data)
                except asyncio.QueueFull:
                    self.new_tokens.get_nowait()
                    self.new_tokens.put_nowait(token_data)
                logger.info(f"[{platform_label}] NEW: ${token_data['symbol']} by {creator[:8]}... (dev buy: {initial_sol:.3f} SOL)")

                # v3.8 FIX: Subscribe to this token's trades so we get buy/sell events
                if hasattr(self, '_pending_trade_subs'):
                    try:
                        self._pending_trade_subs.put_nowait(mint)
                    except asyncio.QueueFull:
                        pass  # will catch on next batch

                # v3.7: Instant snipe on big dev buy at pool creation
                # If dev launches with significant SOL, snipe immediately before others pile in
                if initial_sol >= 7.0:  # HIGH CONVICTION: dev put 7+ SOL in at launch
                    instant_signal = {
                        "mint": mint,
                        "symbol": data.get("symbol", ""),
                        "name": data.get("name", ""),
                        "buys": 1,
                        "sells": 0,
                        "unique_buyers": 1,
                        "velocity": 0,
                        "volume_sol": initial_sol,
                        "age_seconds": 0,
                        "creator": creator,
                        "creator_bought": True,
                        "creator_sol": initial_sol,
                        "largest_buy_sol": initial_sol,
                        "timestamp": time.time(),
                        "platform": platform,
                        "instant_snipe": True,
                        "dev_buy_sol": initial_sol,
                    }
                    try:
                        self.instant_snipe_signals.put_nowait(instant_signal)
                    except asyncio.QueueFull:
                        self.instant_snipe_signals.get_nowait()
                        self.instant_snipe_signals.put_nowait(instant_signal)
                    logger.info(f"[{platform_label}] INSTANT SNIPE SIGNAL: ${data.get('symbol', '?')} dev buy {initial_sol:.3f} SOL — sniping before others!")

        elif data.get("txType") in ("buy", "sell"):
            mint = data.get("mint", "")
            trader = data.get("traderPublicKey", "")
            if mint:
                if mint not in self.hot_tokens:
                    platform = self._detect_platform(data)
                    self.hot_tokens[mint] = {
                        "buys": 0, "sells": 0, "volume_sol": 0,
                        "first_seen": time.time(), "unique_buyers": set(),
                        "creator": "", "creator_bought": False, "creator_sol": 0,
                        "buy_timestamps": [], "symbol": "", "name": "",
                        "largest_buy_sol": 0, "platform": platform,
                    }
                stats = self.hot_tokens[mint]
                sol_amount = data.get("solAmount", 0) / LAMPORTS_PER_SOL if data.get("solAmount") else 0

                # v4.3: Capture real-time price data from PumpPortal trade events
                # These fields give us instant pricing without extra API calls
                market_cap_sol = data.get("marketCapSol", 0)
                if market_cap_sol:
                    stats["market_cap_sol"] = market_cap_sol
                v_sol = data.get("vSolInBondingCurve", 0)
                v_tokens = data.get("vTokensInBondingCurve", 0)
                if v_sol and v_tokens:
                    stats["v_sol"] = v_sol
                    stats["v_tokens"] = v_tokens
                    # Compute real-time price per token in SOL from curve reserves
                    stats["price_sol"] = (v_sol / 1e9) / (v_tokens / 1e6) if v_tokens > 0 else 0
                    stats["price_updated"] = time.time()
                token_amount = data.get("tokenAmount", 0)
                if token_amount and sol_amount > 0:
                    # Implied trade price (backup if curve data missing)
                    stats["last_trade_price_sol"] = sol_amount / (token_amount / 1e6) if token_amount > 0 else 0

                if data["txType"] == "buy":
                    stats["buys"] += 1
                    stats["unique_buyers"].add(trader)
                    stats["buy_timestamps"].append(time.time())
                    if sol_amount > stats.get("largest_buy_sol", 0):
                        stats["largest_buy_sol"] = sol_amount
                    # Check if creator is buying more (bullish signal)
                    if trader == stats.get("creator") and stats["buys"] > 1:
                        stats["creator_bought"] = True
                        stats["creator_sol"] += sol_amount
                    # v3.6: Track wallet→token buys for cabal coordination detection
                    if trader:
                        if trader not in self._wallet_token_buys:
                            self._wallet_token_buys[trader] = set()
                        self._wallet_token_buys[trader].add(mint)
                        if mint not in self._token_buyer_wallets:
                            self._token_buyer_wallets[mint] = set()
                        self._token_buyer_wallets[mint].add(trader)
                else:
                    stats["sells"] += 1
                    # Creator selling = bearish, could be rug
                    if trader == stats.get("creator"):
                        stats["creator_sold"] = True
                stats["volume_sol"] += sol_amount

                # Check sniper signal: fast buys from multiple unique wallets
                self._check_sniper_signal(mint, stats)
                # v3.6: Check cabal coordination signal
                if data["txType"] == "buy":
                    self._check_cabal_coordination(mint, stats)

                # v3.7: Bundle detection — multiple buys within seconds = coordinated
                if data["txType"] == "buy" and mint not in self._bundle_detected:
                    ts_list = stats.get("buy_timestamps", [])
                    now_t = time.time()
                    if len(ts_list) >= 3:
                        # Count buys in last 5 seconds
                        recent = [t for t in ts_list if now_t - t <= 5.0]
                        if len(recent) >= 3:
                            unique_recent = len({b for b in stats.get("unique_buyers", set())})
                            age_s = now_t - stats.get("first_seen", now_t)
                            if age_s < 60 and unique_recent >= 2:  # fresh + multiple wallets
                                platform = stats.get("platform", PLATFORM_PUMP)
                                platform_label = "PUMP" if platform == PLATFORM_PUMP else "BONK" if platform == PLATFORM_BONK else "BAGS"
                                bundle_sig = {
                                    "mint": mint,
                                    "symbol": stats.get("symbol", ""),
                                    "name": stats.get("name", ""),
                                    "buys": stats["buys"],
                                    "sells": stats["sells"],
                                    "unique_buyers": unique_recent,
                                    "velocity": stats["buys"] / (age_s / 60) if age_s > 0 else 0,
                                    "volume_sol": stats["volume_sol"],
                                    "age_seconds": age_s,
                                    "creator": stats.get("creator", ""),
                                    "largest_buy_sol": stats.get("largest_buy_sol", 0),
                                    "timestamp": now_t,
                                    "platform": platform,
                                    "bundle_buys": len(recent),
                                    "dev_buy_sol": 0,
                                    "instant_snipe": True,
                                }
                                try:
                                    self.bundle_signals.put_nowait(bundle_sig)
                                except asyncio.QueueFull:
                                    self.bundle_signals.get_nowait()
                                    self.bundle_signals.put_nowait(bundle_sig)
                                self._bundle_detected.add(mint)
                                logger.info(
                                    f"[{platform_label}] BUNDLE DETECTED: ${stats.get('symbol', '?')} | "
                                    f"{len(recent)} buys in 5s | {unique_recent} unique wallets | "
                                    f"age {age_s:.0f}s — SNIPING"
                                )

                # v3.7: Instant snipe — big buy on a very fresh token (< 15 seconds old)
                # Catches dev buys that come as separate buy txns after create
                if data["txType"] == "buy" and sol_amount >= 7.0 and mint not in self._cabal_signaled:
                    age_s = time.time() - stats.get("first_seen", time.time())
                    if age_s < 15 and stats["buys"] <= 3:
                        platform = stats.get("platform", PLATFORM_PUMP)
                        platform_label = "PUMP" if platform == PLATFORM_PUMP else "BONK" if platform == PLATFORM_BONK else "BAGS"
                        instant_signal = {
                            "mint": mint,
                            "symbol": stats.get("symbol", ""),
                            "name": stats.get("name", ""),
                            "buys": stats["buys"],
                            "sells": stats["sells"],
                            "unique_buyers": len(stats.get("unique_buyers", set())),
                            "velocity": 0,
                            "volume_sol": stats["volume_sol"],
                            "age_seconds": age_s,
                            "creator": stats.get("creator", ""),
                            "creator_bought": trader == stats.get("creator", ""),
                            "creator_sol": sol_amount,
                            "largest_buy_sol": sol_amount,
                            "timestamp": time.time(),
                            "platform": platform,
                            "instant_snipe": True,
                            "dev_buy_sol": sol_amount,
                        }
                        try:
                            self.instant_snipe_signals.put_nowait(instant_signal)
                        except asyncio.QueueFull:
                            self.instant_snipe_signals.get_nowait()
                            self.instant_snipe_signals.put_nowait(instant_signal)
                        logger.info(
                            f"[{platform_label}] INSTANT SNIPE SIGNAL: ${stats.get('symbol', '?')} | "
                            f"big buy {sol_amount:.3f} SOL at {age_s:.0f}s age — sniping NOW"
                        )

        # Cleanup old hot tokens (older than 30 min)
        cutoff = time.time() - 1800
        expired_mints = {k for k, v in self.hot_tokens.items() if v.get("first_seen", 0) <= cutoff}
        self.hot_tokens = {k: v for k, v in self.hot_tokens.items() if k not in expired_mints}
        # Cleanup wallet tracking for expired tokens
        if expired_mints:
            for mint in expired_mints:
                self._token_buyer_wallets.pop(mint, None)
                self._cabal_signaled.discard(mint)
                self._sniper_signaled.discard(mint)
            for wallet in list(self._wallet_token_buys):
                self._wallet_token_buys[wallet] -= expired_mints
                if not self._wallet_token_buys[wallet]:
                    del self._wallet_token_buys[wallet]
        # Cleanup old creator tracking (keep last 100)
        if len(self._tracked_creators) > 500:
            oldest = sorted(self._tracked_creators.keys(), key=lambda c: len(self._tracked_creators[c]))[:250]
            for c in oldest:
                del self._tracked_creators[c]

    def _check_sniper_signal(self, mint: str, stats: dict):
        """Emit a sniper signal if token shows strong early momentum."""
        # v3.9: Dedup — only emit once per token
        if mint in self._sniper_signaled:
            return
        age_seconds = time.time() - stats.get("first_seen", time.time())
        if age_seconds < 5 or age_seconds > 300:  # 5s-5min window
            return

        buys = stats["buys"]
        sells = stats["sells"]
        unique = len(stats.get("unique_buyers", set()))
        velocity = (buys / (age_seconds / 60)) if age_seconds > 0 else 0
        creator_sold = stats.get("creator_sold", False)

        # Sniper criteria: v3.8 loosened — 2 unique buyers, velocity >= 1.0
        platform = stats.get("platform", PLATFORM_PUMP)
        if (buys >= 2 and unique >= 2 and velocity >= 1.0
                and sells <= buys * 0.4 and not creator_sold):
            platform_label = "PUMP" if platform == PLATFORM_PUMP else "BONK" if platform == PLATFORM_BONK else "BAGS"
            signal = {
                "mint": mint,
                "symbol": stats.get("symbol", ""),
                "name": stats.get("name", ""),
                "buys": buys,
                "sells": sells,
                "unique_buyers": unique,
                "velocity": velocity,
                "volume_sol": stats["volume_sol"],
                "age_seconds": age_seconds,
                "creator": stats.get("creator", ""),
                "creator_bought": stats.get("creator_bought", False),
                "creator_sol": stats.get("creator_sol", 0),
                "largest_buy_sol": stats.get("largest_buy_sol", 0),
                "timestamp": time.time(),
                "platform": platform,
            }
            try:
                self.sniper_signals.put_nowait(signal)
            except asyncio.QueueFull:
                self.sniper_signals.get_nowait()
                self.sniper_signals.put_nowait(signal)
            self._sniper_signaled.add(mint)  # v3.9: dedup — don't fire again for same token
            # Keep last 50 signals for dashboard
            self.sniper_history.append(signal)
            if len(self.sniper_history) > 50:
                self.sniper_history = self.sniper_history[-50:]
            logger.info(
                f"[{platform_label}] SNIPER: ${signal['symbol']} | {buys} buys ({unique} unique) | "
                f"{velocity:.1f}/min | {stats['volume_sol']:.2f} SOL vol | age {age_seconds:.0f}s"
            )

    def _check_cabal_coordination(self, mint: str, stats: dict):
        """Detect when connected wallets (sharing buy history) converge on one token.

        If multiple wallets that have been buying the same OTHER tokens now buy THIS token,
        it's likely a coordinated cabal play. Emit a cabal signal for low-cap tokens.
        """
        if mint in self._cabal_signaled:
            return

        buyers = self._token_buyer_wallets.get(mint, set())
        if len(buyers) < 2:
            return

        # Check how many buyer wallets share purchases in OTHER tokens
        # (wallets buying same tokens = likely connected/coordinated)
        buyer_list = list(buyers)
        connected_pairs = 0
        connected_wallets = set()

        for i in range(len(buyer_list)):
            w1_tokens = self._wallet_token_buys.get(buyer_list[i], set())
            for j in range(i + 1, min(len(buyer_list), i + 10)):  # cap comparisons
                w2_tokens = self._wallet_token_buys.get(buyer_list[j], set())
                # Shared buys in OTHER tokens (excluding this mint)
                shared = (w1_tokens & w2_tokens) - {mint}
                if len(shared) >= 1:  # bought at least 1 other token in common
                    connected_pairs += 1
                    connected_wallets.add(buyer_list[i])
                    connected_wallets.add(buyer_list[j])

        # Signal if enough connected wallets are converging (min 3 to avoid noise)
        if len(connected_wallets) >= 3 and connected_pairs >= 2:
            age_seconds = time.time() - stats.get("first_seen", time.time())
            velocity = stats["buys"] / (age_seconds / 60) if age_seconds > 0 else 0
            platform = stats.get("platform", PLATFORM_PUMP)
            signal = {
                "mint": mint,
                "symbol": stats.get("symbol", ""),
                "name": stats.get("name", ""),
                "connected_wallets": len(connected_wallets),
                "connected_pairs": connected_pairs,
                "total_buyers": len(buyers),
                "buys": stats["buys"],
                "velocity": velocity,
                "volume_sol": stats["volume_sol"],
                "age_seconds": age_seconds,
                "creator": stats.get("creator", ""),
                "platform": platform,
                "timestamp": time.time(),
            }
            try:
                self.cabal_signals.put_nowait(signal)
            except asyncio.QueueFull:
                self.cabal_signals.get_nowait()
                self.cabal_signals.put_nowait(signal)
            self._cabal_signaled.add(mint)
            logger.info(
                f"CABAL SIGNAL: ${signal['symbol']} | {len(connected_wallets)} connected wallets "
                f"({connected_pairs} pairs) | {stats['buys']} buys | {velocity:.1f}/min | "
                f"{stats['volume_sol']:.2f} SOL vol"
            )

    def get_buy_velocity(self, mint: str) -> float:
        """Get current buy velocity (buys per minute) for a token."""
        stats = self.hot_tokens.get(mint)
        if not stats:
            return 0
        age_seconds = time.time() - stats.get("first_seen", time.time())
        if age_seconds <= 0:
            return 0
        return stats["buys"] / (age_seconds / 60)

    def get_token_stats(self, mint: str) -> Optional[dict]:
        """Get detailed stats for a token."""
        stats = self.hot_tokens.get(mint)
        if not stats:
            return None
        age_seconds = time.time() - stats.get("first_seen", time.time())
        return {
            "buys": stats["buys"],
            "sells": stats["sells"],
            "unique_buyers": len(stats.get("unique_buyers", set())),
            "volume_sol": stats["volume_sol"],
            "velocity": stats["buys"] / (age_seconds / 60) if age_seconds > 0 else 0,
            "age_seconds": age_seconds,
            "creator_bought": stats.get("creator_bought", False),
            "creator_sold": stats.get("creator_sold", False),
            "largest_buy_sol": stats.get("largest_buy_sol", 0),
        }

    def is_serial_deployer(self, creator: str) -> bool:
        """Check if creator has deployed multiple tokens recently (rug signal)."""
        mints = self._tracked_creators.get(creator, [])
        return len(mints) > 2  # 3+ tokens in 30min = serial deployer

    async def _poll_bags(self):
        """Poll BAGS (bags.fm) API for new token launches."""
        import aiohttp as _aio
        session = None
        while self.running:
            try:
                if not session:
                    bags_key = os.environ.get("BAGS_API_KEY", "")
                    headers = {"x-api-key": bags_key} if bags_key else {}
                    session = _aio.ClientSession(headers=headers)

                # Poll bags.fm for recently created tokens
                url = f"{BAGS_API_URL}/tokens?sort=created&order=desc&limit=10"
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        tokens = data if isinstance(data, list) else data.get("data", data.get("tokens", []))
                        for item in tokens[:5]:
                            mint = item.get("mint", item.get("tokenMint", ""))
                            if not mint or mint in self.hot_tokens:
                                continue
                            symbol = item.get("symbol", "")
                            name = item.get("name", "")
                            creator = item.get("creator", item.get("creatorAddress", ""))
                            self.platform_counts[PLATFORM_BAGS] = self.platform_counts.get(PLATFORM_BAGS, 0) + 1
                            token_data = {
                                "mint": mint, "name": name, "symbol": symbol,
                                "creator": creator, "initial_buy_sol": 0,
                                "timestamp": time.time(), "source": "bonding_curve",
                                "on_bonding_curve": True, "platform": PLATFORM_BAGS,
                            }
                            self.hot_tokens[mint] = {
                                "buys": 0, "sells": 0, "volume_sol": 0,
                                "first_seen": time.time(), "unique_buyers": set(),
                                "creator": creator, "creator_bought": False, "creator_sol": 0,
                                "buy_timestamps": [], "symbol": symbol, "name": name,
                                "largest_buy_sol": 0, "platform": PLATFORM_BAGS,
                            }
                            try:
                                self.new_tokens.put_nowait(token_data)
                            except asyncio.QueueFull:
                                self.new_tokens.get_nowait()
                                self.new_tokens.put_nowait(token_data)
                            logger.info(f"[BAGS] NEW: ${symbol} by {creator[:8] if creator else '?'}...")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"BAGS poll error: {e}")
            await asyncio.sleep(30)  # poll every 30s

        if session:
            await session.close()

    def get_sniper_feed(self, max_age_seconds: float = 900) -> List[dict]:
        """Get recent sniper activity for dashboard - fresh launches only."""
        now = time.time()
        fresh = [s for s in self.sniper_history if now - s.get("timestamp", 0) < max_age_seconds]
        return list(reversed(fresh[-20:]))

    def get_cabal_feed(self, max_age_seconds: float = 900) -> List[dict]:
        """Get recent cabal coordination signals for dashboard - fresh only."""
        now = time.time()
        result = []
        for mint in list(self._cabal_signaled):
            stats = self.hot_tokens.get(mint)
            if not stats:
                continue
            age = now - stats.get("first_seen", now)
            if age > max_age_seconds:
                continue
            buyers = self._token_buyer_wallets.get(mint, set())
            velocity = stats["buys"] / (age / 60) if age > 0 else 0
            result.append({
                "mint": mint,
                "symbol": stats.get("symbol", ""),
                "connected_wallets": len(buyers),
                "buys": stats["buys"],
                "sells": stats["sells"],
                "velocity": velocity,
                "volume_sol": stats["volume_sol"],
                "age_seconds": age,
                "platform": stats.get("platform", PLATFORM_PUMP),
            })
        return result[:15]


# ============================================================
# BONDING CURVE ENGINE (v3.5 - Direct pump.fun trading)
# ============================================================
class BondingCurveEngine:
    """Direct trading on pump.fun bonding curves.

    Bypasses Jupiter/Raydium entirely for pre-graduation tokens.
    Uses PumpPortal local transaction API for speed, with direct
    on-chain instruction fallback.
    """

    def __init__(self, rpc_url: str, fast_rpc_url: str = "", send_rpc_url: str = ""):
        self.rpc_url = rpc_url
        self.fast_rpc_url = fast_rpc_url or rpc_url
        self.send_rpc_url = send_rpc_url or self.fast_rpc_url  # v4.2: Helius staked for sends
        self._curve_cache: Dict[str, Tuple[BondingCurveState, float]] = {}  # mint -> (state, timestamp)
        self._fee_idx = 0

    def _next_fee_recipient(self) -> str:
        """Round-robin fee recipient selection."""
        recipient = PUMP_FEE_RECIPIENTS[self._fee_idx % len(PUMP_FEE_RECIPIENTS)]
        self._fee_idx += 1
        return recipient

    async def get_bonding_curve_state(self, mint: str, session: aiohttp.ClientSession) -> Optional[BondingCurveState]:
        """Fetch and parse the bonding curve account for a token."""
        # Check cache (5 second TTL)
        cached = self._curve_cache.get(mint)
        if cached and time.time() - cached[1] < 5:
            return cached[0]

        try:
            # Derive bonding curve PDA
            from solders.pubkey import Pubkey
            mint_pk = Pubkey.from_string(mint)
            program_pk = Pubkey.from_string(PUMP_PROGRAM_ID)
            curve_pk, _ = Pubkey.find_program_address(
                [b"bonding-curve", bytes(mint_pk)], program_pk
            )

            # Fetch account data via RPC
            payload = {
                "jsonrpc": "2.0", "id": 1,
                "method": "getAccountInfo",
                "params": [str(curve_pk), {"encoding": "base64"}]
            }
            async with session.post(self.fast_rpc_url, json=payload) as resp:
                result = await resp.json()

            account = result.get("result", {}).get("value")
            if not account or not account.get("data"):
                return None

            data_b64 = account["data"][0]
            data = base64.b64decode(data_b64)

            if len(data) < 49:  # 8 discriminator + 5*8 fields + 1 bool
                return None

            # Parse bonding curve layout
            state = BondingCurveState(
                virtual_token_reserves=struct.unpack_from("<Q", data, 8)[0],
                virtual_sol_reserves=struct.unpack_from("<Q", data, 16)[0],
                real_token_reserves=struct.unpack_from("<Q", data, 24)[0],
                real_sol_reserves=struct.unpack_from("<Q", data, 32)[0],
                token_total_supply=struct.unpack_from("<Q", data, 40)[0],
                complete=bool(data[48]),
            )

            self._curve_cache[mint] = (state, time.time())
            return state

        except ImportError:
            logger.error("pip install solders for bonding curve trading")
            return None
        except Exception as e:
            logger.debug(f"Bonding curve fetch error for {mint}: {e}")
            return None

    async def buy_on_curve_pumpportal(
        self, mint: str, sol_amount: float, pubkey: str, keypair,
        session: aiohttp.ClientSession, slippage: int = 15, priority_fee: float = 0.005,
        pool: str = "pump",
    ) -> Optional[str]:
        """Buy a token on the bonding curve via PumpPortal local API.

        Supports multiple launchpads: pool="pump" (pump.fun), pool="bonk" (letsbonk.fun),
        pool="auto" (auto-detect). PumpPortal handles the transaction building.
        """
        try:
            from solders.transaction import VersionedTransaction

            # Get unsigned transaction from PumpPortal
            payload = {
                "publicKey": pubkey,
                "action": "buy",
                "mint": mint,
                "amount": sol_amount,
                "denominatedInSol": "true",
                "slippage": slippage,
                "priorityFee": priority_fee,
                "pool": pool,
            }
            async with session.post(PUMPPORTAL_LOCAL_API, json=payload) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    logger.error(f"PumpPortal buy error: {err}")
                    return None
                tx_bytes = await resp.read()

            if not tx_bytes or len(tx_bytes) < 10:
                logger.error("PumpPortal returned empty transaction")
                return None

            # Deserialize, sign, and send
            tx = VersionedTransaction.from_bytes(tx_bytes)
            signed_tx = VersionedTransaction(tx.message, [keypair])
            signed_b64 = base64.b64encode(bytes(signed_tx)).decode("ascii")

            send_payload = {
                "jsonrpc": "2.0", "id": 1, "method": "sendTransaction",
                "params": [signed_b64, {"encoding": "base64", "skipPreflight": True, "maxRetries": 2}]
            }
            # v4.2: Send via Helius staked RPC for better landing rate
            rpc = self.send_rpc_url
            async with session.post(rpc, json=send_payload) as resp:
                result = await resp.json()

            if "error" in result:
                logger.error(f"Bonding curve TX error: {result['error']}")
                return None

            tx_sig = result.get("result", "")
            if tx_sig:
                logger.info(f"BONDING CURVE BUY: {mint[:12]}... | {sol_amount:.4f} SOL | tx={tx_sig[:20]}...")
            return tx_sig

        except ImportError:
            logger.error("pip install solders for bonding curve trading")
            return None
        except Exception as e:
            logger.error(f"PumpPortal buy error: {e}")
            return None

    async def sell_on_curve_pumpportal(
        self, mint: str, token_amount_pct: float, pubkey: str, keypair,
        session: aiohttp.ClientSession, slippage: int = 15, priority_fee: float = 0.005,
        pool: str = "pump",
    ) -> Optional[str]:
        """Sell tokens on the bonding curve via PumpPortal.

        Supports pool="pump", pool="bonk", pool="auto".
        token_amount_pct: fraction to sell (1.0 = all, 0.5 = half)
        """
        try:
            from solders.transaction import VersionedTransaction

            # First get our token balance
            balance = await self._get_token_balance(mint, pubkey, session)
            if balance <= 0:
                return None

            sell_amount = int(balance * token_amount_pct)
            if sell_amount <= 0:
                return None

            payload = {
                "publicKey": pubkey,
                "action": "sell",
                "mint": mint,
                "amount": sell_amount,
                "denominatedInSol": "false",
                "slippage": slippage,
                "priorityFee": priority_fee,
                "pool": pool,
            }
            async with session.post(PUMPPORTAL_LOCAL_API, json=payload) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    logger.error(f"PumpPortal sell error: {err}")
                    return None
                tx_bytes = await resp.read()

            if not tx_bytes or len(tx_bytes) < 10:
                return None

            tx = VersionedTransaction.from_bytes(tx_bytes)
            signed_tx = VersionedTransaction(tx.message, [keypair])
            signed_b64 = base64.b64encode(bytes(signed_tx)).decode("ascii")

            send_payload = {
                "jsonrpc": "2.0", "id": 1, "method": "sendTransaction",
                "params": [signed_b64, {"encoding": "base64", "skipPreflight": True, "maxRetries": 2}]
            }
            # v4.2: Send via Helius staked RPC for better landing rate
            async with session.post(self.send_rpc_url, json=send_payload) as resp:
                result = await resp.json()

            if "error" in result:
                logger.error(f"Bonding curve sell TX error: {result['error']}")
                return None

            tx_sig = result.get("result", "")
            if tx_sig:
                logger.info(f"BONDING CURVE SELL: {mint[:12]}... | {token_amount_pct:.0%} | tx={tx_sig[:20]}...")
            return tx_sig

        except Exception as e:
            logger.error(f"PumpPortal sell error: {e}")
            return None

    async def _get_token_balance(self, mint: str, owner: str, session: aiohttp.ClientSession) -> int:
        """Get raw token balance for owner."""
        try:
            payload = {
                "jsonrpc": "2.0", "id": 1,
                "method": "getTokenAccountsByOwner",
                "params": [owner, {"mint": mint}, {"encoding": "jsonParsed"}]
            }
            async with session.post(self.fast_rpc_url, json=payload) as resp:
                data = await resp.json()
            accounts = data.get("result", {}).get("value", [])
            if accounts:
                info = accounts[0].get("account", {}).get("data", {}).get("parsed", {}).get("info", {})
                return int(info.get("tokenAmount", {}).get("amount", 0))
        except Exception as e:
            logger.debug(f"Token balance error: {e}")
        return 0

    def is_pre_graduation(self, state: BondingCurveState) -> bool:
        """Check if token is still on bonding curve (hasn't graduated)."""
        return not state.complete and state.real_token_reserves > 0

    def estimate_graduation_time(self, state: BondingCurveState, velocity_sol_per_min: float) -> float:
        """Estimate minutes until graduation based on current buy velocity."""
        remaining_sol = PUMP_GRADUATION_SOL - state.sol_raised
        if remaining_sol <= 0 or velocity_sol_per_min <= 0:
            return 0
        return remaining_sol / velocity_sol_per_min


# ============================================================
# HOLDER WATCHER — Real-time top holder rug detection (v3.9)
# ============================================================
# Known safe token lock / vesting contracts on Solana
# If tokens move to these programs, it's NOT a rug — it's a lock
SAFE_LOCK_PROGRAMS = {
    "strmRqUCoQUgGUan5YhzUZa6KqdzwX5L6FpUxfmKg5m",  # Streamflow Finance
    "8e72pYCDaxu3GqMfeQ5r8wFgoZSYk6oua1Qo9XpsZjX",  # Streamflow Community
    "LocpQgucEQHbqNABEYvBvwoxCPsSbG91A1QaQhQQqjn",  # Jupiter Lock
    "CChTq6PthWU82YZkbveA3WDf7s97BWhBK4Vx9bmsT743",  # Bonfida Token Vesting
    "DRay25Usp3YJAi7beckgpGUC7mGJ2cR1AVPxhYfwVCUX",  # Raydium Burn & Earn (LP locker)
    "11111111111111111111111111111111",                  # System Program (burn)
    "1111111111111111111111111111111111111111111",        # Burn address variant
    "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P",    # Pump.fun program (bonding curve holds)
    "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA",    # PumpSwap AMM (migration target)
    "39azUYFWPz3VHgKCf3VChUwbpURdCHRxjWVowf5jUJjg",   # Pump.fun Migration Authority
}

# Known DEX program IDs — if tokens go here, the holder is SELLING
DEX_SELL_PROGRAMS = {
    "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",  # Raydium AMM v4
    "CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C",  # Raydium CPMM
    "CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK",  # Raydium CLMM
    "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",   # Jupiter v6
    "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc",   # Orca Whirlpool
    "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP",  # Orca v2
    "srmqPvymJeFKQ4zGQed1GFppgkRHL9kaELCbyksJtPX",    # Serum DEX
}

# Known burn / incinerator addresses on Solana — tokens sent here are destroyed permanently
BURN_ADDRESSES = {
    "1nc1nerator11111111111111111111111111111111",    # Solana Incinerator (most common intentional burn)
    "11111111111111111111111111111111",                # System Program (acts as burn)
}

# SPL Token program IDs (for identifying token transfer instructions)
SPL_TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
SPL_TOKEN_2022 = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"


# ============================================================
# BURN SIGNAL MONITOR — Auto-detect LP burns & supply burns (v4.0)
# ============================================================
class BurnSignalMonitor:
    """Monitor Solana burn addresses for LP token burns and supply burns.

    When tokens are burned (LP or supply), it's a strong bullish signal:
    - LP burn = liquidity permanently locked, dev CAN'T rug the pool
    - Supply burn = deflationary, shows dev commitment, reduces sell pressure

    Two detection strategies:
    1. DISCOVERY: Scan recent burn address transactions to find tokens being burned
       → cross-reference with DexScreener/PumpPortal for health check → buy signal
    2. SCORING: For tokens already in our pipeline, check if they have burns → score boost
    """

    def __init__(self, rpc_url: str, fast_rpc_url: str = ""):
        self.rpc_url = rpc_url
        self.fast_rpc = fast_rpc_url or rpc_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.buy_signals: asyncio.Queue = asyncio.Queue(maxsize=50)
        self._running = False
        self._task = None
        self._seen_sigs: set = set()       # tx signatures already processed
        self._seen_mints: set = set()       # mints already signaled (dedup)
        self._burn_history: List[dict] = [] # recent burns for dashboard
        self._burn_buy_max_sol = 0.02       # v4.1: 0.02 SOL per burn-signal buy
        self._healthy_cache: Dict[str, dict] = {}  # mint → health check result (TTL 60s)
        self._burn_buy_timestamps: List[float] = []  # v4.1: cooldown tracking
        self._burn_buy_cooldown_window = 600  # 10 minutes
        self._burn_buy_max_per_window = 2     # max 2 burn buys per 10 minutes
        logger.info("BurnSignalMonitor v4.1 initialized (tightened filters)")

    async def init_session(self, session: aiohttp.ClientSession):
        self.session = session

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("BurnSignalMonitor started — watching incinerator for LP/supply burns")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()

    async def _monitor_loop(self):
        """Scan burn addresses every 12 seconds for fresh token burns."""
        while self._running:
            try:
                await self._scan_burns()
                # v4.0: 8s scan — Alchemy allows ~30 rps, burn scan uses ~3-5 calls per loop
                await asyncio.sleep(8)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"BurnMonitor loop error: {e}")
                await asyncio.sleep(15)

    async def _scan_burns(self):
        """Fetch recent transactions to burn/incinerator addresses and identify token burns."""
        if not self.session:
            return

        for burn_addr in BURN_ADDRESSES:
            try:
                # Get last 25 signatures for this burn address
                payload = {
                    "jsonrpc": "2.0", "id": 1,
                    "method": "getSignaturesForAddress",
                    "params": [burn_addr, {"limit": 25}],
                }
                async with self.session.post(
                    self.fast_rpc, json=payload,
                    timeout=aiohttp.ClientTimeout(total=8),
                ) as resp:
                    if resp.status != 200:
                        continue
                    data = await resp.json()
                    sigs = data.get("result", [])

                for sig_info in sigs:
                    sig = sig_info.get("signature", "")
                    if not sig or sig in self._seen_sigs:
                        continue
                    # Only process recent transactions (< 5 minutes old)
                    block_time = sig_info.get("blockTime", 0)
                    if block_time and time.time() - block_time > 300:
                        self._seen_sigs.add(sig)
                        continue
                    # Skip failed transactions
                    if sig_info.get("err") is not None:
                        self._seen_sigs.add(sig)
                        continue

                    self._seen_sigs.add(sig)
                    await self._process_burn_tx(sig, burn_addr)
                    await asyncio.sleep(0.15)  # rate limit RPC calls

            except Exception as e:
                logger.debug(f"Burn scan error for {burn_addr[:16]}...: {e}")

        # Housekeeping: trim seen signatures
        if len(self._seen_sigs) > 5000:
            self._seen_sigs = set(list(self._seen_sigs)[-2500:])

    async def _process_burn_tx(self, sig: str, burn_addr: str):
        """Parse a transaction to the burn address and extract the token being burned."""
        try:
            payload = {
                "jsonrpc": "2.0", "id": 1,
                "method": "getTransaction",
                "params": [sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}],
            }
            async with self.session.post(
                self.fast_rpc, json=payload,
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()
                tx = data.get("result")
                if not tx:
                    return

            # Parse instructions to find token transfer or burn instructions
            instructions = []
            msg = tx.get("transaction", {}).get("message", {})
            instructions.extend(msg.get("instructions", []))
            # Also check inner instructions
            meta = tx.get("meta", {})
            for inner in (meta.get("innerInstructions") or []):
                instructions.extend(inner.get("instructions", []))

            for ix in instructions:
                program_id = ix.get("programId", "")
                if program_id not in (SPL_TOKEN_PROGRAM, SPL_TOKEN_2022):
                    continue

                parsed = ix.get("parsed", {})
                ix_type = parsed.get("type", "")
                info = parsed.get("info", {})

                token_mint = None
                amount = 0
                burn_type = ""

                if ix_type == "burn" or ix_type == "burnChecked":
                    # Direct SPL burn instruction — supply reduction
                    token_mint = info.get("mint", "")
                    amount = int(info.get("amount", 0) or info.get("tokenAmount", {}).get("amount", 0))
                    burn_type = "supply_burn"

                elif ix_type in ("transfer", "transferChecked"):
                    # Token transfer TO the burn address
                    dest = info.get("destination", "")
                    # We need to check if the destination ATA is owned by burn addr
                    # But since we got this tx from the burn address's history, it's likely a burn
                    token_mint = info.get("mint", "")
                    amount = int(info.get("amount", 0) or info.get("tokenAmount", {}).get("amount", 0))
                    if amount > 0:
                        burn_type = "transfer_burn"

                if not token_mint or token_mint in self._seen_mints or amount <= 0:
                    continue

                # Skip wrapped SOL and other system tokens
                if token_mint in ("So11111111111111111111111111111111111111112",):
                    continue

                logger.info(
                    f"BURN DETECTED: {burn_type} | mint={token_mint[:16]}... | "
                    f"amount={amount} | sig={sig[:20]}..."
                )

                # Record for history
                self._burn_history.append({
                    "mint": token_mint, "burn_type": burn_type,
                    "amount": amount, "sig": sig, "timestamp": time.time(),
                    "burn_addr": burn_addr,
                })
                if len(self._burn_history) > 100:
                    self._burn_history = self._burn_history[-50:]

                # Now check if this token is healthy and worth buying
                await self._evaluate_burn_token(token_mint, burn_type, amount, sig)

        except Exception as e:
            logger.debug(f"Process burn tx error: {e}")

    async def _evaluate_burn_token(self, mint: str, burn_type: str, amount: int, sig: str):
        """Check if a burned token has real value and is healthy enough to buy.

        v4.1 TIGHTENED: Raised all thresholds to stop buying 0-liquidity tokens.
        - Min liquidity $5,000 (was $500)
        - Require active trading in last 5 minutes
        - Require buy/sell ratio > 0.3
        - Require 24h volume > $1,000
        - Require FDV > $10,000
        - Max 2 burn buys per 10 minutes (cooldown)
        """
        if mint in self._seen_mints:
            return
        if not self.session:
            return

        # v4.1: Cooldown check — max 2 burn buys per 10 minutes
        now = time.time()
        self._burn_buy_timestamps = [
            t for t in self._burn_buy_timestamps
            if now - t < self._burn_buy_cooldown_window
        ]
        if len(self._burn_buy_timestamps) >= self._burn_buy_max_per_window:
            logger.debug(f"BURN COOLDOWN: {len(self._burn_buy_timestamps)} buys in last 10min, skipping {mint[:12]}...")
            return

        health = None

        # Strategy 1: Check DexScreener for DEX-listed tokens
        try:
            async with self.session.get(
                f"https://api.dexscreener.com/latest/dex/tokens/{mint}",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    pairs = data.get("pairs") or []
                    if pairs:
                        pair = pairs[0]  # best pair
                        liq = float(pair.get("liquidity", {}).get("usd", 0) or 0)
                        fdv = float(pair.get("fdv", 0) or 0)
                        price_change_5m = float(pair.get("priceChange", {}).get("m5", 0) or 0)
                        vol_24h = float(pair.get("volume", {}).get("h24", 0) or 0)
                        age_str = pair.get("pairCreatedAt", 0)
                        buys_5m = int(pair.get("txns", {}).get("m5", {}).get("buys", 0) or 0)
                        sells_5m = int(pair.get("txns", {}).get("m5", {}).get("sells", 0) or 0)
                        symbol = pair.get("baseToken", {}).get("symbol", "?")

                        # Calculate age in minutes
                        age_min = 9999
                        if age_str:
                            try:
                                age_min = (time.time() * 1000 - float(age_str)) / 60000
                            except (ValueError, TypeError):
                                pass

                        health = {
                            "symbol": symbol, "liquidity": liq, "fdv": fdv,
                            "price_change_5m": price_change_5m, "age_minutes": age_min,
                            "buys_5m": buys_5m, "sells_5m": sells_5m,
                            "volume_24h": vol_24h,
                            "on_dex": True, "on_bonding_curve": False,
                        }
        except Exception:
            pass

        # v4.1: NO MORE fallback for unknown tokens — if not on DEX, skip entirely
        if not health:
            logger.debug(f"BURN SKIP: {mint[:12]}... — not on DEX, no data")
            return

        # v4.1 TIGHTENED HEALTH CHECKS — strict filters to avoid 0-liq tokens
        is_healthy = False
        skip_reason = ""

        if health["on_dex"]:
            liq = health["liquidity"]
            fdv = health["fdv"]
            age = health["age_minutes"]
            buys = health["buys_5m"]
            sells = health["sells_5m"]
            vol = health.get("volume_24h", 0)
            total_txns_5m = buys + sells

            if liq < 5000:
                skip_reason = f"low_liquidity_{liq:.0f}_need_5000"
            elif fdv < 10000:
                skip_reason = f"fdv_too_low_{fdv:.0f}_likely_dead"
            elif fdv > 500000:
                skip_reason = f"fdv_too_high_{fdv:.0f}"
            elif vol < 1000:
                skip_reason = f"low_volume_{vol:.0f}_need_1000"
            elif age > 30:
                skip_reason = f"too_old_{age:.0f}m"
            elif total_txns_5m < 1:
                skip_reason = "no_trades_last_5m"
            elif total_txns_5m > 0 and buys / total_txns_5m < 0.3:
                skip_reason = f"buy_ratio_too_low_{buys}/{total_txns_5m}"
            elif health["price_change_5m"] < -30:
                skip_reason = f"price_dumping_{health['price_change_5m']:.0f}pct"
            else:
                is_healthy = True

        if not is_healthy:
            logger.debug(f"BURN SKIP: {health['symbol']} ({mint[:12]}...) — {skip_reason}")
            return

        # Generate buy signal!
        self._seen_mints.add(mint)
        self._burn_buy_timestamps.append(time.time())
        signal = {
            "mint": mint, "symbol": health["symbol"],
            "burn_type": burn_type, "burn_amount": amount,
            "sig": sig, "health": health,
            "timestamp": time.time(),
            "max_sol": self._burn_buy_max_sol,
        }

        try:
            self.buy_signals.put_nowait(signal)
            logger.info(
                f"BURN BUY SIGNAL: ${health['symbol']} ({mint[:12]}...) | "
                f"type={burn_type} | liq=${health['liquidity']:.0f} | "
                f"fdv=${health['fdv']:.0f} | vol=${health.get('volume_24h', 0):.0f} | "
                f"buys={health['buys_5m']} | cooldown={len(self._burn_buy_timestamps)}/{self._burn_buy_max_per_window}"
            )
        except asyncio.QueueFull:
            pass  # queue full, skip this signal

    async def check_token_burns(self, mint: str) -> dict:
        """Check if a specific token has any burn activity (for scoring boost).

        Returns: {"has_lp_burn": bool, "has_supply_burn": bool, "mint_authority_revoked": bool}
        """
        result = {"has_lp_burn": False, "has_supply_burn": False, "mint_authority_revoked": False}
        if not self.session:
            return result

        # Check if in our recent burn history
        for burn in self._burn_history:
            if burn["mint"] == mint:
                if burn["burn_type"] == "supply_burn":
                    result["has_supply_burn"] = True
                else:
                    result["has_lp_burn"] = True

        # Check mint authority (revoked = can't mint more = bullish)
        try:
            payload = {
                "jsonrpc": "2.0", "id": 1,
                "method": "getAccountInfo",
                "params": [mint, {"encoding": "jsonParsed"}],
            }
            async with self.session.post(
                self.fast_rpc, json=payload,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    acct = data.get("result", {}).get("value", {})
                    parsed = (acct.get("data", {}).get("parsed", {}).get("info", {}))
                    mint_auth = parsed.get("mintAuthority")
                    if mint_auth is None or mint_auth == "":
                        result["mint_authority_revoked"] = True
        except Exception:
            pass

        return result

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "burns_detected": len(self._burn_history),
            "signals_generated": len(self._seen_mints),
            "recent_burns": self._burn_history[-5:],
        }


@dataclass
class HolderSnapshot:
    """Snapshot of a token account holder at a point in time."""
    owner: str           # wallet address that owns the token account
    token_account: str   # the token account (ATA) address
    amount: int          # raw token amount
    ui_amount: float     # human-readable amount
    pct_supply: float    # percentage of total supply this holder has


class HolderWatcher:
    """Real-time on-chain monitoring of top holders for every position we hold.

    The alpha: When a top holder moves tokens and the destination is NOT a known
    token lock contract (Streamflow, Jupiter Lock, Bonfida Vesting, Raydium Burn&Earn),
    we sell IMMEDIATELY because it's likely a dump about to crash the price.

    Architecture:
    - When we buy: snapshot top 10 holders via getTokenLargestAccounts
    - Every 5 seconds: re-check holder balances
    - If any critical holder's balance drops >25%: check where tokens went
    - If destination is DEX program → they're selling → EMERGENCY SELL
    - If destination is unknown wallet → suspicious → SELL
    - If destination is lock contract → SAFE, continue holding

    This catches rugs 10-60 seconds before the price crashes on DexScreener.
    """

    def __init__(self, rpc_url: str, fast_rpc_url: str = ""):
        self.rpc_url = rpc_url
        self.fast_rpc = fast_rpc_url or rpc_url
        self.session: Optional[aiohttp.ClientSession] = None
        # mint → list of HolderSnapshots (top holders at time of buy)
        self.holder_snapshots: Dict[str, List[HolderSnapshot]] = {}
        # mint → set of holder owners we're actively watching
        self.watched_holders: Dict[str, Set[str]] = {}
        # mint → last check timestamp (rate limiting)
        self._last_check: Dict[str, float] = {}
        # mint → set of alerts already fired (dedup)
        self._alerts_fired: Dict[str, Set[str]] = {}
        # Alert queue — DegenTrader reads this to trigger sells
        self.sell_alerts: asyncio.Queue = asyncio.Queue(maxsize=50)
        self._check_interval = 5.0  # seconds between holder checks per token
        self.running = False
        self._task = None

    async def init_session(self, session: aiohttp.ClientSession):
        self.session = session

    async def start(self):
        """Start background holder monitoring loop."""
        self.running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("HolderWatcher started — monitoring top holders for rug detection")

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()

    async def snapshot_holders(self, mint: str, exclude_owner: str = ""):
        """Take initial snapshot of top holders when we buy a token.

        Args:
            mint: Token mint address
            exclude_owner: Our own wallet (exclude from monitoring)
        """
        if not self.session:
            return
        try:
            payload = {
                "jsonrpc": "2.0", "id": 1,
                "method": "getTokenLargestAccounts",
                "params": [mint, {"commitment": "confirmed"}],
            }
            async with self.session.post(self.fast_rpc, json=payload,
                                         timeout=aiohttp.ClientTimeout(total=10)) as resp:
                data = await resp.json()
                accounts = data.get("result", {}).get("value", [])

            if not accounts:
                logger.warning(f"HolderWatch: No holders found for {mint[:12]}")
                return

            # Get total supply for percentage calculation
            total_supply = await self._get_total_supply(mint)
            if total_supply <= 0:
                total_supply = 1_000_000_000_000_000  # pump.fun default: 1B tokens * 10^6 decimals

            snapshots = []
            watched = set()
            for acc in accounts[:10]:  # Top 10 holders
                token_account = acc.get("address", "")
                amount = int(acc.get("amount", "0"))
                ui_amount = float(acc.get("uiAmount", 0) or 0)
                decimals = acc.get("decimals", 6)

                # Get the owner wallet of this token account
                owner = await self._get_token_account_owner(token_account)
                if not owner or owner == exclude_owner:
                    continue

                # Skip if it's a known safe program (bonding curve, lock, etc.)
                if owner in SAFE_LOCK_PROGRAMS:
                    continue

                pct = (amount / total_supply * 100) if total_supply > 0 else 0

                snap = HolderSnapshot(
                    owner=owner, token_account=token_account,
                    amount=amount, ui_amount=ui_amount, pct_supply=round(pct, 2),
                )
                snapshots.append(snap)

                # Only watch holders with >2% of supply — they can move the price
                if pct >= 2.0:
                    watched.add(owner)

            self.holder_snapshots[mint] = snapshots
            self.watched_holders[mint] = watched
            self._alerts_fired[mint] = set()
            self._last_check[mint] = time.time()

            holder_summary = ", ".join(
                f"{s.owner[:8]}..({s.pct_supply:.1f}%)" for s in snapshots[:5]
            )
            logger.info(
                f"HolderWatch: Snapshot {mint[:12]} | {len(snapshots)} holders | "
                f"watching {len(watched)} critical | top: [{holder_summary}]"
            )

        except Exception as e:
            logger.error(f"HolderWatch snapshot error {mint[:12]}: {e}")

    async def unwatch(self, mint: str):
        """Stop watching a token (after we sell)."""
        self.holder_snapshots.pop(mint, None)
        self.watched_holders.pop(mint, None)
        self._last_check.pop(mint, None)
        self._alerts_fired.pop(mint, None)

    async def _monitor_loop(self):
        """Background loop: check all watched positions for holder movements."""
        while self.running:
            try:
                mints = list(self.watched_holders.keys())
                for mint in mints:
                    if not self.running:
                        break
                    # Rate limit: check each token every N seconds
                    last = self._last_check.get(mint, 0)
                    if time.time() - last < self._check_interval:
                        continue
                    self._last_check[mint] = time.time()
                    await self._check_holders(mint)
            except Exception as e:
                logger.error(f"HolderWatch loop error: {e}")
            await asyncio.sleep(1.0)

    async def _check_holders(self, mint: str):
        """Re-check holder balances and detect movements."""
        if not self.session or mint not in self.holder_snapshots:
            return

        original_snapshots = self.holder_snapshots[mint]
        watched = self.watched_holders.get(mint, set())
        if not watched:
            return

        try:
            # Fetch current top holders
            payload = {
                "jsonrpc": "2.0", "id": 1,
                "method": "getTokenLargestAccounts",
                "params": [mint, {"commitment": "confirmed"}],
            }
            async with self.session.post(self.fast_rpc, json=payload,
                                         timeout=aiohttp.ClientTimeout(total=8)) as resp:
                data = await resp.json()
                current_accounts = data.get("result", {}).get("value", [])

            if not current_accounts:
                return

            # Build current balance map: token_account → amount
            current_balances = {}
            for acc in current_accounts:
                current_balances[acc.get("address", "")] = int(acc.get("amount", "0"))

            # Compare against original snapshots
            for snap in original_snapshots:
                if snap.owner not in watched:
                    continue
                if snap.owner in self._alerts_fired.get(mint, set()):
                    continue  # already alerted on this holder

                current_amount = current_balances.get(snap.token_account, None)

                # If token account disappeared from top 20 or balance dropped significantly
                if current_amount is None:
                    # Account not in top 20 anymore — they may have sold everything
                    current_amount = await self._get_account_balance(snap.token_account)

                if current_amount is None:
                    continue

                # Calculate drop percentage
                if snap.amount <= 0:
                    continue
                drop_pct = (snap.amount - current_amount) / snap.amount * 100

                # TRIGGER: Holder dropped >25% of their position
                if drop_pct >= 25:
                    # Check WHERE the tokens went
                    is_safe = await self._check_transfer_destination(snap.owner, mint)

                    if is_safe:
                        logger.info(
                            f"HolderWatch SAFE: {snap.owner[:12]} moved {drop_pct:.0f}% "
                            f"of {mint[:12]} to LOCK CONTRACT — holding"
                        )
                        continue

                    # NOT SAFE — this is a dump. Fire sell alert!
                    self._alerts_fired.setdefault(mint, set()).add(snap.owner)
                    alert = {
                        "mint": mint,
                        "holder": snap.owner,
                        "drop_pct": round(drop_pct, 1),
                        "original_pct_supply": snap.pct_supply,
                        "original_amount": snap.amount,
                        "current_amount": current_amount,
                        "reason": "holder_dump",
                        "timestamp": time.time(),
                    }
                    try:
                        self.sell_alerts.put_nowait(alert)
                    except asyncio.QueueFull:
                        self.sell_alerts.get_nowait()
                        self.sell_alerts.put_nowait(alert)

                    logger.warning(
                        f"HOLDER DUMP DETECTED: {snap.owner[:12]}... dropped "
                        f"{drop_pct:.0f}% of {mint[:12]} (held {snap.pct_supply:.1f}% supply) "
                        f"— NOT a lock contract — SELLING"
                    )

        except Exception as e:
            logger.debug(f"HolderWatch check error {mint[:12]}: {e}")

    async def _check_transfer_destination(self, wallet: str, mint: str) -> bool:
        """Check if a wallet's recent token transfer went to a safe destination (lock contract).

        Returns True if the transfer destination is a known safe program (lock/vesting/burn).
        Returns False if destination is a DEX (selling) or unknown wallet (suspicious).
        """
        if not self.session:
            return False

        try:
            # Get most recent transactions for this wallet
            payload = {
                "jsonrpc": "2.0", "id": 1,
                "method": "getSignaturesForAddress",
                "params": [wallet, {"limit": 3, "commitment": "confirmed"}],
            }
            async with self.session.post(self.fast_rpc, json=payload,
                                         timeout=aiohttp.ClientTimeout(total=8)) as resp:
                data = await resp.json()
                signatures = data.get("result", [])

            if not signatures:
                return False  # can't determine → assume unsafe

            # Parse the most recent transaction
            tx_sig = signatures[0].get("signature", "")
            if not tx_sig:
                return False

            payload = {
                "jsonrpc": "2.0", "id": 1,
                "method": "getTransaction",
                "params": [tx_sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}],
            }
            async with self.session.post(self.fast_rpc, json=payload,
                                         timeout=aiohttp.ClientTimeout(total=10)) as resp:
                data = await resp.json()
                tx = data.get("result")

            if not tx:
                return False

            # Check all programs invoked in this transaction
            message = tx.get("transaction", {}).get("message", {})
            instructions = message.get("instructions", [])
            inner_instructions = tx.get("meta", {}).get("innerInstructions", [])

            all_programs = set()
            for ix in instructions:
                prog = ix.get("programId", "")
                if prog:
                    all_programs.add(prog)
            for inner_group in inner_instructions:
                for ix in inner_group.get("instructions", []):
                    prog = ix.get("programId", "")
                    if prog:
                        all_programs.add(prog)

            # Check if ANY invoked program is a DEX → holder is SELLING
            dex_hit = all_programs & DEX_SELL_PROGRAMS
            if dex_hit:
                logger.info(f"HolderWatch: {wallet[:12]} used DEX program {list(dex_hit)[0][:12]} — DUMP CONFIRMED")
                return False  # NOT safe — selling on DEX

            # Check if destination is a known lock program → SAFE
            lock_hit = all_programs & SAFE_LOCK_PROGRAMS
            if lock_hit:
                logger.info(f"HolderWatch: {wallet[:12]} sent to lock program {list(lock_hit)[0][:12]} — SAFE")
                return True  # Safe — token lock

            # Check for simple SPL token transfers — where did tokens go?
            for ix in instructions:
                if ix.get("program") == "spl-token":
                    parsed = ix.get("parsed", {})
                    ix_type = parsed.get("type", "")
                    if ix_type in ("transfer", "transferChecked"):
                        dest = parsed.get("info", {}).get("destination", "")
                        if dest:
                            # Look up destination account owner
                            dest_owner = await self._get_token_account_owner(dest)
                            if dest_owner and dest_owner in SAFE_LOCK_PROGRAMS:
                                return True

            # Default: unknown destination = assume unsafe
            return False

        except Exception as e:
            logger.debug(f"HolderWatch transfer check error: {e}")
            return False  # can't determine → assume unsafe

    async def _get_total_supply(self, mint: str) -> int:
        """Get total supply of a token."""
        if not self.session:
            return 0
        try:
            payload = {
                "jsonrpc": "2.0", "id": 1,
                "method": "getAccountInfo",
                "params": [mint, {"encoding": "jsonParsed"}],
            }
            async with self.session.post(self.fast_rpc, json=payload,
                                         timeout=aiohttp.ClientTimeout(total=8)) as resp:
                data = await resp.json()
                info = data.get("result", {}).get("value", {}).get("data", {}).get("parsed", {}).get("info", {})
                return int(info.get("supply", 0))
        except Exception:
            return 0

    async def _get_token_account_owner(self, token_account: str) -> Optional[str]:
        """Get the owner wallet of a token account."""
        if not self.session:
            return None
        try:
            payload = {
                "jsonrpc": "2.0", "id": 1,
                "method": "getAccountInfo",
                "params": [token_account, {"encoding": "jsonParsed"}],
            }
            async with self.session.post(self.fast_rpc, json=payload,
                                         timeout=aiohttp.ClientTimeout(total=8)) as resp:
                data = await resp.json()
                parsed = data.get("result", {}).get("value", {}).get("data", {}).get("parsed", {})
                return parsed.get("info", {}).get("owner")
        except Exception:
            return None

    async def _get_account_balance(self, token_account: str) -> Optional[int]:
        """Get current balance of a specific token account."""
        if not self.session:
            return None
        try:
            payload = {
                "jsonrpc": "2.0", "id": 1,
                "method": "getAccountInfo",
                "params": [token_account, {"encoding": "jsonParsed"}],
            }
            async with self.session.post(self.fast_rpc, json=payload,
                                         timeout=aiohttp.ClientTimeout(total=8)) as resp:
                data = await resp.json()
                parsed = data.get("result", {}).get("value", {}).get("data", {}).get("parsed", {})
                amount = parsed.get("info", {}).get("tokenAmount", {}).get("amount")
                return int(amount) if amount is not None else None
        except Exception:
            return None

    def get_status(self) -> Dict:
        """Status for dashboard."""
        return {
            "watching": len(self.watched_holders),
            "tokens_monitored": list(self.watched_holders.keys()),
            "total_holders_tracked": sum(len(v) for v in self.watched_holders.values()),
            "alerts_pending": self.sell_alerts.qsize(),
        }


# ============================================================
# DEX PRICE FEED — Real-time via Solana WebSocket (v4.3)
# ============================================================
class DexPriceFeed:
    """Real-time DEX price feed via Solana WebSocket pool vault subscriptions.

    When we hold a DEX position, subscribes to the Raydium AMM pool's
    token vault accounts. Every swap changes vault balances, triggering
    an instant price update pushed to us — no polling needed.

    Price is computed from: quote_vault_balance / base_vault_balance
    adjusted for token decimals. Gives us sub-second price updates
    vs 15-30s stale data from DexScreener.
    """

    # Raydium AMM V4 account layout offsets (bytes)
    _COIN_DECIMALS_OFFSET = 32     # u64: base token decimals
    _PC_DECIMALS_OFFSET = 40       # u64: quote token decimals
    _COIN_VAULT_OFFSET = 336       # Pubkey (32 bytes): base token vault
    _PC_VAULT_OFFSET = 368         # Pubkey (32 bytes): quote token vault
    _COIN_MINT_OFFSET = 400        # Pubkey (32 bytes): base token mint
    _PC_MINT_OFFSET = 432          # Pubkey (32 bytes): quote token mint

    def __init__(self, rpc_url: str, ws_url: str = ""):
        self.rpc_url = rpc_url
        # Convert HTTP RPC to WebSocket URL
        if ws_url:
            self.ws_url = ws_url
        else:
            self.ws_url = rpc_url.replace("https://", "wss://").replace("http://", "ws://")
        self.ws = None
        self.running = False
        self._task = None
        self._reconnect_count = 0
        # mint -> live price in SOL
        self.live_prices: Dict[str, float] = {}
        # mint -> subscription metadata
        self._subs: Dict[str, dict] = {}
        # ws subscription_id -> (mint, "coin"|"pc")
        self._sub_id_map: Dict[int, tuple] = {}
        # Pending subscribe/unsubscribe requests
        self._pending_subs: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._pending_unsubs: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._next_rpc_id = 1

    async def start(self):
        self.running = True
        self._task = asyncio.create_task(self._listen())
        logger.info("DexPriceFeed started (Raydium pool vault WebSocket subscriptions)")

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass

    async def subscribe(self, mint: str, pool_address: str, session: aiohttp.ClientSession):
        """Subscribe to real-time price for a DEX token via its Raydium pool."""
        if mint in self._subs:
            return True  # already subscribed

        # 1. Read Raydium AMM pool account to get vault addresses + decimals
        vaults = await self._resolve_raydium_vaults(pool_address, session)
        if not vaults:
            logger.debug(f"DexPriceFeed: couldn't resolve vaults for pool {pool_address[:12]}")
            return False

        coin_vault, pc_vault, coin_decimals, pc_decimals, coin_mint = vaults

        # 2. Get initial vault balances
        coin_bal = await self._get_vault_balance(coin_vault, session)
        pc_bal = await self._get_vault_balance(pc_vault, session)

        # 3. Store subscription info
        self._subs[mint] = {
            "pool": pool_address,
            "coin_vault": coin_vault,
            "pc_vault": pc_vault,
            "coin_decimals": coin_decimals,
            "pc_decimals": pc_decimals,
            "coin_mint": coin_mint,
            "coin_balance": coin_bal,
            "pc_balance": pc_bal,
            "coin_sub_id": None,
            "pc_sub_id": None,
        }

        # Compute initial price
        self._update_price(mint)

        # 4. Queue WS subscriptions (handled by _listen loop)
        try:
            self._pending_subs.put_nowait((mint, coin_vault, pc_vault))
        except asyncio.QueueFull:
            pass

        logger.info(
            f"DexPriceFeed: subscribed {mint[:12]}... | "
            f"pool={pool_address[:12]} | price={self.live_prices.get(mint, 0):.10f} SOL"
        )
        return True

    async def unsubscribe(self, mint: str):
        """Unsubscribe from a token's price feed."""
        if mint not in self._subs:
            return
        info = self._subs.pop(mint)
        self.live_prices.pop(mint, None)
        # Queue WS unsubscriptions
        for sub_id in [info.get("coin_sub_id"), info.get("pc_sub_id")]:
            if sub_id is not None:
                try:
                    self._pending_unsubs.put_nowait(sub_id)
                except asyncio.QueueFull:
                    pass
                self._sub_id_map.pop(sub_id, None)
        logger.debug(f"DexPriceFeed: unsubscribed {mint[:12]}")

    def get_price(self, mint: str) -> float:
        """Get the latest real-time price in SOL. Returns 0 if not subscribed."""
        return self.live_prices.get(mint, 0.0)

    def _update_price(self, mint: str):
        """Recompute price from current vault balances."""
        info = self._subs.get(mint)
        if not info:
            return
        coin_bal = info["coin_balance"]
        pc_bal = info["pc_balance"]
        coin_dec = info["coin_decimals"]
        pc_dec = info["pc_decimals"]
        if coin_bal <= 0:
            return
        # Price = (quote_amount / 10^quote_dec) / (base_amount / 10^base_dec)
        price_sol = (pc_bal / 10**pc_dec) / (coin_bal / 10**coin_dec)
        self.live_prices[mint] = price_sol

    async def _resolve_raydium_vaults(self, pool_address: str, session: aiohttp.ClientSession) -> Optional[tuple]:
        """Read Raydium AMM V4 pool account to extract vault pubkeys and decimals."""
        try:
            payload = {
                "jsonrpc": "2.0", "id": 1,
                "method": "getAccountInfo",
                "params": [pool_address, {"encoding": "base64"}],
            }
            async with session.post(self.rpc_url, json=payload, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                account = data.get("result", {}).get("value")
                if not account:
                    return None
                # Verify it's owned by Raydium AMM program
                owner = account.get("owner", "")
                if owner != RAYDIUM_AMM_PROGRAM:
                    logger.debug(f"DexPriceFeed: pool {pool_address[:12]} not Raydium AMM (owner={owner[:12]})")
                    return None
                # Decode base64 account data
                raw = base64.b64decode(account["data"][0])
                if len(raw) < 496:  # minimum Raydium AMM V4 account size
                    return None
                coin_decimals = struct.unpack_from("<Q", raw, self._COIN_DECIMALS_OFFSET)[0]
                pc_decimals = struct.unpack_from("<Q", raw, self._PC_DECIMALS_OFFSET)[0]
                from solders.pubkey import Pubkey as _Pk
                coin_vault = str(_Pk.from_bytes(raw[self._COIN_VAULT_OFFSET:self._COIN_VAULT_OFFSET + 32]))
                pc_vault = str(_Pk.from_bytes(raw[self._PC_VAULT_OFFSET:self._PC_VAULT_OFFSET + 32]))
                coin_mint = str(_Pk.from_bytes(raw[self._COIN_MINT_OFFSET:self._COIN_MINT_OFFSET + 32]))
                return coin_vault, pc_vault, coin_decimals, pc_decimals, coin_mint
        except Exception as e:
            logger.debug(f"DexPriceFeed: vault resolve error: {e}")
            return None

    async def _get_vault_balance(self, vault_pubkey: str, session: aiohttp.ClientSession) -> int:
        """Read SPL token account balance (raw amount)."""
        try:
            payload = {
                "jsonrpc": "2.0", "id": 1,
                "method": "getAccountInfo",
                "params": [vault_pubkey, {"encoding": "base64"}],
            }
            async with session.post(self.rpc_url, json=payload, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    return 0
                data = await resp.json()
                account = data.get("result", {}).get("value")
                if not account:
                    return 0
                raw = base64.b64decode(account["data"][0])
                if len(raw) < 72:
                    return 0
                # SPL token account: amount is u64 at offset 64
                return struct.unpack_from("<Q", raw, 64)[0]
        except Exception as e:
            logger.debug(f"DexPriceFeed: vault balance error: {e}")
            return 0

    async def _listen(self):
        """Main WebSocket loop — receives vault account updates."""
        try:
            import websockets
        except ImportError:
            logger.warning("websockets not installed, DexPriceFeed disabled")
            return

        while self.running:
            try:
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    self.ws = ws
                    self._reconnect_count += 1
                    logger.info(f"DexPriceFeed WS connected (attempt #{self._reconnect_count})")

                    # Start background task to process pending sub/unsub requests
                    asyncio.create_task(self._process_subscriptions(ws))

                    async for msg in ws:
                        if not self.running:
                            break
                        try:
                            data = json.loads(msg)
                            self._handle_ws_message(data)
                        except json.JSONDecodeError:
                            continue

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"DexPriceFeed WS error: {e}, reconnecting in 3s... (reconnects: {self._reconnect_count})")
                await asyncio.sleep(3)

    async def _process_subscriptions(self, ws):
        """Process pending subscribe/unsubscribe requests."""
        while self.running:
            try:
                # Process unsubscriptions
                while not self._pending_unsubs.empty():
                    try:
                        sub_id = self._pending_unsubs.get_nowait()
                        rpc_id = self._next_rpc_id
                        self._next_rpc_id += 1
                        await ws.send(json.dumps({
                            "jsonrpc": "2.0", "id": rpc_id,
                            "method": "accountUnsubscribe", "params": [sub_id],
                        }))
                    except asyncio.QueueEmpty:
                        break
                    except Exception:
                        break

                # Process subscriptions
                while not self._pending_subs.empty():
                    try:
                        mint, coin_vault, pc_vault = self._pending_subs.get_nowait()
                        if mint not in self._subs:
                            continue

                        # Subscribe to coin vault
                        coin_rpc_id = self._next_rpc_id
                        self._next_rpc_id += 1
                        await ws.send(json.dumps({
                            "jsonrpc": "2.0", "id": coin_rpc_id,
                            "method": "accountSubscribe",
                            "params": [coin_vault, {"encoding": "base64", "commitment": "confirmed"}],
                        }))

                        # Subscribe to pc vault
                        pc_rpc_id = self._next_rpc_id
                        self._next_rpc_id += 1
                        await ws.send(json.dumps({
                            "jsonrpc": "2.0", "id": pc_rpc_id,
                            "method": "accountSubscribe",
                            "params": [pc_vault, {"encoding": "base64", "commitment": "confirmed"}],
                        }))

                        # Store pending rpc_id -> mint mapping for subscription confirmation
                        if not hasattr(self, '_pending_confirms'):
                            self._pending_confirms = {}
                        self._pending_confirms[coin_rpc_id] = (mint, "coin")
                        self._pending_confirms[pc_rpc_id] = (mint, "pc")

                    except asyncio.QueueEmpty:
                        break
                    except Exception as e:
                        logger.debug(f"DexPriceFeed sub error: {e}")
                        break

                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(1)

    def _handle_ws_message(self, data: dict):
        """Handle WebSocket messages — subscription confirmations and account updates."""
        # Subscription confirmation: {"jsonrpc":"2.0","result":12345,"id":1}
        if "result" in data and "id" in data and "method" not in data:
            rpc_id = data["id"]
            sub_id = data["result"]
            if hasattr(self, '_pending_confirms') and rpc_id in self._pending_confirms:
                mint, vault_type = self._pending_confirms.pop(rpc_id)
                if mint in self._subs:
                    self._subs[mint][f"{vault_type}_sub_id"] = sub_id
                    self._sub_id_map[sub_id] = (mint, vault_type)
            return

        # Account notification: {"method":"accountNotification","params":{"subscription":12345,"result":{"value":{"data":["base64...","base64"]}}}}
        if data.get("method") == "accountNotification":
            params = data.get("params", {})
            sub_id = params.get("subscription")
            if sub_id not in self._sub_id_map:
                return

            mint, vault_type = self._sub_id_map[sub_id]
            if mint not in self._subs:
                return

            try:
                account_data = params.get("result", {}).get("value", {}).get("data", [])
                if not account_data or not account_data[0]:
                    return
                raw = base64.b64decode(account_data[0])
                if len(raw) < 72:
                    return
                # SPL token account: amount at offset 64 (u64 LE)
                balance = struct.unpack_from("<Q", raw, 64)[0]
                self._subs[mint][f"{vault_type}_balance"] = balance
                self._update_price(mint)
            except Exception as e:
                logger.debug(f"DexPriceFeed: parse error for {mint[:12]}: {e}")


# ============================================================
# WALLET GRAPH ANALYZER
# ============================================================
class WalletAnalyzer:
    """Analyze wallet connections to detect cabals and insider coordination."""

    def __init__(self, rpc_url: str, fast_rpc_url: str = ""):
        self.rpc_url = rpc_url
        self.fast_rpc = fast_rpc_url or rpc_url
        self.wallet_cache: Dict[str, WalletProfile] = {}
        self.session: Optional[aiohttp.ClientSession] = None

    async def init_session(self, session: aiohttp.ClientSession):
        self.session = session

    async def analyze_token_holders(self, mint: str) -> Dict:
        """Analyze top holders of a token for connected wallets."""
        result = {
            "top_holders": [],
            "connected_groups": [],
            "cabal_score": 0.0,
            "concentration": 0.0,
            "dev_holds_pct": 0.0,
        }

        try:
            # Get largest token accounts
            payload = {
                "jsonrpc": "2.0", "id": 1,
                "method": "getTokenLargestAccounts",
                "params": [mint]
            }
            async with self.session.post(self.fast_rpc, json=payload) as resp:
                data = await resp.json()
                accounts = data.get("result", {}).get("value", [])

            if not accounts:
                return result

            # Get top 10 holder addresses
            holder_addresses = []
            total_supply = 0
            top_10_amount = 0

            for acc in accounts[:10]:
                amount = float(acc.get("uiAmount", 0) or 0)
                top_10_amount += amount
                total_supply = max(total_supply, top_10_amount * 2)  # rough estimate

                # Resolve owner of token account
                owner = await self._get_token_account_owner(acc.get("address", ""))
                if owner:
                    holder_addresses.append({"address": owner, "amount": amount})

            if total_supply > 0:
                result["concentration"] = top_10_amount / total_supply

            # Check if top holders share funding sources
            funding_sources = {}
            for holder in holder_addresses[:5]:  # check top 5 for speed
                funder = await self._trace_funding_source(holder["address"])
                if funder:
                    if funder not in funding_sources:
                        funding_sources[funder] = []
                    funding_sources[funder].append(holder["address"])

            # Detect connected groups (wallets funded by same source)
            for source, wallets in funding_sources.items():
                if len(wallets) >= 2:
                    result["connected_groups"].append({
                        "funder": source[:12] + "...",
                        "wallets": len(wallets),
                        "addresses": [w[:12] + "..." for w in wallets],
                    })

            # Calculate cabal score
            connected_wallets = sum(len(g["addresses"]) for g in result["connected_groups"])
            if connected_wallets >= 4:
                result["cabal_score"] = 90
            elif connected_wallets >= 3:
                result["cabal_score"] = 70
            elif connected_wallets >= 2:
                result["cabal_score"] = 50
            else:
                result["cabal_score"] = 10

            # High concentration amplifies cabal score
            if result["concentration"] > 0.5:
                result["cabal_score"] = min(100, result["cabal_score"] * 1.3)

            result["top_holders"] = [
                {"address": h["address"][:12] + "...", "amount": h["amount"]}
                for h in holder_addresses[:5]
            ]

        except Exception as e:
            logger.debug(f"Wallet analysis error for {mint}: {e}")

        return result

    async def _get_token_account_owner(self, token_account: str) -> Optional[str]:
        """Get the owner wallet of a token account."""
        try:
            payload = {
                "jsonrpc": "2.0", "id": 1,
                "method": "getAccountInfo",
                "params": [token_account, {"encoding": "jsonParsed"}]
            }
            async with self.session.post(self.fast_rpc, json=payload) as resp:
                data = await resp.json()
                info = data.get("result", {}).get("value", {})
                if info:
                    return info.get("data", {}).get("parsed", {}).get("info", {}).get("owner")
        except Exception:
            pass
        return None

    async def _trace_funding_source(self, wallet: str) -> Optional[str]:
        """Trace where a wallet got its initial SOL from."""
        try:
            payload = {
                "jsonrpc": "2.0", "id": 1,
                "method": "getSignaturesForAddress",
                "params": [wallet, {"limit": 5}]
            }
            async with self.session.post(self.fast_rpc, json=payload) as resp:
                data = await resp.json()
                sigs = data.get("result", [])

            if not sigs:
                return None

            # Check the earliest transaction for funding source
            earliest_sig = sigs[-1].get("signature", "")
            if not earliest_sig:
                return None

            tx_payload = {
                "jsonrpc": "2.0", "id": 1,
                "method": "getTransaction",
                "params": [earliest_sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}]
            }
            async with self.session.post(self.fast_rpc, json=tx_payload) as resp:
                tx_data = await resp.json()
                tx = tx_data.get("result", {})

            if not tx:
                return None

            # Look for SOL transfer to this wallet in the first transaction
            instructions = (
                tx.get("transaction", {}).get("message", {}).get("instructions", [])
            )
            for ix in instructions:
                parsed = ix.get("parsed", {})
                if parsed.get("type") == "transfer":
                    info = parsed.get("info", {})
                    if info.get("destination") == wallet:
                        return info.get("source", "")

        except Exception:
            pass
        return None

    async def check_whale_buys(self, mint: str) -> List[Dict]:
        """Check if known whale wallets are buying a token."""
        whale_buys = []
        try:
            payload = {
                "jsonrpc": "2.0", "id": 1,
                "method": "getSignaturesForAddress",
                "params": [mint, {"limit": 30}]
            }
            async with self.session.post(self.fast_rpc, json=payload) as resp:
                data = await resp.json()
                sigs = data.get("result", [])

            # Check recent transactions for large swaps
            for sig_info in sigs[:10]:
                sig = sig_info.get("signature", "")
                if not sig:
                    continue
                tx_payload = {
                    "jsonrpc": "2.0", "id": 1,
                    "method": "getTransaction",
                    "params": [sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}]
                }
                async with self.session.post(self.fast_rpc, json=tx_payload) as resp:
                    tx_data = await resp.json()
                    tx = tx_data.get("result")

                if not tx:
                    continue

                # Check pre/post balances for large SOL movements
                pre_balances = tx.get("meta", {}).get("preBalances", [])
                post_balances = tx.get("meta", {}).get("postBalances", [])
                accounts = tx.get("transaction", {}).get("message", {}).get("accountKeys", [])

                for i, (pre, post) in enumerate(zip(pre_balances, post_balances)):
                    sol_change = (post - pre) / LAMPORTS_PER_SOL
                    if sol_change < -1.0:  # spent > 1 SOL on this token
                        account_key = accounts[i] if i < len(accounts) else {}
                        addr = account_key.get("pubkey", "") if isinstance(account_key, dict) else str(account_key)
                        whale_buys.append({
                            "wallet": addr[:12] + "..." if addr else "unknown",
                            "sol_spent": abs(sol_change),
                            "tx": sig[:20] + "...",
                        })
                        break  # one whale per tx

                await asyncio.sleep(0.1)  # rate limiting

        except Exception as e:
            logger.debug(f"Whale buy check error: {e}")

        return whale_buys


# (QuantumTradeAnalyzer replaced by QuantumTradeOracle in v3)


# ============================================================
# SWARM TRADE INTELLIGENCE
# ============================================================
class SwarmTradeIntelligence:
    """Deep integration with shadow agents for trade decisions."""

    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self._shadow_available = None

    async def multi_agent_analysis(self, token: TokenInfo) -> Dict:
        """Run parallel analysis across specialized agents.

        - Grok: X/Twitter sentiment, trending status
        - DeepSeek: Technical analysis, chart patterns
        - Gemini: Multi-factor holistic review
        Returns combined verdict.
        """
        results = {"verdict": "SKIP", "confidence": 0, "reasons": []}

        try:
            from Cosmos.core.collective.persistent_agent import call_shadow_agent
            self._shadow_available = True
        except ImportError:
            self._shadow_available = False
            return await self._api_fallback(token)

        token_brief = (
            f"${token.symbol} | Price: ${token.price_usd:.8f} | "
            f"Liq: ${token.liquidity_usd:.0f} | Age: {token.age_minutes:.0f}m | "
            f"5m: {token.price_change_5m:+.1f}% | FDV: ${token.fdv:.0f} | "
            f"Buys/Sells(5m): {token.buy_count_5m}/{token.sell_count_5m}"
        )

        # Run agents in parallel for speed
        tasks = {
            "grok": call_shadow_agent(
                "grok",
                f"Quick degen scan: {token_brief}. "
                f"Check X/Twitter for ${token.symbol} mentions, hype, cabal activity. "
                f"Reply ONLY: BUY, STRONG_BUY, or SKIP with 1-line reason.",
                max_tokens=100, timeout=8.0,
            ),
            "deepseek": call_shadow_agent(
                "deepseek",
                f"Technical analysis: {token_brief}. "
                f"Analyze buy/sell ratio, liquidity depth, momentum. "
                f"Reply ONLY: BUY, STRONG_BUY, or SKIP with 1-line reason.",
                max_tokens=100, timeout=8.0,
            ),
            "gemini": call_shadow_agent(
                "gemini",
                f"Risk assessment: {token_brief}. "
                f"Evaluate rug risk, holder distribution signal, growth potential. "
                f"Reply ONLY: BUY, STRONG_BUY, or SKIP with 1-line reason.",
                max_tokens=100, timeout=8.0,
            ),
        }

        done = {}
        for name, task in tasks.items():
            try:
                result = await asyncio.wait_for(task, timeout=10.0)
                if result:
                    _, response = result
                    done[name] = response.upper() if response else ""
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug(f"Agent {name} timeout/error: {e}")
                done[name] = ""

        # Tally votes
        buy_votes = 0
        strong_buy_votes = 0
        reasons = []

        for agent, response in done.items():
            if "STRONG_BUY" in response:
                strong_buy_votes += 1
                buy_votes += 1
                reasons.append(f"{agent}: STRONG_BUY")
            elif "BUY" in response:
                buy_votes += 1
                reasons.append(f"{agent}: BUY")
            else:
                reasons.append(f"{agent}: SKIP")

        total = len(done) or 1
        results["reasons"] = reasons

        if strong_buy_votes >= 2:
            results["verdict"] = "STRONG_BUY"
            results["confidence"] = 95
        elif buy_votes >= 2:
            results["verdict"] = "BUY"
            results["confidence"] = 75
        elif buy_votes >= 1:
            results["verdict"] = "WEAK_BUY"
            results["confidence"] = 50
        else:
            results["verdict"] = "SKIP"
            results["confidence"] = 30

        logger.info(f"Swarm verdict on {token.symbol}: {results['verdict']} ({buy_votes}/{total} BUY)")
        return results

    async def _api_fallback(self, token: TokenInfo) -> Dict:
        """Fallback when shadow agents aren't available."""
        try:
            prompt = (
                f"Quick degen check on ${token.symbol}: "
                f"Liq ${token.liquidity_usd:.0f}, Age {token.age_minutes:.0f}m, "
                f"5m {token.price_change_5m:+.1f}%, B/S {token.buy_count_5m}/{token.sell_count_5m}, "
                f"FDV ${token.fdv:.0f}. Reply ONLY 'BUY' or 'SKIP'."
            )
            async with self.session.post(
                "http://localhost:8080/api/chat",
                json={"message": prompt, "bot": "Cosmos", "mode": "quick"},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    reply = (data.get("response") or "").upper()
                    if "BUY" in reply:
                        return {"verdict": "BUY", "confidence": 60, "reasons": ["swarm_api: BUY"]}
        except Exception:
            pass
        return {"verdict": "SKIP", "confidence": 0, "reasons": ["swarm_unavailable"]}


# ============================================================
# COPY TRADE ENGINE
# ============================================================
class CopyTradeEngine:
    """Track and copy-trade the most profitable Solana wallets."""

    def __init__(self, session: aiohttp.ClientSession, fast_rpc: str):
        self.session = session
        self.fast_rpc = fast_rpc
        self.tracked_wallets: Dict[str, TrackedWallet] = {}
        self.copy_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._running = False
        self._monitor_task = None
        self._birdeye_key = os.environ.get("BIRDEYE_API_KEY", "")

    async def start(self):
        self._running = True
        await self._discover_top_wallets()
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(f"CopyTrade engine started, tracking {len(self.tracked_wallets)} wallets")

    async def stop(self):
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()

    async def _discover_top_wallets(self):
        """Find most profitable wallets from GMGN, Birdeye, and DexScreener top holders."""
        await self._fetch_gmgn_wallets()
        if self._birdeye_key:
            await self._fetch_birdeye_wallets()
        await self._fetch_top_holder_wallets()
        logger.info(f"Discovered {len(self.tracked_wallets)} wallets to track")

    async def _fetch_gmgn_wallets(self):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
            }
            params = {"orderby": "smartmoney", "direction": "desc"}
            async with self.session.get(
                GMGN_SMART_MONEY_URL, params=params, headers=headers,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    logger.debug(f"GMGN returned {resp.status}")
                    return
                data = await resp.json()
                rank = data.get("data", {}).get("rank", [])
                for item in rank[:30]:
                    # Extract creator/deployer addresses as smart money
                    creator = item.get("creator_address") or item.get("maker", "")
                    if creator and creator not in self.tracked_wallets:
                        self.tracked_wallets[creator] = TrackedWallet(
                            address=creator,
                            label="gmgn_smart_money",
                            source="gmgn",
                        )
        except Exception as e:
            logger.debug(f"GMGN fetch error: {e}")

    async def _fetch_birdeye_wallets(self):
        try:
            headers = {"X-API-KEY": self._birdeye_key}
            url = f"{BIRDEYE_BASE_URL}/trader/gainers-losers"
            params = {"chain": "solana", "type": "gainers", "sort_by": "PnL", "limit": 20}
            async with self.session.get(
                url, headers=headers, params=params,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()
                traders = data.get("data", {}).get("items", [])
                for trader in traders:
                    addr = trader.get("address", "")
                    if addr and addr not in self.tracked_wallets:
                        self.tracked_wallets[addr] = TrackedWallet(
                            address=addr,
                            label="birdeye_top_gainer",
                            win_rate=float(trader.get("win_rate", 0)),
                            total_pnl=float(trader.get("pnl", 0)),
                            source="birdeye",
                        )
        except Exception as e:
            logger.debug(f"Birdeye fetch error: {e}")

    async def _fetch_top_holder_wallets(self):
        """Get wallets from top holders of recent DexScreener boosted tokens."""
        try:
            async with self.session.get(DEXSCREENER_BOOSTS) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()
                items = data if isinstance(data, list) else []
                sol_tokens = [i for i in items if i.get("chainId") == "solana"][:5]

            for token_item in sol_tokens:
                addr = token_item.get("tokenAddress", "")
                if not addr:
                    continue
                payload = {
                    "jsonrpc": "2.0", "id": 1,
                    "method": "getTokenLargestAccounts",
                    "params": [addr],
                }
                try:
                    async with self.session.post(self.fast_rpc, json=payload) as resp:
                        data = await resp.json()
                        accounts = data.get("result", {}).get("value", [])
                    for acc in accounts[:3]:
                        owner_payload = {
                            "jsonrpc": "2.0", "id": 1,
                            "method": "getAccountInfo",
                            "params": [acc.get("address", ""), {"encoding": "jsonParsed"}],
                        }
                        async with self.session.post(self.fast_rpc, json=owner_payload) as resp:
                            od = await resp.json()
                            owner = (od.get("result", {}).get("value") or {}).get("data", {}).get("parsed", {}).get("info", {}).get("owner", "")
                        if owner and owner not in self.tracked_wallets:
                            self.tracked_wallets[owner] = TrackedWallet(
                                address=owner, label="top_holder", source="dexscreener_boost",
                            )
                    await asyncio.sleep(0.3)
                except Exception:
                    continue
        except Exception as e:
            logger.debug(f"Top holder wallet fetch error: {e}")

    async def _monitor_loop(self):
        """Continuously monitor tracked wallets for new buys."""
        while self._running:
            try:
                wallets = list(self.tracked_wallets.values())
                for wallet in wallets:
                    if not self._running:
                        break
                    new_buys = await self._check_wallet_activity(wallet)
                    for buy in new_buys:
                        try:
                            self.copy_queue.put_nowait(buy)
                        except asyncio.QueueFull:
                            self.copy_queue.get_nowait()
                            self.copy_queue.put_nowait(buy)
                    await asyncio.sleep(0.3)
                # Re-discover every 15 minutes
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Copy trade monitor error: {e}")
                await asyncio.sleep(10)

    async def _check_wallet_activity(self, wallet: TrackedWallet) -> List[dict]:
        """Check a tracked wallet for recent token purchases."""
        new_buys = []
        try:
            payload = {
                "jsonrpc": "2.0", "id": 1,
                "method": "getSignaturesForAddress",
                "params": [wallet.address, {"limit": 5}],
            }
            async with self.session.post(self.fast_rpc, json=payload) as resp:
                data = await resp.json()
                sigs = data.get("result", [])

            for sig_info in sigs:
                block_time = sig_info.get("blockTime", 0)
                # Only recent transactions (last 90 seconds)
                if block_time and time.time() - block_time > 90:
                    continue
                sig = sig_info.get("signature", "")
                if not sig:
                    continue

                tx_payload = {
                    "jsonrpc": "2.0", "id": 1,
                    "method": "getTransaction",
                    "params": [sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}],
                }
                async with self.session.post(self.fast_rpc, json=tx_payload) as resp:
                    tx_data = await resp.json()
                    tx = tx_data.get("result")
                if not tx:
                    continue

                # Detect token buys via post/pre token balance changes
                post_tokens = tx.get("meta", {}).get("postTokenBalances", [])
                pre_tokens = tx.get("meta", {}).get("preTokenBalances", [])

                for post in post_tokens:
                    if post.get("owner") != wallet.address:
                        continue
                    mint = post.get("mint", "")
                    if mint in (SOL_MINT, USDC_MINT):
                        continue
                    post_amt = float(post.get("uiTokenAmount", {}).get("uiAmount", 0) or 0)
                    pre_amt = 0.0
                    for pre in pre_tokens:
                        if pre.get("mint") == mint and pre.get("owner") == wallet.address:
                            pre_amt = float(pre.get("uiTokenAmount", {}).get("uiAmount", 0) or 0)
                            break
                    if post_amt > pre_amt:
                        # Estimate SOL spent from balance changes
                        pre_bals = tx.get("meta", {}).get("preBalances", [])
                        post_bals = tx.get("meta", {}).get("postBalances", [])
                        acc_keys = tx.get("transaction", {}).get("message", {}).get("accountKeys", [])
                        sol_spent = 0
                        for i, (pb, ab) in enumerate(zip(pre_bals, post_bals)):
                            ak = acc_keys[i] if i < len(acc_keys) else {}
                            aa = ak.get("pubkey", "") if isinstance(ak, dict) else str(ak)
                            if aa == wallet.address:
                                sol_spent = (pb - ab) / LAMPORTS_PER_SOL
                                break
                        if sol_spent > 0.01:
                            new_buys.append({
                                "mint": mint, "wallet": wallet.address,
                                "wallet_label": wallet.label, "sol_spent": sol_spent,
                                "timestamp": block_time, "tx": sig,
                            })
                            logger.info(f"COPY SIGNAL: {wallet.label} bought {mint[:12]}... for {sol_spent:.2f} SOL")
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.debug(f"Wallet check error for {wallet.address[:12]}: {e}")
        return new_buys

    def get_copy_signals(self) -> List[dict]:
        signals = []
        while not self.copy_queue.empty():
            try:
                signals.append(self.copy_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return signals


# ============================================================
# X/TWITTER SENTINEL
# ============================================================
class XSentinelMonitor:
    """Monitor X/Twitter for cabal movements and trending memecoins via Grok."""

    def __init__(self):
        self._running = False
        self._task = None
        self.trending_tokens: Dict[str, dict] = {}
        self.cabal_alerts: deque = deque(maxlen=50)
        self._scan_interval = 45

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._scan_loop())
        logger.info("X Sentinel monitor started")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()

    async def _scan_loop(self):
        while self._running:
            try:
                await self._scan_x_sentiment()
                await asyncio.sleep(self._scan_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"X sentinel error: {e}")
                await asyncio.sleep(30)

    async def _scan_x_sentiment(self):
        """Use Grok shadow agent to scan X for memecoin signals."""
        try:
            from Cosmos.core.collective.persistent_agent import call_shadow_agent
        except ImportError:
            return

        prompt = (
            "Scan X/Twitter RIGHT NOW for Solana memecoin activity in the last 30 minutes. "
            "Find: 1) Tokens being shilled by multiple accounts simultaneously (cabal activity), "
            "2) Tokens with sudden spike in mentions, "
            "3) KOL (key opinion leader) calls on new Solana tokens, "
            "4) Dev wallet movements or insider alerts. "
            "Reply in EXACT format, one per line:\n"
            "TOKEN:$SYMBOL|ADDRESS_IF_KNOWN|SIGNAL_TYPE|STRENGTH(1-10)|BRIEF_REASON\n"
            "SIGNAL_TYPE must be one of: CABAL_SHILL, KOL_CALL, TRENDING, INSIDER_ALERT, DEV_MOVEMENT\n"
            "Only include tokens with REAL activity RIGHT NOW. Max 5 tokens."
        )

        try:
            result = await asyncio.wait_for(
                call_shadow_agent("grok", prompt, max_tokens=300, timeout=15.0),
                timeout=18.0,
            )
            if result:
                _, response = result
                self._parse_x_signals(response)
        except (asyncio.TimeoutError, Exception) as e:
            logger.debug(f"X sentinel Grok timeout: {e}")

    # Established tokens that will NEVER pass fresh-launch filters — skip them
    ESTABLISHED_BLACKLIST = {
        "WIF", "BONK", "DOGE", "SHIB", "PEPE", "FLOKI", "MYRO", "MEW",
        "POPCAT", "WEN", "BOME", "SLERF", "SAMO", "ORCA", "RAY", "JUP",
        "JTO", "PYTH", "RENDER", "HNT", "SOL", "USDC", "USDT", "BTC",
        "ETH", "TRUMP", "MELANIA", "PNUT", "GOAT", "MOODENG", "SPX",
        "GIGA", "RETARDIO", "PONKE", "MOTHER", "FWOG", "MICHI", "LUCE",
    }

    def _parse_x_signals(self, response: str):
        if not response:
            return
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line.startswith("TOKEN:"):
                continue
            try:
                parts = line[6:].split("|")
                if len(parts) < 4:
                    continue
                symbol = parts[0].strip().lstrip("$")
                # Skip established tokens — they'll never pass our fresh launch filters
                if symbol.upper() in self.ESTABLISHED_BLACKLIST:
                    logger.debug(f"X sentinel: skipping established token ${symbol}")
                    continue
                address = parts[1].strip() if len(parts[1].strip()) > 20 else ""
                signal_type = parts[2].strip()
                strength = min(10, max(1, int(parts[3].strip())))
                reason = parts[4].strip() if len(parts) > 4 else ""

                key = address if address else symbol.upper()
                self.trending_tokens[key] = {
                    "symbol": symbol, "address": address,
                    "signal_type": signal_type, "strength": strength,
                    "reason": reason, "timestamp": time.time(),
                }
                if signal_type in ("CABAL_SHILL", "INSIDER_ALERT", "DEV_MOVEMENT"):
                    self.cabal_alerts.append({
                        "symbol": symbol, "type": signal_type,
                        "strength": strength, "reason": reason, "time": time.time(),
                    })
                logger.info(f"X SIGNAL: ${symbol} | {signal_type} | strength={strength} | {reason}")
            except (ValueError, IndexError):
                continue
        # Cleanup old signals (>10 min)
        cutoff = time.time() - 600
        self.trending_tokens = {k: v for k, v in self.trending_tokens.items() if v.get("timestamp", 0) > cutoff}

    def get_hot_tokens(self, min_strength: int = 5) -> List[dict]:
        return [v for v in self.trending_tokens.values() if v.get("strength", 0) >= min_strength]

    def get_token_x_boost(self, token_address: str, symbol: str) -> int:
        """Get X sentiment score bonus for a token. Returns 0-20."""
        signal = self.trending_tokens.get(token_address) or self.trending_tokens.get(symbol.upper())
        if not signal:
            return 0
        strength = signal.get("strength", 0)
        if strength >= 8:
            return 20
        elif strength >= 6:
            return 15
        elif strength >= 4:
            return 10
        return 5


# ============================================================
# QUANTUM TRADE ORACLE (enhanced)
# ============================================================
class QuantumTradeOracle:
    """Enhanced quantum analysis: IBM Quantum QAOA, quantum random timing, FarsightProtocol."""

    def __init__(self):
        self._farsight = None
        self._quantum_proof = None

    async def analyze(self, token: TokenInfo, session: aiohttp.ClientSession,
                      historical_patterns: List[dict] = None) -> Dict:
        """Full quantum analysis pipeline. Returns rug_probability, quantum_edge, timing_jitter."""
        result = {
            "rug_probability": 0.5, "quantum_edge": 0,
            "timing_jitter_ms": 0, "pattern_match": None, "confidence": 0.0,
        }

        # 1. Statistical rug probability (fast, always runs)
        result["rug_probability"] = self._statistical_rug_score(token)

        # 2. FarsightProtocol quantum-enhanced analysis
        farsight_result = await self._farsight_analyze(token, session)
        if farsight_result:
            q_rug = farsight_result.get("rug_probability", 0.5)
            result["rug_probability"] = result["rug_probability"] * 0.4 + q_rug * 0.6
            result["confidence"] = farsight_result.get("confidence", 0)

        # 3. Quantum random entry timing jitter
        result["timing_jitter_ms"] = await self._quantum_timing()

        # 4. Pattern matching against historical trades
        if historical_patterns:
            match = self._match_historical_pattern(token, historical_patterns)
            if match:
                result["pattern_match"] = match
                result["quantum_edge"] = match.get("edge", 0)

        # 5. QAOA-inspired position hint via Bell state measurement
        qaoa_hint = await self._qaoa_position_hint()
        if qaoa_hint:
            result["quantum_edge"] += qaoa_hint

        return result

    def _statistical_rug_score(self, token: TokenInfo) -> float:
        signals = []
        if token.fdv > 0:
            liq_ratio = token.liquidity_usd / token.fdv
            signals.append(0.7 if liq_ratio < 0.05 else 0.4 if liq_ratio < 0.1 else 0.1)
        else:
            signals.append(0.5)
        if token.age_minutes < 5:
            signals.append(0.8)
        elif token.age_minutes < 15:
            signals.append(0.5)
        elif token.age_minutes < 60:
            signals.append(0.3)
        else:
            signals.append(0.15)
        total_txns = token.buy_count_5m + token.sell_count_5m
        if total_txns > 0:
            signals.append(min(1.0, (token.sell_count_5m / total_txns) * 1.2))
        else:
            signals.append(0.5)
        if token.price_change_5m < -30:
            signals.append(0.9)
        elif token.price_change_5m < -15:
            signals.append(0.6)
        else:
            signals.append(0.1)
        return sum(signals) / len(signals) if signals else 0.5

    async def _farsight_analyze(self, token: TokenInfo, session: aiohttp.ClientSession) -> Optional[Dict]:
        try:
            farsight = await self._get_farsight()
            if not farsight:
                return None
            # Try analyze_token first (purpose-built for crypto)
            try:
                result = await asyncio.wait_for(farsight.analyze_token(token.address), timeout=10.0)
                if result:
                    return {
                        "rug_probability": result.get("rug_probability", 0.5),
                        "confidence": result.get("confidence", 0),
                    }
            except (asyncio.TimeoutError, Exception):
                pass
            # Fallback to predict() with quantum enabled
            prediction = await asyncio.wait_for(
                farsight.predict(
                    f"Token ${token.symbol}: liq ${token.liquidity_usd:.0f}, age {token.age_minutes:.0f}m, "
                    f"FDV ${token.fdv:.0f}, B/S(5m) {token.buy_count_5m}/{token.sell_count_5m}. "
                    f"Probability of rug pull in next 1 hour?",
                    category="crypto", include_visual=False, include_quantum=True,
                ),
                timeout=12.0,
            )
            if prediction and prediction.simulation_outcomes:
                rug_prob = prediction.simulation_outcomes.get("rug_pull", 0)
                return {
                    "rug_probability": rug_prob if rug_prob > 0 else 0.5,
                    "confidence": prediction.farsight_confidence,
                }
        except asyncio.TimeoutError:
            logger.debug("FarsightProtocol timeout")
        except Exception as e:
            logger.debug(f"FarsightProtocol error: {e}")
        return None

    async def _quantum_timing(self) -> int:
        """Quantum-random timing jitter to avoid predictable entry patterns."""
        try:
            qp = await self._get_quantum_proof()
            if qp:
                job = await asyncio.wait_for(qp.run_quantum_random(num_bits=8, shots=1), timeout=5.0)
                if job and job.results:
                    bits = list(job.results.keys())[0] if job.results else "00000000"
                    return int(bits, 2) * 8  # 0-2040ms
        except (asyncio.TimeoutError, Exception):
            pass
        return int.from_bytes(os.urandom(2), "big") % 2000

    def _match_historical_pattern(self, token: TokenInfo, patterns: List[dict]) -> Optional[dict]:
        """Find closest matching historical trade pattern."""
        best_match = None
        best_sim = 0.0
        for p in patterns:
            sim = 0.0
            if p.get("source") == token.source:
                sim += 0.2
            p_liq = p.get("liquidity_at_entry", 0)
            if p_liq > 0 and token.liquidity_usd > 0:
                sim += min(p_liq, token.liquidity_usd) / max(p_liq, token.liquidity_usd) * 0.25
            p_age = p.get("age_at_entry", 0)
            if p_age > 0 and token.age_minutes > 0:
                sim += min(p_age, token.age_minutes) / max(p_age, token.age_minutes) * 0.2
            p_score = p.get("entry_score", 0)
            if p_score > 0 and token.score > 0:
                sim += min(p_score, token.score) / max(p_score, token.score) * 0.15
            if p.get("cabal_score", 0) > 50 and token.cabal_score > 50:
                sim += 0.1
            if p.get("outcome") == "win":
                sim += 0.1
            if sim > best_sim:
                best_sim = sim
                edge = 0
                if p.get("outcome") == "win":
                    edge = min(20, int(p.get("pnl_multiple", 1) * 5))
                elif p.get("outcome") == "loss":
                    edge = -10
                elif p.get("outcome") == "rug":
                    edge = -15
                best_match = {
                    "similarity": round(sim, 2), "outcome": p.get("outcome", "unknown"),
                    "pnl_multiple": p.get("pnl_multiple", 1),
                    "edge": max(-20, min(20, edge)), "pattern_symbol": p.get("symbol", "?"),
                }
        return best_match if best_match and best_sim > 0.5 else None

    async def _qaoa_position_hint(self) -> int:
        """QAOA-inspired hint via Bell state measurement."""
        try:
            qp = await self._get_quantum_proof()
            if not qp:
                return 0
            job = await asyncio.wait_for(qp.run_bell_state(shots=20), timeout=5.0)
            if not job or not job.results:
                return 0
            counts = job.results
            total = sum(counts.values())
            if total == 0:
                return 0
            correlated = counts.get("00", 0) + counts.get("11", 0)
            ratio = correlated / total
            if ratio > 0.7:
                return 5
            elif ratio < 0.3:
                return -5
            return 0
        except (asyncio.TimeoutError, Exception):
            return 0

    async def _get_farsight(self):
        if self._farsight is None:
            try:
                from Cosmos.integration.hackathon.farsight_protocol import FarsightProtocol
                self._farsight = FarsightProtocol()
            except ImportError:
                self._farsight = False
        return self._farsight if self._farsight else None

    async def _get_quantum_proof(self):
        if self._quantum_proof is None:
            try:
                from Cosmos.integration.hackathon.quantum_proof import QuantumProof
                self._quantum_proof = QuantumProof()
            except ImportError:
                self._quantum_proof = False
        return self._quantum_proof if self._quantum_proof else None


# ============================================================
# QUANTUM WALLET PREDICTION ENGINE (v3.7)
# ============================================================
class QuantumWalletPredictor:
    """Predict wallet movements BEFORE they happen using quantum correlation.

    Tracks wallet behavior patterns across the PumpPortal stream:
    - Wallet buy sequences (what tokens a wallet buys in order)
    - Time-between-buys patterns (cadence detection)
    - Cross-wallet correlation (when wallet A buys, wallet B follows)
    - Token affinity scoring (which token characteristics attract which wallets)

    Uses quantum Bell state correlations to weight prediction confidence:
    strongly correlated (|00⟩+|11⟩) = high confidence prediction
    anti-correlated (|01⟩+|10⟩) = inverse/contrarian signal
    """

    def __init__(self):
        # wallet -> list of (mint, timestamp, sol_amount) in order
        self.wallet_sequences: Dict[str, List[Tuple[str, float, float]]] = {}
        # wallet_pair -> correlation score (how often they buy same tokens)
        self.wallet_correlations: Dict[str, float] = {}
        # wallet -> average time between buys in seconds
        self.wallet_cadence: Dict[str, float] = {}
        # Predictions: mint -> {predicted_buyers, confidence, quantum_correlation}
        self.predictions: Dict[str, dict] = {}
        self._quantum_proof = None
        self._max_wallet_history = 20
        self._max_predictions = 30
        self.prediction_hits = 0
        self.prediction_misses = 0

    def record_buy(self, wallet: str, mint: str, sol_amount: float):
        """Record a wallet buy and update correlation patterns."""
        now = time.time()
        if wallet not in self.wallet_sequences:
            self.wallet_sequences[wallet] = []
        seq = self.wallet_sequences[wallet]
        seq.append((mint, now, sol_amount))
        if len(seq) > self._max_wallet_history:
            self.wallet_sequences[wallet] = seq[-self._max_wallet_history:]

        # Update cadence (average time between buys)
        if len(seq) >= 2:
            deltas = [seq[i][1] - seq[i-1][1] for i in range(1, len(seq))]
            self.wallet_cadence[wallet] = sum(deltas) / len(deltas)

        # Check if this was a predicted buy
        pred = self.predictions.get(mint)
        if pred and wallet in pred.get("predicted_buyers", set()):
            self.prediction_hits += 1
            logger.info(f"QUANTUM PREDICTION HIT: {wallet[:8]}... bought ${mint[:8]} as predicted (conf: {pred.get('confidence', 0):.0%})")

        # Update wallet-pair correlations
        for other_wallet, other_seq in self.wallet_sequences.items():
            if other_wallet == wallet:
                continue
            other_mints = {s[0] for s in other_seq}
            my_mints = {s[0] for s in seq}
            shared = my_mints & other_mints
            if len(shared) >= 2:
                pair_key = tuple(sorted([wallet, other_wallet]))
                correlation = len(shared) / max(len(my_mints), len(other_mints))
                self.wallet_correlations[str(pair_key)] = correlation

    async def predict_next_buys(self, hot_tokens: Dict[str, dict],
                                wallet_token_buys: Dict[str, set]) -> List[dict]:
        """Predict which tokens are about to get bought and by whom.

        Uses wallet cadence, correlation patterns, and quantum Bell states
        to predict the next likely buys.
        """
        predictions = []
        now = time.time()

        # Find wallets that are "due" for their next buy (based on cadence)
        active_wallets = []
        for wallet, cadence in self.wallet_cadence.items():
            if cadence <= 0 or cadence > 600:  # skip if > 10 min cadence
                continue
            seq = self.wallet_sequences.get(wallet, [])
            if not seq:
                continue
            last_buy_time = seq[-1][1]
            time_since = now - last_buy_time
            # Wallet is "due" if time since last buy > 80% of their cadence
            if time_since > cadence * 0.8:
                active_wallets.append({
                    "wallet": wallet,
                    "cadence": cadence,
                    "overdue_ratio": time_since / cadence,
                    "last_tokens": [s[0] for s in seq[-5:]],
                    "avg_sol": sum(s[2] for s in seq) / len(seq) if seq else 0,
                })

        if not active_wallets:
            return predictions

        # For each hot token, calculate probability of being bought next
        for mint, stats in hot_tokens.items():
            if stats.get("buys", 0) < 1:
                continue
            age_s = now - stats.get("first_seen", now)
            if age_s > 900:  # only predict for tokens < 15 min
                continue

            predicted_buyers = set()
            total_confidence = 0

            for aw in active_wallets[:20]:  # cap computation
                wallet = aw["wallet"]
                wallet_mints = wallet_token_buys.get(wallet, set())

                # Check if correlated wallets already bought this token
                token_buyers = set()
                for buyer in stats.get("unique_buyers", set()):
                    if buyer != wallet:
                        pair_key = str(tuple(sorted([wallet, buyer])))
                        corr = self.wallet_correlations.get(pair_key, 0)
                        if corr > 0.3:
                            token_buyers.add(buyer)

                if token_buyers:
                    # Correlated wallets bought this → predict this wallet will too
                    overdue = min(2.0, aw["overdue_ratio"])
                    corr_strength = len(token_buyers) / max(1, len(stats.get("unique_buyers", set())))
                    confidence = min(0.95, overdue * 0.3 + corr_strength * 0.5)
                    if confidence > 0.3:
                        predicted_buyers.add(wallet)
                        total_confidence += confidence

            if predicted_buyers and total_confidence > 0.3:
                # Quantum enhancement: use Bell state to weight confidence
                q_factor = await self._quantum_correlation_factor()
                final_confidence = min(0.95, (total_confidence / len(predicted_buyers)) * (0.7 + q_factor * 0.3))

                pred = {
                    "mint": mint,
                    "symbol": stats.get("symbol", ""),
                    "predicted_buyers": predicted_buyers,
                    "predicted_count": len(predicted_buyers),
                    "confidence": final_confidence,
                    "quantum_factor": q_factor,
                    "age_seconds": age_s,
                    "current_buys": stats.get("buys", 0),
                    "timestamp": now,
                }
                predictions.append(pred)
                self.predictions[mint] = pred

        # Cleanup old predictions
        cutoff = now - 300  # 5 min
        self.predictions = {k: v for k, v in self.predictions.items() if v.get("timestamp", 0) > cutoff}

        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        return predictions[:self._max_predictions]

    async def _quantum_correlation_factor(self) -> float:
        """Use Bell state measurement to get quantum correlation factor (0-1)."""
        try:
            qp = await self._get_quantum()
            if not qp:
                return 0.5  # neutral
            job = await asyncio.wait_for(qp.run_bell_state(shots=10), timeout=3.0)
            if not job or not job.results:
                return 0.5
            counts = job.results
            total = sum(counts.values())
            if total == 0:
                return 0.5
            correlated = counts.get("00", 0) + counts.get("11", 0)
            return correlated / total  # 0 = anti-correlated, 1 = strongly correlated
        except (asyncio.TimeoutError, Exception):
            return 0.5

    async def _get_quantum(self):
        if self._quantum_proof is None:
            try:
                from Cosmos.integration.hackathon.quantum_proof import QuantumProof
                self._quantum_proof = QuantumProof()
            except ImportError:
                self._quantum_proof = False
        return self._quantum_proof if self._quantum_proof else None

    def get_prediction_feed(self, max_items: int = 10) -> List[dict]:
        """Get current predictions for dashboard display."""
        now = time.time()
        result = []
        for mint, pred in sorted(self.predictions.items(),
                                  key=lambda x: x[1].get("confidence", 0), reverse=True):
            if now - pred.get("timestamp", 0) > 300:
                continue
            result.append({
                "mint": mint,
                "symbol": pred.get("symbol", ""),
                "predicted_buyers": pred.get("predicted_count", 0),
                "confidence": round(pred.get("confidence", 0), 2),
                "quantum_factor": round(pred.get("quantum_factor", 0.5), 2),
                "age_seconds": round(now - pred.get("timestamp", now)),
                "current_buys": pred.get("current_buys", 0),
            })
        return result[:max_items]

    def cleanup(self):
        """Clean up old data to prevent memory growth."""
        if len(self.wallet_sequences) > 1000:
            # Keep only most active wallets
            sorted_wallets = sorted(self.wallet_sequences.keys(),
                                    key=lambda w: len(self.wallet_sequences[w]), reverse=True)
            keep = set(sorted_wallets[:500])
            self.wallet_sequences = {w: s for w, s in self.wallet_sequences.items() if w in keep}
            self.wallet_cadence = {w: c for w, c in self.wallet_cadence.items() if w in keep}
        if len(self.wallet_correlations) > 5000:
            # Keep top correlations only
            top = sorted(self.wallet_correlations.items(), key=lambda x: x[1], reverse=True)[:2500]
            self.wallet_correlations = dict(top)


# ============================================================
# TRADING MEMORY (learns from every trade)
# ============================================================
class TradingMemory:
    """Trading-specific memory layer. Stores outcomes, learns patterns, improves scoring."""

    def __init__(self):
        self._memory = None
        self._initialized = False
        self._trade_patterns: List[dict] = []
        self._score_adjustments: Dict[str, float] = {}

    async def initialize(self):
        try:
            from Cosmos.memory import MemorySystem
            self._memory = MemorySystem()
            await self._memory.initialize()
            self._initialized = True
            await self._load_trade_patterns()
            logger.info(f"TradingMemory initialized, {len(self._trade_patterns)} historical patterns loaded")
        except Exception as e:
            logger.warning(f"TradingMemory init failed (running without memory): {e}")
            self._initialized = False

    async def record_trade(self, entry: TradeMemoryEntry):
        """Store a trade outcome in archival memory + knowledge graph."""
        # Always cache locally
        pattern = {
            "symbol": entry.symbol, "entry_score": entry.entry_score,
            "rug_probability": entry.rug_probability, "cabal_score": entry.cabal_score,
            "source": entry.source, "outcome": entry.outcome,
            "pnl_multiple": entry.pnl_multiple, "hold_minutes": entry.hold_minutes,
            "liquidity_at_entry": entry.liquidity_at_entry, "age_at_entry": entry.age_at_entry,
        }
        self._trade_patterns.append(pattern)
        self._update_score_adjustments()

        if not self._initialized:
            return

        try:
            content = (
                f"TRADE {entry.action.upper()} ${entry.symbol} ({entry.token_address[:12]}...): "
                f"Score={entry.entry_score:.0f}, Rug={entry.rug_probability:.0%}, "
                f"Swarm={entry.swarm_sentiment}, Cabal={entry.cabal_score:.0f}, "
                f"Source={entry.source}, Outcome={entry.outcome}, "
                f"PnL={entry.pnl_multiple:.2f}x in {entry.hold_minutes:.0f}m, "
                f"Liq=${entry.liquidity_at_entry:.0f}, Age={entry.age_at_entry:.0f}m"
            )
            tags = [
                "trading", f"outcome_{entry.outcome}", f"source_{entry.source}",
                f"action_{entry.action}", entry.symbol,
            ]
            if entry.cabal_score > 50:
                tags.append("cabal_detected")
            if entry.pnl_multiple >= 2.0:
                tags.append("big_win")
            elif entry.pnl_multiple <= 0.3:
                tags.append("big_loss")

            importance = 0.7 if entry.outcome in ("win", "rug") else 0.5
            valence = 0.8 if entry.outcome == "win" else -0.5 if entry.outcome in ("loss", "rug") else 0.0

            await self._memory.remember(
                content=content, tags=tags, importance=importance,
                metadata={
                    "type": "trade_outcome", "token_address": entry.token_address,
                    "symbol": entry.symbol, "entry_score": entry.entry_score,
                    "rug_probability": entry.rug_probability, "cabal_score": entry.cabal_score,
                    "source": entry.source, "outcome": entry.outcome,
                    "pnl_multiple": entry.pnl_multiple, "hold_minutes": entry.hold_minutes,
                    "liquidity_at_entry": entry.liquidity_at_entry,
                    "age_at_entry": entry.age_at_entry,
                    "timestamp": entry.timestamp or time.time(),
                },
                emotional_valence=valence,
            )

            # Add to knowledge graph
            try:
                kg = self._memory.knowledge_graph
                if kg:
                    await kg.add_entity(
                        name=entry.symbol, entity_type="token",
                        properties={"address": entry.token_address, "last_outcome": entry.outcome, "last_pnl": entry.pnl_multiple},
                    )
                    await kg.add_relationship(
                        source=entry.symbol, target=f"outcome_{entry.outcome}",
                        relation_type="resulted_in", weight=entry.pnl_multiple, evidence=content,
                    )
                    if entry.source:
                        await kg.add_relationship(
                            source=entry.symbol, target=f"source_{entry.source}",
                            relation_type="detected_by", weight=1.0,
                        )
            except Exception:
                pass

            logger.info(f"Trade recorded to memory: ${entry.symbol} {entry.outcome} {entry.pnl_multiple:.2f}x")
        except Exception as e:
            logger.debug(f"TradingMemory record error: {e}")

    async def recall_similar_trades(self, token: TokenInfo) -> List[dict]:
        """Find past trades with similar characteristics."""
        if not self._initialized:
            return self._trade_patterns
        try:
            results = await self._memory.recall(
                query=f"memecoin trade ${token.symbol} liquidity ${token.liquidity_usd:.0f} age {token.age_minutes:.0f}m source {token.source}",
                top_k=20, min_score=0.2,
            )
            patterns = [r.metadata for r in results if (r.metadata or {}).get("type") == "trade_outcome"]
            return patterns if patterns else self._trade_patterns
        except Exception:
            return self._trade_patterns

    def get_learned_adjustment(self, feature: str) -> float:
        return self._score_adjustments.get(feature, 0)

    def get_historical_patterns(self) -> List[dict]:
        return self._trade_patterns

    def _update_score_adjustments(self):
        if len(self._trade_patterns) < 5:
            return
        source_outcomes = defaultdict(list)
        cabal_outcomes = []
        for p in self._trade_patterns:
            source_outcomes[p.get("source", "unknown")].append(p.get("pnl_multiple", 1.0))
            if p.get("cabal_score", 0) > 50:
                cabal_outcomes.append(p.get("pnl_multiple", 1.0))

        for source, pnls in source_outcomes.items():
            avg = sum(pnls) / len(pnls)
            if avg > 1.5:
                self._score_adjustments[f"source_{source}"] = min(15, int((avg - 1) * 10))
            elif avg < 0.5:
                self._score_adjustments[f"source_{source}"] = max(-15, -int((1 - avg) * 10))
        if cabal_outcomes:
            avg_cabal = sum(cabal_outcomes) / len(cabal_outcomes)
            self._score_adjustments["cabal_detected"] = 10 if avg_cabal > 1.3 else -10 if avg_cabal < 0.6 else 0

    async def _load_trade_patterns(self):
        if not self._initialized:
            return
        try:
            results = await self._memory.recall(query="memecoin trade outcome win loss rug", top_k=100, min_score=0.1)
            for r in results:
                meta = r.metadata or {}
                if meta.get("type") == "trade_outcome":
                    self._trade_patterns.append(meta)
            self._update_score_adjustments()
        except Exception as e:
            logger.debug(f"Trade pattern load error: {e}")

    async def get_win_rate_by_source(self) -> Dict[str, float]:
        source_stats = defaultdict(lambda: {"wins": 0, "total": 0})
        for p in self._trade_patterns:
            source = p.get("source", "unknown")
            source_stats[source]["total"] += 1
            if p.get("outcome") == "win":
                source_stats[source]["wins"] += 1
        return {
            source: round(s["wins"] / s["total"] * 100, 1) if s["total"] > 0 else 0
            for source, s in source_stats.items()
        }


# ============================================================
# ADAPTIVE LEARNER (v4.0 — Collective Intelligence Engine)
# ============================================================
class AdaptiveLearner:
    """Real-time adaptive learning engine that uses the swarm collective to
    analyze trade patterns and auto-tune strategy parameters.

    - Records rich per-trade data (velocity, buys, curve progress, holder concentration, etc.)
    - Categorizes failure modes (creator_rug, velocity_death, sell_pressure, stop_loss, timeout)
    - Runs local statistical analysis: win rates by condition buckets
    - Periodically asks the swarm collective (Grok/DeepSeek/Gemini) for deeper pattern insights
    - Auto-adjusts TraderConfig thresholds toward what's actually profitable
    """

    # Failure categories for pattern analysis
    FAILURE_CATEGORIES = {
        "creator_rug": ["creator_rug", "creator_sold"],
        "holder_dump": ["holder_dump"],
        "velocity_death": ["velocity_death", "curve_velocity_death", "dead_momentum"],
        "sell_pressure": ["sell_pressure", "curve_sell_pressure"],
        "stop_loss": ["stop_loss"],
        "time_exit": ["time_exit", "max_hold_curve"],
        "liquidity_rug": ["liquidity_rug"],
        "data_unavailable": ["data_unavailable"],
    }

    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        self.session = session
        self._trades: List[dict] = []          # rich trade records
        self._adjustments: Dict[str, float] = {}  # learned config adjustments
        self._failure_counts: Dict[str, int] = defaultdict(int)
        self._condition_stats: Dict[str, Dict] = {}  # condition → {wins, losses, avg_pnl}
        self._collective_insights: List[dict] = []  # swarm analysis results
        self._last_analysis_time = 0.0
        self._last_collective_time = 0.0
        self._trades_since_analysis = 0
        self._analysis_task = None
        self._running = False
        # Learnable parameters with their bounds
        self._tunable_params = {
            "min_score": {"min": 25, "max": 70, "step": 3, "default": 40},
            "bonding_curve_min_buys": {"min": 1, "max": 8, "step": 1, "default": 2},
            "bonding_curve_min_velocity": {"min": 0.3, "max": 5.0, "step": 0.3, "default": 1.0},
            "bonding_curve_max_progress": {"min": 20, "max": 80, "step": 5, "default": 50},
            "quick_take_profit": {"min": 1.05, "max": 1.5, "step": 0.05, "default": 1.15},
            "quick_take_profit_2": {"min": 1.1, "max": 2.0, "step": 0.05, "default": 1.25},
            "stop_loss": {"min": 0.5, "max": 0.9, "step": 0.05, "default": 0.7},
            "max_hold_minutes": {"min": 5, "max": 60, "step": 5, "default": 20},
            "max_age_minutes": {"min": 5, "max": 30, "step": 2, "default": 15},
            "velocity_drop_sell_pct": {"min": 0.2, "max": 0.7, "step": 0.05, "default": 0.4},
            "instant_snipe_min_dev_sol": {"min": 0.5, "max": 10, "step": 0.5, "default": 2.0},
            "cabal_follow_min_wallets": {"min": 2, "max": 5, "step": 1, "default": 2},
        }
        logger.info("AdaptiveLearner v4.0 initialized — collective intelligence engine ready")

    async def start(self):
        """Start background analysis loop."""
        self._running = True
        self._analysis_task = asyncio.create_task(self._analysis_loop())
        logger.info("AdaptiveLearner background analysis started")

    async def stop(self):
        self._running = False
        if self._analysis_task:
            self._analysis_task.cancel()

    def record_trade(self, trade_data: dict):
        """Record a completed trade with rich context for learning.

        trade_data should include:
            symbol, source, outcome, pnl_multiple, pnl_sol, sol_spent, sol_received,
            hold_minutes, sell_reason, entry_velocity, entry_buys, entry_unique_buyers,
            entry_curve_progress, on_bonding_curve, entry_score, rug_probability,
            cabal_score, liquidity_at_entry, age_at_entry, fdv_at_entry,
            holder_concentration, timestamp
        """
        trade_data.setdefault("timestamp", time.time())
        self._trades.append(trade_data)
        self._trades_since_analysis += 1

        # Categorize failure
        reason = trade_data.get("sell_reason", "")
        outcome = trade_data.get("outcome", "")
        if outcome in ("loss", "rug", "timeout"):
            category = self._categorize_failure(reason)
            self._failure_counts[category] += 1

        # Run quick local analysis after every 3 trades
        if self._trades_since_analysis >= 3:
            self._analyze_patterns_local()

        logger.info(
            f"LEARNER: recorded {trade_data.get('symbol', '?')} "
            f"outcome={outcome} pnl={trade_data.get('pnl_sol', 0):+.4f} "
            f"reason={reason} | total_trades={len(self._trades)}"
        )

    def _categorize_failure(self, reason: str) -> str:
        """Map a sell reason string to a failure category."""
        reason_lower = reason.lower()
        for category, keywords in self.FAILURE_CATEGORIES.items():
            for kw in keywords:
                if kw in reason_lower:
                    return category
        return "other"

    def _analyze_patterns_local(self):
        """Fast local statistical analysis — runs after every few trades."""
        if len(self._trades) < 3:
            return

        self._trades_since_analysis = 0
        self._last_analysis_time = time.time()
        sells = [t for t in self._trades if t.get("outcome") in ("win", "loss", "rug", "timeout")]
        if not sells:
            return

        # Win rate by source
        source_stats = defaultdict(lambda: {"wins": 0, "total": 0, "pnl_sum": 0.0})
        for t in sells:
            src = t.get("source", "unknown")
            source_stats[src]["total"] += 1
            source_stats[src]["pnl_sum"] += t.get("pnl_sol", 0)
            if t.get("outcome") == "win":
                source_stats[src]["wins"] += 1

        # Win rate by velocity bucket
        vel_buckets = {"low_vel(<1)": [], "med_vel(1-3)": [], "high_vel(3+)": []}
        for t in sells:
            v = t.get("entry_velocity", 0)
            if v < 1:
                vel_buckets["low_vel(<1)"].append(t)
            elif v < 3:
                vel_buckets["med_vel(1-3)"].append(t)
            else:
                vel_buckets["high_vel(3+)"].append(t)

        # Win rate by curve vs DEX
        curve_trades = [t for t in sells if t.get("on_bonding_curve")]
        dex_trades = [t for t in sells if not t.get("on_bonding_curve")]

        # Win rate by hold time bucket
        hold_buckets = {"<2min": [], "2-5min": [], "5-15min": [], "15+min": []}
        for t in sells:
            h = t.get("hold_minutes", 0)
            if h < 2:
                hold_buckets["<2min"].append(t)
            elif h < 5:
                hold_buckets["2-5min"].append(t)
            elif h < 15:
                hold_buckets["5-15min"].append(t)
            else:
                hold_buckets["15+min"].append(t)

        # Compute stats for all buckets
        self._condition_stats = {}
        for label, trades_list in [
            *[("source_" + k, v) for k, v in [
                (src, [t for t in sells if t.get("source") == src]) for src in source_stats
            ]],
            *vel_buckets.items(),
            ("bonding_curve", curve_trades),
            ("dex_pool", dex_trades),
            *hold_buckets.items(),
        ]:
            if not trades_list:
                continue
            wins = sum(1 for t in trades_list if t.get("outcome") == "win")
            total = len(trades_list)
            avg_pnl = sum(t.get("pnl_sol", 0) for t in trades_list) / total
            self._condition_stats[label] = {
                "wins": wins, "total": total,
                "win_rate": round(wins / total * 100, 1),
                "avg_pnl_sol": round(avg_pnl, 6),
            }

        # Derive config adjustments from patterns
        self._derive_adjustments(sells, source_stats, vel_buckets, hold_buckets)

        logger.info(
            f"LEARNER ANALYSIS: {len(sells)} trades | "
            f"failures={dict(self._failure_counts)} | "
            f"adjustments={self._adjustments}"
        )

    def _derive_adjustments(self, sells, source_stats, vel_buckets, hold_buckets):
        """Derive config parameter adjustments from statistical patterns."""
        total = len(sells)
        if total < 5:
            return

        overall_wr = sum(1 for t in sells if t.get("outcome") == "win") / total

        # 1. Velocity threshold: if low velocity trades consistently lose, raise minimum
        low_vel = vel_buckets.get("low_vel(<1)", [])
        if len(low_vel) >= 3:
            low_wr = sum(1 for t in low_vel if t.get("outcome") == "win") / len(low_vel)
            if low_wr < 0.15:  # <15% win rate at low velocity → raise min velocity
                self._adjustments["bonding_curve_min_velocity"] = 0.3  # bump up
            elif low_wr > overall_wr:
                self._adjustments["bonding_curve_min_velocity"] = -0.2  # loosen

        # 2. Hold time: if long holds always lose, shorten max_hold
        long_holds = hold_buckets.get("15+min", [])
        if len(long_holds) >= 3:
            long_wr = sum(1 for t in long_holds if t.get("outcome") == "win") / len(long_holds)
            if long_wr < 0.1:
                self._adjustments["max_hold_minutes"] = -5  # reduce max hold
            elif long_wr > 0.4:
                self._adjustments["max_hold_minutes"] = 5  # can hold longer

        # 3. Stop loss: if most losses are tiny (0.7-0.9x), our stop is fine.
        #    If we're getting stopped out and price recovers, loosen stop.
        stop_loss_trades = [t for t in sells if "stop_loss" in t.get("sell_reason", "")]
        if len(stop_loss_trades) >= 3:
            sl_pnl = sum(t.get("pnl_sol", 0) for t in stop_loss_trades) / len(stop_loss_trades)
            if sl_pnl < -0.01:  # average stop loss is big → tighten
                self._adjustments["stop_loss"] = 0.05  # raise stop loss (less loss per trade)
            # If stop losses are a big % of all trades → maybe too tight
            if len(stop_loss_trades) / total > 0.4:
                self._adjustments["stop_loss"] = -0.05  # loosen

        # 4. Take profit: if wins are tiny, maybe take profit too early
        wins = [t for t in sells if t.get("outcome") == "win"]
        if len(wins) >= 3:
            avg_win_mult = sum(t.get("pnl_multiple", 1) for t in wins) / len(wins)
            if avg_win_mult < 1.1:  # winning only 10% on average → hold longer
                self._adjustments["quick_take_profit"] = 0.05
                self._adjustments["quick_take_profit_2"] = 0.05
            elif avg_win_mult > 1.5:  # big wins → current TP is good or can tighten
                self._adjustments["quick_take_profit"] = -0.03

        # 5. Score threshold: if low-score trades always lose
        low_score_trades = [t for t in sells if t.get("entry_score", 100) < 35]
        if len(low_score_trades) >= 3:
            ls_wr = sum(1 for t in low_score_trades if t.get("outcome") == "win") / len(low_score_trades)
            if ls_wr < 0.1:
                self._adjustments["min_score"] = 5  # raise min score
            elif ls_wr > overall_wr:
                self._adjustments["min_score"] = -3

        # 6. Failure-specific: if creator rugs are >30% of losses, need better creator screening
        losses = [t for t in sells if t.get("outcome") != "win"]
        if losses:
            rug_pct = self._failure_counts.get("creator_rug", 0) / len(losses)
            if rug_pct > 0.3:
                # Raise min buys — creators with more community buys less likely to rug
                self._adjustments["bonding_curve_min_buys"] = 1

            vel_death_pct = self._failure_counts.get("velocity_death", 0) / len(losses)
            if vel_death_pct > 0.3:
                # Velocity death is common → tighten velocity drop sell
                self._adjustments["velocity_drop_sell_pct"] = 0.05

    def apply_to_config(self, config: 'TraderConfig') -> Dict[str, str]:
        """Apply learned adjustments to the live config. Returns dict of changes made."""
        if not self._adjustments:
            return {}

        changes = {}
        for param, delta in self._adjustments.items():
            if param not in self._tunable_params:
                continue
            bounds = self._tunable_params[param]
            current = getattr(config, param, bounds["default"])
            new_val = current + delta

            # Clamp to bounds
            new_val = max(bounds["min"], min(bounds["max"], new_val))

            # Round for cleaner values
            if isinstance(bounds["step"], int):
                new_val = int(round(new_val))
            else:
                new_val = round(new_val, 2)

            if new_val != current:
                setattr(config, param, new_val)
                changes[param] = f"{current} → {new_val}"
                logger.info(f"LEARNER TUNE: {param} {current} → {new_val}")

        if changes:
            logger.info(f"LEARNER: Applied {len(changes)} config adjustments: {changes}")

        # Clear adjustments after applying
        self._adjustments.clear()
        return changes

    async def ask_collective(self, config: 'TraderConfig') -> Optional[dict]:
        """Ask the swarm collective to analyze our trade patterns and suggest improvements.

        Calls the local Cosmos API which routes through Grok/DeepSeek/Gemini.
        """
        if not self.session or len(self._trades) < 5:
            return None

        self._last_collective_time = time.time()
        sells = [t for t in self._trades if t.get("outcome") in ("win", "loss", "rug", "timeout")]
        if len(sells) < 3:
            return None

        # Build a compact summary for the collective
        total = len(sells)
        wins = sum(1 for t in sells if t.get("outcome") == "win")
        total_pnl = sum(t.get("pnl_sol", 0) for t in sells)
        avg_hold = sum(t.get("hold_minutes", 0) for t in sells) / total

        # Recent trades (last 10) for context
        recent = sells[-10:]
        recent_summary = "\n".join([
            f"  ${t.get('symbol','?')}: {t.get('outcome')} {t.get('pnl_sol',0):+.4f} SOL "
            f"({t.get('sell_reason','?')}) vel={t.get('entry_velocity',0):.1f} "
            f"buys={t.get('entry_buys',0)} hold={t.get('hold_minutes',0):.0f}m "
            f"source={t.get('source','?')}"
            for t in recent
        ])

        failure_summary = ", ".join(f"{k}={v}" for k, v in self._failure_counts.items() if v > 0)

        prompt = (
            f"TRADING PATTERN ANALYSIS — Analyze our Solana memecoin trade data and suggest parameter adjustments.\n\n"
            f"STATS: {total} trades, {wins} wins ({wins/total*100:.0f}%), "
            f"net PnL {total_pnl:+.4f} SOL, avg hold {avg_hold:.1f}min\n"
            f"FAILURES: {failure_summary}\n\n"
            f"CURRENT CONFIG:\n"
            f"  min_score={config.min_score}, min_velocity={config.bonding_curve_min_velocity}, "
            f"min_buys={config.bonding_curve_min_buys}, max_progress={config.bonding_curve_max_progress}%\n"
            f"  take_profit={config.quick_take_profit}/{config.quick_take_profit_2}, "
            f"stop_loss=0.7, max_hold={config.max_hold_minutes}min, "
            f"vel_drop_sell={config.velocity_drop_sell_pct}\n\n"
            f"RECENT TRADES:\n{recent_summary}\n\n"
            f"CONDITION WIN RATES:\n"
            + "\n".join(f"  {k}: {v['win_rate']}% ({v['total']} trades, avg {v['avg_pnl_sol']:+.6f} SOL)"
                       for k, v in self._condition_stats.items())
            + "\n\nBased on this data, suggest SPECIFIC numeric parameter changes. "
            "Reply in this EXACT format (only include params that should change):\n"
            "ADJUST min_score=X\nADJUST bonding_curve_min_velocity=X\n"
            "ADJUST quick_take_profit=X\nADJUST max_hold_minutes=X\n"
            "REASONING: <1-2 sentences explaining why>\n"
        )

        try:
            async with self.session.post(
                "http://localhost:8080/api/chat",
                json={"message": prompt, "bot": "Grok", "mode": "quick"},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    reply = data.get("response", "")
                    insight = self._parse_collective_response(reply)
                    if insight:
                        self._collective_insights.append({
                            "timestamp": time.time(),
                            "adjustments": insight.get("adjustments", {}),
                            "reasoning": insight.get("reasoning", ""),
                        })
                        # Merge collective suggestions into our adjustments
                        for param, val in insight.get("adjustments", {}).items():
                            if param in self._tunable_params:
                                current = self._tunable_params[param]["default"]
                                self._adjustments[param] = val - current
                        logger.info(f"COLLECTIVE INSIGHT: {insight}")
                        return insight
        except Exception as e:
            logger.debug(f"Collective analysis failed: {e}")

        return None

    def _parse_collective_response(self, reply: str) -> Optional[dict]:
        """Parse the swarm's response into actionable adjustments."""
        if not reply:
            return None

        adjustments = {}
        reasoning = ""

        for line in reply.split("\n"):
            line = line.strip()
            if line.startswith("ADJUST "):
                try:
                    parts = line[7:].split("=")
                    if len(parts) == 2:
                        param = parts[0].strip()
                        val = float(parts[1].strip())
                        if param in self._tunable_params:
                            bounds = self._tunable_params[param]
                            val = max(bounds["min"], min(bounds["max"], val))
                            adjustments[param] = val
                except (ValueError, IndexError):
                    pass
            elif line.startswith("REASONING:"):
                reasoning = line[10:].strip()

        if adjustments:
            return {"adjustments": adjustments, "reasoning": reasoning}
        return None

    async def _analysis_loop(self):
        """Background loop: run local analysis + periodic collective queries."""
        while self._running:
            try:
                await asyncio.sleep(60)  # check every minute

                # Local analysis every 5 minutes if we have new trades
                if self._trades_since_analysis > 0 and time.time() - self._last_analysis_time > 300:
                    self._analyze_patterns_local()

                # Collective analysis every 15 minutes if we have enough data
                if (len(self._trades) >= 5 and
                        time.time() - self._last_collective_time > 900):
                    # We need a config reference — will be called from DegenTrader
                    pass  # collective is called explicitly by DegenTrader

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"AdaptiveLearner loop error: {e}")
                await asyncio.sleep(30)

    def get_status(self) -> dict:
        """Return learner status for dashboard."""
        sells = [t for t in self._trades if t.get("outcome") in ("win", "loss", "rug", "timeout")]
        total = len(sells)
        wins = sum(1 for t in sells if t.get("outcome") == "win")
        return {
            "total_trades": total,
            "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
            "failure_breakdown": dict(self._failure_counts),
            "condition_stats": self._condition_stats,
            "pending_adjustments": self._adjustments,
            "collective_insights": len(self._collective_insights),
            "last_insight": self._collective_insights[-1] if self._collective_insights else None,
            "trades_since_analysis": self._trades_since_analysis,
        }

    def get_learnings_summary(self) -> str:
        """Human-readable summary of what we've learned."""
        if not self._trades:
            return "No trades recorded yet."

        sells = [t for t in self._trades if t.get("outcome") in ("win", "loss", "rug", "timeout")]
        if not sells:
            return "No completed trades yet."

        total = len(sells)
        wins = sum(1 for t in sells if t.get("outcome") == "win")
        total_pnl = sum(t.get("pnl_sol", 0) for t in sells)

        lines = [
            f"Trades: {total} | Wins: {wins} ({wins/total*100:.0f}%) | Net PnL: {total_pnl:+.4f} SOL",
        ]

        if self._failure_counts:
            top_failures = sorted(self._failure_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            lines.append(f"Top failures: {', '.join(f'{k}({v})' for k,v in top_failures)}")

        if self._condition_stats:
            best = max(self._condition_stats.items(), key=lambda x: x[1].get("win_rate", 0))
            worst = min(self._condition_stats.items(), key=lambda x: x[1].get("win_rate", 100))
            if best[1]["total"] >= 2:
                lines.append(f"Best condition: {best[0]} ({best[1]['win_rate']}% WR)")
            if worst[1]["total"] >= 2:
                lines.append(f"Worst condition: {worst[0]} ({worst[1]['win_rate']}% WR)")

        if self._collective_insights:
            last = self._collective_insights[-1]
            if last.get("reasoning"):
                lines.append(f"Collective says: {last['reasoning']}")

        return " | ".join(lines)


# ============================================================
# WHALE HUNTER — Discover, Track, Learn from Top Wallets
# ============================================================
WHALE_DB_PATH = Path(__file__).parent / ".whale_db.json"

# Known Solana mixer / privacy programs and bridge contracts
KNOWN_MIXER_PROGRAMS = {
    "mix1111111111111111111111111111111111111111",   # Placeholder for Elusiv/Light Protocol
    "2b1kV6DkPAnxd5ixfnxCpjxmKwqjjaYmCZfHsFu24GR",  # Elusiv program (deprecated)
    "E1us1vDbqqZkHRSHioh2UNR6FreAjSczw3KgnBPUNXch",  # Elusiv V2 (if active)
    "worm2ZoG2kUd4vFXhvjh93UUH596ayRfgQ2MgjNMTth",   # Wormhole Bridge
    "DZnkkTmCiFWfYTfT41X3Rd1kDgozqzxWaHqsw6W4x2oe",  # DeBridge
    "br1xwubggTiEZ6b7iNZUwfA3cvvzMFMoHkoM8ojeGBz",   # AllBridge
}


class WhaleHunter:
    """Discover, track, and learn from the most profitable wallets on Solana.

    Features:
    - Discovery: GMGN smart money, Birdeye top traders, Jito bundle detection
    - Persistent DB: .whale_db.json with win rates, PnL, connected wallets
    - Connected wallet tracing: funding source analysis to find wallet clusters
    - Mixer monitoring: track SOL flowing through privacy protocols
    - Signal generation: whale convergence boosts token scores
    """

    def __init__(self, rpc_url: str, fast_rpc_url: str = ""):
        self.rpc_url = rpc_url
        self.fast_rpc = fast_rpc_url or rpc_url
        self.session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._task = None
        self._birdeye_key = os.environ.get("BIRDEYE_API_KEY", "")
        self._discovery_interval = 900  # 15 minutes

        # Whale database (persistent)
        self.wallets: Dict[str, dict] = {}  # address -> whale profile
        self._load_db()

        # Recent whale buys for signal generation (in-memory, rolling)
        self._recent_whale_buys: Dict[str, List[dict]] = defaultdict(list)  # mint -> [buy events]
        self._whale_buy_signals: asyncio.Queue = asyncio.Queue(maxsize=100)

        # Mixer monitoring
        self._mixer_outflows: List[dict] = []  # recent outflows from mixers
        self._mixer_fresh_wallets: Dict[str, float] = {}  # wallet -> timestamp first seen from mixer
        self._mixer_buy_signals: asyncio.Queue = asyncio.Queue(maxsize=50)

        logger.info(f"WhaleHunter initialized — {len(self.wallets)} wallets in DB")

    def _load_db(self):
        """Load persistent whale database."""
        try:
            if WHALE_DB_PATH.exists():
                data = json.loads(WHALE_DB_PATH.read_text())
                self.wallets = data.get("wallets", {})
                logger.info(f"WhaleHunter loaded {len(self.wallets)} tracked wallets from DB")
        except Exception as e:
            logger.debug(f"WhaleHunter DB load error: {e}")
            self.wallets = {}

    def _save_db(self):
        """Persist whale database to disk."""
        try:
            WHALE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {"wallets": self.wallets, "updated_at": time.time()}
            WHALE_DB_PATH.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.debug(f"WhaleHunter DB save error: {e}")

    async def init_session(self, session: aiohttp.ClientSession):
        self.session = session

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._hunter_loop())
        logger.info("WhaleHunter started — hunting for smart money wallets")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
        self._save_db()

    async def _hunter_loop(self):
        """Main loop: discover whales every 15 min, monitor activity continuously."""
        while self._running:
            try:
                await self._discover_whales()
                await self._monitor_whale_activity()
                await self._monitor_mixer_outflows()
                self._save_db()
                await asyncio.sleep(self._discovery_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"WhaleHunter loop error: {e}")
                await asyncio.sleep(60)

    # ----------------------------------------------------------
    # DISCOVERY — Find profitable wallets
    # ----------------------------------------------------------
    async def _discover_whales(self):
        """Discover top wallets from GMGN, Birdeye, and Jito analysis."""
        if not self.session:
            return

        discovered = 0

        # Source 1: GMGN smart money ranking
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
            }
            params = {"orderby": "smartmoney", "direction": "desc"}
            async with self.session.get(
                GMGN_SMART_MONEY_URL, params=params, headers=headers,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    rank = data.get("data", {}).get("rank", [])
                    for item in rank[:30]:
                        addr = item.get("creator_address") or item.get("maker", "")
                        if addr and addr not in self.wallets:
                            self.wallets[addr] = {
                                "label": "gmgn_smart_money",
                                "first_seen": time.time(),
                                "last_active": time.time(),
                                "win_rate": 0,
                                "total_pnl_sol": 0,
                                "avg_hold_minutes": 0,
                                "tokens_traded": 0,
                                "connected_wallets": [],
                                "funding_source": "",
                                "is_frontrunner": False,
                                "tags": ["smart_money"],
                            }
                            discovered += 1
        except Exception as e:
            logger.debug(f"WhaleHunter GMGN discovery error: {e}")

        # Source 2: Birdeye top gainers
        if self._birdeye_key:
            try:
                headers = {"X-API-KEY": self._birdeye_key}
                url = f"{BIRDEYE_BASE_URL}/trader/gainers-losers"
                params = {"chain": "solana", "type": "gainers", "sort_by": "PnL", "limit": 20}
                async with self.session.get(
                    url, headers=headers, params=params,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        traders = data.get("data", {}).get("items", [])
                        for trader in traders:
                            addr = trader.get("address", "")
                            if addr:
                                pnl = float(trader.get("pnl", 0))
                                wr = float(trader.get("win_rate", 0))
                                if addr not in self.wallets:
                                    self.wallets[addr] = {
                                        "label": "birdeye_top_gainer",
                                        "first_seen": time.time(),
                                        "last_active": time.time(),
                                        "win_rate": wr,
                                        "total_pnl_sol": pnl,
                                        "avg_hold_minutes": 0,
                                        "tokens_traded": int(trader.get("trade_count", 0)),
                                        "connected_wallets": [],
                                        "funding_source": "",
                                        "is_frontrunner": False,
                                        "tags": ["high_pnl"],
                                    }
                                    discovered += 1
                                else:
                                    # Update existing
                                    self.wallets[addr]["win_rate"] = wr
                                    self.wallets[addr]["total_pnl_sol"] = pnl
                                    self.wallets[addr]["last_active"] = time.time()
            except Exception as e:
                logger.debug(f"WhaleHunter Birdeye discovery error: {e}")

        # Trace connected wallets for top performers
        await self._discover_connected_wallets()

        if discovered > 0:
            logger.info(f"WhaleHunter discovered {discovered} new wallets, total tracked: {len(self.wallets)}")

    async def _discover_connected_wallets(self):
        """For top wallets, trace funding source to find connected wallet clusters."""
        if not self.session:
            return

        # Only trace wallets that haven't been traced yet
        untraced = [
            (addr, w) for addr, w in self.wallets.items()
            if not w.get("funding_source") and w.get("label") in ("gmgn_smart_money", "birdeye_top_gainer")
        ][:10]  # max 10 per cycle

        for addr, profile in untraced:
            try:
                payload = {
                    "jsonrpc": "2.0", "id": 1,
                    "method": "getSignaturesForAddress",
                    "params": [addr, {"limit": 5}],
                }
                async with self.session.post(
                    self.fast_rpc, json=payload,
                    timeout=aiohttp.ClientTimeout(total=8),
                ) as resp:
                    if resp.status != 200:
                        continue
                    data = await resp.json()
                    sigs = data.get("result", [])

                if not sigs:
                    continue

                # Check earliest tx for funding source
                earliest_sig = sigs[-1].get("signature", "")
                if not earliest_sig:
                    continue

                tx_payload = {
                    "jsonrpc": "2.0", "id": 1,
                    "method": "getTransaction",
                    "params": [earliest_sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}],
                }
                async with self.session.post(
                    self.fast_rpc, json=tx_payload,
                    timeout=aiohttp.ClientTimeout(total=8),
                ) as resp:
                    if resp.status != 200:
                        continue
                    tx_data = await resp.json()
                    tx = tx_data.get("result", {})

                if not tx:
                    continue

                instructions = tx.get("transaction", {}).get("message", {}).get("instructions", [])
                for ix in instructions:
                    parsed = ix.get("parsed", {})
                    if parsed.get("type") == "transfer":
                        info = parsed.get("info", {})
                        if info.get("destination") == addr:
                            funder = info.get("source", "")
                            if funder:
                                profile["funding_source"] = funder
                                # Find other wallets funded by same source
                                for other_addr, other_w in self.wallets.items():
                                    if other_addr != addr and other_w.get("funding_source") == funder:
                                        if other_addr not in profile.get("connected_wallets", []):
                                            profile.setdefault("connected_wallets", []).append(other_addr)
                                        if addr not in other_w.get("connected_wallets", []):
                                            other_w.setdefault("connected_wallets", []).append(addr)
                            break

                await asyncio.sleep(0.2)  # rate limit
            except Exception:
                continue

    # ----------------------------------------------------------
    # WHALE ACTIVITY MONITORING
    # ----------------------------------------------------------
    async def _monitor_whale_activity(self):
        """Check recent transactions of tracked whales for buy signals."""
        if not self.session:
            return

        # Sample up to 20 whales per cycle (rotate through all)
        whale_addrs = list(self.wallets.keys())
        if not whale_addrs:
            return

        # Rotate: check a different subset each cycle
        sample_size = min(20, len(whale_addrs))
        cycle_offset = int(time.time() / self._discovery_interval) % max(1, len(whale_addrs) // sample_size)
        start = cycle_offset * sample_size
        sample = whale_addrs[start:start + sample_size]

        for addr in sample:
            try:
                payload = {
                    "jsonrpc": "2.0", "id": 1,
                    "method": "getSignaturesForAddress",
                    "params": [addr, {"limit": 5}],
                }
                async with self.session.post(
                    self.fast_rpc, json=payload,
                    timeout=aiohttp.ClientTimeout(total=8),
                ) as resp:
                    if resp.status != 200:
                        continue
                    data = await resp.json()
                    sigs = data.get("result", [])

                for sig_info in sigs:
                    block_time = sig_info.get("blockTime", 0)
                    # Only look at txs from last 10 minutes
                    if block_time and time.time() - block_time > 600:
                        continue
                    if sig_info.get("err") is not None:
                        continue

                    sig = sig_info.get("signature", "")
                    if not sig:
                        continue

                    # Parse transaction for token buys
                    tx_payload = {
                        "jsonrpc": "2.0", "id": 1,
                        "method": "getTransaction",
                        "params": [sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}],
                    }
                    async with self.session.post(
                        self.fast_rpc, json=tx_payload,
                        timeout=aiohttp.ClientTimeout(total=8),
                    ) as resp2:
                        if resp2.status != 200:
                            continue
                        tx_data = await resp2.json()
                        tx = tx_data.get("result")

                    if not tx:
                        continue

                    # Check pre/post token balances for new token acquisitions
                    pre_tokens = tx.get("meta", {}).get("preTokenBalances", [])
                    post_tokens = tx.get("meta", {}).get("postTokenBalances", [])

                    # Detect new token buys (post has token balance that pre doesn't)
                    pre_mints = {b.get("mint", ""): b for b in pre_tokens}
                    for post_b in post_tokens:
                        mint = post_b.get("mint", "")
                        owner = post_b.get("owner", "")
                        if owner != addr or not mint or mint == SOL_MINT:
                            continue

                        post_amt = float(post_b.get("uiTokenAmount", {}).get("uiAmount", 0) or 0)
                        pre_amt = 0
                        if mint in pre_mints:
                            pre_amt = float(pre_mints[mint].get("uiTokenAmount", {}).get("uiAmount", 0) or 0)

                        if post_amt > pre_amt and post_amt - pre_amt > 0:
                            # This whale bought tokens!
                            buy_event = {
                                "whale": addr,
                                "mint": mint,
                                "tokens_bought": post_amt - pre_amt,
                                "timestamp": block_time or time.time(),
                                "label": self.wallets.get(addr, {}).get("label", ""),
                            }
                            self._recent_whale_buys[mint].append(buy_event)
                            # Trim old entries
                            cutoff = time.time() - 600  # 10 min window
                            self._recent_whale_buys[mint] = [
                                b for b in self._recent_whale_buys[mint] if b["timestamp"] > cutoff
                            ]

                            # Update whale last_active
                            if addr in self.wallets:
                                self.wallets[addr]["last_active"] = time.time()

                            break  # one buy per tx

                await asyncio.sleep(0.15)
            except Exception:
                continue

    # ----------------------------------------------------------
    # MIXER / PRIVACY MONITORING
    # ----------------------------------------------------------
    async def _monitor_mixer_outflows(self):
        """Monitor known mixer programs for SOL outflows to fresh wallets."""
        if not self.session:
            return

        for mixer_prog in list(KNOWN_MIXER_PROGRAMS)[:3]:  # check top 3 per cycle
            try:
                payload = {
                    "jsonrpc": "2.0", "id": 1,
                    "method": "getSignaturesForAddress",
                    "params": [mixer_prog, {"limit": 15}],
                }
                async with self.session.post(
                    self.fast_rpc, json=payload,
                    timeout=aiohttp.ClientTimeout(total=8),
                ) as resp:
                    if resp.status != 200:
                        continue
                    data = await resp.json()
                    sigs = data.get("result", [])

                for sig_info in sigs:
                    block_time = sig_info.get("blockTime", 0)
                    if block_time and time.time() - block_time > 300:  # last 5 min
                        continue
                    if sig_info.get("err") is not None:
                        continue
                    sig = sig_info.get("signature", "")
                    if not sig:
                        continue

                    # Parse tx for SOL transfers OUT of mixer
                    tx_payload = {
                        "jsonrpc": "2.0", "id": 1,
                        "method": "getTransaction",
                        "params": [sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}],
                    }
                    async with self.session.post(
                        self.fast_rpc, json=tx_payload,
                        timeout=aiohttp.ClientTimeout(total=8),
                    ) as resp2:
                        if resp2.status != 200:
                            continue
                        tx_data = await resp2.json()
                        tx = tx_data.get("result")

                    if not tx:
                        continue

                    # Check post balances for wallets that received SOL
                    pre_balances = tx.get("meta", {}).get("preBalances", [])
                    post_balances = tx.get("meta", {}).get("postBalances", [])
                    accounts = tx.get("transaction", {}).get("message", {}).get("accountKeys", [])

                    for i, (pre, post) in enumerate(zip(pre_balances, post_balances)):
                        sol_received = (post - pre) / LAMPORTS_PER_SOL
                        if sol_received > 0.1:  # meaningful SOL outflow
                            acct_key = accounts[i] if i < len(accounts) else {}
                            dest_addr = acct_key.get("pubkey", "") if isinstance(acct_key, dict) else str(acct_key)
                            if dest_addr and dest_addr != mixer_prog:
                                self._mixer_fresh_wallets[dest_addr] = time.time()
                                self._mixer_outflows.append({
                                    "wallet": dest_addr,
                                    "sol_amount": sol_received,
                                    "mixer": mixer_prog[:16],
                                    "timestamp": time.time(),
                                })
                                # Flag existing tracked whale if using mixer
                                if dest_addr in self.wallets:
                                    self.wallets[dest_addr]["tags"] = list(set(
                                        self.wallets[dest_addr].get("tags", []) + ["mixer_user"]
                                    ))

                    await asyncio.sleep(0.15)
            except Exception:
                continue

        # Trim old outflows (keep last 30 min)
        cutoff = time.time() - 1800
        self._mixer_outflows = [o for o in self._mixer_outflows if o["timestamp"] > cutoff]
        # Trim old fresh wallets (keep 30 min window)
        self._mixer_fresh_wallets = {
            w: t for w, t in self._mixer_fresh_wallets.items() if time.time() - t < 1800
        }

        # Check if any mixer-funded fresh wallets are now buying tokens
        await self._check_mixer_wallet_buys()

    async def _check_mixer_wallet_buys(self):
        """Check if fresh wallets from mixer outflows are now buying tokens.
        If SOL came from mixer into wallet that buys a token with good liquidity
        and price action → generate buy signal for 0.05 SOL."""
        if not self.session or not self._mixer_fresh_wallets:
            return

        # Sample up to 10 fresh mixer wallets
        fresh_wallets = list(self._mixer_fresh_wallets.keys())[:10]

        for wallet_addr in fresh_wallets:
            try:
                payload = {
                    "jsonrpc": "2.0", "id": 1,
                    "method": "getSignaturesForAddress",
                    "params": [wallet_addr, {"limit": 5}],
                }
                async with self.session.post(
                    self.fast_rpc, json=payload,
                    timeout=aiohttp.ClientTimeout(total=8),
                ) as resp:
                    if resp.status != 200:
                        continue
                    data = await resp.json()
                    sigs = data.get("result", [])

                for sig_info in sigs:
                    block_time = sig_info.get("blockTime", 0)
                    if block_time and time.time() - block_time > 300:
                        continue
                    sig = sig_info.get("signature", "")
                    if not sig:
                        continue

                    tx_payload = {
                        "jsonrpc": "2.0", "id": 1,
                        "method": "getTransaction",
                        "params": [sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}],
                    }
                    async with self.session.post(
                        self.fast_rpc, json=tx_payload,
                        timeout=aiohttp.ClientTimeout(total=8),
                    ) as resp2:
                        if resp2.status != 200:
                            continue
                        tx_data = await resp2.json()
                        tx = tx_data.get("result")

                    if not tx:
                        continue

                    # Detect token buys
                    pre_tokens = tx.get("meta", {}).get("preTokenBalances", [])
                    post_tokens = tx.get("meta", {}).get("postTokenBalances", [])
                    pre_mints = {b.get("mint", ""): b for b in pre_tokens}

                    for post_b in post_tokens:
                        mint = post_b.get("mint", "")
                        owner = post_b.get("owner", "")
                        if owner != wallet_addr or not mint or mint == SOL_MINT:
                            continue
                        post_amt = float(post_b.get("uiTokenAmount", {}).get("uiAmount", 0) or 0)
                        pre_amt = 0
                        if mint in pre_mints:
                            pre_amt = float(pre_mints[mint].get("uiTokenAmount", {}).get("uiAmount", 0) or 0)

                        if post_amt > pre_amt:
                            # Mixer-funded wallet bought a token! Check health via DexScreener
                            try:
                                async with self.session.get(
                                    f"{DEXSCREENER_TOKENS}/{mint}",
                                    timeout=aiohttp.ClientTimeout(total=5),
                                ) as ds_resp:
                                    if ds_resp.status == 200:
                                        ds_data = await ds_resp.json()
                                        pairs = ds_data.get("pairs") or []
                                        if pairs:
                                            pair = pairs[0]
                                            liq = float(pair.get("liquidity", {}).get("usd", 0) or 0)
                                            vol_24h = float(pair.get("volume", {}).get("h24", 0) or 0)
                                            price_change_5m = float(pair.get("priceChange", {}).get("m5", 0) or 0)
                                            buys_5m = int(pair.get("txns", {}).get("m5", {}).get("buys", 0) or 0)
                                            symbol = pair.get("baseToken", {}).get("symbol", "?")

                                            # Good liquidity + positive price action
                                            if liq >= 5000 and vol_24h >= 1000 and price_change_5m > -10 and buys_5m >= 3:
                                                signal = {
                                                    "mint": mint,
                                                    "symbol": symbol,
                                                    "source": "mixer_wallet",
                                                    "wallet": wallet_addr[:12] + "...",
                                                    "liquidity": liq,
                                                    "volume_24h": vol_24h,
                                                    "price_change_5m": price_change_5m,
                                                    "timestamp": time.time(),
                                                    "max_sol": 0.02,
                                                }
                                                try:
                                                    self._mixer_buy_signals.put_nowait(signal)
                                                    logger.info(
                                                        f"MIXER WALLET BUY: ${symbol} ({mint[:12]}...) | "
                                                        f"wallet from mixer buying | liq=${liq:.0f} | "
                                                        f"vol=${vol_24h:.0f} | Δ5m={price_change_5m:+.1f}%"
                                                    )
                                                except asyncio.QueueFull:
                                                    pass

                                                # Add fresh wallet to tracked whales
                                                if wallet_addr not in self.wallets:
                                                    self.wallets[wallet_addr] = {
                                                        "label": "mixer_funded",
                                                        "first_seen": time.time(),
                                                        "last_active": time.time(),
                                                        "win_rate": 0,
                                                        "total_pnl_sol": 0,
                                                        "avg_hold_minutes": 0,
                                                        "tokens_traded": 1,
                                                        "connected_wallets": [],
                                                        "funding_source": "mixer",
                                                        "is_frontrunner": False,
                                                        "tags": ["mixer_funded", "fresh"],
                                                    }
                            except Exception:
                                pass
                            break

                await asyncio.sleep(0.15)
            except Exception:
                continue

    # ----------------------------------------------------------
    # SIGNAL GENERATION — for scoring pipeline
    # ----------------------------------------------------------
    def get_whale_score_boost(self, mint: str) -> int:
        """Get score boost for a token based on tracked whale activity.
        Returns: 0 (no signal), +15-25 (single whale), +35 (convergence)"""
        whale_buys = self._recent_whale_buys.get(mint, [])
        if not whale_buys:
            return 0

        # Deduplicate by whale address
        unique_whales = set(b["whale"] for b in whale_buys)
        cutoff = time.time() - 600  # last 10 minutes
        recent = [b for b in whale_buys if b["timestamp"] > cutoff]
        recent_whales = set(b["whale"] for b in recent)

        if len(recent_whales) >= 2:
            return 35  # CONVERGENCE: 2+ tracked whales on same token
        elif len(recent_whales) == 1:
            whale_addr = list(recent_whales)[0]
            whale_profile = self.wallets.get(whale_addr, {})
            wr = whale_profile.get("win_rate", 0)
            if wr >= 60:
                return 25  # high win-rate whale
            return 15  # any tracked whale
        return 0

    def get_whale_feed(self) -> List[dict]:
        """Get recent whale activity for the dashboard feed."""
        feed = []
        cutoff = time.time() - 600
        for mint, buys in self._recent_whale_buys.items():
            for b in buys:
                if b["timestamp"] > cutoff:
                    feed.append({
                        "mint": mint,
                        "whale": b["whale"][:12] + "...",
                        "label": b.get("label", ""),
                        "tokens": b.get("tokens_bought", 0),
                        "timestamp": b["timestamp"],
                    })
        feed.sort(key=lambda x: x["timestamp"], reverse=True)
        return feed[:20]

    def get_status(self) -> dict:
        return {
            "tracked_wallets": len(self.wallets),
            "recent_whale_buys": sum(len(v) for v in self._recent_whale_buys.values()),
            "mixer_fresh_wallets": len(self._mixer_fresh_wallets),
            "mixer_outflows": len(self._mixer_outflows),
            "labels": dict(defaultdict(int, {
                w.get("label", "unknown"): 1 for w in self.wallets.values()
            })),
        }


# ============================================================
# DEGEN TRADER (MAIN)
# ============================================================
class DegenTrader:
    """High-frequency Solana memecoin trader powered by collective intelligence."""

    def __init__(self, config: Optional[TraderConfig] = None, wallet_name: str = "degen_trader"):
        # Auth lock — refuse to run outside Cosmos
        _CosmosAuthLock.lock_check()
        self.config = config or TraderConfig()
        self.wallet_name = wallet_name
        self.keypair = None
        self.pubkey = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.seen_tokens: set = set()
        self.running = False
        self.total_pnl_sol = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.start_balance = 0.0
        self.total_invested_sol = 0.0   # v3.9: total SOL spent on buys
        self.total_lost_sol = 0.0       # v3.9: total SOL lost (realized losses)
        self._scan_count = 0

        # Intelligence layers
        self.pump_monitor: Optional[PumpFunMonitor] = None
        self.wallet_analyzer: Optional[WalletAnalyzer] = None
        self.quantum_oracle: Optional[QuantumTradeOracle] = None
        self.swarm_intel: Optional[SwarmTradeIntelligence] = None
        # v3 layers
        self.copy_engine: Optional[CopyTradeEngine] = None
        self.x_sentinel: Optional[XSentinelMonitor] = None
        self.trading_memory: Optional[TradingMemory] = None
        # v3.5: Bonding curve direct trading
        self.curve_engine: Optional[BondingCurveEngine] = None
        self._sniper_bought: set = set()  # mints already sniped
        # v3.7: Re-entry watch list — tokens we sold that we keep monitoring
        self._reentry_watchlist: Dict[str, dict] = {}  # mint -> {sold_at, sold_price, peak_vel, symbol}
        # v3.7: Quantum wallet prediction
        self.wallet_predictor: Optional[QuantumWalletPredictor] = None
        # v3.9: Holder rug detection — monitors top holders of our positions
        self.holder_watcher: Optional[HolderWatcher] = None
        # v4.0: Burn signal monitor — detects LP/supply burns for buy signals
        self.burn_monitor: Optional[BurnSignalMonitor] = None
        # v4.0: Adaptive learning engine — uses collective to learn and auto-tune
        self.adaptive_learner: Optional[AdaptiveLearner] = None
        # v4.1: Whale hunter — discover, track, learn from top wallets + mixer monitoring
        self.whale_hunter: Optional[WhaleHunter] = None
        self._last_collective_query = 0.0  # timestamp of last collective query
        # v4.3: Real-time DEX price feed via Raydium pool vault WebSocket
        self.dex_price_feed: Optional[DexPriceFeed] = None
        # v4.4: Quantum Trading Cortex — fused quantum+EMA+collective signals
        self._quantum_signals: Dict[str, dict] = {}  # token_address -> latest quantum signal
        self._quantum_signal_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        # v3.8: Dynamic adaptation state
        self._quiet_cycles = 0          # consecutive cycles with 0 qualifying tokens
        self._hot_cycles = 0            # consecutive cycles with 3+ qualifying tokens
        self._adapt_score_offset = 0    # dynamic score adjustment (negative = looser)
        self._adapt_vel_mult = 1.0      # dynamic velocity multiplier (< 1 = looser)
        # v4.3: Paper trading virtual state
        self._paper_balance = 0.0                     # virtual SOL balance
        self._paper_token_holdings: Dict[str, int] = {}  # mint -> virtual raw token amount

    async def initialize(self):
        """Load wallet, start session, initialize intelligence layers."""
        # v4.3: Paper trading — set virtual balance, still load wallet for pubkey (read-only)
        if self.config.paper_trade:
            self._paper_balance = self.config.paper_start_balance
            logger.info(f"PAPER TRADE MODE — virtual balance: {self._paper_balance:.4f} SOL (no real SOL will be spent)")

        wallet_path = WALLET_DIR / f"{self.wallet_name}.json"
        if not wallet_path.exists():
            pubkey, _ = create_wallet(self.wallet_name)
            logger.info(f"New wallet generated: {pubkey}")
            if not self.config.paper_trade:
                logger.info(f"Fund this wallet with SOL before starting trades")

        self.keypair = load_wallet(self.wallet_name)
        self.pubkey = str(self.keypair.pubkey())

        # Resolve Alchemy RPC for bulk reads
        if not self.config.fast_rpc_url:
            alchemy_key = os.environ.get("ALCHEMY_API_KEY", "")
            if alchemy_key:
                self.config.fast_rpc_url = f"https://solana-mainnet.g.alchemy.com/v2/{alchemy_key}"
                logger.info("Using Alchemy RPC for bulk reads (30M CU/month)")
            else:
                self.config.fast_rpc_url = self.config.rpc_url

        # v4.2: Resolve Helius staked RPC for sendTransaction (better landing rate)
        if not self.config.helius_api_key:
            self.config.helius_api_key = os.environ.get("HELIUS_API_KEY", "")
        if self.config.helius_api_key and not self.config.helius_rpc_url:
            self.config.helius_rpc_url = f"https://mainnet.helius-rpc.com/?api-key={self.config.helius_api_key}"
            logger.info("Using Helius staked RPC for sendTransaction (better tx landing)")

        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"User-Agent": "CosmosDegenTrader/3.5"}
        )

        balance = await self.get_sol_balance()
        self.start_balance = balance
        logger.info(f"Wallet: {self.pubkey}")
        logger.info(f"Balance: {balance:.4f} SOL")

        # Initialize intelligence layers
        if self.config.use_pumpfun:
            self.pump_monitor = PumpFunMonitor()
            await self.pump_monitor.start()

        if self.config.use_wallet_analysis:
            self.wallet_analyzer = WalletAnalyzer(self.config.rpc_url, self.config.fast_rpc_url)
            await self.wallet_analyzer.init_session(self.session)

        if self.config.use_quantum:
            self.quantum_oracle = QuantumTradeOracle()

        self.swarm_intel = SwarmTradeIntelligence(self.session)

        # v3: Copy trading engine
        if self.config.use_copy_trading:
            self.copy_engine = CopyTradeEngine(self.session, self.config.fast_rpc_url)
            await self.copy_engine.start()

        # v3: X sentinel
        if self.config.use_x_sentinel:
            self.x_sentinel = XSentinelMonitor()
            await self.x_sentinel.start()

        # v3: Trading memory
        if self.config.use_trading_memory:
            self.trading_memory = TradingMemory()
            await self.trading_memory.initialize()

        # v3.5: Bonding curve engine (v4.2: Helius staked for sends)
        if self.config.use_bonding_curve:
            self.curve_engine = BondingCurveEngine(
                self.config.rpc_url, self.config.fast_rpc_url,
                send_rpc_url=self.config.helius_rpc_url or self.config.fast_rpc_url
            )
            logger.info("Bonding curve engine enabled (direct pump.fun trading)")

        # v3.7: Quantum wallet prediction engine
        if self.config.use_quantum:
            self.wallet_predictor = QuantumWalletPredictor()
            logger.info("Quantum wallet prediction engine enabled (Bell state correlations)")

        # v3.9: Holder rug detection — monitors top holders of our positions in real-time
        self.holder_watcher = HolderWatcher(self.config.rpc_url, self.config.fast_rpc_url)
        await self.holder_watcher.init_session(self.session)
        await self.holder_watcher.start()
        logger.info("HolderWatcher enabled — monitoring top holders for rug detection")

        # v4.0: Burn signal monitor — detects LP/supply burns for auto-buy
        self.burn_monitor = BurnSignalMonitor(self.config.rpc_url, self.config.fast_rpc_url)
        await self.burn_monitor.init_session(self.session)
        await self.burn_monitor.start()
        logger.info("BurnSignalMonitor v4.1 enabled — tightened filters, watching incinerator for bullish burns")

        # v4.0: Adaptive learning engine — collective intelligence auto-tuning
        self.adaptive_learner = AdaptiveLearner(session=self.session)
        await self.adaptive_learner.start()
        logger.info("AdaptiveLearner v4.0 enabled — collective intelligence auto-tuning")

        # v4.1: Whale hunter — discover, track, and learn from top wallets
        self.whale_hunter = WhaleHunter(self.config.rpc_url, self.config.fast_rpc_url)
        await self.whale_hunter.init_session(self.session)
        await self.whale_hunter.start()
        logger.info(f"WhaleHunter v4.1 enabled — tracking {len(self.whale_hunter.wallets)} wallets + mixer monitoring")

        # v4.3: Real-time DEX price feed — subscribe to Raydium pool vaults via WebSocket
        ws_rpc = self.config.helius_rpc_url or self.config.fast_rpc_url or self.config.rpc_url
        ws_url = ws_rpc.replace("https://", "wss://").replace("http://", "ws://")
        self.dex_price_feed = DexPriceFeed(self.config.fast_rpc_url or self.config.rpc_url, ws_url)
        await self.dex_price_feed.start()
        logger.info(f"DexPriceFeed v4.3 enabled — real-time Raydium pool vault subscriptions")

        # v4.4: Subscribe to Quantum Trading Cortex signals via Nexus
        try:
            from Cosmos.core.nexus import Nexus, SignalType
            nexus = Nexus._instance
            if nexus:
                await nexus.subscribe(
                    SignalType.QUANTUM_SIGNAL_GENERATED,
                    self._on_quantum_signal
                )
                logger.info("DegenTrader: Subscribed to quantum trading signals via Nexus")
        except Exception as e:
            logger.debug(f"DegenTrader: Quantum signal subscription skipped: {e}")

        self._load_state()
        return self.pubkey

    async def _on_quantum_signal(self, signal):
        """
        v4.4: Handle quantum trading signals from QuantumTradingCortex.
        Factors quantum signals into trade decisions as an additional input.
        High confidence LONG + existing buy conditions = stronger entry.
        High confidence SHORT = exit or skip.
        """
        try:
            payload = signal.payload if hasattr(signal, 'payload') else signal
            if not isinstance(payload, dict):
                return

            token_address = payload.get("token_address", "")
            direction = payload.get("direction", "HOLD")
            confidence = float(payload.get("confidence", 0))
            strength = int(payload.get("strength", 1))

            # Store latest signal for this token
            self._quantum_signals[token_address] = {
                "direction": direction,
                "confidence": confidence,
                "strength": strength,
                "quantum_bull_prob": float(payload.get("quantum_bull_prob", 0.5)),
                "momentum_score": float(payload.get("momentum_score", 0)),
                "reasoning": payload.get("reasoning", ""),
                "ts": time.time(),
            }

            # High-confidence SHORT on a held position → consider exiting
            if direction == "SHORT" and confidence > 0.7 and strength >= 3:
                if token_address in self.positions:
                    logger.info(
                        f"[QUANTUM] Strong SHORT signal for held position "
                        f"{token_address[:8]}.. (confidence={confidence:.0%}, strength={strength})"
                    )

            # Queue signal for main trading loop to process
            try:
                self._quantum_signal_queue.put_nowait(payload)
            except asyncio.QueueFull:
                self._quantum_signal_queue.get_nowait()
                self._quantum_signal_queue.put_nowait(payload)

        except Exception as e:
            logger.debug(f"DegenTrader: Quantum signal handler error: {e}")

    def get_quantum_boost(self, token_address: str) -> float:
        """
        v4.4: Get quantum signal boost for a token's buy/sell score.
        Returns a score modifier: positive = bullish boost, negative = bearish.
        Called by the main trading logic to factor quantum signals.
        """
        sig = self._quantum_signals.get(token_address)
        if not sig or time.time() - sig.get("ts", 0) > 600:  # signals expire after 10 min
            return 0.0

        direction = sig.get("direction", "HOLD")
        confidence = sig.get("confidence", 0)
        strength = sig.get("strength", 1)

        if direction == "LONG":
            return confidence * strength * 2  # max ~10 point boost
        elif direction == "SHORT":
            return -confidence * strength * 2  # max ~-10 penalty
        return 0.0

    async def shutdown(self):
        """Clean shutdown of all systems."""
        self.running = False
        self._save_state()
        if self.pump_monitor:
            await self.pump_monitor.stop()
        if self.copy_engine:
            await self.copy_engine.stop()
        if self.x_sentinel:
            await self.x_sentinel.stop()
        if self.holder_watcher:
            await self.holder_watcher.stop()
        if self.burn_monitor:
            await self.burn_monitor.stop()
        if self.adaptive_learner:
            await self.adaptive_learner.stop()
        if self.whale_hunter:
            await self.whale_hunter.stop()
        if self.dex_price_feed:
            await self.dex_price_feed.stop()
        if self.session:
            await self.session.close()
        logger.info("Trader shut down cleanly")

    # ----------------------------------------------------------
    # BALANCE & RPC
    # ----------------------------------------------------------
    async def get_sol_balance(self) -> float:
        # v4.3: Paper mode — return virtual balance
        if self.config.paper_trade:
            return self._paper_balance
        payload = {"jsonrpc": "2.0", "id": 1, "method": "getBalance", "params": [self.pubkey]}
        async with self.session.post(self.config.rpc_url, json=payload) as resp:
            data = await resp.json()
            return data.get("result", {}).get("value", 0) / LAMPORTS_PER_SOL

    async def get_token_accounts(self) -> Dict[str, float]:
        payload = {
            "jsonrpc": "2.0", "id": 1,
            "method": "getTokenAccountsByOwner",
            "params": [self.pubkey, {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"}, {"encoding": "jsonParsed"}]
        }
        try:
            async with self.session.post(self.config.rpc_url, json=payload) as resp:
                data = await resp.json()
                holdings = {}
                for acc in data.get("result", {}).get("value", []):
                    info = acc.get("account", {}).get("data", {}).get("parsed", {}).get("info", {})
                    mint = info.get("mint", "")
                    amount = float(info.get("tokenAmount", {}).get("uiAmount", 0) or 0)
                    if amount > 0:
                        holdings[mint] = amount
                return holdings
        except Exception as e:
            logger.error(f"Token accounts error: {e}")
            return {}

    # ----------------------------------------------------------
    # TOKEN SCANNING (DexScreener + Pump.fun)
    # ----------------------------------------------------------
    async def scan_new_tokens(self) -> List[TokenInfo]:
        """Scan all sources for promising new Solana launches.

        Multi-source scanning with fallbacks to avoid rate limits:
        1. PumpPortal WSS (real-time, primary for fresh launches)
        2. DexScreener boosted + profiles (backup, slower indexing)
        3. GMGN new pairs (backup, catches tokens DexScreener misses)
        4. Birdeye new tokens (backup, requires API key)
        """
        tokens = []

        # Source 1: Pump.fun real-time (fastest)
        if self.pump_monitor:
            pf_tokens = await self._drain_pumpfun_queue()
            tokens.extend(pf_tokens)

        # Source 2: DexScreener boosted + profiles
        dx_tokens = await self._scan_dexscreener()
        tokens.extend(dx_tokens)

        # Source 3: GMGN new pairs — catches what DexScreener misses
        # Official limit: 2 rps. At 3s scan interval = 0.33 rps → well under limit. Scan every cycle.
        gmgn_tokens = await self._scan_gmgn_new_pairs()
        tokens.extend(gmgn_tokens)

        # Source 4: Birdeye new tokens (requires API key)
        # Official limit: 1 rps (free). At 6s cadence = 0.17 rps → safe. Every 2nd cycle.
        if self._scan_count % 2 == 0 and os.environ.get("BIRDEYE_API_KEY"):
            birdeye_tokens = await self._scan_birdeye_new()
            tokens.extend(birdeye_tokens)

        # Source 5: Pump.fun trending (king-of-the-hill + latest)
        # No documented rate limit on trending page. Scan every cycle.
        pf_trending = await self._scan_pumpfun_trending()
        tokens.extend(pf_trending)

        # Deduplicate by address
        seen = set()
        unique = []
        for t in tokens:
            if t.address not in seen and t.address not in self.seen_tokens:
                seen.add(t.address)
                unique.append(t)

        self._scan_count += 1
        return unique

    async def _drain_pumpfun_queue(self) -> List[TokenInfo]:
        """Get new tokens from pump.fun monitor.

        v3.5: Enhanced to pull bonding curve state for fresh tokens
        instead of waiting for DexScreener indexing.
        """
        tokens = []
        while not self.pump_monitor.new_tokens.empty():
            try:
                pf = self.pump_monitor.new_tokens.get_nowait()
                mint = pf.get("mint", "")
                if not mint or mint in self.seen_tokens or mint in self._sniper_bought:
                    continue

                age_seconds = time.time() - pf.get("timestamp", time.time())
                hot = self.pump_monitor.hot_tokens.get(mint, {})
                buys = hot.get("buys", 0)

                # v3.5: For fresh tokens still on bonding curve, build TokenInfo
                # from pump.fun data + curve state (don't wait for DexScreener)
                if age_seconds < 120 and self.curve_engine:
                    if buys >= self.config.bonding_curve_min_buys:
                        curve_state = await self.curve_engine.get_bonding_curve_state(mint, self.session)
                        if curve_state and not curve_state.complete:
                            velocity = self.pump_monitor.get_buy_velocity(mint)
                            unique = len(hot.get("unique_buyers", set()))
                            token_data = TokenInfo(
                                address=mint,
                                symbol=pf.get("symbol", hot.get("symbol", "???")),
                                name=pf.get("name", hot.get("name", "Unknown")),
                                pair_address="",
                                price_usd=curve_state.price_sol * 150,  # rough SOL->USD
                                liquidity_usd=curve_state.sol_raised * 150,
                                volume_24h=hot.get("volume_sol", 0) * 150,
                                age_minutes=age_seconds / 60,
                                holders=unique,
                                fdv=curve_state.price_sol * 1e9 * 150,  # rough
                                buy_count_5m=buys,
                                sell_count_5m=hot.get("sells", 0),
                                source="bonding_curve",
                                creator_wallet=pf.get("creator", ""),
                                on_bonding_curve=True,
                                curve_progress=curve_state.progress_pct,
                                curve_sol_raised=curve_state.sol_raised,
                                buy_velocity_per_min=velocity,
                                initial_buy_sol=pf.get("initial_buy_sol", 0),
                            )
                            tokens.append(token_data)
                            continue

                # Fallback: try DexScreener for older tokens
                if age_seconds >= 60 or buys >= 5:
                    token_data = await self._fetch_token_data(mint)
                    if token_data:
                        token_data.source = "pumpfun"
                        token_data.creator_wallet = pf.get("creator", "")
                        # Enrich with pump.fun stats
                        pf_stats = self.pump_monitor.get_token_stats(mint)
                        if pf_stats:
                            token_data.buy_velocity_per_min = pf_stats.get("velocity", 0)
                        tokens.append(token_data)
            except asyncio.QueueEmpty:
                break
        return tokens

    async def _scan_dexscreener(self) -> List[TokenInfo]:
        """Scan DexScreener for trending tokens.

        v4.0: Parallel fetch boosts+profiles (60 rpm each = safe at 3s cycle).
        Token lookups use 300 rpm limit = can do ~5/sec. Batch addresses to minimize calls.
        """
        tokens = []
        try:
            # Fetch boosts + profiles in parallel (saves ~2s)
            async def fetch_list(url):
                try:
                    async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=6)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            return data if isinstance(data, list) else []
                        elif resp.status == 429:
                            logger.debug(f"DexScreener rate limited: {url}")
                except Exception:
                    pass
                return []

            boosts, profiles = await asyncio.gather(
                fetch_list(DEXSCREENER_BOOSTS),
                fetch_list(DEXSCREENER_PROFILES),
            )

            # Collect unique Solana addresses from both sources
            addresses = []
            for item in (boosts[:20] + profiles[:15]):
                if item.get("chainId") == "solana":
                    addr = item.get("tokenAddress", "")
                    if addr and addr not in self.seen_tokens and addr not in [a for a in addresses]:
                        addresses.append(addr)

            # Fetch token data — limit to 8 lookups per cycle (300 rpm = 5/sec, we do 8 in ~3s)
            for addr in addresses[:8]:
                token = await self._fetch_token_data(addr)
                if token:
                    tokens.append(token)
                await asyncio.sleep(0.2)  # 200ms gap = 5 rps, under 300 rpm limit

        except Exception as e:
            logger.error(f"DexScreener scan error: {e}")
        return tokens

    async def _scan_gmgn_new_pairs(self) -> List[TokenInfo]:
        """Backup scan: GMGN new Solana pairs (catches tokens DexScreener misses)."""
        tokens = []
        try:
            url = "https://gmgn.ai/defi/quotation/v1/pairs/sol/new_pairs?limit=15&orderby=open_timestamp&direction=desc"
            headers = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}
            async with self.session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return tokens
                data = await resp.json()
                pairs = data.get("data", {}).get("pairs", []) if isinstance(data.get("data"), dict) else []
                for pair in pairs[:10]:
                    addr = pair.get("base_address", pair.get("token_address", ""))
                    if not addr or addr in self.seen_tokens:
                        continue
                    # Quick build from GMGN data
                    created = pair.get("open_timestamp", 0)
                    age_min = (time.time() - created) / 60 if created else 999
                    if age_min > self.config.max_age_minutes:
                        continue
                    fdv = pair.get("fdv", 0) or 0
                    liq = pair.get("liquidity", 0) or 0
                    token = TokenInfo(
                        address=addr,
                        symbol=pair.get("base_symbol", pair.get("symbol", "?")),
                        name=pair.get("base_name", pair.get("name", "")),
                        pair_address=pair.get("pair_address", ""),
                        price_usd=pair.get("price", 0) or 0,
                        liquidity_usd=liq,
                        volume_24h=pair.get("volume_24h", 0) or 0,
                        age_minutes=age_min,
                        fdv=fdv,
                        buy_count_5m=pair.get("buys_5m", 0) or 0,
                        sell_count_5m=pair.get("sells_5m", 0) or 0,
                        source="gmgn",
                    )
                    tokens.append(token)
        except Exception as e:
            logger.debug(f"GMGN new pairs scan error: {e}")
        if tokens:
            logger.info(f"GMGN backup: found {len(tokens)} fresh pairs")
        return tokens

    async def _scan_birdeye_new(self) -> List[TokenInfo]:
        """Backup scan: Birdeye new tokens (requires BIRDEYE_API_KEY)."""
        tokens = []
        try:
            key = os.environ.get("BIRDEYE_API_KEY", "")
            if not key:
                return tokens
            url = f"{BIRDEYE_BASE_URL}/defi/v3/token/new_listing"
            headers = {"X-API-KEY": key, "x-chain": "solana"}
            params = {"limit": 15, "sort_by": "created_at", "sort_type": "desc"}
            async with self.session.get(url, headers=headers, params=params,
                                        timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return tokens
                data = await resp.json()
                items = data.get("data", {}).get("items", []) if isinstance(data.get("data"), dict) else []
                for item in items[:10]:
                    addr = item.get("address", "")
                    if not addr or addr in self.seen_tokens:
                        continue
                    token = await self._fetch_token_data(addr)
                    if token:
                        token.source = "birdeye"
                        tokens.append(token)
        except Exception as e:
            logger.debug(f"Birdeye new tokens scan error: {e}")
        if tokens:
            logger.info(f"Birdeye backup: found {len(tokens)} fresh tokens")
        return tokens

    async def _scan_pumpfun_trending(self) -> List[TokenInfo]:
        """v3.8: Scan pump.fun trending/king-of-the-hill for fresh momentum plays."""
        tokens = []
        try:
            endpoints = [
                f"{PUMPFUN_API_URL}/coins/king-of-the-hill?includeNsfw=false",
                f"{PUMPFUN_API_URL}/coins/latest?offset=0&limit=10&includeNsfw=false",
            ]
            headers = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}
            for url in endpoints:
                try:
                    async with self.session.get(url, headers=headers,
                                                timeout=aiohttp.ClientTimeout(total=8)) as resp:
                        if resp.status != 200:
                            continue
                        data = await resp.json()
                        items = data if isinstance(data, list) else data.get("data", data.get("coins", []))
                        for item in items[:8]:
                            mint = item.get("mint", item.get("address", ""))
                            if not mint or mint in self.seen_tokens:
                                continue
                            # Check if already tracked in hot_tokens
                            if self.pump_monitor and mint in self.pump_monitor.hot_tokens:
                                continue
                            created = item.get("created_timestamp", item.get("createdAt", 0))
                            if isinstance(created, str):
                                try:
                                    from datetime import datetime as _dt
                                    created = _dt.fromisoformat(created.replace("Z", "+00:00")).timestamp()
                                except Exception:
                                    created = 0
                            age_min = (time.time() - created) / 60 if created and created > 1e9 else 999
                            if age_min > self.config.max_age_minutes:
                                continue
                            mc = item.get("usd_market_cap", item.get("market_cap", 0)) or 0
                            if mc > self.config.max_fdv and mc > 0:
                                continue
                            symbol = item.get("symbol", "?")
                            # Register in hot_tokens for velocity tracking
                            if self.pump_monitor:
                                self.pump_monitor.hot_tokens[mint] = {
                                    "buys": 0, "sells": 0, "volume_sol": 0,
                                    "first_seen": time.time(), "unique_buyers": set(),
                                    "creator": item.get("creator", ""), "creator_bought": False,
                                    "creator_sol": 0, "buy_timestamps": [],
                                    "symbol": symbol, "name": item.get("name", ""),
                                    "largest_buy_sol": 0, "platform": PLATFORM_PUMP,
                                }
                            token = TokenInfo(
                                address=mint,
                                symbol=symbol,
                                name=item.get("name", ""),
                                pair_address="",
                                price_usd=0,
                                liquidity_usd=mc * 0.1 if mc else 0,  # rough estimate
                                volume_24h=0,
                                age_minutes=age_min,
                                fdv=mc,
                                source="pumpfun_trending",
                                on_bonding_curve=True,
                            )
                            tokens.append(token)
                except Exception as e:
                    logger.debug(f"Pump.fun trending endpoint error: {e}")
        except Exception as e:
            logger.debug(f"Pump.fun trending scan error: {e}")
        if tokens:
            logger.info(f"Pump.fun trending: found {len(tokens)} fresh tokens")
        return tokens

    async def _fetch_token_data(self, address: str) -> Optional[TokenInfo]:
        """Fetch detailed token data from DexScreener, with Birdeye fallback."""
        try:
            url = f"{DEXSCREENER_TOKENS}/{address}"
            async with self.session.get(url) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                pairs = data.get("pairs")
                if not pairs:
                    return None

                pair = max(pairs, key=lambda p: float(p.get("liquidity", {}).get("usd", 0) or 0))
                created_at = pair.get("pairCreatedAt", 0)
                age_min = max(0, (time.time() * 1000 - created_at) / 60000) if created_at else 999
                txns_5m = pair.get("txns", {}).get("m5", {})
                base = pair.get("baseToken", {})

                return TokenInfo(
                    address=address,
                    symbol=base.get("symbol", "???"),
                    name=base.get("name", "Unknown"),
                    pair_address=pair.get("pairAddress", ""),
                    price_usd=float(pair.get("priceUsd", 0) or 0),
                    liquidity_usd=float(pair.get("liquidity", {}).get("usd", 0) or 0),
                    volume_24h=float(pair.get("volume", {}).get("h24", 0) or 0),
                    price_change_5m=float(pair.get("priceChange", {}).get("m5", 0) or 0),
                    price_change_1h=float(pair.get("priceChange", {}).get("h1", 0) or 0),
                    age_minutes=age_min,
                    fdv=float(pair.get("fdv", 0) or 0),
                    buy_count_5m=int(txns_5m.get("buys", 0) or 0),
                    sell_count_5m=int(txns_5m.get("sells", 0) or 0),
                )
        except Exception as e:
            logger.debug(f"DexScreener fetch error {address}: {e}")

        # Birdeye fallback (if DexScreener rate-limited or failed)
        try:
            birdeye_key = os.environ.get("BIRDEYE_API_KEY", "")
            if birdeye_key:
                url = f"{BIRDEYE_BASE_URL}/defi/v3/token/overview"
                headers = {"X-API-KEY": birdeye_key, "x-chain": "solana"}
                params = {"address": address}
                async with self.session.get(url, headers=headers, params=params,
                                            timeout=aiohttp.ClientTimeout(total=8)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        d = data.get("data", {})
                        if d:
                            return TokenInfo(
                                address=address,
                                symbol=d.get("symbol", "???"),
                                name=d.get("name", "Unknown"),
                                pair_address="",
                                price_usd=float(d.get("price", 0) or 0),
                                liquidity_usd=float(d.get("liquidity", 0) or 0),
                                volume_24h=float(d.get("v24hUSD", 0) or 0),
                                age_minutes=0,
                                fdv=float(d.get("mc", 0) or 0),  # market cap as FDV proxy
                                buy_count_5m=int(d.get("buy24h", 0) or 0),
                                sell_count_5m=int(d.get("sell24h", 0) or 0),
                                source="birdeye_fallback",
                            )
        except Exception as e:
            logger.debug(f"Birdeye fallback error {address}: {e}")

        return None

    async def _get_jupiter_price(self, mint: str) -> float:
        """v4.3: Fast real-time price via Jupiter Price API v2 (~200ms, reads AMM pools directly).

        Returns price in USD, or 0.0 if unavailable.
        Much faster than DexScreener (15-30s stale) for position management and paper PnL.
        Requires x-api-key header (free tier from portal.jup.ag).
        """
        try:
            params = {"ids": mint}  # no vsToken = returns USD price
            headers = {}
            if self.config.jupiter_api_key:
                headers["x-api-key"] = self.config.jupiter_api_key
            async with self.session.get(
                JUPITER_PRICE_URL, params=params,
                headers=headers if headers else None,
                timeout=aiohttp.ClientTimeout(total=3)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    token_data = data.get("data", {}).get(mint, {})
                    price = float(token_data.get("price", 0) or 0)
                    if price > 0:
                        return price
                elif resp.status == 401:
                    logger.warning("Jupiter Price API returned 401 — set JUPITER_API_KEY in .env (free at portal.jup.ag)")
        except Exception as e:
            logger.debug(f"Jupiter price error {mint[:12]}: {e}")
        return 0.0

    async def _get_fast_price(self, mint: str) -> float:
        """v4.3: Fastest available price — Jupiter first, DexScreener fallback.

        Jupiter Price API v2 returns real-time AMM pool prices in ~200ms.
        DexScreener data is 15-30s stale. This method tries Jupiter first.
        """
        # 1. Jupiter Price API — real-time (~200ms)
        price = await self._get_jupiter_price(mint)
        if price > 0:
            return price

        # 2. DexScreener fallback — slower but wider coverage
        token = await self._fetch_token_data(mint)
        if token and token.price_usd > 0:
            return token.price_usd

        return 0.0

    # ----------------------------------------------------------
    # TOKEN SCORING (enhanced)
    # ----------------------------------------------------------
    def score_token(self, token: TokenInfo) -> float:
        """Score a token 0-100 based on multiple degen signals."""
        score = 0.0

        # FDV cap - low cap only
        if token.fdv > self.config.max_fdv and token.fdv > 0:
            return 0  # HARD REJECT - not low cap

        # Liquidity sweet spot (skip for bonding curve tokens - they have no traditional liquidity)
        if not token.on_bonding_curve:
            if token.liquidity_usd < self.config.min_liquidity:
                return 0
            if token.liquidity_usd > self.config.max_liquidity:
                return 0

            if 5000 <= token.liquidity_usd <= 50000:
                score += 20  # sweet spot for fresh low caps
            elif 1000 <= token.liquidity_usd < 5000:
                score += 15  # early liquidity
            elif 50000 < token.liquidity_usd <= 100000:
                score += 10
            else:
                score += 5
        else:
            # Bonding curve: score based on SOL raised
            if token.curve_sol_raised >= 10:
                score += 20
            elif token.curve_sol_raised >= 3:
                score += 15
            elif token.curve_sol_raised >= 0.5:
                score += 10
            else:
                score += 5

        # Age - FRESH LAUNCHES ONLY (hard reject anything over max_age_minutes)
        if not token.on_bonding_curve and token.age_minutes < self.config.min_age_minutes:
            return 0
        if token.age_minutes > self.config.max_age_minutes:
            return 0  # HARD REJECT - too old, we only trade fresh launches
        if token.age_minutes <= 3:
            score += 30  # ultra-fresh = maximum alpha
        elif token.age_minutes <= 7:
            score += 25  # still very early
        elif token.age_minutes <= 15:
            score += 15  # acceptable freshness

        # Buy/sell pressure
        total_txns = token.buy_count_5m + token.sell_count_5m
        if total_txns > 0:
            buy_ratio = token.buy_count_5m / total_txns
            if buy_ratio > 0.7:
                score += 20
            elif buy_ratio > 0.55:
                score += 10
            elif buy_ratio < 0.3:
                return 0

        # Volume/liquidity ratio
        if token.liquidity_usd > 0:
            vol_ratio = token.volume_24h / token.liquidity_usd
            if vol_ratio > 5:
                score += 15
            elif vol_ratio > 2:
                score += 10
            elif vol_ratio > 0.5:
                score += 5

        # 5-minute momentum
        if token.price_change_5m > 10:
            score += 10
        elif token.price_change_5m > 5:
            score += 5
        elif token.price_change_5m < -20:
            score -= 10

        # FDV (micro/low cap = huge upside on fresh launches)
        if 0 < token.fdv < 50000:
            score += 15  # micro cap, maximum upside
        elif token.fdv < 150000:
            score += 10  # low cap, great upside
        elif token.fdv < 500000:
            score += 5   # still acceptable

        # Pump.fun source bonus (earliest detection)
        if token.source == "pumpfun":
            score += 10

        # v3.5: Bonding curve scoring (pre-graduation plays)
        if token.on_bonding_curve:
            score += 15  # base bonus for being early
            # Buy velocity = momentum (v3.8: loosened thresholds)
            if token.buy_velocity_per_min >= 4.0:
                score += 20  # extremely hot
            elif token.buy_velocity_per_min >= 2.0:
                score += 15
            elif token.buy_velocity_per_min >= 0.8:
                score += 10
            # Early in curve = more upside
            if token.curve_progress < 10:
                score += 15  # very early
            elif token.curve_progress < 25:
                score += 10
            elif token.curve_progress < 50:
                score += 5
            # Dev bought their own token = skin in the game
            if token.initial_buy_sol >= 0.5:
                score += 10
            elif token.initial_buy_sol >= 0.1:
                score += 5

        # Cabal bonus/penalty
        if token.top_holders_connected and self.config.cabal_is_bullish:
            score += 15  # organized money backing it
            # Extra boost for low-cap cabal plays (under 100k FDV = coordinated pump)
            if token.fdv > 0 and token.fdv < self.config.cabal_follow_max_fdv:
                score += 10  # low-cap + connected wallets = strong signal

        # v3: X sentinel boost
        if self.x_sentinel:
            x_boost = self.x_sentinel.get_token_x_boost(token.address, token.symbol)
            if x_boost > 0:
                score += x_boost
                logger.debug(f"X boost for {token.symbol}: +{x_boost}")

        # v3: Learned adjustments from trading memory
        if self.trading_memory:
            src_adj = self.trading_memory.get_learned_adjustment(f"source_{token.source}")
            if src_adj != 0:
                score += src_adj
            if token.cabal_score > 50:
                cabal_adj = self.trading_memory.get_learned_adjustment("cabal_detected")
                if cabal_adj != 0:
                    score += cabal_adj

        # v4.0: Burn signal bonus — if token has recent burns in our history, big boost
        if self.burn_monitor:
            for burn in self.burn_monitor._burn_history:
                if burn["mint"] == token.address:
                    if burn["burn_type"] == "supply_burn":
                        score += 15  # supply burn = deflationary = bullish
                    else:
                        score += 20  # LP burn = can't rug liquidity = very bullish
                    break  # one boost per token

        # v4.1: Whale hunter signal boost — tracked whales buying this token
        if self.whale_hunter:
            whale_boost = self.whale_hunter.get_whale_score_boost(token.address)
            if whale_boost > 0:
                score += whale_boost
                logger.debug(f"Whale boost for {token.symbol}: +{whale_boost}")

        token.score = max(0, min(100, score))
        return token.score

    # ----------------------------------------------------------
    # DEEP ANALYSIS PIPELINE
    # ----------------------------------------------------------
    async def deep_analyze(self, token: TokenInfo) -> bool:
        """Run full analysis pipeline: wallet graph + quantum + swarm.

        Returns True if token passes all checks.
        """
        # 1. Wallet analysis (detect cabals, check concentration)
        if self.wallet_analyzer and self.config.use_wallet_analysis:
            try:
                analysis = await asyncio.wait_for(
                    self.wallet_analyzer.analyze_token_holders(token.address),
                    timeout=10.0,
                )
                token.cabal_score = analysis.get("cabal_score", 0)
                token.top_holders_connected = len(analysis.get("connected_groups", [])) > 0

                # Check for whale buys (parallel with above)
                whale_buys = await asyncio.wait_for(
                    self.wallet_analyzer.check_whale_buys(token.address),
                    timeout=8.0,
                )
                if whale_buys:
                    total_whale_sol = sum(w.get("sol_spent", 0) for w in whale_buys)
                    logger.info(f"Whale activity on {token.symbol}: {len(whale_buys)} buys, {total_whale_sol:.1f} SOL")
                    token.score = min(100, token.score + 15)  # whale boost

                if analysis.get("concentration", 0) > 0.8:
                    logger.info(f"SKIP {token.symbol}: top holders own >80%")
                    return False

            except asyncio.TimeoutError:
                logger.debug(f"Wallet analysis timeout for {token.symbol}")

        # 2. Quantum oracle (rug probability + pattern matching + QAOA hint)
        if self.quantum_oracle and self.config.use_quantum:
            try:
                patterns = self.trading_memory.get_historical_patterns() if self.trading_memory else []
                q_result = await asyncio.wait_for(
                    self.quantum_oracle.analyze(token, self.session, patterns),
                    timeout=15.0,
                )
                token.rug_probability = q_result.get("rug_probability", 0.5)
                q_edge = q_result.get("quantum_edge", 0)
                if q_edge != 0:
                    token.score = max(0, min(100, token.score + q_edge))
                    logger.info(f"Quantum edge for {token.symbol}: {q_edge:+d}")
                if q_result.get("pattern_match"):
                    pm = q_result["pattern_match"]
                    logger.info(f"Pattern match for {token.symbol}: ~{pm['pattern_symbol']} ({pm['outcome']}, {pm['similarity']:.0%} similar)")
                if token.rug_probability > self.config.max_rug_probability:
                    logger.info(f"SKIP {token.symbol}: rug probability {token.rug_probability:.1%}")
                    return False
                logger.info(f"Rug probability for {token.symbol}: {token.rug_probability:.1%}")
                # Apply timing jitter for less predictable entry
                jitter = q_result.get("timing_jitter_ms", 0)
                if jitter > 0:
                    await asyncio.sleep(jitter / 1000.0)
            except asyncio.TimeoutError:
                logger.debug(f"Quantum analysis timeout for {token.symbol}")

        # 3. Swarm multi-agent analysis
        if self.swarm_intel and self.config.use_swarm:
            try:
                verdict = await asyncio.wait_for(
                    self.swarm_intel.multi_agent_analysis(token),
                    timeout=15.0,
                )
                token.swarm_sentiment = verdict.get("verdict", "SKIP")
                if verdict["verdict"] == "SKIP":
                    logger.info(f"Swarm SKIP on {token.symbol}: {verdict.get('reasons', [])}")
                    return False
                if verdict["verdict"] == "STRONG_BUY":
                    token.score = min(100, token.score + 20)
            except asyncio.TimeoutError:
                logger.debug(f"Swarm analysis timeout for {token.symbol}")

        return True

    # ----------------------------------------------------------
    # BONDING CURVE SNIPER (v3.5)
    # ----------------------------------------------------------
    async def execute_sniper_buy(self, signal: dict, amount_sol: float) -> Optional[Trade]:
        """Ultra-fast bonding curve buy from sniper signal. Minimal checks."""
        mint = signal.get("mint", "")
        symbol = signal.get("symbol", "???")

        if not self.curve_engine or not self.session:
            return None
        if mint in self._sniper_bought or mint in self.positions:
            return None

        # Quick rug checks (fast, no deep analysis)
        creator = signal.get("creator", "")
        if creator and self.pump_monitor and self.pump_monitor.is_serial_deployer(creator):
            logger.info(f"SNIPER SKIP {symbol}: serial deployer {creator[:8]}...")
            self._sniper_bought.add(mint)
            return None

        # Check bonding curve state
        curve_state = await self.curve_engine.get_bonding_curve_state(mint, self.session)
        if not curve_state:
            logger.debug(f"SNIPER SKIP {symbol}: can't read curve state")
            return None
        if curve_state.complete:
            logger.debug(f"SNIPER SKIP {symbol}: already graduated")
            return None
        if curve_state.progress_pct > self.config.bonding_curve_max_progress:
            logger.info(f"SNIPER SKIP {symbol}: curve {curve_state.progress_pct:.1f}% (max {self.config.bonding_curve_max_progress}%)")
            return None

        # Determine PumpPortal pool based on platform
        platform = signal.get("platform", PLATFORM_PUMP)
        pool = platform if platform in (PLATFORM_PUMP, PLATFORM_BONK) else "auto"
        platform_label = "PUMP" if platform == PLATFORM_PUMP else "BONK" if platform == PLATFORM_BONK else "BAGS"

        # Execute via PumpPortal (fastest path)
        logger.info(
            f"{'[PAPER] ' if self.config.paper_trade else ''}[{platform_label}] SNIPER BUY ${symbol} | {amount_sol:.4f} SOL | "
            f"curve {curve_state.progress_pct:.1f}% | {signal.get('buys', 0)} buys | "
            f"{signal.get('velocity', 0):.1f}/min | {signal.get('unique_buyers', 0)} unique"
        )

        # v4.3: Paper trade — simulate sniper buy
        if self.config.paper_trade:
            if self._paper_balance < amount_sol + self.config.reserve_sol:
                logger.info(f"[PAPER] SNIPER BLOCKED: insufficient virtual balance ({self._paper_balance:.4f} SOL)")
                return None
            tx_sig = f"PAPER_SNIPE_{mint[:8]}_{int(time.time())}"
            self._paper_balance -= amount_sol
            self._paper_token_holdings[mint] = 1_000_000_000
        else:
            tx_sig = None
            if self.config.use_pumpportal:
                tx_sig = await self.curve_engine.buy_on_curve_pumpportal(
                    mint, amount_sol, self.pubkey, self.keypair, self.session,
                    pool=pool,
                )

            if not tx_sig:
                # Fallback to Jupiter (token might have just graduated)
                tx_sig = await self._jupiter_swap(SOL_MINT, mint, int(amount_sol * LAMPORTS_PER_SOL))

        if tx_sig:
            self._sniper_bought.add(mint)
            self.total_invested_sol += amount_sol  # v3.9: track total invested
            entry_vel = signal.get("velocity", 0)
            self.positions[mint] = Position(
                token_address=mint, symbol=symbol,
                entry_price=amount_sol,  # v3.9: use SOL spent as reference (real PnL from balance diff)
                amount_tokens=0, amount_sol_spent=amount_sol,
                entry_time=time.time(),
                take_profit_levels=[1.15, 1.25, 1.5],  # v3.8: micro scalp — pull at 15-25%
                stop_loss=0.7,  # v3.8: tight stop — cut at 30% loss
                source="bonding_curve",
                entry_velocity=entry_vel,
                peak_velocity=entry_vel,
                on_bonding_curve=True,  # v3.9: mark as bonding curve token
            )
            self.seen_tokens.add(mint)
            trade = Trade(
                timestamp=time.time(), action="buy", token_address=mint,
                symbol=symbol, amount_sol=amount_sol,
                price_usd=curve_state.progress_pct,  # v3.9: store curve progress for reference
                tx_signature=tx_sig,
                reason=f"SNIPER curve={curve_state.progress_pct:.0f}% buys={signal.get('buys', 0)} vel={signal.get('velocity', 0):.1f}/min",
            )
            self.trades.append(trade)
            self.total_trades += 1
            self._save_state()
            logger.info(f"SNIPER BUY OK: ${symbol} tx={tx_sig[:20]}...")

            # v3.9: Snapshot top holders for rug detection
            if self.holder_watcher:
                asyncio.create_task(self.holder_watcher.snapshot_holders(mint, exclude_owner=self.pubkey))

            # Record to memory
            if self.trading_memory:
                await self.trading_memory.record_trade(TradeMemoryEntry(
                    token_address=mint, symbol=symbol, action="buy",
                    entry_score=80, rug_probability=0.1,
                    swarm_sentiment="SNIPER", cabal_score=0,
                    source="bonding_curve", outcome="pending", pnl_multiple=1.0,
                    hold_minutes=0, liquidity_at_entry=curve_state.sol_raised * 150,  # rough USD
                    age_at_entry=signal.get("age_seconds", 0) / 60,
                    timestamp=time.time(),
                ))
            return trade

        logger.warning(f"SNIPER BUY FAILED: ${symbol}")
        self._sniper_bought.add(mint)  # don't retry
        return None

    async def check_graduation_sells(self):
        """Check if any bonding curve positions have graduated and sell partial."""
        if not self.curve_engine or not self.session:
            return

        for addr in list(self.positions.keys()):
            pos = self.positions.get(addr)
            if not pos or pos.source != "bonding_curve":
                continue

            curve_state = await self.curve_engine.get_bonding_curve_state(addr, self.session)
            if not curve_state:
                continue

            # Token graduated! Sell configured percentage for guaranteed profit
            if curve_state.complete and pos.partial_sells == 0:
                sell_pct = self.config.graduation_sell_pct
                logger.info(f"{'[PAPER] ' if self.config.paper_trade else ''}GRADUATION DETECTED: ${pos.symbol} | Selling {sell_pct:.0%}")
                # v4.3: Paper trade — simulate graduation sell
                if self.config.paper_trade:
                    tx_sig = f"PAPER_GRAD_{addr[:8]}_{int(time.time())}"
                    grad_sol = pos.amount_sol_spent * sell_pct * 1.2  # assume 20% gain at graduation
                    self._paper_balance += grad_sol
                else:
                    tx_sig = await self.curve_engine.sell_on_curve_pumpportal(
                        addr, sell_pct, self.pubkey, self.keypair, self.session,
                    )
                if tx_sig:
                    pos.partial_sells += 1
                    trade = Trade(
                        timestamp=time.time(), action="sell", token_address=addr,
                        symbol=pos.symbol, amount_sol=pos.amount_sol_spent * sell_pct,
                        price_usd=0, tx_signature=tx_sig,
                        reason=f"GRADUATION_SELL {sell_pct:.0%}",
                    )
                    self.trades.append(trade)
                    self.total_trades += 1
                    self._save_state()

    # ----------------------------------------------------------
    # TRADE EXECUTION (Jupiter)
    # ----------------------------------------------------------
    async def execute_buy(self, token: TokenInfo, amount_sol: float) -> Optional[Trade]:
        amount_lamports = int(amount_sol * LAMPORTS_PER_SOL)

        # v4.1: Pre-buy safety check — verify token has real liquidity and trading
        if not token.on_bonding_curve and token.liquidity_usd < 2000:
            logger.info(f"BUY BLOCKED: {token.symbol} — liquidity ${token.liquidity_usd:.0f} too low, skipping")
            return None

        logger.info(f"{'[PAPER] ' if self.config.paper_trade else ''}BUY {token.symbol} | {amount_sol:.4f} SOL | score={token.score:.0f} | rug={token.rug_probability:.0%} | swarm={token.swarm_sentiment}")

        # v4.3: Paper trade — simulate buy without real transaction
        if self.config.paper_trade:
            if self._paper_balance < amount_sol + self.config.reserve_sol:
                logger.info(f"[PAPER] BUY BLOCKED: insufficient virtual balance ({self._paper_balance:.4f} SOL)")
                return None
            tx_sig = f"PAPER_{token.address[:8]}_{int(time.time())}"
            self._paper_balance -= amount_sol
            # Store virtual token amount (arbitrary positive value — real amount doesn't matter for paper PnL)
            self._paper_token_holdings[token.address] = 1_000_000_000
        else:
            # v3.5: Use bonding curve for pre-graduation tokens
            tx_sig = None
            if token.on_bonding_curve and self.curve_engine and self.config.use_pumpportal:
                tx_sig = await self.curve_engine.buy_on_curve_pumpportal(
                    token.address, amount_sol, self.pubkey, self.keypair, self.session,
                )
            if not tx_sig:
                tx_sig = await self._jupiter_swap(SOL_MINT, token.address, amount_lamports)
        if tx_sig:
            self.total_invested_sol += amount_sol  # v3.9: track total invested
            self.positions[token.address] = Position(
                token_address=token.address, symbol=token.symbol,
                entry_price=token.price_usd, amount_tokens=0,
                amount_sol_spent=amount_sol, entry_time=time.time(),
                source=token.source,
                on_bonding_curve=getattr(token, 'on_bonding_curve', False),
            )
            self.seen_tokens.add(token.address)
            trade = Trade(
                timestamp=time.time(), action="buy", token_address=token.address,
                symbol=token.symbol, amount_sol=amount_sol, price_usd=token.price_usd,
                tx_signature=tx_sig,
                reason=f"score={token.score:.0f} rug={token.rug_probability:.0%} swarm={token.swarm_sentiment} src={token.source}",
            )
            self.trades.append(trade)
            self.total_trades += 1
            self._save_state()
            logger.info(f"BUY OK: {token.symbol} tx={tx_sig[:20]}...")

            # v3.9: Snapshot top holders for rug detection
            if self.holder_watcher:
                asyncio.create_task(self.holder_watcher.snapshot_holders(token.address, exclude_owner=self.pubkey))

            # v4.3: Subscribe to real-time DEX price feed for this token
            if self.dex_price_feed and token.pair_address and not token.on_bonding_curve:
                asyncio.create_task(self.dex_price_feed.subscribe(token.address, token.pair_address, self.session))

            # v3: Record buy to trading memory
            if self.trading_memory:
                await self.trading_memory.record_trade(TradeMemoryEntry(
                    token_address=token.address, symbol=token.symbol, action="buy",
                    entry_score=token.score, rug_probability=token.rug_probability,
                    swarm_sentiment=token.swarm_sentiment, cabal_score=token.cabal_score,
                    source=token.source, outcome="pending", pnl_multiple=1.0,
                    hold_minutes=0, liquidity_at_entry=token.liquidity_usd,
                    age_at_entry=token.age_minutes, timestamp=time.time(),
                ))
            return trade

        logger.warning(f"BUY FAILED: {token.symbol}")
        return None

    async def execute_sell(self, token_address: str, reason: str = "manual") -> Optional[Trade]:
        pos = self.positions.get(token_address)
        if not pos:
            return None

        raw_amount = await self._get_raw_token_balance(token_address)
        if raw_amount <= 0:
            self.positions.pop(token_address, None)
            return None

        # v4.3: Paper trade — simulate sell using price data
        if self.config.paper_trade:
            logger.info(f"[PAPER] SELL {pos.symbol} | reason={reason}")
            tx_sig = f"PAPER_SELL_{token_address[:8]}_{int(time.time())}"
            # Estimate SOL received from current market price (Jupiter real-time → DexScreener fallback)
            sol_received = pos.amount_sol_spent  # default: break even
            if not pos.on_bonding_curve:
                current_price = await self._get_fast_price(token_address)
                if current_price > 0 and pos.entry_price > 0:
                    price_mult = current_price / pos.entry_price
                    sol_received = pos.amount_sol_spent * price_mult
                elif current_price == 0:
                    sol_received = 0  # token dead / rugged
            else:
                # Bonding curve token — use real-time PumpPortal price data
                if self.pump_monitor and token_address in self.pump_monitor.hot_tokens:
                    stats = self.pump_monitor.hot_tokens[token_address]
                    # v4.3: Use real curve price if available (from PumpPortal WS events)
                    curve_price = stats.get("price_sol", 0)
                    entry_price_sol = pos.entry_price  # SOL spent at entry (used as reference for bonding curve)
                    if curve_price > 0 and entry_price_sol > 0:
                        # Read curve state for exact sell simulation
                        if self.curve_engine:
                            curve_state = await self.curve_engine.get_bonding_curve_state(token_address, self.session)
                            if curve_state and not curve_state.complete:
                                # Simulate selling our tokens on the curve
                                virtual_tokens = self._paper_token_holdings.get(token_address, 0)
                                if virtual_tokens > 0:
                                    estimated_tokens = curve_state.calc_tokens_for_sol(pos.amount_sol_spent)
                                    sol_back_lamports = curve_state.calc_sol_for_tokens(estimated_tokens)
                                    sol_received = sol_back_lamports / 1e9
                                else:
                                    sol_received = pos.amount_sol_spent  # break even fallback
                            else:
                                # Graduated or state unavailable — use market cap ratio
                                mcap = stats.get("market_cap_sol", 0)
                                sol_received = pos.amount_sol_spent * 1.2 if mcap > 0 else pos.amount_sol_spent
                        else:
                            sol_received = pos.amount_sol_spent  # no engine available
                    else:
                        # Fallback: momentum-based estimate
                        buys = stats.get("buys", 0)
                        sells = stats.get("sells", 0)
                        if sells > buys * 0.6:
                            sol_received = pos.amount_sol_spent * 0.5
                        elif buys > 5:
                            sol_received = pos.amount_sol_spent * 1.1
            # Update virtual balance
            self._paper_balance += sol_received
            self._paper_token_holdings.pop(token_address, None)
            real_pnl = sol_received - pos.amount_sol_spent
        else:
            # v4.0: SAFETY — verify the token mint is a valid SPL token before interacting
            try:
                payload = {
                    "jsonrpc": "2.0", "id": 1,
                    "method": "getAccountInfo",
                    "params": [token_address, {"encoding": "jsonParsed"}],
                }
                rpc = self.config.fast_rpc_url or self.config.rpc_url
                async with self.session.post(rpc, json=payload, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        acct_data = await resp.json()
                        acct_info = acct_data.get("result", {}).get("value", {})
                        owner_program = acct_info.get("owner", "")
                        # Must be owned by SPL Token program — reject anything else
                        if owner_program not in (SPL_TOKEN_PROGRAM, SPL_TOKEN_2022):
                            logger.error(
                                f"SAFETY BLOCK: {pos.symbol} mint owned by {owner_program} — NOT SPL Token! "
                                f"Possible malicious contract. Skipping sell to avoid token drain."
                            )
                            # Remove position to avoid retrying
                            self.positions.pop(token_address, None)
                            return None
            except Exception as e:
                logger.debug(f"Token safety check error (proceeding with caution): {e}")

            # v3.9: Get SOL balance BEFORE sell to calculate real PnL
            pre_sell_balance = await self.get_sol_balance()

            logger.info(f"SELL {pos.symbol} | reason={reason}")

            # v3.9: Smart routing — try PumpPortal first for bonding curve tokens (best price on pump pool)
            tx_sig = None
            if pos.on_bonding_curve and self.curve_engine:
                # Sell directly on the bonding curve via PumpPortal — best route for pre-migration tokens
                logger.info(f"SELL via PumpPortal (bonding curve): {pos.symbol}")
                tx_sig = await self.curve_engine.sell_on_curve_pumpportal(
                    token_address, 1.0, self.pubkey, self.keypair, self.session,
                )

            if not tx_sig:
                # v4.0: Smart route — parallel Jupiter + Raydium quotes, pick the better one
                tx_sig = await self._smart_sell(token_address, raw_amount)

            if not tx_sig:
                logger.warning(f"SELL FAILED: {pos.symbol}")
                return None

            # v3.9: Real PnL — measure actual SOL received instead of trusting DexScreener prices
            await asyncio.sleep(1.5)  # wait for balance to update
            post_sell_balance = await self.get_sol_balance()
            sol_received = max(0, post_sell_balance - pre_sell_balance)

            # v4.0: SAFETY — detect balance drain (if SOL dropped instead of increasing)
            if post_sell_balance < pre_sell_balance - 0.001:
                # Balance DECREASED after sell — possible token drain attack
                drain_amount = pre_sell_balance - post_sell_balance
                logger.error(
                    f"SAFETY ALERT: SOL balance DROPPED by {drain_amount:.6f} after selling {pos.symbol}! "
                    f"Pre={pre_sell_balance:.6f} Post={post_sell_balance:.6f}. "
                    f"Possible malicious token interaction. tx={tx_sig}"
                )
                # Still record the trade but flag it
                sol_received = 0  # treat as total loss

            real_pnl = sol_received - pos.amount_sol_spent

        # --- Shared path: record trade for both paper and real modes ---
        if real_pnl > 0:
            self.winning_trades += 1
            self.total_pnl_sol += real_pnl
        else:
            self.total_pnl_sol += real_pnl  # negative
            self.total_lost_sol += abs(real_pnl)

        price_mult = sol_received / pos.amount_sol_spent if pos.amount_sol_spent > 0 else 0

        trade = Trade(
            timestamp=time.time(), action="sell", token_address=token_address,
            symbol=pos.symbol, amount_sol=pos.amount_sol_spent, price_usd=sol_received,
            tx_signature=tx_sig, reason=reason,
            pnl_sol=round(real_pnl, 6),
        )
        self.trades.append(trade)
        self.total_trades += 1
        self.positions.pop(token_address, None)
        # v3.9: Stop watching holders for this token
        if self.holder_watcher:
            await self.holder_watcher.unwatch(token_address)
        # v4.3: Unsubscribe from DEX price feed
        if self.dex_price_feed:
            await self.dex_price_feed.unsubscribe(token_address)
        self._save_state()
        logger.info(
            f"{'[PAPER] ' if self.config.paper_trade else ''}SELL OK: {pos.symbol} tx={tx_sig[:20]}... | "
            f"spent={pos.amount_sol_spent:.4f} received={sol_received:.4f} pnl={real_pnl:+.4f} SOL ({price_mult:.2f}x)"
        )

        # Compute outcome + hold time
        hold_min = (time.time() - pos.entry_time) / 60
        outcome = "win" if real_pnl > 0 else "loss"
        if "rug" in reason or "liquidity" in reason or "holder_dump" in reason:
            outcome = "rug"
        elif "time" in reason:
            outcome = "timeout"

        # Gather rich context from PumpPortal hot_tokens data
        entry_buys = 0
        entry_unique_buyers = 0
        entry_curve_progress = 0.0
        holder_concentration = 0.0
        if self.pump_monitor and token_address in self.pump_monitor.hot_tokens:
            stats = self.pump_monitor.hot_tokens[token_address]
            entry_buys = stats.get("buys", 0)
            entry_unique_buyers = len(stats.get("unique_buyers", set()))
            entry_curve_progress = stats.get("curve_progress", 0)
        if self.holder_watcher:
            snaps = self.holder_watcher.holder_snapshots.get(token_address, [])
            if snaps:
                holder_concentration = max(s.pct_supply for s in snaps)

        # v3: Record to TradingMemory (archival + knowledge graph)
        if self.trading_memory:
            await self.trading_memory.record_trade(TradeMemoryEntry(
                token_address=token_address, symbol=pos.symbol, action="sell",
                entry_score=0, rug_probability=0, swarm_sentiment="",
                cabal_score=0, source=pos.source, outcome=outcome,
                pnl_multiple=round(price_mult, 3), hold_minutes=round(hold_min, 1),
                liquidity_at_entry=0, age_at_entry=0, timestamp=time.time(),
                entry_velocity=pos.entry_velocity, entry_buys=entry_buys,
                entry_unique_buyers=entry_unique_buyers,
                entry_curve_progress=entry_curve_progress,
                on_bonding_curve=pos.on_bonding_curve, sell_reason=reason,
                sol_spent=pos.amount_sol_spent, sol_received=sol_received,
                holder_concentration=holder_concentration,
            ))

        # v4.0: Feed rich data to adaptive learner
        if self.adaptive_learner:
            self.adaptive_learner.record_trade({
                "symbol": pos.symbol, "source": pos.source,
                "outcome": outcome, "pnl_multiple": round(price_mult, 3),
                "pnl_sol": round(real_pnl, 6),
                "sol_spent": pos.amount_sol_spent, "sol_received": sol_received,
                "hold_minutes": round(hold_min, 1), "sell_reason": reason,
                "entry_velocity": pos.entry_velocity, "entry_buys": entry_buys,
                "entry_unique_buyers": entry_unique_buyers,
                "entry_curve_progress": entry_curve_progress,
                "on_bonding_curve": pos.on_bonding_curve,
                "entry_score": 0, "rug_probability": 0, "cabal_score": 0,
                "liquidity_at_entry": 0, "age_at_entry": 0,
                "holder_concentration": holder_concentration,
                "fdv_at_entry": 0, "timestamp": time.time(),
            })

        return trade

    async def _get_raw_token_balance(self, mint: str) -> int:
        # v4.3: Paper mode — return virtual token balance
        if self.config.paper_trade:
            return self._paper_token_holdings.get(mint, 0)
        payload = {
            "jsonrpc": "2.0", "id": 1,
            "method": "getTokenAccountsByOwner",
            "params": [self.pubkey, {"mint": mint}, {"encoding": "jsonParsed"}]
        }
        try:
            async with self.session.post(self.config.rpc_url, json=payload) as resp:
                data = await resp.json()
                accounts = data.get("result", {}).get("value", [])
                if accounts:
                    return int(accounts[0]["account"]["data"]["parsed"]["info"]["tokenAmount"]["amount"])
        except Exception as e:
            logger.error(f"Raw balance error: {e}")
        return 0

    async def _smart_sell(self, token_address: str, raw_amount: int) -> Optional[str]:
        """v4.0: Smart routing — parallel Jupiter + Raydium quotes, rate-limit aware.

        Rate limits (official docs):
        - Jupiter free: 1 req/sec (60/min). Pro: 10+ rps.
        - Raydium: undocumented, ~2-5 rps safe.
        We fetch both quotes in parallel to save time, respecting 1s min gap for Jupiter.
        """
        jup_key = os.environ.get("JUP_API_KEY", "")
        quote_url = "https://api.jup.ag/swap/v1/quote" if jup_key else JUPITER_QUOTE_URL
        headers = {"x-api-key": jup_key} if jup_key else {}

        # Fetch Jupiter + Raydium quotes in PARALLEL (saves ~4s vs sequential)
        async def get_jup_quote():
            try:
                params = {
                    "inputMint": token_address, "outputMint": SOL_MINT,
                    "amount": str(raw_amount), "slippageBps": str(self.config.slippage_bps),
                }
                async with self.session.get(quote_url, params=params, headers=headers,
                                            timeout=aiohttp.ClientTimeout(total=6)) as resp:
                    if resp.status == 200:
                        q = await resp.json()
                        return int(q.get("outAmount", 0))
                    elif resp.status == 429:
                        logger.debug("Jupiter rate limited on quote — using Raydium only")
            except Exception:
                pass
            return 0

        async def get_ray_quote():
            try:
                params = {
                    "inputMint": token_address, "outputMint": SOL_MINT,
                    "amount": str(raw_amount), "slippageBps": str(self.config.slippage_bps),
                    "txVersion": "V0",
                }
                async with self.session.get(RAYDIUM_QUOTE_URL, params=params,
                                            timeout=aiohttp.ClientTimeout(total=6)) as resp:
                    if resp.status == 200:
                        q = await resp.json()
                        return int(q.get("data", {}).get("outputAmount", q.get("outputAmount", 0)) or 0)
            except Exception:
                pass
            return 0

        jup_out, ray_out = await asyncio.gather(get_jup_quote(), get_ray_quote())

        # Pick best route
        if jup_out >= ray_out and jup_out > 0:
            logger.info(f"SMART ROUTE: Jupiter wins (jup={jup_out} ray={ray_out})")
            tx = await self._jupiter_swap_inner(token_address, SOL_MINT, raw_amount)
            if tx:
                return tx
        if ray_out > 0:
            logger.info(f"SMART ROUTE: Raydium {'wins' if ray_out > jup_out else 'fallback'} (jup={jup_out} ray={ray_out})")
            tx = await self._raydium_swap(token_address, SOL_MINT, raw_amount)
            if tx:
                return tx

        # Both failed — try Jupiter as final fallback
        return await self._jupiter_swap(token_address, SOL_MINT, raw_amount)

    async def _jupiter_swap(self, input_mint: str, output_mint: str, amount: int) -> Optional[str]:
        """Swap via Jupiter (lite-api) with Raydium fallback. v3.8: updated for new endpoints."""
        # Try Jupiter first, then Raydium
        tx_sig = await self._jupiter_swap_inner(input_mint, output_mint, amount)
        if tx_sig:
            return tx_sig
        # Fallback to Raydium
        logger.info("Jupiter failed — trying Raydium fallback...")
        return await self._raydium_swap(input_mint, output_mint, amount)

    async def _jupiter_swap_inner(self, input_mint: str, output_mint: str, amount: int) -> Optional[str]:
        """Jupiter swap via lite-api.jup.ag (v3.8 migration from dead quote-api.jup.ag)."""
        try:
            from solders.transaction import VersionedTransaction

            # Quote via new endpoint
            params = {
                "inputMint": input_mint, "outputMint": output_mint,
                "amount": str(amount), "slippageBps": str(self.config.slippage_bps),
            }
            # If JUP_API_KEY is set, use api.jup.ag (faster, long-term); otherwise lite-api
            jup_key = os.environ.get("JUP_API_KEY", "")
            if jup_key:
                quote_url = "https://api.jup.ag/swap/v1/quote"
                swap_url = "https://api.jup.ag/swap/v1/swap"
                headers = {"x-api-key": jup_key}
            else:
                quote_url = JUPITER_QUOTE_URL
                swap_url = JUPITER_SWAP_URL
                headers = {}

            async with self.session.get(quote_url, params=params, headers=headers,
                                        timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    logger.warning(f"Jupiter quote error ({resp.status}): {err[:200]}")
                    return None
                quote = await resp.json()

            if "error" in quote:
                logger.warning(f"Jupiter quote: {quote['error']}")
                return None

            # v4.2: Use Helius priority fee if available, otherwise static config
            priority_fee = self.config.priority_fee_lamports
            helius_fee = await self._get_helius_priority_fee([input_mint, output_mint])
            if helius_fee is not None:
                # Helius returns microlamports, Jupiter expects lamports
                priority_fee = max(helius_fee, self.config.priority_fee_lamports)

            # Swap transaction
            swap_body = {
                "quoteResponse": quote, "userPublicKey": self.pubkey,
                "wrapAndUnwrapSol": True,
                "prioritizationFeeLamports": priority_fee,
            }
            async with self.session.post(swap_url, json=swap_body, headers=headers,
                                         timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    logger.warning(f"Jupiter swap error ({resp.status}): {err[:200]}")
                    return None
                swap_data = await resp.json()

            swap_tx_b64 = swap_data.get("swapTransaction")
            if not swap_tx_b64:
                return None

            return await self._sign_and_send_tx(swap_tx_b64)

        except ImportError:
            logger.error("pip install solders")
            return None
        except Exception as e:
            logger.warning(f"Jupiter swap error: {e}")
            return None

    async def _raydium_swap(self, input_mint: str, output_mint: str, amount: int) -> Optional[str]:
        """Raydium Trade API fallback — no API key needed. v3.8."""
        try:
            from solders.transaction import VersionedTransaction

            # Quote
            params = {
                "inputMint": input_mint, "outputMint": output_mint,
                "amount": str(amount), "slippageBps": str(self.config.slippage_bps),
                "txVersion": "V0",
            }
            async with self.session.get(RAYDIUM_QUOTE_URL, params=params,
                                        timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    logger.warning(f"Raydium quote error ({resp.status}): {err[:200]}")
                    return None
                quote_data = await resp.json()

            if not quote_data.get("success", True):
                logger.warning(f"Raydium quote failed: {quote_data}")
                return None

            # v4.2: Use Helius priority fee if available
            priority_fee = self.config.priority_fee_lamports
            helius_fee = await self._get_helius_priority_fee([input_mint, output_mint])
            if helius_fee is not None:
                priority_fee = max(helius_fee, self.config.priority_fee_lamports)

            # Build swap transaction
            swap_body = {
                "computeUnitPriceMicroLamports": str(priority_fee),
                "swapResponse": quote_data,
                "txVersion": "V0",
                "wallet": self.pubkey,
                "wrapSol": input_mint == SOL_MINT,
                "unwrapSol": output_mint == SOL_MINT,
            }
            async with self.session.post(RAYDIUM_SWAP_URL, json=swap_body,
                                         timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    logger.warning(f"Raydium swap error ({resp.status}): {err[:200]}")
                    return None
                swap_data = await resp.json()

            # Raydium returns data array of base64 transactions
            txns = swap_data.get("data", [])
            if not txns:
                logger.warning("Raydium: no transaction data returned")
                return None

            # Sign and send first transaction
            swap_tx_b64 = txns[0].get("transaction", txns[0]) if isinstance(txns[0], dict) else txns[0]
            result = await self._sign_and_send_tx(swap_tx_b64)
            if result:
                logger.info(f"Raydium swap OK: {result[:20]}...")
            return result

        except ImportError:
            logger.error("pip install solders")
            return None
        except Exception as e:
            logger.warning(f"Raydium swap error: {e}")
            return None

    # v4.2: Helius priority fee estimation + staked send
    async def _get_helius_priority_fee(self, account_keys: List[str] = None) -> Optional[int]:
        """Get optimal priority fee from Helius getPriorityFeeEstimate API.

        Returns fee in microlamports, or None if Helius unavailable.
        Uses Helius-exclusive RPC method for accurate fee estimation.
        """
        if not self.config.helius_rpc_url:
            return None
        try:
            payload = {
                "jsonrpc": "2.0", "id": 1,
                "method": "getPriorityFeeEstimate",
                "params": [{
                    "accountKeys": account_keys or [PUMP_PROGRAM_ID],
                    "options": {"priorityLevel": "High"}
                }]
            }
            async with self.session.post(
                self.config.helius_rpc_url, json=payload,
                timeout=aiohttp.ClientTimeout(total=3)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    fee = data.get("result", {}).get("priorityFeeEstimate")
                    if fee is not None:
                        fee_int = int(fee)
                        logger.debug(f"Helius priority fee: {fee_int} microlamports")
                        return fee_int
        except Exception as e:
            logger.debug(f"Helius priority fee error: {e}")
        return None

    def _get_send_rpc(self) -> str:
        """Get the best RPC for sendTransaction. Helius staked > Alchemy > default."""
        if self.config.helius_rpc_url:
            return self.config.helius_rpc_url
        return self.config.fast_rpc_url or self.config.rpc_url

    # v4.0: Trusted programs — only sign transactions that interact with these
    TRUSTED_PROGRAMS = {
        SPL_TOKEN_PROGRAM,                                     # SPL Token
        SPL_TOKEN_2022,                                        # Token-2022
        "11111111111111111111111111111111",                     # System Program
        "ComputeBudget111111111111111111111111111111",          # Compute Budget
        "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL",      # Associated Token Account
        "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",       # Jupiter v6
        "JUP4Fb2cqiRUcaTHdrPC8h2gNsA2ETXiPGeSyBD8TAc",       # Jupiter v4
        "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",      # Raydium AMM v4
        "CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C",      # Raydium CPMM
        "CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK",      # Raydium CLMM
        "routeUGWgWzqBWFcrCfv8tritsqukccJPu3q5GPP3xS",       # Raydium Router
        "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc",       # Orca Whirlpool
        "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP",      # Orca v2
        "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P",       # Pump.fun
        "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA",       # PumpSwap AMM
        "srmqPvymJeFKQ4zGQed1GFppgkRHL9kaELCbyksJtPX",       # Serum DEX
        "LBUZKhRxPF3XUpBCjp4YzTKgLccjZhTSDM9YuVaPwxo",      # Meteora LB
        "Eo7WjKq67rjJQSZxS6z3YkapzY3eMj6Xy8X5EQVn5UaB",     # Meteora Pools
        "SSwapUtytfBdBn1b9NUGG6foMVPtcWgpRU32HToDUZr",       # Saros Swap
        "MERLuDFBMmsHnsBPZw2sDQZHvXFMwp8EdjudcU2HKky",       # Mercurial
    }

    async def _sign_and_send_tx(self, swap_tx_b64: str) -> Optional[str]:
        """Sign a base64 transaction and send to RPC. Shared by Jupiter + Raydium.

        v4.0: SAFETY — validates transaction only touches trusted programs before signing.
        Prevents interaction with malicious contracts that could drain the wallet.
        """
        try:
            from solders.transaction import VersionedTransaction

            raw_tx = base64.b64decode(swap_tx_b64)
            tx = VersionedTransaction.from_bytes(raw_tx)

            # v4.0: SAFETY — check all program IDs in the transaction
            try:
                msg = tx.message
                # Get account keys from the message (includes lookup table resolved keys)
                account_keys = [str(k) for k in msg.account_keys]
                # Check instructions reference only trusted programs
                for ix in msg.instructions:
                    program_idx = ix.program_id_index
                    if program_idx < len(account_keys):
                        program_id = account_keys[program_idx]
                        if program_id not in self.TRUSTED_PROGRAMS:
                            logger.error(
                                f"SAFETY BLOCK: Transaction contains untrusted program {program_id}! "
                                f"Refusing to sign. This could be a malicious contract."
                            )
                            return None
            except Exception as e:
                # If we can't parse programs, log warning but allow through
                # (Jupiter/Raydium sometimes use address lookup tables)
                logger.debug(f"TX safety check partial: {e} — proceeding (from trusted API)")

            signed_tx = VersionedTransaction(tx.message, [self.keypair])
            signed_bytes = base64.b64encode(bytes(signed_tx)).decode("ascii")

            send_payload = {
                "jsonrpc": "2.0", "id": 1, "method": "sendTransaction",
                "params": [signed_bytes, {"encoding": "base64", "skipPreflight": False, "maxRetries": 3}]
            }
            # v4.2: Send via Helius staked RPC for better landing rate
            rpc = self._get_send_rpc()
            async with self.session.post(rpc, json=send_payload) as resp:
                result = await resp.json()

            if "error" in result:
                logger.error(f"TX send error: {result['error']}")
                return None

            tx_sig = result.get("result", "")
            if tx_sig:
                await self._confirm_transaction(tx_sig)
            return tx_sig

        except Exception as e:
            logger.error(f"Sign/send error: {e}")
            return None

    async def _confirm_transaction(self, signature: str, timeout: int = 30):
        start = time.time()
        while time.time() - start < timeout:
            payload = {
                "jsonrpc": "2.0", "id": 1, "method": "getSignatureStatuses",
                "params": [[signature], {"searchTransactionHistory": True}]
            }
            try:
                async with self.session.post(self.config.fast_rpc_url or self.config.rpc_url, json=payload) as resp:
                    data = await resp.json()
                    statuses = data.get("result", {}).get("value", [])
                    if statuses and statuses[0]:
                        s = statuses[0]
                        if s.get("confirmationStatus") in ("confirmed", "finalized"):
                            return s.get("err") is None
            except Exception:
                pass
            await asyncio.sleep(0.5)
        return False

    # ----------------------------------------------------------
    # POSITION MANAGEMENT
    # ----------------------------------------------------------
    async def manage_positions(self):
        if not self.positions:
            return

        for addr in list(self.positions.keys()):
            pos = self.positions.get(addr)
            if not pos:
                continue

            hold_min = (time.time() - pos.entry_time) / 60

            # v3.9: Bonding curve tokens — use PumpPortal data + curve state, NOT DexScreener
            if pos.on_bonding_curve:
                await self._manage_bonding_curve_position(addr, pos, hold_min)
                continue

            # v4.3: Real-time price — Jupiter USD (~200ms) + DexScreener (liquidity/rug data) in parallel
            # WS feed provides instant SOL-denominated price for change detection
            token_task = asyncio.create_task(self._fetch_token_data(addr))
            jup_task = asyncio.create_task(self._get_jupiter_price(addr))
            token, jup_price = await asyncio.gather(token_task, jup_task)
            if jup_price > 0 and token:
                token.price_usd = jup_price  # real-time Jupiter USD overrides stale DexScreener
            if not token:
                if time.time() - pos.entry_time > 600:
                    await self.execute_sell(addr, reason="data_unavailable")
                continue

            if pos.entry_price <= 0:
                continue

            price_mult = token.price_usd / pos.entry_price

            # Stop loss
            if price_mult <= pos.stop_loss:
                await self.execute_sell(addr, reason=f"stop_loss_{price_mult:.2f}x")
                continue

            # v3.8: Dynamic take profits from position's levels (quick scalp style)
            tp_levels = pos.take_profit_levels if pos.take_profit_levels else [1.4, 1.5, 2.0]
            sold_tp = False
            for i, tp in enumerate(reversed(tp_levels)):
                sell_idx = len(tp_levels) - 1 - i  # check highest TP first
                if price_mult >= tp and pos.partial_sells < sell_idx + 1:
                    await self.execute_sell(addr, reason=f"tp_{tp:.1f}x_{price_mult:.2f}x")
                    sold_tp = True
                    break
            if sold_tp:
                continue

            # v4.1: Tightened time exit — don't hold losers, take any profit after 5min
            max_hold = self.config.max_hold_minutes
            if hold_min > max_hold and price_mult < 1.1:
                await self.execute_sell(addr, reason=f"time_exit_{hold_min:.0f}m_{price_mult:.2f}x")
                continue
            # v4.1: If losing after 8 min, cut it — momentum is gone
            if hold_min > 8 and price_mult < 1.0:
                await self.execute_sell(addr, reason=f"stale_loser_{hold_min:.0f}m_{price_mult:.2f}x")
                continue
            # v4.1: If in profit after 5min, just take it (was 10min)
            if hold_min > 5 and price_mult >= 1.03:
                await self.execute_sell(addr, reason=f"micro_tp_{hold_min:.0f}m_{price_mult:.2f}x")
                continue

            # Rug detection: liquidity vanishing
            if token.liquidity_usd < self.config.min_liquidity * 0.3:
                await self.execute_sell(addr, reason="liquidity_rug")
                continue

            # Heavy sells
            total = token.buy_count_5m + token.sell_count_5m
            if total > 5 and token.sell_count_5m / total > 0.8:
                await self.execute_sell(addr, reason="sell_pressure")
                continue

            # v3.6: Velocity-drop sell - exit when buy momentum dies
            if self.pump_monitor and pos.entry_velocity > 0:
                current_velocity = self.pump_monitor.get_buy_velocity(addr)
                # Track peak velocity
                if current_velocity > pos.peak_velocity:
                    pos.peak_velocity = current_velocity
                # Sell when velocity drops below threshold of peak (momentum dying)
                if pos.peak_velocity > 0 and hold_min >= 1.0:  # v3.8: check after 1 min (was 2)
                    velocity_ratio = current_velocity / pos.peak_velocity
                    if velocity_ratio <= self.config.velocity_drop_sell_pct:
                        logger.info(
                            f"VELOCITY DROP: ${pos.symbol} | vel {current_velocity:.1f}/min "
                            f"(peak {pos.peak_velocity:.1f}/min, now {velocity_ratio:.0%}) | "
                            f"PnL {price_mult:.2f}x | selling — adding to RE-ENTRY watchlist"
                        )
                        # Add to re-entry watchlist BEFORE selling so we keep watching
                        if self.config.reentry_enabled:
                            self._reentry_watchlist[addr] = {
                                "symbol": pos.symbol,
                                "sold_at": time.time(),
                                "sold_price": token.price_usd if token else 0,
                                "peak_velocity": pos.peak_velocity,
                                "entry_velocity": pos.entry_velocity,
                                "source": pos.source,
                                "pnl_at_sell": price_mult,
                            }
                        await self.execute_sell(addr, reason=f"velocity_drop_{velocity_ratio:.0%}_of_peak")
                        continue

    async def _manage_bonding_curve_position(self, addr: str, pos, hold_min: float):
        """v3.9: Manage bonding curve positions using curve state + PumpPortal data.

        DexScreener returns priceUsd=0 for pre-migration tokens, causing instant false stop_loss.
        Instead, use the actual bonding curve state to determine if we should hold or sell.
        """
        # Check if token has graduated — if so, switch to normal management
        if self.curve_engine:
            curve_state = await self.curve_engine.get_bonding_curve_state(addr, self.session)
            if curve_state and curve_state.complete:
                # Token graduated! It now has a Raydium pool. Switch to normal management.
                pos.on_bonding_curve = False
                logger.info(f"GRADUATED: ${pos.symbol} — switching to DEX price tracking")
                return  # will be handled normally next cycle

        # Use PumpPortal hot_tokens data for buy momentum tracking
        velocity = 0.0
        buys = 0
        sells = 0
        creator_sold = False
        curve_price_sol = 0.0
        if self.pump_monitor and addr in self.pump_monitor.hot_tokens:
            stats = self.pump_monitor.hot_tokens[addr]
            age_s = time.time() - stats.get("first_seen", time.time())
            buys = stats.get("buys", 0)
            sells = stats.get("sells", 0)
            velocity = (buys / (age_s / 60)) if age_s > 60 else buys
            creator_sold = stats.get("creator_sold", False)
            # v4.3: Real-time curve price from PumpPortal WS events
            curve_price_sol = stats.get("price_sol", 0)

            # Track peak velocity
            if velocity > pos.peak_velocity:
                pos.peak_velocity = velocity

        # v4.3: Price-based PnL for bonding curve tokens (if we have real-time price)
        # Use curve state for exact sell simulation
        if curve_price_sol > 0 and self.curve_engine:
            curve_state = await self.curve_engine.get_bonding_curve_state(addr, self.session) if not (hasattr(self, '_last_curve_state') and addr in getattr(self, '_last_curve_state', {})) else None
            if curve_state and not curve_state.complete:
                estimated_tokens = curve_state.calc_tokens_for_sol(pos.amount_sol_spent)
                if estimated_tokens > 0:
                    sol_back = curve_state.calc_sol_for_tokens(estimated_tokens) / 1e9
                    curve_pnl_mult = sol_back / pos.amount_sol_spent if pos.amount_sol_spent > 0 else 1.0
                    # Take profit on curve tokens if real PnL is solid
                    if curve_pnl_mult >= 1.5 and hold_min >= 0.5:
                        logger.info(f"CURVE TP: ${pos.symbol} | {curve_pnl_mult:.2f}x (real curve PnL) — taking profit")
                        await self.execute_sell(addr, reason=f"curve_tp_{curve_pnl_mult:.2f}x")
                        return
                    # Stop loss on curve tokens
                    if curve_pnl_mult <= 0.6 and hold_min >= 1.0:
                        logger.info(f"CURVE SL: ${pos.symbol} | {curve_pnl_mult:.2f}x (real curve PnL) — cutting loss")
                        await self.execute_sell(addr, reason=f"curve_sl_{curve_pnl_mult:.2f}x")
                        return

        # SELL CONDITIONS for bonding curve tokens:

        # 1. Creator rugged — sold their tokens
        if creator_sold:
            logger.info(f"CREATOR SOLD: ${pos.symbol} — selling immediately")
            await self.execute_sell(addr, reason="creator_rug")
            return

        # 2. Heavy sell pressure (more sells than buys)
        if buys > 3 and sells > buys * 0.6:
            logger.info(f"SELL PRESSURE on curve: ${pos.symbol} {sells}s/{buys}b — selling")
            await self.execute_sell(addr, reason=f"curve_sell_pressure_{sells}s_{buys}b")
            return

        # 3. Velocity died — momentum gone (v4.1: check sooner at 1.5min)
        if hold_min >= 1.5 and pos.peak_velocity > 0:
            velocity_ratio = velocity / pos.peak_velocity if pos.peak_velocity > 0 else 0
            if velocity_ratio <= 0.3:  # velocity dropped to 30% of peak
                logger.info(
                    f"CURVE VEL DROP: ${pos.symbol} | vel {velocity:.1f}/min "
                    f"(peak {pos.peak_velocity:.1f}/min, now {velocity_ratio:.0%}) — selling"
                )
                await self.execute_sell(addr, reason=f"curve_vel_drop_{velocity_ratio:.0%}")
                return

        # 4. Max hold time (v4.1: tightened from 10 to 7 min for bonding curve tokens)
        if hold_min > 7:
            logger.info(f"CURVE TIME EXIT: ${pos.symbol} | held {hold_min:.0f}m — selling")
            await self.execute_sell(addr, reason=f"curve_time_exit_{hold_min:.0f}m")
            return

        # 5. No activity — token is dead (v4.1: tightened from 3min to 2min, vel 0.5 to 0.8)
        if hold_min >= 2.0 and velocity < 0.8:
            logger.info(f"CURVE DEAD: ${pos.symbol} | vel={velocity:.1f}/min — selling")
            await self.execute_sell(addr, reason="curve_dead_momentum")
            return

    def reset_pnl(self):
        """v3.9: Reset all PnL counters for a fresh start."""
        self.total_pnl_sol = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_invested_sol = 0.0
        self.total_lost_sol = 0.0
        self.trades.clear()
        self._save_state()
        logger.info("PnL RESET — starting fresh")

    # ----------------------------------------------------------
    # MAIN TRADING LOOP
    # ----------------------------------------------------------
    async def run(self):
        _CosmosAuthLock.lock_check()  # double-check at runtime
        await self.initialize()
        self.running = True

        balance = await self.get_sol_balance()
        logger.info("=" * 60)
        if self.config.paper_trade:
            logger.info("COSMOS DEGEN TRADER v4.3 - PAPER TRADE MODE (NO REAL SOL)")
        else:
            logger.info("COSMOS DEGEN TRADER v4.3 - LIVE TRADING (REAL SOL)")
        logger.info("=" * 60)
        logger.info(f"Mode:       {'PAPER TRADE' if self.config.paper_trade else 'LIVE'}")
        logger.info(f"Wallet:     {self.pubkey}")
        logger.info(f"Balance:    {balance:.4f} SOL{' (virtual)' if self.config.paper_trade else ''}")
        logger.info(f"RPC:        {self.config.rpc_url[:40]}...")
        logger.info(f"Read RPC:   {self.config.fast_rpc_url[:40]}...")
        logger.info(f"Send RPC:   {(self.config.helius_rpc_url or self.config.fast_rpc_url)[:40]}...")
        logger.info(f"Helius:     {'ON (staked + priority fees)' if self.config.helius_rpc_url else 'OFF'}")
        logger.info(f"Pump.fun:   {'ON' if self.pump_monitor else 'OFF'}")
        logger.info(f"BondCurve:  {'ON' if self.curve_engine else 'OFF'} (PumpPortal: {'ON' if self.config.use_pumpportal else 'OFF'})")
        logger.info(f"InstSnipe:  {'ON' if self.config.instant_snipe else 'OFF'} (dev buy >= {self.config.instant_snipe_min_dev_sol} SOL → instant {self.config.instant_snipe_max_sol} SOL)")
        logger.info(f"BundleSnipe:{'ON' if self.config.bundle_snipe else 'OFF'} ({self.config.bundle_min_buys}+ buys in {self.config.bundle_window_sec}s → {self.config.bundle_snipe_max_sol} SOL)")
        logger.info(f"Re-Entry:   {'ON' if self.config.reentry_enabled else 'OFF'} (vel>={self.config.reentry_velocity_min}/min | qPredict>50% → {self.config.reentry_max_sol} SOL, SL {self.config.reentry_stop_loss:.0%})")
        logger.info(f"Sniper:     {'ON' if self.config.sniper_mode else 'OFF'} (max {self.config.bonding_curve_max_sol} SOL, <{self.config.bonding_curve_max_progress}% curve)")
        logger.info(f"FreshOnly:  <{self.config.max_age_minutes}min | FDV cap: ${self.config.max_fdv:,.0f} | Liq: ${self.config.min_liquidity:,.0f}-${self.config.max_liquidity:,.0f}")
        logger.info(f"CabalFollow: {'ON' if self.config.use_cabal_follow else 'OFF'} (FDV<${self.config.cabal_follow_max_fdv:,.0f}, {self.config.cabal_follow_min_wallets}+ wallets, vel-drop sell at {self.config.velocity_drop_sell_pct:.0%})")
        logger.info(f"Scalper:    TP at {self.config.quick_take_profit}x/{self.config.quick_take_profit_2}x | max hold {self.config.max_hold_minutes:.0f}m")
        logger.info(f"DynAdapt:   {'ON' if self.config.dynamic_adapt else 'OFF'} (loosen after {self.config.adapt_quiet_cycles} quiet, tighten after {self.config.adapt_hot_cycles} hot)")
        logger.info(f"Wallets:    {'ON' if self.wallet_analyzer else 'OFF'}")
        logger.info(f"Quantum:    {'ON' if self.quantum_oracle else 'OFF'}")
        logger.info(f"QPredict:   {'ON' if self.wallet_predictor else 'OFF'} (Bell state wallet correlation → pre-buy before crowd)")
        logger.info(f"Swarm:      {'ON' if self.config.use_swarm else 'OFF'}")
        logger.info(f"CopyTrade:  {'ON' if self.copy_engine else 'OFF'}")
        logger.info(f"X Sentinel: {'ON' if self.x_sentinel else 'OFF'}")
        logger.info(f"Memory:     {'ON' if self.trading_memory and self.trading_memory._initialized else 'OFF'}")
        logger.info(f"BurnWatch:  {'ON' if self.burn_monitor else 'OFF'} (incinerator scan 8s, auto-buy 0.05 SOL on healthy burns)")
        logger.info(f"Learner:    {'ON' if self.adaptive_learner else 'OFF'} (collective auto-tune every 15m, local analysis every 3 trades)")
        logger.info(f"Max trade:  {self.config.max_position_sol} SOL")
        logger.info(f"Max pos:    {self.config.max_positions}")
        logger.info(f"Grad sell:  {self.config.graduation_sell_pct:.0%} at graduation")
        logger.info("=" * 60)

        if balance < self.config.reserve_sol + 0.01:
            logger.error(f"Insufficient balance ({balance:.4f} SOL). Fund the wallet first.")
            self.running = False
            return

        cycle = 0
        while self.running:
            try:
                cycle += 1
                paper_tag = "[PAPER] " if self.config.paper_trade else ""
                logger.info(f"--- {paper_tag}Cycle {cycle} | Pos: {len(self.positions)}/{self.config.max_positions} | PnL: {self.total_pnl_sol:+.4f} SOL | Trades: {self.total_trades} | Bal: {await self.get_sol_balance():.4f} SOL ---")

                # Manage existing positions
                await self.manage_positions()

                # v3.9: Process holder dump alerts — EMERGENCY SELL if top holder is dumping
                if self.holder_watcher:
                    while not self.holder_watcher.sell_alerts.empty():
                        try:
                            alert = self.holder_watcher.sell_alerts.get_nowait()
                            mint = alert.get("mint", "")
                            if mint in self.positions:
                                holder = alert.get("holder", "")[:12]
                                drop = alert.get("drop_pct", 0)
                                supply_pct = alert.get("original_pct_supply", 0)
                                logger.warning(
                                    f"HOLDER RUG ALERT: ${self.positions[mint].symbol} | "
                                    f"holder {holder}... ({supply_pct:.1f}% supply) dropped {drop:.0f}% — EMERGENCY SELL"
                                )
                                await self.execute_sell(
                                    mint,
                                    reason=f"holder_dump_{holder}_{drop:.0f}pct_{supply_pct:.1f}pct_supply"
                                )
                        except asyncio.QueueEmpty:
                            break

                # v4.0: Process burn buy signals — tokens with LP/supply burns = strong bullish
                if self.burn_monitor:
                    while not self.burn_monitor.buy_signals.empty():
                        try:
                            sig = self.burn_monitor.buy_signals.get_nowait()
                            mint = sig.get("mint", "")
                            symbol = sig.get("symbol", "?")
                            burn_type = sig.get("burn_type", "?")
                            health = sig.get("health", {})

                            if mint in self.positions or mint in self._sniper_bought or mint in self.seen_tokens:
                                continue
                            if len(self.positions) >= self.config.max_positions:
                                break

                            balance = await self.get_sol_balance()
                            if balance - self.config.reserve_sol < 0.05:
                                break

                            # Cross-reference with PumpPortal for pre-bonding tokens
                            is_pre_bonding = False
                            if self.pump_monitor and mint in self.pump_monitor.hot_tokens:
                                stats = self.pump_monitor.hot_tokens[mint]
                                buys = stats.get("buys", 0)
                                vel = stats.get("velocity", 0)
                                symbol = stats.get("symbol", symbol)
                                is_pre_bonding = True
                                # Extra health check for pre-bonding
                                if buys < 2 or vel < 0.5:
                                    logger.debug(f"BURN SKIP prebond {symbol}: buys={buys} vel={vel:.1f} too weak")
                                    continue

                            # For DEX tokens, health was already checked by BurnSignalMonitor
                            # For pre-bonding, we just validated above
                            if not health.get("on_dex") and not is_pre_bonding:
                                continue  # unknown token — skip

                            buy_sol = 0.02  # v4.1: 0.02 SOL for burn buys
                            logger.info(
                                f"BURN BUY: ${symbol} ({mint[:12]}...) | "
                                f"type={burn_type} | {'PRE-BOND' if is_pre_bonding else 'DEX'} | "
                                f"buying {buy_sol} SOL"
                            )

                            if is_pre_bonding and self.curve_engine:
                                # Buy on bonding curve via PumpPortal
                                trade = await self.execute_sniper_buy(
                                    {"mint": mint, "symbol": symbol, "buys": health.get("buys_5m", 0)},
                                    buy_sol,
                                )
                                if trade and mint in self.positions:
                                    self.positions[mint].source = f"burn_{burn_type}"
                            else:
                                # Buy on DEX via Jupiter
                                token = TokenInfo(
                                    address=mint, symbol=symbol, name=symbol,
                                    pair_address="", price_usd=0,
                                    liquidity_usd=health.get("liquidity", 0),
                                    volume_24h=0, source=f"burn_{burn_type}",
                                )
                                token.score = 70  # burn signal = high confidence
                                trade = await self.execute_buy(token, buy_sol)

                            await asyncio.sleep(0.3)
                        except asyncio.QueueEmpty:
                            break
                        except Exception as e:
                            logger.debug(f"Burn buy error: {e}")

                # v4.1: Process mixer wallet buy signals — fresh wallets from mixer buying tokens
                if self.whale_hunter:
                    while not self.whale_hunter._mixer_buy_signals.empty():
                        try:
                            sig = self.whale_hunter._mixer_buy_signals.get_nowait()
                            mint = sig.get("mint", "")
                            symbol = sig.get("symbol", "?")

                            if mint in self.positions or mint in self._sniper_bought or mint in self.seen_tokens:
                                continue
                            if len(self.positions) >= self.config.max_positions:
                                break

                            balance = await self.get_sol_balance()
                            if balance - self.config.reserve_sol < 0.05:
                                break

                            buy_sol = 0.02  # v4.1: 0.02 SOL for mixer wallet signals
                            logger.info(
                                f"MIXER WALLET BUY: ${symbol} ({mint[:12]}...) | "
                                f"wallet={sig.get('wallet', '?')} | "
                                f"liq=${sig.get('liquidity', 0):.0f} | buying {buy_sol} SOL"
                            )

                            token = TokenInfo(
                                address=mint, symbol=symbol, name=symbol,
                                pair_address="", price_usd=0,
                                liquidity_usd=sig.get("liquidity", 0),
                                volume_24h=sig.get("volume_24h", 0),
                                source="mixer_wallet",
                            )
                            token.score = 65  # mixer wallet signal = moderate confidence
                            await self.execute_buy(token, buy_sol)

                            await asyncio.sleep(0.3)
                        except asyncio.QueueEmpty:
                            break
                        except Exception as e:
                            logger.debug(f"Mixer buy error: {e}")

                # Check capacity
                balance = await self.get_sol_balance()
                available = balance - self.config.reserve_sol
                can_trade = (
                    len(self.positions) < self.config.max_positions
                    and available >= self.config.max_position_sol
                )

                if can_trade:
                    # v3.7: INSTANT SNIPE — big dev buy at pool creation, buy BEFORE others
                    if (self.pump_monitor and self.curve_engine
                            and self.config.instant_snipe and self.config.sniper_mode):
                        instant_signals = []
                        while not self.pump_monitor.instant_snipe_signals.empty():
                            try:
                                sig = self.pump_monitor.instant_snipe_signals.get_nowait()
                                instant_signals.append(sig)
                            except asyncio.QueueEmpty:
                                break
                        for signal in instant_signals[:2]:  # max 2 instant snipes per cycle
                            mint = signal.get("mint", "")
                            dev_sol = signal.get("dev_buy_sol", 0)
                            if mint in self.positions or mint in self._sniper_bought:
                                continue
                            if dev_sol < self.config.instant_snipe_min_dev_sol:
                                continue  # below threshold
                            if len(self.positions) >= self.config.max_positions:
                                break
                            balance = await self.get_sol_balance()
                            if balance - self.config.reserve_sol < self.config.instant_snipe_max_sol:
                                break
                            logger.info(
                                f"INSTANT SNIPE: ${signal.get('symbol', '?')} | dev buy {dev_sol:.3f} SOL | "
                                f"buying {self.config.instant_snipe_max_sol} SOL BEFORE the crowd"
                            )
                            await self.execute_sniper_buy(signal, self.config.instant_snipe_max_sol)
                            await asyncio.sleep(0.2)  # minimal delay — speed is everything

                    # v3.7: BUNDLE SNIPE — coordinated buys detected
                    if (self.pump_monitor and self.curve_engine
                            and self.config.bundle_snipe and self.config.sniper_mode):
                        bundle_sigs = []
                        while not self.pump_monitor.bundle_signals.empty():
                            try:
                                sig = self.pump_monitor.bundle_signals.get_nowait()
                                bundle_sigs.append(sig)
                            except asyncio.QueueEmpty:
                                break
                        for signal in bundle_sigs[:2]:
                            mint = signal.get("mint", "")
                            if mint in self.positions or mint in self._sniper_bought:
                                continue
                            if len(self.positions) >= self.config.max_positions:
                                break
                            balance = await self.get_sol_balance()
                            if balance - self.config.reserve_sol < self.config.bundle_snipe_max_sol:
                                break
                            logger.info(
                                f"BUNDLE SNIPE: ${signal.get('symbol', '?')} | "
                                f"{signal.get('bundle_buys', 0)} buys in 5s | "
                                f"{signal.get('unique_buyers', 0)} unique wallets | "
                                f"buying {self.config.bundle_snipe_max_sol} SOL"
                            )
                            await self.execute_sniper_buy(signal, self.config.bundle_snipe_max_sol)
                            await asyncio.sleep(0.2)

                    # v3.5: Process sniper signals (momentum-based, bonding curve)
                    if self.pump_monitor and self.curve_engine and self.config.sniper_mode:
                        sniper_signals = []
                        while not self.pump_monitor.sniper_signals.empty():
                            try:
                                sig = self.pump_monitor.sniper_signals.get_nowait()
                                sniper_signals.append(sig)
                            except asyncio.QueueEmpty:
                                break
                        for signal in sniper_signals[:3]:  # max 3 sniper buys per cycle
                            mint = signal.get("mint", "")
                            if mint in self.positions or mint in self._sniper_bought:
                                continue
                            if len(self.positions) >= self.config.max_positions:
                                break
                            balance = await self.get_sol_balance()
                            if balance - self.config.reserve_sol < self.config.bonding_curve_max_sol:
                                break
                            await self.execute_sniper_buy(signal, self.config.bonding_curve_max_sol)
                            await asyncio.sleep(0.3)  # small delay between buys

                    # v3.5: Check graduation sells for bonding curve positions
                    if self.curve_engine:
                        await self.check_graduation_sells()

                    # v3.6: Process cabal coordination signals (connected wallets converging)
                    if self.pump_monitor and self.config.use_cabal_follow:
                        cabal_sigs = []
                        while not self.pump_monitor.cabal_signals.empty():
                            try:
                                sig = self.pump_monitor.cabal_signals.get_nowait()
                                cabal_sigs.append(sig)
                            except asyncio.QueueEmpty:
                                break
                        for signal in cabal_sigs[:3]:
                            mint = signal.get("mint", "")
                            if not mint or mint in self.positions or mint in self.seen_tokens or mint in self._sniper_bought:
                                continue
                            # v4.0: Enforce min wallets from config (PumpFunMonitor uses hardcoded 3)
                            if signal.get("connected_wallets", 0) < self.config.cabal_follow_min_wallets:
                                logger.debug(f"SKIP cabal {signal.get('symbol','')}: only {signal.get('connected_wallets',0)} wallets < {self.config.cabal_follow_min_wallets}")
                                continue
                            if len(self.positions) >= self.config.max_positions:
                                break
                            balance = await self.get_sol_balance()
                            if balance - self.config.reserve_sol < self.config.cabal_follow_max_sol:
                                break
                            # Fetch token data and enforce low-cap + fresh filter
                            token = await self._fetch_token_data(mint)
                            if not token:
                                continue
                            if token.age_minutes > self.config.max_age_minutes:
                                logger.debug(f"SKIP cabal {token.symbol}: too old ({token.age_minutes:.0f}m)")
                                continue
                            if token.fdv > self.config.cabal_follow_max_fdv and token.fdv > 0:
                                logger.debug(f"SKIP cabal {token.symbol}: FDV ${token.fdv:.0f} > ${self.config.cabal_follow_max_fdv:.0f}")
                                continue
                            # Tag and score
                            token.source = "cabal_follow"
                            token.cabal_score = min(100, signal.get("connected_wallets", 2) * 30)
                            token.top_holders_connected = True
                            self.score_token(token)
                            token.score = min(100, token.score + 25)  # cabal coordination bonus
                            velocity = signal.get("velocity", 0)
                            # v4.0: Enforce minimum score — cabal bonus shouldn't bypass quality filter
                            effective_min = self.config.min_score + self._adapt_score_offset
                            if token.score < effective_min:
                                logger.debug(f"SKIP cabal {token.symbol}: score {token.score:.0f} < min {effective_min:.0f}")
                                continue
                            logger.info(
                                f"CABAL FOLLOW: ${token.symbol} | score={token.score:.0f} | {signal.get('connected_wallets', 0)} connected wallets | "
                                f"FDV ${token.fdv:.0f} | age {token.age_minutes:.0f}m | vel {velocity:.1f}/min"
                            )
                            buy_result = await self.execute_buy(token, self.config.cabal_follow_max_sol)
                            if buy_result and mint in self.positions:
                                # Record entry velocity for velocity-drop sell
                                self.positions[mint].entry_velocity = velocity
                                self.positions[mint].peak_velocity = velocity
                            await asyncio.sleep(0.3)

                    # v3: Process copy trade signals (second fastest alpha)
                    if self.copy_engine:
                        copy_signals = self.copy_engine.get_copy_signals()
                        for signal in copy_signals[:2]:
                            mint = signal.get("mint", "")
                            if mint in self.positions or mint in self.seen_tokens:
                                continue
                            if len(self.positions) >= self.config.max_positions:
                                break
                            balance = await self.get_sol_balance()
                            if balance - self.config.reserve_sol < self.config.copy_trade_max_sol:
                                break
                            token = await self._fetch_token_data(mint)
                            if token and token.liquidity_usd >= self.config.min_liquidity:
                                # Enforce fresh launch filter on copy trades too
                                if token.age_minutes > self.config.max_age_minutes:
                                    logger.debug(f"SKIP copy {token.symbol}: too old ({token.age_minutes:.0f}m > {self.config.max_age_minutes}m)")
                                    continue
                                token.source = f"copy_{signal.get('wallet_label', 'unknown')}"
                                self.score_token(token)
                                token.score = min(100, token.score + 20)  # copy trade bonus
                                logger.info(f"COPY TRADE: ${token.symbol} from {signal.get('wallet_label', '?')} ({signal.get('sol_spent', 0):.2f} SOL) age:{token.age_minutes:.0f}m")
                                await self.execute_buy(token, self.config.copy_trade_max_sol)
                                await asyncio.sleep(0.5)

                    # v3: Check X sentinel for hot tokens
                    if self.x_sentinel:
                        hot = self.x_sentinel.get_hot_tokens(min_strength=5)  # v3.8: loosened from 7
                        for signal in hot[:2]:
                            addr = signal.get("address", "")
                            if not addr or addr in self.positions or addr in self.seen_tokens:
                                continue
                            if len(self.positions) >= self.config.max_positions:
                                break
                            token = await self._fetch_token_data(addr)
                            if token and token.liquidity_usd >= self.config.min_liquidity:
                                # Enforce fresh launch filter on X sentinel too
                                if token.age_minutes > self.config.max_age_minutes:
                                    logger.debug(f"SKIP X signal {token.symbol}: too old ({token.age_minutes:.0f}m > {self.config.max_age_minutes}m)")
                                    self.seen_tokens.add(addr)
                                    continue
                                token.source = f"x_{signal.get('signal_type', 'trending')}"
                                self.score_token(token)
                                approved = await self.deep_analyze(token)
                                if approved:
                                    await self.execute_buy(token, self.config.max_position_sol)
                                    await asyncio.sleep(0.5)
                                else:
                                    self.seen_tokens.add(addr)

                    # v3.7: Feed wallet buys to quantum predictor + run predictions
                    if self.wallet_predictor and self.pump_monitor:
                        # Feed recent buy data from PumpPortal stream
                        for mint, stats in self.pump_monitor.hot_tokens.items():
                            for buyer in stats.get("unique_buyers", set()):
                                self.wallet_predictor.record_buy(buyer, mint, stats.get("largest_buy_sol", 0))
                        # Run quantum predictions
                        predictions = await self.wallet_predictor.predict_next_buys(
                            self.pump_monitor.hot_tokens,
                            self.pump_monitor._wallet_token_buys,
                        )
                        # Act on high-confidence predictions (pre-buy before wallets move)
                        for pred in predictions[:2]:
                            mint = pred.get("mint", "")
                            conf = pred.get("confidence", 0)
                            if conf < 0.5 or mint in self.positions or mint in self._sniper_bought or mint in self.seen_tokens:
                                continue
                            if len(self.positions) >= self.config.max_positions:
                                break
                            balance = await self.get_sol_balance()
                            if balance - self.config.reserve_sol < self.config.instant_snipe_max_sol:
                                break
                            logger.info(
                                f"QUANTUM PREDICTION BUY: ${pred.get('symbol', '?')} | "
                                f"{pred.get('predicted_count', 0)} wallets predicted to buy | "
                                f"conf: {conf:.0%} | q_factor: {pred.get('quantum_factor', 0.5):.2f} | "
                                f"already {pred.get('current_buys', 0)} buys"
                            )
                            # Build a sniper-compatible signal and execute
                            signal = {
                                "mint": mint,
                                "symbol": pred.get("symbol", ""),
                                "buys": pred.get("current_buys", 0),
                                "unique_buyers": pred.get("predicted_count", 0),
                                "velocity": 0,
                                "creator": "",
                                "platform": "pump",
                                "instant_snipe": True,
                                "dev_buy_sol": 0,
                            }
                            await self.execute_sniper_buy(signal, self.config.instant_snipe_max_sol)
                            await asyncio.sleep(0.3)
                        # Cleanup old predictor data periodically
                        if self._scan_count % 10 == 0:
                            self.wallet_predictor.cleanup()

                    # Scan all standard sources
                    tokens = await self.scan_new_tokens()
                    # Pre-filter: fresh launches only
                    fresh = [t for t in tokens if t.age_minutes <= self.config.max_age_minutes]
                    stale = len(tokens) - len(fresh)
                    if stale > 0:
                        logger.info(f"Found {len(tokens)} tokens, filtered {stale} older than {self.config.max_age_minutes}m → {len(fresh)} fresh launches")
                    else:
                        logger.info(f"Found {len(fresh)} fresh tokens (all under {self.config.max_age_minutes}m)")

                    # Score with dynamic adaptation (v3.8)
                    effective_min_score = max(25, self.config.min_score + self._adapt_score_offset)
                    scored = []
                    for t in fresh:
                        s = self.score_token(t)
                        if s >= effective_min_score:
                            scored.append(t)
                    scored.sort(key=lambda t: t.score, reverse=True)

                    # v3.8: Dynamic adaptation — auto-adjust based on market activity
                    if self.config.dynamic_adapt:
                        if len(scored) == 0:
                            self._quiet_cycles += 1
                            self._hot_cycles = 0
                            if self._quiet_cycles >= self.config.adapt_quiet_cycles:
                                # Market is quiet — loosen thresholds
                                old_offset = self._adapt_score_offset
                                self._adapt_score_offset = max(-20, self._adapt_score_offset - 3)
                                if self._adapt_score_offset != old_offset:
                                    logger.info(
                                        f"ADAPT: {self._quiet_cycles} quiet cycles → loosening score "
                                        f"(min_score {effective_min_score} → {max(25, self.config.min_score + self._adapt_score_offset)})"
                                    )
                        elif len(scored) >= 3:
                            self._hot_cycles += 1
                            self._quiet_cycles = 0
                            if self._hot_cycles >= self.config.adapt_hot_cycles:
                                # Market is hot — tighten slightly to pick best only
                                old_offset = self._adapt_score_offset
                                self._adapt_score_offset = min(10, self._adapt_score_offset + 2)
                                if self._adapt_score_offset != old_offset:
                                    logger.info(
                                        f"ADAPT: {self._hot_cycles} hot cycles → tightening score "
                                        f"(min_score {effective_min_score} → {max(25, self.config.min_score + self._adapt_score_offset)})"
                                    )
                        else:
                            # Normal activity — slowly drift back to baseline
                            if self._adapt_score_offset > 0:
                                self._adapt_score_offset -= 1
                            elif self._adapt_score_offset < 0:
                                self._adapt_score_offset += 1
                            self._quiet_cycles = 0
                            self._hot_cycles = 0

                    # Deep analyze and trade top picks
                    for token in scored[:3]:
                        if token.address in self.positions:
                            continue
                        if len(self.positions) >= self.config.max_positions:
                            break

                        balance = await self.get_sol_balance()
                        if balance - self.config.reserve_sol < self.config.max_position_sol:
                            break

                        # Full analysis pipeline
                        approved = await self.deep_analyze(token)
                        if not approved:
                            self.seen_tokens.add(token.address)
                            continue

                        # Re-score after analysis enhancements
                        self.score_token(token)

                        await self.execute_buy(token, self.config.max_position_sol)
                        await asyncio.sleep(1)

                    # v3.7: RE-ENTRY CHECK — watch dumped tokens, buy back on strength
                    if self.config.reentry_enabled and self._reentry_watchlist and self.pump_monitor:
                        expired = []
                        for mint, watch in self._reentry_watchlist.items():
                            # Expire after 15 min
                            if time.time() - watch.get("sold_at", 0) > 900:
                                expired.append(mint)
                                continue
                            # Already re-entered
                            if mint in self.positions:
                                expired.append(mint)
                                continue
                            if len(self.positions) >= self.config.max_positions:
                                break
                            # Check current velocity from PumpPortal
                            current_vel = self.pump_monitor.get_buy_velocity(mint)
                            stats = self.pump_monitor.hot_tokens.get(mint, {})
                            if not stats:
                                continue
                            buys_now = stats.get("buys", 0)
                            unique_now = len(stats.get("unique_buyers", set()))

                            # Re-entry triggers:
                            # 1. Velocity resurgence (exceeds our min threshold)
                            # 2. Quantum wallet prediction targets this token
                            # 3. More unique buyers piling in
                            velocity_strong = current_vel >= self.config.reentry_velocity_min
                            quantum_predicted = (self.wallet_predictor
                                                 and mint in self.wallet_predictor.predictions
                                                 and self.wallet_predictor.predictions[mint].get("confidence", 0) > 0.5)

                            if velocity_strong or quantum_predicted:
                                balance = await self.get_sol_balance()
                                if balance - self.config.reserve_sol < self.config.reentry_max_sol:
                                    break
                                reason_parts = []
                                if velocity_strong:
                                    reason_parts.append(f"vel={current_vel:.1f}/min")
                                if quantum_predicted:
                                    qconf = self.wallet_predictor.predictions[mint].get("confidence", 0)
                                    reason_parts.append(f"qPredict={qconf:.0%}")
                                reason = " + ".join(reason_parts)
                                logger.info(
                                    f"RE-ENTRY: ${watch.get('symbol', '?')} | {reason} | "
                                    f"{buys_now} buys ({unique_now} unique) | "
                                    f"was sold at {watch.get('pnl_at_sell', 0):.2f}x — buying back with stop loss"
                                )
                                # Build signal and buy
                                signal = {
                                    "mint": mint,
                                    "symbol": watch.get("symbol", ""),
                                    "buys": buys_now,
                                    "unique_buyers": unique_now,
                                    "velocity": current_vel,
                                    "creator": "",
                                    "platform": stats.get("platform", PLATFORM_PUMP),
                                    "instant_snipe": True,
                                    "dev_buy_sol": 0,
                                }
                                result = await self.execute_sniper_buy(signal, self.config.reentry_max_sol)
                                if result and mint in self.positions:
                                    # Apply tight stop loss for re-entry
                                    self.positions[mint].stop_loss = self.config.reentry_stop_loss
                                    self.positions[mint].source = "reentry"
                                    self.positions[mint].entry_velocity = current_vel
                                    self.positions[mint].peak_velocity = current_vel
                                    logger.info(f"RE-ENTRY OK: ${watch.get('symbol', '?')} | stop loss at {self.config.reentry_stop_loss:.0%}")
                                expired.append(mint)
                                await asyncio.sleep(0.3)
                        for mint in expired:
                            self._reentry_watchlist.pop(mint, None)

                # v4.0: Adaptive learning — apply local learnings + periodic collective query
                if self.adaptive_learner and cycle % 10 == 0:
                    # Apply any pending local adjustments every 10 cycles
                    changes = self.adaptive_learner.apply_to_config(self.config)
                    if changes:
                        logger.info(f"ADAPTIVE TUNE: {changes}")

                    # Query collective every 15 minutes for deep analysis
                    if (time.time() - self._last_collective_query > 900
                            and len(self.adaptive_learner._trades) >= 5):
                        self._last_collective_query = time.time()
                        try:
                            insight = await self.adaptive_learner.ask_collective(self.config)
                            if insight:
                                logger.info(f"COLLECTIVE INSIGHT: {insight.get('reasoning', '')}")
                                # Apply collective's suggestions
                                coll_changes = self.adaptive_learner.apply_to_config(self.config)
                                if coll_changes:
                                    logger.info(f"COLLECTIVE TUNE: {coll_changes}")
                        except Exception as e:
                            logger.debug(f"Collective query failed: {e}")

                await asyncio.sleep(self.config.scan_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cycle error: {e}", exc_info=True)
                await asyncio.sleep(5)

        await self.shutdown()

    # ----------------------------------------------------------
    # STATE
    # ----------------------------------------------------------
    def _save_state(self):
        state = {
            "positions": {k: asdict(v) for k, v in self.positions.items()},
            "trades": [asdict(t) for t in self.trades[-200:]],
            "seen_tokens": list(self.seen_tokens)[-5000:],
            "total_pnl_sol": self.total_pnl_sol,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "start_balance": self.start_balance,
            "total_invested_sol": self.total_invested_sol,
            "total_lost_sol": self.total_lost_sol,
            "saved_at": time.time(),
            # v4.3: Paper trading state
            "paper_trade": self.config.paper_trade,
            "paper_balance": self._paper_balance,
            "paper_token_holdings": self._paper_token_holdings,
        }
        try:
            STATE_FILE.write_text(json.dumps(state, indent=2, default=str))
        except Exception as e:
            logger.error(f"State save error: {e}")

    def _load_state(self):
        if not STATE_FILE.exists():
            return
        try:
            state = json.loads(STATE_FILE.read_text())
            self.total_pnl_sol = state.get("total_pnl_sol", 0)
            self.total_trades = state.get("total_trades", 0)
            self.winning_trades = state.get("winning_trades", 0)
            self.start_balance = state.get("start_balance", 0)
            self.total_invested_sol = state.get("total_invested_sol", 0)
            self.total_lost_sol = state.get("total_lost_sol", 0)
            self.seen_tokens = set(state.get("seen_tokens", []))
            for addr, pdata in state.get("positions", {}).items():
                self.positions[addr] = Position(**pdata)
            for tdata in state.get("trades", []):
                self.trades.append(Trade(**tdata))
            # v4.3: Restore paper trading state
            if self.config.paper_trade and state.get("paper_trade"):
                self._paper_balance = state.get("paper_balance", self.config.paper_start_balance)
                self._paper_token_holdings = state.get("paper_token_holdings", {})
            logger.info(f"Loaded: {self.total_trades} trades, {len(self.positions)} positions" +
                        (f" | Paper balance: {self._paper_balance:.4f} SOL" if self.config.paper_trade else ""))
        except Exception as e:
            logger.error(f"State load error: {e}")

    # ----------------------------------------------------------
    # STATUS
    # ----------------------------------------------------------
    def _get_scan_feed(self) -> List[dict]:
        """Get top active tokens being scanned — live data for dashboard feeds."""
        if not self.pump_monitor:
            return []
        now = time.time()
        result = []
        for mint, stats in self.pump_monitor.hot_tokens.items():
            age_s = now - stats.get("first_seen", now)
            if age_s > self.config.max_age_minutes * 60:
                continue
            buys = stats.get("buys", 0)
            if buys < 1:
                continue
            unique = len(stats.get("unique_buyers", set()))
            velocity = buys / (age_s / 60) if age_s > 0 else 0
            result.append({
                "mint": mint,
                "symbol": stats.get("symbol", "???"),
                "buys": buys,
                "sells": stats.get("sells", 0),
                "unique_buyers": unique,
                "velocity": round(velocity, 1),
                "volume_sol": round(stats.get("volume_sol", 0), 3),
                "age_seconds": round(age_s),
                "largest_buy_sol": round(stats.get("largest_buy_sol", 0), 3),
                "creator_bought": stats.get("creator_bought", False),
                "platform": stats.get("platform", PLATFORM_PUMP),
            })
        result.sort(key=lambda x: x["velocity"], reverse=True)
        return result[:15]

    async def _fetch_live_prices(self) -> Dict[str, dict]:
        """Batch fetch live prices from DexScreener for all open positions.
        Returns: {mint: {"price_usd": float, "price_sol": float}} """
        if not self.session or not self.positions:
            return {}

        prices = {}
        mints = list(self.positions.keys())

        # DexScreener supports comma-separated tokens (max 30 per call)
        for i in range(0, len(mints), 30):
            batch = mints[i:i + 30]
            batch_str = ",".join(batch)
            try:
                async with self.session.get(
                    f"{DEXSCREENER_TOKENS}/{batch_str}",
                    timeout=aiohttp.ClientTimeout(total=8),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for pair in (data.get("pairs") or []):
                            mint = pair.get("baseToken", {}).get("address", "")
                            if mint and mint in self.positions:
                                price_usd = float(pair.get("priceUsd", 0) or 0)
                                price_native = float(pair.get("priceNative", 0) or 0)
                                prices[mint] = {
                                    "price_usd": price_usd,
                                    "price_sol": price_native,
                                }
            except Exception as e:
                logger.debug(f"Live price fetch error: {e}")

        return prices

    async def status(self) -> Dict:
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

        # v4.1: Fetch live prices for PnL calculation
        live_prices = await self._fetch_live_prices()

        # Build positions dict with live PnL
        positions_data = {}
        total_unrealized_pnl = 0.0
        for addr, p in self.positions.items():
            pos_data = {
                "symbol": p.symbol, "entry_price": p.entry_price,
                "sol_spent": p.amount_sol_spent, "source": p.source,
                "hold_minutes": round((time.time() - p.entry_time) / 60, 1),
                "entry_velocity": round(p.entry_velocity, 1),
                "peak_velocity": round(p.peak_velocity, 1),
                "current_velocity": round(self.pump_monitor.get_buy_velocity(addr), 1) if self.pump_monitor else 0,
                "on_curve": p.on_bonding_curve,
                "current_price_usd": 0,
                "unrealized_pnl_sol": 0,
                "unrealized_pnl_pct": 0,
            }

            # Calculate live PnL if we have price data
            if addr in live_prices:
                price_sol = live_prices[addr].get("price_sol", 0)
                price_usd = live_prices[addr].get("price_usd", 0)
                pos_data["current_price_usd"] = price_usd

                if price_sol > 0 and p.amount_tokens > 0:
                    current_value_sol = p.amount_tokens * price_sol
                    unrealized = current_value_sol - p.amount_sol_spent
                    pnl_pct = ((current_value_sol / p.amount_sol_spent) - 1) * 100 if p.amount_sol_spent > 0 else 0
                    pos_data["unrealized_pnl_sol"] = round(unrealized, 4)
                    pos_data["unrealized_pnl_pct"] = round(pnl_pct, 1)
                    total_unrealized_pnl += unrealized

            positions_data[addr] = pos_data

        return {
            "running": self.running,
            "wallet": self.pubkey,
            "total_pnl_sol": round(self.total_pnl_sol, 4),
            "total_invested_sol": round(self.total_invested_sol, 4),
            "total_lost_sol": round(self.total_lost_sol, 4),
            "total_unrealized_pnl_sol": round(total_unrealized_pnl, 4),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": round(win_rate, 1),
            "open_positions": len(self.positions),
            "positions": positions_data,
            "recent_trades": [
                {"action": t.action, "symbol": t.symbol, "sol": t.amount_sol,
                 "pnl_sol": t.pnl_sol, "reason": t.reason,
                 "time": datetime.fromtimestamp(t.timestamp).isoformat()}
                for t in self.trades[-10:]
            ],
            "scan_count": self._scan_count,
            "intelligence": {
                "pumpfun": self.pump_monitor is not None and self.pump_monitor.running,
                "wallet_analysis": self.wallet_analyzer is not None,
                "quantum_oracle": self.quantum_oracle is not None,
                "swarm": self.config.use_swarm,
                "copy_trading": self.copy_engine is not None,
                "x_sentinel": self.x_sentinel is not None,
                "trading_memory": self.trading_memory is not None and self.trading_memory._initialized,
                "bonding_curve": self.curve_engine is not None,
                "sniper_mode": self.config.sniper_mode,
                "instant_snipe": self.config.instant_snipe,
                "instant_snipe_min_dev_sol": self.config.instant_snipe_min_dev_sol,
                "ws_reconnects": self.pump_monitor._reconnect_count if self.pump_monitor else 0,
                "tracked_wallets": len(self.copy_engine.tracked_wallets) if self.copy_engine else 0,
                "x_signals_active": len(self.x_sentinel.trending_tokens) if self.x_sentinel else 0,
                "learned_patterns": len(self.trading_memory.get_historical_patterns()) if self.trading_memory else 0,
                "fast_rpc": bool(self.config.fast_rpc_url and self.config.fast_rpc_url != self.config.rpc_url),
                "helius_staked": bool(self.config.helius_rpc_url),
                "sniper_buys": len(self._sniper_bought),
                "cabal_follow": self.config.use_cabal_follow,
                "cabal_follow_max_fdv": self.config.cabal_follow_max_fdv,
                "cabal_signals_seen": len(self.pump_monitor._cabal_signaled) if self.pump_monitor else 0,
                "velocity_drop_sell_pct": self.config.velocity_drop_sell_pct,
                "hot_tokens_tracked": len(self.pump_monitor.hot_tokens) if self.pump_monitor else 0,
                "wallets_tracked": len(self.pump_monitor._wallet_token_buys) if self.pump_monitor else 0,
                "platform_counts": self.pump_monitor.platform_counts if self.pump_monitor else {},
                "bundles_detected": len(self.pump_monitor._bundle_detected) if self.pump_monitor else 0,
                "reentry_watchlist": len(self._reentry_watchlist),
                "holder_watcher": self.holder_watcher.get_status() if self.holder_watcher else {},
                "adaptive_learner": self.adaptive_learner.get_status() if self.adaptive_learner else {},
                "burn_monitor": self.burn_monitor.get_status() if self.burn_monitor else {},
                "whale_hunter": self.whale_hunter.get_status() if self.whale_hunter else {},
            },
            "sniper_feed": self.pump_monitor.get_sniper_feed(max_age_seconds=self.config.max_age_minutes * 60) if self.pump_monitor else [],
            "cabal_feed": self.pump_monitor.get_cabal_feed(max_age_seconds=self.config.max_age_minutes * 60) if self.pump_monitor else [],
            "scan_feed": self._get_scan_feed() if self.pump_monitor else [],
            "whale_feed": self.whale_hunter.get_whale_feed() if self.whale_hunter else [],
            "prediction_feed": self.wallet_predictor.get_prediction_feed() if self.wallet_predictor else [],
            "prediction_stats": {
                "hits": self.wallet_predictor.prediction_hits if self.wallet_predictor else 0,
                "misses": self.wallet_predictor.prediction_misses if self.wallet_predictor else 0,
                "wallets_modeled": len(self.wallet_predictor.wallet_sequences) if self.wallet_predictor else 0,
                "correlations": len(self.wallet_predictor.wallet_correlations) if self.wallet_predictor else 0,
                "active_predictions": len(self.wallet_predictor.predictions) if self.wallet_predictor else 0,
            },
            "x_feed": [
                {"symbol": v.get("symbol", ""), "signal_type": v.get("signal_type", ""),
                 "strength": v.get("strength", 0), "reason": v.get("reason", ""),
                 "address": v.get("address", ""), "timestamp": v.get("timestamp", 0)}
                for v in (self.x_sentinel.trending_tokens.values() if self.x_sentinel else [])
            ][:10],
            "config": {
                "max_age_minutes": self.config.max_age_minutes,
                "max_fdv": self.config.max_fdv,
                "cabal_follow_max_fdv": self.config.cabal_follow_max_fdv,
                "velocity_drop_sell_pct": self.config.velocity_drop_sell_pct,
                "instant_snipe": self.config.instant_snipe,
                "instant_snipe_min_dev_sol": self.config.instant_snipe_min_dev_sol,
                "instant_snipe_max_sol": self.config.instant_snipe_max_sol,
                "quick_take_profit": self.config.quick_take_profit,
                "quick_take_profit_2": self.config.quick_take_profit_2,
                "max_hold_minutes": self.config.max_hold_minutes,
                "dynamic_adapt": self.config.dynamic_adapt,
                "adapt_score_offset": self._adapt_score_offset,
                "effective_min_score": max(25, self.config.min_score + self._adapt_score_offset),
                "bonding_curve_min_velocity": self.config.bonding_curve_min_velocity,
                "bonding_curve_min_buys": self.config.bonding_curve_min_buys,
                "min_score": self.config.min_score,
                "stop_loss_default": 0.7,
            },
            "learner": self.adaptive_learner.get_status() if self.adaptive_learner else {},
            "learner_summary": self.adaptive_learner.get_learnings_summary() if self.adaptive_learner else "disabled",
        }


# ============================================================
# ENTRY POINTS
# ============================================================
async def start_trader(
    rpc_url: str = DEFAULT_RPC,
    wallet_name: str = "degen_trader",
    **overrides,
):
    # v4.0: Use TraderConfig class defaults (tightened in v4.0)
    # Any kwargs passed override specific fields
    config_kwargs = {"rpc_url": rpc_url}
    config_kwargs.update(overrides)
    config = TraderConfig(**config_kwargs)
    trader = DegenTrader(config=config, wallet_name=wallet_name)
    await trader.run()
    return trader


if __name__ == "__main__":
    # Auth lock: refuse direct standalone execution
    print("\n" + "=" * 60)
    print("COSMOS DEGEN TRADER - AUTH LOCK")
    print("=" * 60)
    print("This trading engine is locked to the Cosmos AI Swarm.")
    print("It cannot be run standalone.")
    print("\nStart via the Cosmos server:")
    print("  python -m cosmos.web.server")
    print("  Then POST to /api/trading/start")
    print("=" * 60 + "\n")

    # Only allow --create-wallet as standalone
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Cosmos Degen Trader v3.8 - LOCKED TO SWARM")
    parser.add_argument("--rpc", default=os.environ.get("SOLANA_RPC_URL", DEFAULT_RPC))
    parser.add_argument("--wallet", default="degen_trader")
    parser.add_argument("--max-sol", type=float, default=0.1, help="Max SOL per standard trade")
    parser.add_argument("--sniper-sol", type=float, default=0.08, help="Max SOL per bonding curve snipe")
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--interval", type=int, default=8)
    parser.add_argument("--no-swarm", action="store_true")
    parser.add_argument("--no-quantum", action="store_true")
    parser.add_argument("--no-pumpfun", action="store_true")
    parser.add_argument("--no-copy-trading", action="store_true")
    parser.add_argument("--no-x-sentinel", action="store_true")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--no-sniper", action="store_true", help="Disable bonding curve sniper mode")
    parser.add_argument("--no-bonding-curve", action="store_true", help="Disable direct bonding curve trading")
    parser.add_argument("--live", action="store_true", help="Disable paper trading and use REAL SOL (requires explicit opt-in)")
    parser.add_argument("--paper-balance", type=float, default=1.0, help="Starting virtual SOL for paper trading (default: 1.0)")
    parser.add_argument("--create-wallet", action="store_true")
    args = parser.parse_args()

    if args.create_wallet:
        pubkey, path = create_wallet(args.wallet)
        print(f"\nWallet created!")
        print(f"  Address: {pubkey}")
        print(f"  Keypair: {path}")
        print(f"\nStart via Cosmos server, then POST to /api/trading/start")
    else:
        # Auth lock: standalone trading blocked
        print("\nERROR: Direct trading execution is disabled.")
        print("The Degen Trader requires the full Cosmos swarm to operate.")
        print("\nTo trade, start the Cosmos server:")
        print("  python -m cosmos.web.server")
        print("  curl -X POST http://localhost:8080/api/trading/start")
        import sys
        sys.exit(1)
