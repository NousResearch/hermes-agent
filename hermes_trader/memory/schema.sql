-- Hermes Agentic Trader — trade_episodes schema (P3)
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trade_episodes (
    episode_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    chain TEXT NOT NULL DEFAULT '',
    strategy_tag TEXT,
    gate_decision TEXT NOT NULL,
    gate_reason TEXT,
    intent_json TEXT NOT NULL,
    decision_json TEXT,
    execution_json TEXT,
    market_summary_json TEXT,
    liquidity_usd REAL,
    token_address TEXT NOT NULL DEFAULT '',
    tx_hash TEXT,
    entry_price REAL,
    exit_price REAL,
    pnl_usd REAL,
    holding_hours REAL,
    embedding_id TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trade_episodes_strategy
    ON trade_episodes(strategy_tag);
CREATE INDEX IF NOT EXISTS idx_trade_episodes_gate
    ON trade_episodes(gate_decision);
CREATE INDEX IF NOT EXISTS idx_trade_episodes_chain
    ON trade_episodes(chain);
CREATE INDEX IF NOT EXISTS idx_trade_episodes_token
    ON trade_episodes(token_address);
CREATE INDEX IF NOT EXISTS idx_trade_episodes_created
    ON trade_episodes(created_at DESC);