-- AGC-1663 schema-30 opt-in Telegram DDL fixture; no profile data.
--
-- Provenance:
--   source commit: d337b736aa1e8ebecfab043842d13e4a2d2f48a3
--   census: /private/tmp/agc1663-M021-os-isolated-census-root-20261017/schema30-census.json
--   census sha256: 6195d518ae9b90c48a5228958538b736fa30c9ee24c38f7f914ad109ff4bddd8
--   reviewed policy: /private/tmp/agc1663-schema30-optional-telegram-policy-review-independent-20261017/review.json
--   reviewed policy sha256: 973357eed254fdf4d34adb4cf0d673b9de0949941578754b41ba24d20ebd144b
--   optional DDL shape sha256: dbf04ec3c2082cae3eef95629d7b76c2afe0b8124c58f6bdab12d06fec9e3f50
--
-- The two CREATE TABLE bodies preserve the actual30 census types, nullability,
-- defaults, composite primary-key positions, and the bindings -> sessions FK.
-- The explicit indexes are the only non-table optional objects in the census.

CREATE TABLE telegram_dm_topic_bindings (
    profile_name TEXT NOT NULL DEFAULT 'default',
    chat_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    session_key TEXT NOT NULL,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    managed_mode TEXT NOT NULL DEFAULT 'auto',
    linked_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (profile_name, chat_id, thread_id)
);

CREATE TABLE telegram_dm_topic_mode (
    profile_name TEXT NOT NULL DEFAULT 'default',
    chat_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    activated_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    has_topics_enabled INTEGER,
    allows_users_to_create_topics INTEGER,
    capability_checked_at REAL,
    intro_message_id TEXT,
    pinned_message_id TEXT,
    PRIMARY KEY (profile_name, chat_id)
);

CREATE UNIQUE INDEX idx_telegram_dm_topic_bindings_session
    ON telegram_dm_topic_bindings(session_id);

CREATE INDEX idx_telegram_dm_topic_bindings_user
    ON telegram_dm_topic_bindings(profile_name, user_id, chat_id);
