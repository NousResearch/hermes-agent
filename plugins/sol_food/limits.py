"""Frozen transport ceilings for the Sol food workflow.

These constants are integration ceilings frozen by the reviewed migration
contract. They are deliberately spelled as exact literals — every one of
them has boundary tests at ``limit-1 / limit / limit+1`` — and none may be
raised (or "temporarily" bypassed) without a new reviewed contract. The
canonical Health food-commit contract (``health.food_commit.v1`` /
``health.food_meal.v1``, v3) may impose stricter payload limits; this
transport layer may not weaken either side.
"""

from __future__ import annotations

# ── Text ────────────────────────────────────────────────────────────────
#: Max Unicode characters of a food text description after normalization.
#: Over-limit input stays ordinary Sol conversation; it never writes.
FOOD_TEXT_MAX_CHARS = 4096
#: Max Unicode characters a Telegram photo caption may contribute.
FOOD_CAPTION_MAX_CHARS = 1024

# ── Photos ──────────────────────────────────────────────────────────────
#: Exactly one image per proposal.
FOOD_IMAGES_PER_PROPOSAL = 1
#: Max downloaded bytes for the one accepted image (10 MiB).
FOOD_IMAGE_MAX_BYTES = 10 * 1024 * 1024
#: Accepted image formats — magic-validated, not extension-validated.
FOOD_IMAGE_FORMATS = ("jpeg", "png", "webp")
#: Decoded-image safety bounds, enforced from header metadata BEFORE any
#: full decode or model submission (decompression-bomb rejection).
FOOD_IMAGE_MAX_PIXELS = 32_000_000
FOOD_IMAGE_MAX_SIDE = 7_900

# ── Parsing ─────────────────────────────────────────────────────────────
#: One total deadline per proposal (seconds) …
FOOD_PARSE_DEADLINE_SECONDS = 90
#: … and at most this many model attempts against the same single
#: immutable origin update/message. Parsing never commits.
FOOD_PARSE_MAX_ATTEMPTS = 2

# ── Food image cache ────────────────────────────────────────────────────
FOOD_CACHE_DIR_MODE = 0o700
FOOD_CACHE_FILE_MODE = 0o600
#: A food image is deleted on success/error/cancel/timeout, and no later
#: than this many seconds after terminal parsing.
FOOD_CACHE_TERMINAL_DELETE_SECONDS = 60

# ── Candidate sets ──────────────────────────────────────────────────────
#: One active candidate set per single origin update/message.
FOOD_CANDIDATE_MAX_CHOICES = 4
#: Max normalized structured items per candidate. This equals the frozen
#: Health payload bound (``health.food_meal.v1`` items: [1..24]).
FOOD_CANDIDATE_MAX_ITEMS = 24
#: Max Unicode characters per UI label.
FOOD_LABEL_MAX_CHARS = 120
#: Max canonical transient proposal JSON bytes.
FOOD_PROPOSAL_JSON_MAX_BYTES = 64 * 1024
#: Telegram candidate display cap (leaves error/receipt headroom under the
#: platform's 4096-char message limit).
FOOD_DISPLAY_MAX_CHARS = 3500

# ── Candidate lifetime ──────────────────────────────────────────────────
#: 30 minutes from issuance. Editing creates a new version INSIDE the same
#: lifetime and immediately expires all earlier-version tokens.
FOOD_PROPOSAL_TTL_SECONDS = 30 * 60

# ── Callback token ──────────────────────────────────────────────────────
#: Exactly "sf1:" + unpadded 22-char base64url of 128 random bits.
FOOD_CALLBACK_PREFIX = "sf1:"
FOOD_CALLBACK_RANDOM_BYTES = 16  # 128 bits
FOOD_CALLBACK_B64_CHARS = 22
#: Total token length in ASCII bytes (4 + 22). Below Telegram's official
#: 1–64-byte callback-data limit.
FOOD_CALLBACK_TOKEN_BYTES = 26

# ── Dedup retention ─────────────────────────────────────────────────────
#: Value-free consumed-action / update-to-receipt linkage retention
#: (covers Telegram's at-most-24-hour update retention window).
FOOD_DEDUP_RETENTION_SECONDS = 48 * 3600

# ── Health client framing (transport side of the v3 contract) ───────────
#: The Health food endpoint accepts exactly one Content-Length in 1..8192.
HEALTH_FOOD_BODY_MAX_BYTES = 8192
#: HEALTH_FOOD_COMMIT_TOKEN: 43 canonical unpadded base64url chars
#: decoding to exactly 32 random bytes.
HEALTH_FOOD_TOKEN_CHARS = 43
HEALTH_FOOD_TOKEN_BYTES = 32
