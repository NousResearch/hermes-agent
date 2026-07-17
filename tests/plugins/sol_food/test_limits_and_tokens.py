"""Frozen ceilings are exact; sf1 tokens are exact."""

import re

from plugins.sol_food import limits
from plugins.sol_food.tokens import mint_token, parse_token


class TestFrozenCeilings:
    """The migration contract freezes these exact literals."""

    def test_text_and_caption(self):
        assert limits.FOOD_TEXT_MAX_CHARS == 4096
        assert limits.FOOD_CAPTION_MAX_CHARS == 1024

    def test_photo_bounds(self):
        assert limits.FOOD_IMAGES_PER_PROPOSAL == 1
        assert limits.FOOD_IMAGE_MAX_BYTES == 10 * 1024 * 1024
        assert limits.FOOD_IMAGE_FORMATS == ("jpeg", "png", "webp")
        assert limits.FOOD_IMAGE_MAX_PIXELS == 32_000_000
        assert limits.FOOD_IMAGE_MAX_SIDE == 7_900

    def test_parsing_bounds(self):
        assert limits.FOOD_PARSE_DEADLINE_SECONDS == 90
        assert limits.FOOD_PARSE_MAX_ATTEMPTS == 2

    def test_cache_bounds(self):
        assert limits.FOOD_CACHE_DIR_MODE == 0o700
        assert limits.FOOD_CACHE_FILE_MODE == 0o600
        assert limits.FOOD_CACHE_TERMINAL_DELETE_SECONDS == 60

    def test_candidate_bounds(self):
        assert limits.FOOD_CANDIDATE_MAX_CHOICES == 4
        assert limits.FOOD_CANDIDATE_MAX_ITEMS == 24
        assert limits.FOOD_LABEL_MAX_CHARS == 120
        assert limits.FOOD_PROPOSAL_JSON_MAX_BYTES == 64 * 1024
        assert limits.FOOD_DISPLAY_MAX_CHARS == 3500

    def test_lifetime_and_dedup(self):
        assert limits.FOOD_PROPOSAL_TTL_SECONDS == 30 * 60
        assert limits.FOOD_DEDUP_RETENTION_SECONDS == 48 * 3600

    def test_callback_token_shape(self):
        assert limits.FOOD_CALLBACK_PREFIX == "sf1:"
        assert limits.FOOD_CALLBACK_RANDOM_BYTES == 16  # 128 bits
        assert limits.FOOD_CALLBACK_B64_CHARS == 22
        assert limits.FOOD_CALLBACK_TOKEN_BYTES == 26
        # Below Telegram's 1–64-byte callback-data limit.
        assert 1 <= limits.FOOD_CALLBACK_TOKEN_BYTES <= 64

    def test_health_framing(self):
        assert limits.HEALTH_FOOD_BODY_MAX_BYTES == 8192
        assert limits.HEALTH_FOOD_TOKEN_CHARS == 43
        assert limits.HEALTH_FOOD_TOKEN_BYTES == 32


class TestTokens:
    def test_mint_shape(self):
        token = mint_token()
        assert len(token) == 26
        assert len(token.encode("ascii")) == 26
        assert re.fullmatch(r"sf1:[A-Za-z0-9_-]{22}", token)

    def test_mint_unique(self):
        seen = {mint_token() for _ in range(256)}
        assert len(seen) == 256

    def test_parse_roundtrip(self):
        token = mint_token()
        assert parse_token(token) == token

    def test_parse_rejects_wrong_length(self):
        token = mint_token()
        assert parse_token(token[:-1]) is None  # 25 bytes
        assert parse_token(token + "A") is None  # 27 bytes

    def test_parse_rejects_bad_prefix(self):
        token = mint_token()
        assert parse_token("sf2:" + token[4:]) is None
        assert parse_token("SF1:" + token[4:]) is None

    def test_parse_rejects_padding_and_bad_chars(self):
        assert parse_token("sf1:" + "A" * 21 + "=") is None
        assert parse_token("sf1:" + "A" * 21 + "+") is None
        assert parse_token("sf1:" + "A" * 21 + "/") is None

    def test_parse_rejects_non_string(self):
        assert parse_token(None) is None
        assert parse_token(b"sf1:" + b"A" * 22) is None
        assert parse_token(26) is None

    def test_token_is_opaque(self):
        # Nothing but the fixed prefix and random base64url: no colon
        # separators beyond the prefix, no embedded fields.
        token = mint_token()
        assert token[4:].count(":") == 0
