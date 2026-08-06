"""Tests for hermes_token_codec — bit-packed messages.token_count codec.

Covers:
  * pack/unpack round-trip for both buckets
  * boundary saturation (values clamped to field width)
  * the F=sign-bit invariant (packed rows are negative, legacy non-negative)
  * legacy compatibility (non-negative token_count read as a raw count)
  * role-aware resolution via resolve_message_tokens
"""
import pytest

from hermes_token_codec import (
    TAG_OUTPUT,
    TAG_REASONING,
    TAG_TOTAL_INPUT,
    TAG_CACHE_READ,
    pack_token_count,
    unpack_token_count,
    pack_assistant_tokens,
    pack_input_tokens,
    resolve_message_tokens,
    _V1_MAX,
    _V2_MAX,
)


# --- round-trip -----------------------------------------------------------

def test_assistant_round_trip():
    packed = pack_assistant_tokens(1234, 567)
    assert packed < 0  # F=1 -> stored negative
    d = unpack_token_count(packed)
    assert d == {"output_tokens": 1234, "reasoning_tokens": 567}


def test_input_round_trip():
    packed = pack_input_tokens(98765, 43210)
    assert packed < 0
    d = unpack_token_count(packed)
    assert d == {"total_input_tokens": 98765, "cache_read_tokens": 43210}


def test_round_trip_zeros():
    packed = pack_assistant_tokens(0, 0)
    # F flag still set -> negative even with zero payload
    assert packed < 0
    assert unpack_token_count(packed) == {"output_tokens": 0, "reasoning_tokens": 0}


@pytest.mark.parametrize(
    "tag1,v1,tag2,v2,name1,name2",
    [
        (TAG_OUTPUT, 10, TAG_REASONING, 20, "output_tokens", "reasoning_tokens"),
        (TAG_TOTAL_INPUT, 30, TAG_CACHE_READ, 40, "total_input_tokens", "cache_read_tokens"),
    ],
)
def test_generic_pack_round_trip(tag1, v1, tag2, v2, name1, name2):
    d = unpack_token_count(pack_token_count(tag1, v1, tag2, v2))
    assert d[name1] == v1
    assert d[name2] == v2


# --- boundary saturation --------------------------------------------------

def test_value1_saturates_at_27_bits():
    packed = pack_token_count(TAG_TOTAL_INPUT, _V1_MAX + 1000, TAG_CACHE_READ, 5)
    d = unpack_token_count(packed)
    assert d["total_input_tokens"] == _V1_MAX
    assert d["cache_read_tokens"] == 5


def test_value2_saturates_at_28_bits():
    packed = pack_token_count(TAG_TOTAL_INPUT, 5, TAG_CACHE_READ, _V2_MAX + 1000)
    d = unpack_token_count(packed)
    assert d["total_input_tokens"] == 5
    assert d["cache_read_tokens"] == _V2_MAX


def test_negative_inputs_clamped_to_zero():
    packed = pack_token_count(TAG_OUTPUT, -5, TAG_REASONING, -9)
    d = unpack_token_count(packed)
    assert d["output_tokens"] == 0
    assert d["reasoning_tokens"] == 0


def test_max_values_do_not_bleed_across_fields():
    # Both fields at max must decode independently (no carry/overlap).
    packed = pack_token_count(TAG_TOTAL_INPUT, _V1_MAX, TAG_CACHE_READ, _V2_MAX)
    d = unpack_token_count(packed)
    assert d["total_input_tokens"] == _V1_MAX
    assert d["cache_read_tokens"] == _V2_MAX


# --- F == sign bit --------------------------------------------------------

def test_format_flag_is_sign_bit():
    # Every packed value is negative; this is the read-time discriminator.
    for a, b in [(0, 0), (1, 1), (_V1_MAX, _V2_MAX), (42, 99)]:
        assert pack_token_count(TAG_OUTPUT, a, TAG_REASONING, b) < 0


def test_packed_value_in_signed64_range():
    packed = pack_token_count(TAG_TOTAL_INPUT, _V1_MAX, TAG_CACHE_READ, _V2_MAX)
    assert -(2 ** 63) <= packed <= (2 ** 63) - 1


# --- legacy compatibility -------------------------------------------------

def test_legacy_non_negative_is_raw_count():
    assert unpack_token_count(0) == {"legacy": 0}
    assert unpack_token_count(1500) == {"legacy": 1500}


def test_none_unpacks_to_empty():
    assert unpack_token_count(None) == {}


def test_unknown_tag_falls_back_to_hex_name():
    # tag 0x7 has no registered name -> tag_0x7 key
    packed = pack_token_count(0x7, 11, 0x0, 22)
    d = unpack_token_count(packed)
    assert d["tag_0x7"] == 11
    assert d["output_tokens"] == 22


# --- resolve_message_tokens ----------------------------------------------

def test_resolve_assistant_packed():
    packed = pack_assistant_tokens(1000, 250)
    out = resolve_message_tokens("assistant", packed)
    assert out == {"input": 0, "output": 1000, "cache_read": 0, "reasoning": 250}


def test_resolve_input_packed():
    packed = pack_input_tokens(8000, 6000)
    out = resolve_message_tokens("user", packed)
    assert out == {"input": 8000, "output": 0, "cache_read": 6000, "reasoning": 0}


def test_resolve_legacy_assistant_counts_as_output():
    out = resolve_message_tokens("assistant", 1234)
    assert out["output"] == 1234
    assert out["input"] == 0


def test_resolve_legacy_non_assistant_is_ignored():
    # A legacy non-negative count on a user/tool row is ambiguous; only
    # assistant rows historically carried output counts, so others -> 0.
    out = resolve_message_tokens("user", 1234)
    assert out == {"input": 0, "output": 0, "cache_read": 0, "reasoning": 0}


def test_resolve_none_is_all_zero():
    assert resolve_message_tokens("assistant", None) == {
        "input": 0, "output": 0, "cache_read": 0, "reasoning": 0,
    }
