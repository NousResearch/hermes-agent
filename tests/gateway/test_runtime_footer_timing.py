"""Tests for runtime-footer timing breakdown fields."""

from __future__ import annotations

import pytest

from gateway.runtime_footer import (
    _format_duration,
    build_footer_line,
    format_runtime_footer,
)
from hermes_cli.config import DEFAULT_CONFIG


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (None, ""),
        (-1.0, ""),
        (0.4, "0.4s"),
        (9.99, "10.0s"),
        (10, "10s"),
        (42.7, "43s"),
        (60, "1m00s"),
        (125, "2m05s"),
        (3599, "59m59s"),
        (3600, "1h00m"),
        (3725, "1h02m"),
    ],
)
def test_format_duration(seconds, expected):
    assert _format_duration(seconds) == expected


def test_default_config_enables_all_timing_fields_when_footer_is_opted_in():
    footer = DEFAULT_CONFIG["display"]["runtime_footer"]
    assert footer["enabled"] is False
    assert footer["fields"] == [
        "model",
        "context_pct",
        "cwd",
        "turn_time",
        "api_time",
        "tool_time",
        "overhead_time",
        "api_calls",
    ]


def test_format_footer_full_timing_breakdown():
    out = format_runtime_footer(
        model="MiniMax-M3",
        context_tokens=20_000,
        context_length=200_000,
        turn_time=70.0,
        api_time=42.0,
        tool_time=15.0,
        api_calls=3,
        fields=(
            "model",
            "context_pct",
            "turn_time",
            "api_time",
            "tool_time",
            "overhead_time",
            "api_calls",
        ),
    )
    assert out == (
        "MiniMax-M3 · 10% · 1m10s · api 42s · tools 15s · "
        "other 13s · 3 calls"
    )


def test_format_footer_timing_fields_are_independently_optional():
    out = format_runtime_footer(
        model="x",
        context_tokens=0,
        context_length=100,
        api_time=38.2,
        tool_time=3.1,
        api_calls=1,
        fields=("api_time", "tool_time", "api_calls"),
    )
    assert out == "api 38s · tools 3.1s · 1 call"


def test_format_footer_overhead_is_clamped_to_zero():
    out = format_runtime_footer(
        model="x",
        context_tokens=0,
        context_length=100,
        turn_time=10.0,
        api_time=15.0,
        tool_time=5.0,
        fields=("overhead_time",),
    )
    assert out == "other 0.0s"


def test_format_footer_omits_breakdown_when_component_timings_are_unavailable():
    out = format_runtime_footer(
        model="x",
        context_tokens=0,
        context_length=100,
        turn_time=70.0,
        api_time=None,
        tool_time=None,
        fields=("turn_time", "api_time", "tool_time", "overhead_time"),
    )
    assert out == "1m10s"


def test_build_footer_wires_timing_values_end_to_end():
    user = {
        "display": {
            "runtime_footer": {
                "enabled": True,
                "fields": [
                    "turn_time",
                    "api_time",
                    "tool_time",
                    "overhead_time",
                    "api_calls",
                ],
            }
        }
    }
    out = build_footer_line(
        user_config=user,
        platform_key="discord",
        model="x",
        context_tokens=0,
        context_length=100,
        turn_time=70.0,
        api_time=42.0,
        tool_time=15.0,
        api_calls=3,
    )
    assert out == "1m10s · api 42s · tools 15s · other 13s · 3 calls"
