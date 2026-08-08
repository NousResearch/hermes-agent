"""Gateway reasoning recap verbosity contracts (#79884)."""

import gateway.run as gateway_run


def _reasoning_lines(count: int) -> str:
    return "\n".join(f"line {index}" for index in range(1, count + 1))


def test_reasoning_recap_keeps_legacy_fifteen_line_default():
    recap = gateway_run._format_reasoning_recap(
        _reasoning_lines(18),
        user_config={},
        platform_key="telegram",
    )

    assert "line 15" in recap
    assert "line 16" not in recap
    assert "_... (3 more lines)_" in recap


def test_reasoning_recap_honors_global_full_setting():
    recap = gateway_run._format_reasoning_recap(
        _reasoning_lines(18),
        user_config={"display": {"reasoning_full": True}},
        platform_key="telegram",
    )

    assert "line 18" in recap
    assert "more lines" not in recap


def test_reasoning_recap_honors_platform_override_over_global():
    recap = gateway_run._format_reasoning_recap(
        _reasoning_lines(18),
        user_config={
            "display": {
                "reasoning_full": True,
                "platforms": {"telegram": {"reasoning_full": False}},
            }
        },
        platform_key="telegram",
    )

    assert "line 16" not in recap
    assert "_... (3 more lines)_" in recap
