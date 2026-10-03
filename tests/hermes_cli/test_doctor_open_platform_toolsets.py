"""Coverage for the doctor warning about publicly reachable powerful tools."""

import contextlib
import io

from hermes_cli import doctor_open_policy


def _run(config, env):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        finding = doctor_open_policy._check_open_platform_toolsets_for_config(
            config, env.get
        )
    return finding, output.getvalue()


def test_warns_for_open_platform_with_high_impact_toolsets():
    finding, output = _run(
        {
            "platforms": {"weixin": {"enabled": True, "dm_policy": "open"}},
            "platform_toolsets": {"weixin": ["terminal", "file"]},
        },
        {"WEIXIN_ALLOW_ALL_USERS": "true"},
    )

    assert "weixin" in output
    assert "file, terminal" in output
    assert finding.manual_issues


def test_closed_policy_does_not_warn_even_with_powerful_toolsets():
    finding, output = _run(
        {
            "platforms": {"weixin": {"enabled": True, "dm_policy": "allowlist"}},
            "platform_toolsets": {"weixin": ["terminal", "file"]},
        },
        {"WEIXIN_ALLOW_ALL_USERS": "true"},
    )

    assert output == ""
    assert finding.manual_issues == []


def test_open_policy_needs_allow_all_opt_in_before_warning():
    finding, output = _run(
        {
            "platforms": {"weixin": {"enabled": True, "dm_policy": "open"}},
            "platform_toolsets": {"weixin": ["terminal"]},
        },
        {},
    )

    assert output == ""
    assert finding.manual_issues == []


def test_open_platform_without_high_impact_toolsets_is_not_actionable():
    finding, output = _run(
        {
            "platforms": {"weixin": {"enabled": True, "dm_policy": "open"}},
            "platform_toolsets": {"weixin": ["web", "memory"]},
        },
        {"WEIXIN_ALLOW_ALL_USERS": "yes"},
    )

    assert output == ""
    assert finding.manual_issues == []
