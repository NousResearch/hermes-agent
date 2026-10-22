"""Coverage for the doctor warning about publicly reachable powerful tools."""

import contextlib
import io

import pytest

from hermes_cli import doctor_open_policy


def _run(config, env):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        finding = doctor_open_policy._check_open_platform_toolsets_for_config(
            config, env.get
        )
    return finding, output.getvalue()


def _strict_env_get(env):
    def get(name):
        assert isinstance(name, str), f"environment variable name must be a string: {name!r}"
        return env.get(name)

    return get


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


@pytest.mark.parametrize("platforms", [
    {"qqbot": {"enabled": True, "dm_policy": "open"}, "weixin": {"enabled": True, "dm_policy": "open"}},
    {"weixin": {"enabled": True, "dm_policy": "open"}, "qqbot": {"enabled": True, "dm_policy": "open"}},
])
@pytest.mark.parametrize("should_fix", [False, True])
def test_decorated_check_reports_qqbot_and_other_open_platforms_with_strict_env_names(
    monkeypatch, platforms, should_fix,
):
    config = {
        "platforms": platforms,
        "platform_toolsets": {"qqbot": ["terminal"], "weixin": ["file"]},
    }
    env_get = _strict_env_get({
        "QQ_ALLOW_ALL_USERS": "true",
        "WEIXIN_ALLOW_ALL_USERS": "true",
    })
    from hermes_cli import config as config_module

    monkeypatch.setattr(config_module, "load_config", lambda: config)
    monkeypatch.setattr(config_module, "get_env_value", env_get)

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        finding = doctor_open_policy._check_open_platform_toolsets(should_fix)

    assert "qqbot" in output.getvalue()
    assert "weixin" in output.getvalue()
    assert any(issue.startswith("Review qqbot ") for issue in finding.manual_issues)
    assert any(issue.startswith("Review weixin ") for issue in finding.manual_issues)
    assert finding.issues == []
    assert finding.fixed == 0
