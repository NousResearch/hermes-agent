import json

import hermes_cli.xsearch_command as xsearch_cmd


def test_status_reports_ready_when_toolset_and_oauth_are_available(monkeypatch):
    cfg = {
        "platform_toolsets": {"cli": ["x_search"]},
        "x_search": {
            "model": "grok-4.20-reasoning",
            "timeout_seconds": "90",
            "retries": "1",
        },
    }

    monkeypatch.setattr(xsearch_cmd, "load_config", lambda: cfg)
    monkeypatch.setattr(xsearch_cmd, "_get_platform_toolsets", lambda *_args, **_kwargs: {"x_search"})
    monkeypatch.setattr(xsearch_cmd, "_oauth_status", lambda: {"logged_in": True})
    monkeypatch.setattr(xsearch_cmd, "get_env_value", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(xsearch_cmd, "_check_x_search_requirements", lambda: True)

    result = xsearch_cmd.run_xsearch_command("/xsearch status")

    assert "Ready: yes" in result.output
    assert "Toolset enabled: yes" in result.output
    assert "Preferred credential source: xai-oauth" in result.output


def test_setup_enables_toolset_sets_default_model_and_requests_auth(monkeypatch):
    cfg = {
        "platform_toolsets": {"cli": []},
        "x_search": {},
    }
    saved = {"called": False}

    monkeypatch.setattr(xsearch_cmd, "load_config", lambda: cfg)
    monkeypatch.setattr(
        xsearch_cmd,
        "_get_platform_toolsets",
        lambda config, platform: set(config.get("platform_toolsets", {}).get(platform, [])),
    )

    def _fake_apply(config, platform, toolset_names, action):
        assert action == "enable"
        config.setdefault("platform_toolsets", {}).setdefault(platform, [])
        for name in toolset_names:
            if name not in config["platform_toolsets"][platform]:
                config["platform_toolsets"][platform].append(name)

    monkeypatch.setattr(xsearch_cmd, "_apply_toolset_toggle", _fake_apply)
    monkeypatch.setattr(xsearch_cmd, "save_config", lambda _cfg: saved.__setitem__("called", True))
    monkeypatch.setattr(xsearch_cmd, "_oauth_status", lambda: {"logged_in": False})
    monkeypatch.setattr(xsearch_cmd, "get_env_value", lambda *_args, **_kwargs: None)

    result = xsearch_cmd.run_xsearch_command("/xsearch setup")

    assert result.reset_session is True
    assert saved["called"] is True
    assert cfg["x_search"]["model"] == xsearch_cmd.DEFAULT_X_SEARCH_MODEL
    assert "enabled x_search for cli" in result.output
    assert "hermes auth add xai-oauth" in result.output


def test_search_parses_flags_and_formats_result(monkeypatch):
    cfg = {
        "platform_toolsets": {"cli": ["x_search"]},
        "x_search": {"model": "grok-4.20-reasoning"},
    }
    captured = {}

    monkeypatch.setattr(xsearch_cmd, "load_config", lambda: cfg)
    monkeypatch.setattr(xsearch_cmd, "_get_platform_toolsets", lambda *_args, **_kwargs: {"x_search"})

    def _fake_tool(**kwargs):
        captured.update(kwargs)
        return json.dumps(
            {
                "success": True,
                "model": "grok-4.20-reasoning",
                "credential_source": "xai-oauth",
                "answer": "Summary",
                "citations": [{"title": "xAI", "url": "https://x.com/xai/status/1"}],
                "inline_citations": [],
                "degraded": False,
            }
        )

    monkeypatch.setattr(xsearch_cmd, "_run_x_search_tool", _fake_tool)

    result = xsearch_cmd.run_xsearch_command(
        "/xsearch latest grok launch reactions --from xai,openai --since 2026-05-20 --images"
    )

    assert captured["query"] == "latest grok launch reactions"
    assert captured["allowed_x_handles"] == ["xai", "openai"]
    assert captured["from_date"] == "2026-05-20"
    assert captured["enable_image_understanding"] is True
    assert "X Search via grok-4.20-reasoning (xai-oauth)" in result.output
    assert "Sources:" in result.output


def test_search_refuses_when_toolset_is_disabled(monkeypatch):
    cfg = {"platform_toolsets": {"cli": []}, "x_search": {}}

    monkeypatch.setattr(xsearch_cmd, "load_config", lambda: cfg)
    monkeypatch.setattr(xsearch_cmd, "_get_platform_toolsets", lambda *_args, **_kwargs: set())

    result = xsearch_cmd.run_xsearch_command("/xsearch latest xai posts")

    assert "x_search is disabled for cli" in result.output
