"""Tests for opt-in host sleep prevention while Hermes is running."""

import threading

from hermes_cli import power_sleep


def test_default_config_exposes_disabled_power_setting():
    from hermes_cli.config import DEFAULT_CONFIG

    assert power_sleep.prevent_sleep_config(DEFAULT_CONFIG) == {
        "enabled": False,
        "mode": "system",
        "surfaces": ["desktop", "gateway"],
    }


def test_prevent_sleep_disabled_by_default():
    assert power_sleep.should_prevent_sleep("gateway", config={}) is False
    assert power_sleep.should_prevent_sleep("desktop", config={"power": {}}) is False


def test_prevent_sleep_enabled_for_configured_surfaces():
    config = {
        "power": {
            "prevent_sleep": {
                "enabled": True,
                "surfaces": ["desktop", "gateway"],
                "mode": "system",
            }
        }
    }

    assert power_sleep.should_prevent_sleep("desktop", config=config) is True
    assert power_sleep.should_prevent_sleep("gateway", config=config) is True
    assert power_sleep.should_prevent_sleep("cron", config=config) is False
    assert power_sleep.prevent_sleep_mode(config=config) == "system"


def test_boolean_shorthand_uses_default_surfaces():
    config = {"power": {"prevent_sleep": True}}

    assert power_sleep.should_prevent_sleep("desktop", config=config) is True
    assert power_sleep.should_prevent_sleep("gateway", config=config) is True


def test_explicit_empty_surfaces_disable_every_surface():
    config = {"power": {"prevent_sleep": {"enabled": True, "surfaces": []}}}

    assert power_sleep.should_prevent_sleep("desktop", config=config) is False
    assert power_sleep.should_prevent_sleep("gateway", config=config) is False


def test_env_expanded_config_uses_canonical_loader(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("PREVENT_SLEEP_ENABLED", "true")
    monkeypatch.setenv("PREVENT_SLEEP_SURFACES", "desktop,gateway")
    monkeypatch.setenv("PREVENT_SLEEP_MODE", "display")
    (tmp_path / "config.yaml").write_text(
        "power:\n"
        "  prevent_sleep:\n"
        "    enabled: ${PREVENT_SLEEP_ENABLED}\n"
        "    surfaces: ${PREVENT_SLEEP_SURFACES}\n"
        "    mode: ${PREVENT_SLEEP_MODE}\n",
        encoding="utf-8",
    )

    from hermes_cli.config import load_config

    config = load_config()
    assert config["power"]["prevent_sleep"]["enabled"] == "true"
    assert power_sleep.should_prevent_sleep("desktop", config=config) is True
    assert power_sleep.should_prevent_sleep("gateway", config=config) is True
    assert power_sleep.prevent_sleep_mode(config=config) == "display"


def test_start_without_config_uses_canonical_readonly_loader(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("PREVENT_SLEEP_ENABLED", "true")
    (tmp_path / "config.yaml").write_text(
        "power:\n"
        "  prevent_sleep:\n"
        "    enabled: ${PREVENT_SLEEP_ENABLED}\n"
        "    surfaces: [gateway]\n"
        "    mode: display\n",
        encoding="utf-8",
    )

    handle = power_sleep.start_prevent_sleep("gateway", platform="linux")

    assert handle.reason == "unsupported-platform:linux"
    assert handle.mode == "display"


def test_canonical_loader_preserves_last_known_good_on_malformed_yaml(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "power:\n"
        "  prevent_sleep:\n"
        "    enabled: true\n"
        "    surfaces: [gateway]\n"
        "    mode: display\n",
        encoding="utf-8",
    )

    first = power_sleep.start_prevent_sleep("gateway", platform="linux")
    config_path.write_text("power: [\n", encoding="utf-8")
    second = power_sleep.start_prevent_sleep("gateway", platform="linux")

    assert first.reason == "unsupported-platform:linux"
    assert second.reason == "unsupported-platform:linux"
    assert second.mode == "display"


def test_invalid_mode_falls_back_to_system():
    config = {"power": {"prevent_sleep": {"enabled": True, "mode": "invalid"}}}

    assert power_sleep.prevent_sleep_mode(config=config) == "system"


def test_unresolved_boolean_template_fails_closed():
    config = {"power": {"prevent_sleep": {"enabled": "${MISSING_BOOL}"}}}

    assert power_sleep.should_prevent_sleep("gateway", config=config) is False


def test_windows_execution_state_flags_are_started_and_cleared():
    calls = []
    previous_state = power_sleep.ES_CONTINUOUS | power_sleep.ES_DISPLAY_REQUIRED

    def fake_set_thread_execution_state(flags):
        calls.append(flags)
        return previous_state

    handle = power_sleep.start_prevent_sleep(
        "gateway",
        config={"power": {"prevent_sleep": {"enabled": True, "mode": "system"}}},
        platform="win32",
        set_thread_execution_state=fake_set_thread_execution_state,
    )

    assert handle.started is True
    assert calls == [power_sleep.ES_CONTINUOUS | power_sleep.ES_SYSTEM_REQUIRED]
    assert handle.stop() is True
    assert calls[-1] == previous_state
    assert handle.stop() is False


def test_windows_handle_rejects_cross_thread_stop_and_retries_on_owner():
    calls = []

    def fake_set_thread_execution_state(flags):
        calls.append(flags)
        return power_sleep.ES_CONTINUOUS

    handle = power_sleep.start_prevent_sleep(
        "gateway",
        config={"power": {"prevent_sleep": {"enabled": True}}},
        platform="win32",
        set_thread_execution_state=fake_set_thread_execution_state,
    )
    stopped = []
    thread = threading.Thread(target=lambda: stopped.append(handle.stop()))

    thread.start()
    thread.join()

    assert stopped == [False]
    assert handle.started is True
    assert len(calls) == 1
    assert handle.stop() is True
    assert len(calls) == 2


def test_display_mode_adds_display_required_flag():
    calls = []

    handle = power_sleep.start_prevent_sleep(
        "desktop",
        config={"power": {"prevent_sleep": {"enabled": True, "mode": "display"}}},
        platform="win32",
        set_thread_execution_state=lambda flags: calls.append(flags) or 1,
    )

    assert handle.started is True
    assert calls == [
        power_sleep.ES_CONTINUOUS
        | power_sleep.ES_SYSTEM_REQUIRED
        | power_sleep.ES_DISPLAY_REQUIRED
    ]


def test_windows_api_zero_return_is_not_reported_as_started():
    handle = power_sleep.start_prevent_sleep(
        "gateway",
        config={"power": {"prevent_sleep": {"enabled": True}}},
        platform="win32",
        set_thread_execution_state=lambda _flags: 0,
    )

    assert handle.started is False
    assert handle.reason == "api-failed"


def test_failed_clear_can_be_retried():
    results = iter([1, 0, 1])
    handle = power_sleep.start_prevent_sleep(
        "gateway",
        config={"power": {"prevent_sleep": {"enabled": True}}},
        platform="win32",
        set_thread_execution_state=lambda _flags: next(results),
    )

    assert handle.stop() is False
    assert handle.started is True
    assert handle.stop() is True
    assert handle.started is False


def test_non_windows_enabled_config_returns_inactive_handle_without_calling_api():
    calls = []

    handle = power_sleep.start_prevent_sleep(
        "gateway",
        config={"power": {"prevent_sleep": {"enabled": True}}},
        platform="linux",
        set_thread_execution_state=lambda flags: calls.append(flags) or 1,
    )

    assert handle.started is False
    assert calls == []
