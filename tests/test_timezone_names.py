"""Keep the clock's server-local fallback out of sandbox TZ overrides."""

import json
from pathlib import Path

import pytest

import hermes_time
from hermes_constants import get_hermes_home


@pytest.fixture(autouse=True)
def reset_timezone_cache():
    hermes_time.reset_cache()
    yield
    hermes_time.reset_cache()


def configure_timezone(monkeypatch, source, name):
    if source == "environment":
        monkeypatch.setenv("HERMES_TIMEZONE", name)
    else:
        monkeypatch.delenv("HERMES_TIMEZONE", raising=False)
        (Path(get_hermes_home()) / "config.yaml").write_text(
            json.dumps({"timezone": name}), encoding="utf-8"
        )


@pytest.mark.parametrize("source", ["environment", "config"])
@pytest.mark.parametrize(
    "name,expected", [("Not/AZone", ""), ("", ""), ("Asia/Kolkata", "Asia/Kolkata")]
)
def test_timezone_name_exports_only_a_resolved_zone(
    monkeypatch, source, name, expected
):
    configure_timezone(monkeypatch, source, name)

    assert hermes_time.get_timezone_name() == expected
    zone = hermes_time.get_timezone()
    assert (str(zone) if zone is not None else "") == expected
    assert hermes_time.now().tzinfo is not None


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("source", ["environment", "config"])
def test_execute_code_does_not_override_local_time_with_an_invalid_zone(
    monkeypatch, source
):
    from tools.code_kernel import shutdown_all_kernels
    from tools.code_execution_tool import execute_code

    configure_timezone(monkeypatch, source, "Not/AZone")
    monkeypatch.setenv("TERMINAL_ENV", "local")
    try:
        result = json.loads(
            execute_code(
                code='import json, os; print(json.dumps({"tz": os.getenv("TZ"), "setting": os.getenv("HERMES_TIMEZONE")}))',
                task_id="invalid-timezone-test",
                enabled_tools=[],
            )
        )

        assert result["status"] == "success", result
        exported = json.loads(result["output"].strip())
        assert exported["tz"] is None
        assert exported["setting"] is None
    finally:
        shutdown_all_kernels()
