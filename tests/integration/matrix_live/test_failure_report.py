"""A failed live test's report includes the log of each gateway that the test used."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.integration.matrix_live.live_gateway import LiveGateway


def test_a_failed_test_report_includes_the_gateway_log(request: pytest.FixtureRequest, tmp_path: Path) -> None:
    """Test modules import ``LiveGateway`` by its dotted path, and pytest loads the conftest,
    which defines the hook, as a plugin module under another name. The hook must still
    recognise their gateways."""
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "gateway.log").write_text("gateway log line\n")
    item = SimpleNamespace(funcargs={"rtc_gateway": LiveGateway(None, None, tmp_path)})
    report = pytest.TestReport("live::test", ("live", 0, "test"), {}, "failed", None, "call")
    conftest = request.config.pluginmanager.getplugin(str(Path(__file__).with_name("conftest.py")))

    hook = conftest.pytest_runtest_makereport(item, None)
    next(hook)
    with pytest.raises(StopIteration) as finished:
        hook.send(report)

    assert finished.value.value.sections == [("gateway.log", "gateway log line")]
