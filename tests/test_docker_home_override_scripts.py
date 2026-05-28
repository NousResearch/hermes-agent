"""Regression tests for Docker HOME overrides under s6/with-contenv."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_RUN = REPO_ROOT / "docker" / "s6-rc.d" / "dashboard" / "run"


def test_dashboard_run_resets_home_before_dropping_privileges() -> None:
    text = DASHBOARD_RUN.read_text(encoding="utf-8")

    assert "#!/command/with-contenv sh" in text
    assert "export HOME=/opt/data" in text
    assert "exec s6-setuidgid hermes hermes dashboard" in text
