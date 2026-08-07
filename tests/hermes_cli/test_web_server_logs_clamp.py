"""Regression tests for dashboard + CLI log line-count clamping."""

import pytest

from hermes_cli.logs import (
    LOG_LINES_DEFAULT_CLI,
    LOG_LINES_DEFAULT_DASHBOARD,
    clamp_log_lines,
    tail_log,
)


class TestClampLogLines:
    def test_invalid_value_uses_default(self):
        assert clamp_log_lines("bad") == LOG_LINES_DEFAULT_CLI
        assert clamp_log_lines("bad", default=LOG_LINES_DEFAULT_DASHBOARD) == 100

    def test_zero_clamped_to_one(self):
        assert clamp_log_lines(0) == 1

    def test_negative_clamped_to_one(self):
        assert clamp_log_lines(-1) == 1

    def test_excessive_lines_capped(self):
        assert clamp_log_lines(9999) == 500

    def test_valid_value_unchanged(self):
        assert clamp_log_lines(50) == 50


class TestApiLogsLinesClamp:
    """Endpoint-level tests so both the normal and search paths stay clamped."""

    @pytest.fixture(autouse=True)
    def _setup_client(self, monkeypatch, _isolate_hermes_home):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        import hermes_state
        from hermes_constants import get_hermes_home
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")

        logs_dir = get_hermes_home() / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        # 20 multi-line entries so lines=-1 would otherwise return ~19 lines
        # via Python negative-slice semantics instead of a 1-line tail.
        (logs_dir / "agent.log").write_text(
            "".join(f"2026-01-01 00:00:00 INFO agent.run: line-{i} unique-{i}\n" for i in range(20)),
            encoding="utf-8",
        )

        self.client = TestClient(app)
        self.client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
        self.home = get_hermes_home()

    def test_get_logs_negative_lines_returns_single_tail_line(self):
        resp = self.client.get("/api/logs?lines=-1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["lines"]) == 1
        assert "line-19" in data["lines"][0]

    def test_get_logs_search_negative_lines_clamps_post_filter(self):
        resp = self.client.get("/api/logs?lines=-1&search=unique")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["lines"]) == 1
        assert "unique-19" in data["lines"][0]

    def test_get_logs_huge_lines_capped(self):
        resp = self.client.get("/api/logs?lines=9999")
        assert resp.status_code == 200
        assert len(resp.json()["lines"]) == 20  # file only has 20 lines


class TestCliTailLogLinesClamp:
    def test_tail_log_negative_lines_prints_single_line(self, monkeypatch, _isolate_hermes_home, capsys):
        from hermes_constants import get_hermes_home

        logs_dir = get_hermes_home() / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / "agent.log").write_text(
            "".join(f"line-{i}\n" for i in range(20)),
            encoding="utf-8",
        )

        monkeypatch.setattr("sys.exit", lambda code=0: (_ for _ in ()).throw(SystemExit(code)))
        tail_log("agent", num_lines=-1)
        out = capsys.readouterr().out
        assert "line-19" in out
        # Must not dump nearly the whole file (negative-slice footgun).
        assert out.count("line-") == 1
