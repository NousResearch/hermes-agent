"""Tests for the /rpe Discord slash command.

AC coverage:
  AC1  - /rpe is registered in the Discord adapter's _register_slash_commands
  AC2  - rpe is a required integer parameter constrained to 1-10; out-of-range rejected
  AC3  - notes is an optional free-text parameter forwarded verbatim
  AC4  - date is an optional parameter; defaults to today's date when omitted
  AC5  - POSTs to perf-coach feel-entry endpoint using bearer token from config (not hardcoded)
  AC6  - Hermes does not parse or interpret feedback content
  AC7  - Non-2xx/unreachable endpoint → clear human-readable error; no silent failure
  AC8  - 409 from endpoint treated as idempotent no-op or clear duplicate notice
  AC9  - Every invocation audit-logged; bearer token never appears in the log
"""
from __future__ import annotations

import datetime
import json
import os
import sys
import threading
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Config tests (AC5 — token from config, never hardcoded)
# ---------------------------------------------------------------------------

class TestConfig:
    def test_get_perf_coach_url_from_env(self, monkeypatch):
        monkeypatch.setenv("PERF_COACH_URL", "https://perf.example.com")
        from services.hermes.config import get_perf_coach_url
        assert get_perf_coach_url() == "https://perf.example.com"

    def test_get_perf_coach_url_missing_returns_none(self, monkeypatch):
        monkeypatch.delenv("PERF_COACH_URL", raising=False)
        from services.hermes.config import get_perf_coach_url
        assert get_perf_coach_url() is None

    def test_get_perf_coach_token_from_env(self, monkeypatch):
        monkeypatch.setenv("PERF_COACH_BEARER_TOKEN", "secret-tok")
        from services.hermes.config import get_perf_coach_token
        assert get_perf_coach_token() == "secret-tok"

    def test_get_perf_coach_token_missing_returns_none(self, monkeypatch):
        monkeypatch.delenv("PERF_COACH_BEARER_TOKEN", raising=False)
        from services.hermes.config import get_perf_coach_token
        assert get_perf_coach_token() is None

    def test_get_perf_coach_user_from_env(self, monkeypatch):
        monkeypatch.setenv("PERF_COACH_USER", "jane doe")
        from services.hermes.config import get_perf_coach_user
        assert get_perf_coach_user() == "jane doe"

    def test_get_perf_coach_user_missing_returns_none(self, monkeypatch):
        monkeypatch.delenv("PERF_COACH_USER", raising=False)
        from services.hermes.config import get_perf_coach_user
        assert get_perf_coach_user() is None


# ---------------------------------------------------------------------------
# Audit log tests (AC9)
# ---------------------------------------------------------------------------

class TestAuditLog:
    def test_log_writes_required_fields(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from services.hermes import audit
        import importlib
        importlib.reload(audit)

        audit.log_rpe_invocation(
            user_id="u123",
            rpe=7,
            has_notes=True,
            target_date="2026-07-14",
            http_outcome="200 OK",
        )

        log_path = tmp_path / "logs" / "rpe-audit.log"
        assert log_path.exists()
        line = json.loads(log_path.read_text().strip())
        assert line["user_id"] == "u123"
        assert line["rpe"] == 7
        assert line["has_notes"] is True
        assert line["target_date"] == "2026-07-14"
        assert line["http_outcome"] == "200 OK"
        assert "ts" in line

    def test_log_records_failure_outcome(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from services.hermes import audit
        import importlib
        importlib.reload(audit)

        audit.log_rpe_invocation(
            user_id="u456",
            rpe=5,
            has_notes=False,
            target_date="2026-07-14",
            http_outcome="503 Connection Error",
        )
        log_path = tmp_path / "logs" / "rpe-audit.log"
        line = json.loads(log_path.read_text().strip())
        assert line["http_outcome"] == "503 Connection Error"
        assert line["user_id"] == "u456"

    def test_log_never_contains_bearer_token(self, tmp_path, monkeypatch):
        """AC9: bearer token must never appear in the audit log."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("PERF_COACH_BEARER_TOKEN", "super-secret-bearer")
        from services.hermes import audit
        import importlib
        importlib.reload(audit)

        audit.log_rpe_invocation(
            user_id="u789",
            rpe=3,
            has_notes=False,
            target_date="2026-07-14",
            http_outcome="200 OK",
        )
        log_content = (tmp_path / "logs" / "rpe-audit.log").read_text()
        assert "super-secret-bearer" not in log_content

    def test_log_survives_write_error(self, tmp_path, monkeypatch, caplog):
        """Audit log failures must not raise — auth/logging must never block commands."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from services.hermes import audit
        import importlib
        importlib.reload(audit)

        with patch("builtins.open", side_effect=OSError("disk full")):
            # Must not raise
            audit.log_rpe_invocation(
                user_id="u000",
                rpe=5,
                has_notes=False,
                target_date="2026-07-14",
                http_outcome="200 OK",
            )


# ---------------------------------------------------------------------------
# Handler tests — tests for the core RPE logic
# ---------------------------------------------------------------------------

def _make_feel_entry_response(status_code: int, json_body: dict | None = None):
    """Build a minimal mock HTTP response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = (200 <= status_code < 300)
    resp.json.return_value = json_body or {}
    return resp


class TestRpeHandler:
    """Tests for handle_rpe() — the core function that POSTs to feel-entry."""

    def setup_method(self):
        # Clear any cached imports so env patches take effect
        for mod in list(sys.modules):
            if mod.startswith("services.hermes"):
                del sys.modules[mod]

    def test_success_posts_correct_payload(self, monkeypatch, tmp_path):
        """AC5 — POSTs to feel-entry with feel_date, rpe_1_to_10, notes; bearer token in header."""
        monkeypatch.setenv("PERF_COACH_URL", "https://perf.example.com")
        monkeypatch.setenv("PERF_COACH_BEARER_TOKEN", "tok-abc")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("PERF_COACH_USER", raising=False)

        mock_resp = _make_feel_entry_response(200)
        with patch("urllib.request.urlopen") as mock_open_url:
            mock_open_url.return_value.__enter__ = lambda s: mock_open_url.return_value
            mock_open_url.return_value.__exit__ = MagicMock(return_value=False)
            mock_open_url.return_value.status = 200
            mock_open_url.return_value.read.return_value = b"{}"

            from services.hermes.discord import handle_rpe
            result = handle_rpe(
                user_id="u1",
                rpe=7,
                notes=None,
                date_str=None,
                today=datetime.date(2026, 7, 14),
            )

        assert result["success"] is True
        assert "error" not in result or result.get("error") is None
        # The HTTP call must have been made
        mock_open_url.assert_called_once()
        req = mock_open_url.call_args[0][0]
        body = json.loads(req.data)
        assert body["rpe_1_to_10"] == 7
        assert body["feel_date"] == "2026-07-14"
        # notes absent or null when not provided
        assert body.get("notes") is None or body.get("notes") == ""
        # Old contract keys must not be present.
        assert "rpe" not in body
        assert "date" not in body
        assert "user_id" not in body
        assert req.get_header("Authorization") == "Bearer tok-abc"
        # No PERF_COACH_USER set → no ``user`` query param on the URL.
        assert "?user=" not in req.full_url
        assert req.full_url == "https://perf.example.com/feel-entry"

    def test_notes_forwarded_verbatim(self, monkeypatch, tmp_path):
        """AC3 — notes forwarded exactly as given, not parsed or interpreted."""
        monkeypatch.setenv("PERF_COACH_URL", "https://perf.example.com")
        monkeypatch.setenv("PERF_COACH_BEARER_TOKEN", "tok-abc")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("PERF_COACH_USER", raising=False)

        with patch("urllib.request.urlopen") as mock_open_url:
            mock_open_url.return_value.__enter__ = lambda s: mock_open_url.return_value
            mock_open_url.return_value.__exit__ = MagicMock(return_value=False)
            mock_open_url.return_value.status = 200
            mock_open_url.return_value.read.return_value = b"{}"

            from services.hermes.discord import handle_rpe
            handle_rpe(
                user_id="u1",
                rpe=5,
                notes="Felt tired",
                date_str=None,
                today=datetime.date(2026, 7, 14),
            )

        req = mock_open_url.call_args[0][0]
        body = json.loads(req.data)
        assert body["notes"] == "Felt tired"

    def test_date_defaults_to_today(self, monkeypatch, tmp_path):
        """AC4 — when date_str is None, payload uses today's date."""
        monkeypatch.setenv("PERF_COACH_URL", "https://perf.example.com")
        monkeypatch.setenv("PERF_COACH_BEARER_TOKEN", "tok")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("PERF_COACH_USER", raising=False)

        with patch("urllib.request.urlopen") as mock_open_url:
            mock_open_url.return_value.__enter__ = lambda s: mock_open_url.return_value
            mock_open_url.return_value.__exit__ = MagicMock(return_value=False)
            mock_open_url.return_value.status = 200
            mock_open_url.return_value.read.return_value = b"{}"

            from services.hermes.discord import handle_rpe
            today = datetime.date(2026, 7, 14)
            handle_rpe(user_id="u1", rpe=6, notes=None, date_str=None, today=today)

        req = mock_open_url.call_args[0][0]
        body = json.loads(req.data)
        assert body["feel_date"] == "2026-07-14"

    def test_explicit_date_used_when_provided(self, monkeypatch, tmp_path):
        """AC4 — explicit date_str is forwarded to the endpoint."""
        monkeypatch.setenv("PERF_COACH_URL", "https://perf.example.com")
        monkeypatch.setenv("PERF_COACH_BEARER_TOKEN", "tok")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("PERF_COACH_USER", raising=False)

        with patch("urllib.request.urlopen") as mock_open_url:
            mock_open_url.return_value.__enter__ = lambda s: mock_open_url.return_value
            mock_open_url.return_value.__exit__ = MagicMock(return_value=False)
            mock_open_url.return_value.status = 200
            mock_open_url.return_value.read.return_value = b"{}"

            from services.hermes.discord import handle_rpe
            handle_rpe(
                user_id="u1",
                rpe=5,
                notes="Felt tired",
                date_str="2026-07-10",
                today=datetime.date(2026, 7, 14),
            )

        req = mock_open_url.call_args[0][0]
        body = json.loads(req.data)
        assert body["feel_date"] == "2026-07-10"

    def test_rpe_below_minimum_rejected(self, monkeypatch, tmp_path):
        """AC2 — rpe=0 is rejected; no HTTP call made."""
        monkeypatch.setenv("PERF_COACH_URL", "https://perf.example.com")
        monkeypatch.setenv("PERF_COACH_BEARER_TOKEN", "tok")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("PERF_COACH_USER", raising=False)

        with patch("urllib.request.urlopen") as mock_open_url:
            from services.hermes.discord import handle_rpe
            result = handle_rpe(
                user_id="u1", rpe=0, notes=None, date_str=None,
                today=datetime.date(2026, 7, 14),
            )

        assert result["success"] is False
        assert "1" in result["error"] and "10" in result["error"]
        mock_open_url.assert_not_called()

    def test_rpe_above_maximum_rejected(self, monkeypatch, tmp_path):
        """AC2 — rpe=11 is rejected; no HTTP call made."""
        monkeypatch.setenv("PERF_COACH_URL", "https://perf.example.com")
        monkeypatch.setenv("PERF_COACH_BEARER_TOKEN", "tok")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("PERF_COACH_USER", raising=False)

        with patch("urllib.request.urlopen") as mock_open_url:
            from services.hermes.discord import handle_rpe
            result = handle_rpe(
                user_id="u1", rpe=11, notes=None, date_str=None,
                today=datetime.date(2026, 7, 14),
            )

        assert result["success"] is False
        assert "1" in result["error"] and "10" in result["error"]
        mock_open_url.assert_not_called()

    def test_non_2xx_response_returns_error(self, monkeypatch, tmp_path):
        """AC7 — non-2xx response → clear error message, not a silent success."""
        import urllib.error
        monkeypatch.setenv("PERF_COACH_URL", "https://perf.example.com")
        monkeypatch.setenv("PERF_COACH_BEARER_TOKEN", "tok")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("PERF_COACH_USER", raising=False)

        http_err = urllib.error.HTTPError(
            url="https://perf.example.com/feel-entry",
            code=503,
            msg="Service Unavailable",
            hdrs={},
            fp=None,
        )

        with patch("urllib.request.urlopen", side_effect=http_err):
            from services.hermes.discord import handle_rpe
            result = handle_rpe(
                user_id="u1", rpe=7, notes=None, date_str=None,
                today=datetime.date(2026, 7, 14),
            )

        assert result["success"] is False
        assert result["error"]  # must not be empty
        # Must be human-readable, not a raw exception repr
        assert "perf" in result["error"].lower() or "service" in result["error"].lower() or "unavailable" in result["error"].lower() or "try again" in result["error"].lower()

    def test_connection_error_returns_friendly_message(self, monkeypatch, tmp_path):
        """AC7 — unreachable endpoint → friendly error; no raw traceback exposed."""
        import urllib.error
        monkeypatch.setenv("PERF_COACH_URL", "https://perf.example.com")
        monkeypatch.setenv("PERF_COACH_BEARER_TOKEN", "tok")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("PERF_COACH_USER", raising=False)

        url_err = urllib.error.URLError("Connection refused")
        with patch("urllib.request.urlopen", side_effect=url_err):
            from services.hermes.discord import handle_rpe
            result = handle_rpe(
                user_id="u1", rpe=6, notes=None, date_str=None,
                today=datetime.date(2026, 7, 14),
            )

        assert result["success"] is False
        assert result["error"]
        # Must NOT expose raw exception string (no "URLError", "Traceback", etc.)
        assert "URLError" not in result["error"]
        assert "Traceback" not in result["error"]

    def test_409_treated_as_idempotent(self, monkeypatch, tmp_path):
        """AC8 — 409 Conflict from endpoint → success/duplicate notice, not an error."""
        import urllib.error
        monkeypatch.setenv("PERF_COACH_URL", "https://perf.example.com")
        monkeypatch.setenv("PERF_COACH_BEARER_TOKEN", "tok")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("PERF_COACH_USER", raising=False)

        http_err = urllib.error.HTTPError(
            url="https://perf.example.com/feel-entry",
            code=409,
            msg="Conflict",
            hdrs={},
            fp=None,
        )

        with patch("urllib.request.urlopen", side_effect=http_err):
            from services.hermes.discord import handle_rpe
            result = handle_rpe(
                user_id="u1", rpe=7, notes=None, date_str=None,
                today=datetime.date(2026, 7, 14),
            )

        # Must succeed (idempotent) or clearly note the duplicate
        assert result["success"] is True or (
            result["success"] is False and "duplicate" in result.get("error", "").lower()
            or "already" in result.get("error", "").lower()
        )

    def test_missing_url_config_returns_error(self, monkeypatch, tmp_path):
        """AC5 — missing PERF_COACH_URL config → clear error, no HTTP call."""
        monkeypatch.delenv("PERF_COACH_URL", raising=False)
        monkeypatch.setenv("PERF_COACH_BEARER_TOKEN", "tok")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("PERF_COACH_USER", raising=False)

        with patch("urllib.request.urlopen") as mock_open_url:
            from services.hermes.discord import handle_rpe
            result = handle_rpe(
                user_id="u1", rpe=7, notes=None, date_str=None,
                today=datetime.date(2026, 7, 14),
            )

        assert result["success"] is False
        assert result["error"]
        mock_open_url.assert_not_called()

    def test_missing_token_config_returns_error(self, monkeypatch, tmp_path):
        """AC5 — missing PERF_COACH_BEARER_TOKEN → clear error, no HTTP call."""
        monkeypatch.setenv("PERF_COACH_URL", "https://perf.example.com")
        monkeypatch.delenv("PERF_COACH_BEARER_TOKEN", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("PERF_COACH_USER", raising=False)

        with patch("urllib.request.urlopen") as mock_open_url:
            from services.hermes.discord import handle_rpe
            result = handle_rpe(
                user_id="u1", rpe=7, notes=None, date_str=None,
                today=datetime.date(2026, 7, 14),
            )

        assert result["success"] is False
        assert result["error"]
        mock_open_url.assert_not_called()

    def test_audit_log_written_on_success(self, monkeypatch, tmp_path):
        """AC9 — audit log entry written on every successful invocation."""
        monkeypatch.setenv("PERF_COACH_URL", "https://perf.example.com")
        monkeypatch.setenv("PERF_COACH_BEARER_TOKEN", "tok-xyz")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("PERF_COACH_USER", raising=False)

        with patch("urllib.request.urlopen") as mock_open_url:
            mock_open_url.return_value.__enter__ = lambda s: mock_open_url.return_value
            mock_open_url.return_value.__exit__ = MagicMock(return_value=False)
            mock_open_url.return_value.status = 200
            mock_open_url.return_value.read.return_value = b"{}"

            from services.hermes.discord import handle_rpe
            from services.hermes import audit
            import importlib
            importlib.reload(audit)

            handle_rpe(
                user_id="u99",
                rpe=8,
                notes="Great session",
                date_str=None,
                today=datetime.date(2026, 7, 14),
            )

        log_path = tmp_path / "logs" / "rpe-audit.log"
        assert log_path.exists()
        line = json.loads(log_path.read_text().strip())
        assert line["user_id"] == "u99"
        assert line["rpe"] == 8
        assert line["has_notes"] is True
        assert line["target_date"] == "2026-07-14"
        assert "200" in line["http_outcome"]
        assert "tok-xyz" not in log_path.read_text()

    def test_audit_log_written_on_failure(self, monkeypatch, tmp_path):
        """AC9 — audit log entry written even when HTTP call fails."""
        import urllib.error
        monkeypatch.setenv("PERF_COACH_URL", "https://perf.example.com")
        monkeypatch.setenv("PERF_COACH_BEARER_TOKEN", "tok")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("PERF_COACH_USER", raising=False)

        http_err = urllib.error.HTTPError("url", 500, "Server Error", {}, None)
        with patch("urllib.request.urlopen", side_effect=http_err):
            from services.hermes.discord import handle_rpe
            from services.hermes import audit
            import importlib
            importlib.reload(audit)

            handle_rpe(
                user_id="u1", rpe=7, notes=None, date_str=None,
                today=datetime.date(2026, 7, 14),
            )

        log_path = tmp_path / "logs" / "rpe-audit.log"
        assert log_path.exists()
        line = json.loads(log_path.read_text().strip())
        assert "500" in line["http_outcome"] or "error" in line["http_outcome"].lower()

    def test_no_interpretation_of_notes_content(self, monkeypatch, tmp_path):
        """AC6 — notes content is forwarded as-is; handler doesn't parse or act on it."""
        monkeypatch.setenv("PERF_COACH_URL", "https://perf.example.com")
        monkeypatch.setenv("PERF_COACH_BEARER_TOKEN", "tok")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("PERF_COACH_USER", raising=False)

        special_notes = "URGENT: stop all training immediately"

        with patch("urllib.request.urlopen") as mock_open_url:
            mock_open_url.return_value.__enter__ = lambda s: mock_open_url.return_value
            mock_open_url.return_value.__exit__ = MagicMock(return_value=False)
            mock_open_url.return_value.status = 200
            mock_open_url.return_value.read.return_value = b"{}"

            from services.hermes.discord import handle_rpe
            result = handle_rpe(
                user_id="u1",
                rpe=5,
                notes=special_notes,
                date_str=None,
                today=datetime.date(2026, 7, 14),
            )

        # Must succeed — notes content doesn't affect handler behavior
        assert result["success"] is True
        req = mock_open_url.call_args[0][0]
        body = json.loads(req.data)
        assert body["notes"] == special_notes


# ---------------------------------------------------------------------------
# PERF_COACH_USER query-param tests
# ---------------------------------------------------------------------------

class TestRpeHandlerPerfCoachUser:
    """Tests for the ``?user=`` query param sourced from PERF_COACH_USER."""

    def setup_method(self):
        # Clear any cached imports so env patches take effect
        for mod in list(sys.modules):
            if mod.startswith("services.hermes"):
                del sys.modules[mod]

    def test_perf_coach_user_set_appends_urlencoded_query_param(self, monkeypatch, tmp_path):
        """PERF_COACH_USER set (with a space) → ?user=<urlencoded> appended to the URL."""
        monkeypatch.setenv("PERF_COACH_URL", "https://perf.example.com")
        monkeypatch.setenv("PERF_COACH_BEARER_TOKEN", "tok")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("PERF_COACH_USER", "jane doe")

        with patch("urllib.request.urlopen") as mock_open_url:
            mock_open_url.return_value.__enter__ = lambda s: mock_open_url.return_value
            mock_open_url.return_value.__exit__ = MagicMock(return_value=False)
            mock_open_url.return_value.status = 200
            mock_open_url.return_value.read.return_value = b"{}"

            from services.hermes.discord import handle_rpe
            result = handle_rpe(
                user_id="u1",
                rpe=7,
                notes=None,
                date_str=None,
                today=datetime.date(2026, 7, 14),
            )

        assert result["success"] is True
        req = mock_open_url.call_args[0][0]
        assert req.full_url == "https://perf.example.com/feel-entry?user=jane%20doe"

    def test_perf_coach_user_unset_no_query_param(self, monkeypatch, tmp_path):
        """PERF_COACH_USER unset → URL has no ``user`` query param."""
        monkeypatch.setenv("PERF_COACH_URL", "https://perf.example.com")
        monkeypatch.setenv("PERF_COACH_BEARER_TOKEN", "tok")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("PERF_COACH_USER", raising=False)

        with patch("urllib.request.urlopen") as mock_open_url:
            mock_open_url.return_value.__enter__ = lambda s: mock_open_url.return_value
            mock_open_url.return_value.__exit__ = MagicMock(return_value=False)
            mock_open_url.return_value.status = 200
            mock_open_url.return_value.read.return_value = b"{}"

            from services.hermes.discord import handle_rpe
            result = handle_rpe(
                user_id="u1",
                rpe=7,
                notes=None,
                date_str=None,
                today=datetime.date(2026, 7, 14),
            )

        assert result["success"] is True
        req = mock_open_url.call_args[0][0]
        assert "?user=" not in req.full_url
        assert req.full_url == "https://perf.example.com/feel-entry"

    def test_payload_contains_new_keys_and_excludes_old_keys(self, monkeypatch, tmp_path):
        """Payload has feel_date/rpe_1_to_10; must NOT have rpe/date/user_id keys."""
        monkeypatch.setenv("PERF_COACH_URL", "https://perf.example.com")
        monkeypatch.setenv("PERF_COACH_BEARER_TOKEN", "tok")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("PERF_COACH_USER", raising=False)

        with patch("urllib.request.urlopen") as mock_open_url:
            mock_open_url.return_value.__enter__ = lambda s: mock_open_url.return_value
            mock_open_url.return_value.__exit__ = MagicMock(return_value=False)
            mock_open_url.return_value.status = 200
            mock_open_url.return_value.read.return_value = b"{}"

            from services.hermes.discord import handle_rpe
            handle_rpe(
                user_id="u1",
                rpe=4,
                notes="ok",
                date_str="2026-07-01",
                today=datetime.date(2026, 7, 14),
            )

        req = mock_open_url.call_args[0][0]
        body = json.loads(req.data)
        assert body["feel_date"] == "2026-07-01"
        assert body["rpe_1_to_10"] == 4
        assert body["notes"] == "ok"
        assert set(body.keys()) == {"feel_date", "rpe_1_to_10", "notes"}
        assert "rpe" not in body
        assert "date" not in body
        assert "user_id" not in body


# ---------------------------------------------------------------------------
# Registration test (AC1)
# ---------------------------------------------------------------------------

class TestRpeRegistration:
    """Verify /rpe appears in the Discord command tree after _register_slash_commands."""

    def test_rpe_command_registered(self):
        """AC1 — /rpe is registered in _register_slash_commands."""
        if "discord" not in sys.modules:
            discord_mod = MagicMock()
            discord_mod.Intents.default.return_value = MagicMock()
            discord_mod.DMChannel = type("DMChannel", (), {})
            discord_mod.Thread = type("Thread", (), {})
            discord_mod.ForumChannel = type("ForumChannel", (), {})
            discord_mod.Interaction = object

            class _FakeGroup:
                def __init__(self, *, name, description, parent=None):
                    self.name = name
                    self.description = description
                    self.parent = parent
                    self._children: dict = {}
                    if parent is not None:
                        parent.add_command(self)

                def add_command(self, cmd):
                    self._children[cmd.name] = cmd

            class _FakeCommand:
                def __init__(self, *, name, description, callback, parent=None):
                    self.name = name
                    self.description = description
                    self.callback = callback
                    self.parent = parent

            discord_mod.app_commands = SimpleNamespace(
                describe=lambda **kwargs: (lambda fn: fn),
                choices=lambda **kwargs: (lambda fn: fn),
                autocomplete=lambda **kwargs: (lambda fn: fn),
                Choice=lambda **kwargs: SimpleNamespace(**kwargs),
                Group=_FakeGroup,
                Command=_FakeCommand,
            )

            ext_mod = MagicMock()
            commands_mod = MagicMock()
            commands_mod.Bot = MagicMock
            ext_mod.commands = commands_mod
            sys.modules["discord"] = discord_mod
            sys.modules.setdefault("discord.ext", ext_mod)
            sys.modules.setdefault("discord.ext.commands", commands_mod)

        # Import the registration function
        from services.hermes.discord import register_rpe_command

        class FakeTree:
            def __init__(self):
                self.commands: dict = {}

            def command(self, *, name, description):
                def decorator(fn):
                    self.commands[name] = fn
                    return fn
                return decorator

            def add_command(self, cmd):
                self.commands[cmd.name] = cmd

            def get_commands(self):
                return [SimpleNamespace(name=n) for n in self.commands]

        tree = FakeTree()
        register_rpe_command(tree)
        assert "rpe" in tree.commands, "/rpe not found in command tree after register_rpe_command()"

    def test_rpe_command_has_required_rpe_param(self):
        """AC2 — the registered command exposes rpe, notes, date parameters."""
        # The parameter contract is verified via the handler signature
        import inspect
        from services.hermes.discord import handle_rpe
        sig = inspect.signature(handle_rpe)
        params = set(sig.parameters)
        assert "rpe" in params
        assert "notes" in params
        assert "date_str" in params
