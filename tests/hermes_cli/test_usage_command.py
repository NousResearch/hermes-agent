from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import pytest

from agent.account_usage import (
    AccountUsageMetric,
    AccountUsageSnapshot,
    AccountUsageWindow,
)
from hermes_state import SessionDB
from hermes_cli import usage
from hermes_cli.auth import AuthError
from hermes_cli.nous_account import (
    NousPaidServiceAccessInfo,
    NousPortalAccountInfo,
    NousPortalSubscriptionInfo,
)


def test_default_json_report_does_not_select_a_persisted_session(monkeypatch, capsys):
    monkeypatch.setattr(
        usage,
        "SessionDB",
        lambda: pytest.fail("default usage must not open the session database"),
    )
    monkeypatch.setattr(usage, "_resolve_configured_provider", lambda: None)
    monkeypatch.setattr(
        usage, "_collect_nous_account", lambda warnings: usage._empty_nous_account()
    )

    with pytest.raises(SystemExit) as exc:
        usage.cmd_usage(
            argparse.Namespace(json=True, session=None, latest_session=False)
        )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert exc.value.code == 0
    assert captured.err == ""
    assert report["schema_version"] == 1
    assert report["session"]["status"] == "not_requested"
    assert report["session"]["id"] is None
    assert report["accounts"]["provider"]["status"] == "not_configured"
    assert report["warnings"] == []


def _run_json(args, capsys):
    with pytest.raises(SystemExit) as exc:
        usage.cmd_usage(args)
    captured = capsys.readouterr()
    return exc.value.code, json.loads(captured.out), captured.err


def test_session_selectors_are_explicit_and_latest_uses_user_facing_projection(
    tmp_path, monkeypatch, capsys
):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("visible-older", source="tui")
    db.append_message("visible-older", "user", "hello")
    db.create_session("visible-newest", source="cli")
    db.append_message("visible-newest", "user", "hello")
    db.create_session("cron-newest", source="cron")
    db.append_message("cron-newest", "user", "scheduled")
    db.create_session("shared-prefix-one", source="cli")
    db.create_session("shared-prefix-two", source="cli")

    monkeypatch.setattr(usage, "SessionDB", lambda: db)
    monkeypatch.setattr(db, "close", lambda: None)
    monkeypatch.setattr(
        usage,
        "_resolve_configured_provider",
        lambda: pytest.fail(
            "selected providerless sessions must not use runtime config"
        ),
    )
    monkeypatch.setattr(
        usage, "_collect_nous_account", lambda warnings: usage._empty_nous_account()
    )

    code, exact, err = _run_json(
        argparse.Namespace(json=True, session="visible-older", latest_session=False),
        capsys,
    )
    assert (code, err) == (0, "")
    assert exact["session"]["id"] == "visible-older"

    code, prefix, _ = _run_json(
        argparse.Namespace(json=True, session="visible-new", latest_session=False),
        capsys,
    )
    assert code == 0
    assert prefix["session"]["id"] == "visible-newest"

    code, latest, _ = _run_json(
        argparse.Namespace(json=True, session=None, latest_session=True), capsys
    )
    assert code == 0
    assert latest["session"]["id"] == "visible-newest"

    monkeypatch.setattr(
        usage,
        "_collect_provider_account",
        lambda provider: pytest.fail("invalid selectors must not start account calls"),
        raising=False,
    )
    code, missing, _ = _run_json(
        argparse.Namespace(json=True, session="missing", latest_session=False), capsys
    )
    assert code == 2
    assert missing["session"]["status"] == "not_found"
    assert missing["accounts"]["provider"]["status"] == "not_requested"
    assert missing["accounts"]["nous"]["status"] == "not_requested"

    code, empty, _ = _run_json(
        argparse.Namespace(json=True, session="", latest_session=False), capsys
    )
    assert code == 2
    assert empty["session"]["status"] == "not_found"

    code, ambiguous, _ = _run_json(
        argparse.Namespace(json=True, session="shared-prefix", latest_session=False),
        capsys,
    )
    assert code == 2
    assert ambiguous["session"]["status"] == "ambiguous"

    db.close()


def test_latest_session_reloads_the_projected_id_for_durable_counters(monkeypatch):
    class FakeDB:
        def list_sessions_rich(self, **kwargs):
            assert kwargs == {
                "exclude_sources": ["cron"],
                "limit": 1,
                "min_message_count": 1,
                "order_by_last_active": True,
                "compact_rows": True,
            }
            return [{"id": "compression-tip", "api_call_count": 1}]

        def get_session(self, session_id):
            assert session_id == "compression-tip"
            return {
                "id": session_id,
                "source": "cli",
                "model": "durable-model",
                "billing_provider": "anthropic",
                "message_count": 4,
                "api_call_count": 3,
                "input_tokens": 99,
                "output_tokens": 7,
                "reasoning_tokens": 2,
            }

        def get_latest_main_model_usage(self, session_id):
            return None

        def close(self):
            pass

    monkeypatch.setattr(usage, "SessionDB", FakeDB)

    session, code = usage._select_session(None, True)

    assert code == 0
    assert session["id"] == "compression-tip"
    assert session["api_calls"] == 3
    assert session["tokens"]["input"] == 99


def test_latest_compression_continuation_reports_tip_counters(
    tmp_path, monkeypatch, capsys
):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="root", source="cli", model="root-model")
    db.append_message("root", "user", "root")
    db.update_token_counts(
        "root",
        input_tokens=11,
        output_tokens=2,
        api_call_count=1,
        model="root-model",
        billing_provider="openrouter",
    )
    db.end_session("root", end_reason="compression")
    db.create_session(
        session_id="tip",
        source="cli",
        model="tip-model",
        parent_session_id="root",
    )
    db.append_message("tip", "user", "tip")
    db.update_token_counts(
        "tip",
        input_tokens=99,
        output_tokens=8,
        api_call_count=3,
        model="tip-model",
        billing_provider="anthropic",
    )
    monkeypatch.setattr(usage, "SessionDB", lambda: db)
    monkeypatch.setattr(db, "close", lambda: None)
    monkeypatch.setattr(
        usage,
        "_collect_provider_account",
        lambda provider, warnings: usage._empty_provider_account("ok", provider),
    )
    monkeypatch.setattr(
        usage,
        "_collect_nous_account",
        lambda warnings: usage._empty_nous_account(),
    )

    code, report, err = _run_json(
        argparse.Namespace(json=True, session=None, latest_session=True), capsys
    )

    assert (code, err) == (0, "")
    selected = report["session"]
    assert selected["id"] == "tip"
    assert selected["tokens"]["input"] == 99
    assert selected["api_calls"] == 3
    assert (selected["provider"], selected["model"]) == (
        "anthropic",
        "tip-model",
    )


def test_persisted_session_uses_durable_counters_and_latest_main_route(
    tmp_path, monkeypatch, capsys
):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(
        "persisted",
        source="discord",
        model="old-model",
    )
    db.update_token_counts(
        "persisted",
        input_tokens=10,
        output_tokens=4,
        reasoning_tokens=1,
        api_call_count=1,
        model="old-model",
        billing_provider="openrouter",
    )
    db.update_token_counts(
        "persisted",
        input_tokens=20,
        output_tokens=6,
        reasoning_tokens=2,
        api_call_count=1,
        model="new-model",
        billing_provider="anthropic",
    )

    monkeypatch.setattr(usage, "SessionDB", lambda: db)
    monkeypatch.setattr(db, "close", lambda: None)
    monkeypatch.setattr(usage, "_resolve_configured_provider", lambda: "openrouter")
    provider_calls = []
    monkeypatch.setattr(
        usage,
        "fetch_account_usage",
        lambda provider, report_failures=True: provider_calls.append(provider),
    )
    monkeypatch.setattr(
        usage, "_collect_nous_account", lambda warnings: usage._empty_nous_account()
    )

    code, report, err = _run_json(
        argparse.Namespace(json=True, session="persisted", latest_session=False),
        capsys,
    )

    session = report["session"]
    assert (code, err) == (0, "")
    assert session["provider"] == "anthropic"
    assert session["model"] == "new-model"
    assert session["api_calls"] == 2
    assert session["tokens"]["input"] == 30
    assert session["tokens"]["output"] == 10
    assert session["tokens"]["reasoning"] == 3
    assert session["tokens"]["prompt"] is None
    assert session["tokens"]["completion"] is None
    assert session["tokens"]["total"] is None
    assert session["duration_seconds"] is None
    assert session["context"]["status"] == "not_applicable"
    assert provider_calls == ["anthropic"]

    db.close()


def test_account_failure_is_partial_and_structured_provider_data_is_allowlisted(
    monkeypatch, capsys
):
    forbidden = [
        "secret-token",
        "user@example.com",
        "did:privy:private-user",
        "org-private",
        "client-private",
        "product-private",
        "pool-label-private",
        "https://private.example/api",
        "/private/repository/path",
        "private-branch",
        "private-session-title",
        "raw-provider-detail",
    ]
    sentinel = " ".join(forbidden)
    snapshot = AccountUsageSnapshot(
        provider="openrouter",
        source="credits_api",
        fetched_at=datetime(2026, 7, 22, 2, 29, 59, tzinfo=timezone.utc),
        plan="payg",
        windows=(
            AccountUsageWindow(
                id="api_key_quota",
                label="API key quota",
                used_percent=25.0,
                reset_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
            ),
        ),
        metrics=(
            AccountUsageMetric("credit_balance", 9.99, unit=sentinel),
            AccountUsageMetric("api_key_usage_total", float("nan")),
            AccountUsageMetric("raw-provider-detail", 123.0),
        ),
        details=(sentinel,),
    )
    monkeypatch.setattr(usage, "SessionDB", lambda: pytest.fail("no selector"))
    monkeypatch.setattr(usage, "_resolve_configured_provider", lambda: "openrouter")
    monkeypatch.setattr(
        usage,
        "fetch_account_usage",
        lambda provider, report_failures=True: snapshot,
    )
    monkeypatch.setattr(
        usage,
        "get_nous_portal_account_info",
        lambda force_fresh=True: (_ for _ in ()).throw(TimeoutError(sentinel)),
    )

    code, report, err = _run_json(
        argparse.Namespace(json=True, session=None, latest_session=False), capsys
    )

    assert (code, err) == (0, "")
    assert report["accounts"]["provider"] == {
        "status": "ok",
        "provider": "openrouter",
        "plan": "payg",
        "fetched_at": "2026-07-22T02:29:59Z",
        "windows": [
            {
                "id": "api_key_quota",
                "label": "API key quota",
                "used_percent": 25.0,
                "remaining_percent": 75.0,
                "resets_at": "2026-08-01T00:00:00Z",
            }
        ],
        "metrics": [{"name": "credit_balance", "value": 9.99, "unit": "usd"}],
    }
    assert report["accounts"]["nous"]["status"] == "unavailable"
    assert report["warnings"] == [
        {
            "code": "nous_timeout",
            "source": "nous",
            "message": "Nous Portal usage timed out.",
        }
    ]
    serialized = json.dumps(report, allow_nan=False)
    assert all(value not in serialized for value in forbidden)
    assert all(value not in err for value in forbidden)


def test_logged_out_nous_error_is_unavailable_with_a_sanitized_warning(monkeypatch):
    sentinel = "private/path user@example.com"
    monkeypatch.setattr(
        usage,
        "get_nous_portal_account_info",
        lambda force_fresh=True: NousPortalAccountInfo(
            logged_in=False,
            source="account_api",
            fresh=False,
            error=sentinel,
        ),
    )
    warnings = []

    result = usage._collect_nous_account(warnings)

    assert result["status"] == "unavailable"
    assert warnings == [
        {
            "code": "nous_unavailable",
            "source": "nous",
            "message": "Nous Portal usage is unavailable.",
        }
    ]
    assert sentinel not in json.dumps({"account": result, "warnings": warnings})


@pytest.mark.parametrize(
    ("paid", "has_subscription", "topup", "expected_access", "expected_status"),
    [
        (True, True, 5.0, "subscription_and_topup", "ok"),
        (True, True, 0.0, "subscription", "ok"),
        (True, False, 5.0, "topup_only", "ok"),
        (False, False, 0.0, "free", "ok"),
        (False, True, 0.0, "depleted", "ok"),
        (True, False, 0.0, "unknown", "partial"),
    ],
)
def test_nous_account_access_states(
    monkeypatch,
    paid,
    has_subscription,
    topup,
    expected_access,
    expected_status,
):
    subscription = (
        NousPortalSubscriptionInfo(
            plan="Pro",
            monthly_credits=100.0,
            credits_remaining=40.0,
            current_period_end="2026-08-01T02:00:00+02:00",
        )
        if has_subscription
        else None
    )
    access = NousPaidServiceAccessInfo(
        paid_access=paid,
        has_active_subscription=has_subscription,
        active_subscription_is_paid=has_subscription,
        subscription_credits_remaining=(40.0 if has_subscription else None),
        purchased_credits_remaining=topup,
        total_usable_credits=(40.0 if has_subscription else 0.0) + topup,
    )
    account = NousPortalAccountInfo(
        logged_in=True,
        source="account_api",
        fresh=True,
        subscription=subscription,
        paid_service_access=paid,
        paid_service_access_info=access,
    )
    monkeypatch.setattr(
        usage, "get_nous_portal_account_info", lambda force_fresh=True: account
    )

    warnings = []
    result = usage._collect_nous_account(warnings)

    assert warnings == []
    assert result["status"] == expected_status
    assert result["access"] == expected_access
    if has_subscription:
        assert result["subscription"] == {
            "remaining_usd": 40.0,
            "allowance_usd": 100.0,
            "used_percent": 60.0,
            "renews_at": "2026-08-01T00:00:00Z",
        }
    assert result["topup"]["remaining_usd"] == topup


def test_nous_account_helper_can_resolve_credential_pool_auth(monkeypatch):
    account = NousPortalAccountInfo(
        logged_in=True,
        source="account_api",
        fresh=True,
        paid_service_access=False,
        paid_service_access_info=NousPaidServiceAccessInfo(
            paid_access=False,
            total_usable_credits=0.0,
        ),
    )
    calls = []
    monkeypatch.setattr(
        usage,
        "get_nous_portal_account_info",
        lambda force_fresh=True: calls.append(force_fresh) or account,
    )

    result = usage._collect_nous_account([])

    assert calls == [True]
    assert result["status"] == "ok"
    assert result["access"] == "free"
    assert result["total_spendable_usd"] == 0.0


def test_nous_inference_key_without_portal_oauth_is_not_connected(monkeypatch):
    monkeypatch.setattr(
        usage,
        "get_nous_portal_account_info",
        lambda force_fresh=True: NousPortalAccountInfo(
            logged_in=False,
            source="inference_key",
            fresh=False,
            error="portal_oauth_missing",
        ),
    )
    warnings = []

    result = usage._collect_nous_account(warnings)

    assert result["status"] == "not_connected"
    assert warnings == []


@pytest.mark.parametrize(
    ("has_subscription", "expected_access"),
    [(False, "free"), (True, "depleted")],
)
def test_known_free_or_depleted_nous_balance_is_zero(
    monkeypatch, has_subscription, expected_access
):
    monkeypatch.setattr(
        usage,
        "get_nous_portal_account_info",
        lambda force_fresh=True: NousPortalAccountInfo(
            logged_in=True,
            source="account_api",
            fresh=True,
            paid_service_access=False,
            paid_service_access_info=NousPaidServiceAccessInfo(
                paid_access=False,
                has_active_subscription=has_subscription,
            ),
        ),
    )

    result = usage._collect_nous_account([])

    assert result["status"] == "ok"
    assert result["access"] == expected_access
    assert result["total_spendable_usd"] == 0.0


def test_usage_parser_registers_mutually_exclusive_session_selectors():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    usage_parser = usage.build_usage_parser(subparsers)

    args = parser.parse_args(["usage", "--json", "--session", "abc"])
    assert args.command == "usage"
    assert args.json is True
    assert args.session == "abc"
    assert args.latest_session is False
    assert args.func is usage.cmd_usage
    help_text = " ".join(usage_parser.format_help().split())
    assert "does not select a persisted session" in help_text
    assert "live network account checks" in help_text
    assert "schema-versioned JSON" in help_text

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["usage", "--session", "abc", "--latest-session"])
    assert exc.value.code == 2


def test_json_mode_suppresses_dependency_stdout(monkeypatch, capsys):
    snapshot = AccountUsageSnapshot(
        provider="openrouter",
        source="credits_api",
        fetched_at=datetime.now(timezone.utc),
    )
    monkeypatch.setattr(usage, "_resolve_configured_provider", lambda: "openrouter")
    monkeypatch.setattr(
        usage,
        "fetch_account_usage",
        lambda provider, report_failures=True: (
            print("credential refresh"),
            print("credential warning", file=sys.stderr),
            snapshot,
        )[-1],
    )
    monkeypatch.setattr(
        usage,
        "_collect_nous_account",
        lambda warnings: usage._empty_nous_account(),
    )

    code, report, err = _run_json(
        argparse.Namespace(json=True, session=None, latest_session=False), capsys
    )

    assert (code, err) == (0, "")
    assert report["schema_version"] == 1
    assert report["accounts"]["provider"]["status"] == "ok"


def test_local_session_failure_is_sanitized_and_operational(monkeypatch, capsys):
    sentinel = "private/path user@example.com"
    monkeypatch.setattr(
        usage, "SessionDB", lambda: (_ for _ in ()).throw(RuntimeError(sentinel))
    )
    monkeypatch.setattr(usage, "_resolve_configured_provider", lambda: None)
    monkeypatch.setattr(
        usage,
        "_collect_nous_account",
        lambda warnings: usage._empty_nous_account(),
    )

    code, report, err = _run_json(
        argparse.Namespace(json=True, session="abc", latest_session=False), capsys
    )

    assert (code, err) == (1, "")
    assert report["session"]["status"] == "unavailable"
    assert report["warnings"] == [
        {
            "code": "session_unavailable",
            "source": "session",
            "message": "Session usage is unavailable.",
        }
    ]
    assert sentinel not in json.dumps(report)


def test_unexpected_handler_failure_still_emits_one_sanitized_json_envelope(
    monkeypatch, capsys
):
    sentinel = "private/path user@example.com secret-token"
    monkeypatch.setattr(
        usage,
        "collect_usage",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError(sentinel)),
    )

    code, report, err = _run_json(
        argparse.Namespace(json=True, session=None, latest_session=False), capsys
    )

    assert (code, err) == (1, "")
    assert report["schema_version"] == 1
    assert report["session"]["status"] == "unavailable"
    assert report["accounts"]["provider"]["status"] == "unavailable"
    assert report["accounts"]["nous"]["status"] == "unavailable"
    assert report["warnings"] == [
        {
            "code": "usage_unavailable",
            "source": "usage",
            "message": "Usage reporting is unavailable.",
        }
    ]
    assert sentinel not in json.dumps(report)


def test_runtime_provider_resolution_failure_is_a_sanitized_local_failure(
    monkeypatch, capsys
):
    sentinel = "private-config-path user@example.com"
    monkeypatch.setattr(
        usage,
        "_resolve_configured_provider",
        lambda: (_ for _ in ()).throw(RuntimeError(sentinel)),
    )

    code, report, err = _run_json(
        argparse.Namespace(json=True, session=None, latest_session=False), capsys
    )

    assert (code, err) == (1, "")
    assert report["session"]["status"] == "unavailable"
    assert report["warnings"][0]["code"] == "usage_unavailable"
    assert sentinel not in json.dumps(report)


def test_human_report_has_stable_sections_and_actionable_empty_hints(
    monkeypatch, capsys
):
    monkeypatch.setattr(usage, "_resolve_configured_provider", lambda: None)
    monkeypatch.setattr(
        usage,
        "_collect_nous_account",
        lambda warnings: usage._empty_nous_account(),
    )

    with pytest.raises(SystemExit) as exc:
        usage.cmd_usage(
            argparse.Namespace(json=False, session=None, latest_session=False)
        )

    output = capsys.readouterr().out
    assert exc.value.code == 0
    assert "Session usage" in output
    assert "Provider account" in output
    assert "Nous Portal" in output
    assert "--session ID" in output
    assert "hermes login" in output


def test_human_local_failure_writes_only_a_concise_sanitized_error(monkeypatch, capsys):
    sentinel = "private/path user@example.com secret-token"
    monkeypatch.setattr(
        usage,
        "collect_usage",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError(sentinel)),
    )

    with pytest.raises(SystemExit) as exc:
        usage.cmd_usage(
            argparse.Namespace(json=False, session=None, latest_session=False)
        )

    captured = capsys.readouterr()
    assert exc.value.code == 1
    assert captured.err == "Usage reporting encountered a local error.\n"
    assert sentinel not in captured.out + captured.err


def test_runtime_provider_auto_resolves_when_config_has_no_explicit_provider(
    monkeypatch,
):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"model": {}})
    calls = []
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_provider",
        lambda requested: calls.append(requested) or "openai-codex",
    )

    assert usage._resolve_configured_provider() == "openai-codex"
    assert calls == [None]


def test_runtime_provider_no_configuration_is_not_an_operational_failure(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"model": {}})
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_provider",
        lambda requested: (_ for _ in ()).throw(
            AuthError("none configured", code="no_provider_configured")
        ),
    )

    assert usage._resolve_configured_provider() is None


def test_blank_slate_real_command_returns_not_configured_json(tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    env = os.environ.copy()
    env["HERMES_HOME"] = str(hermes_home)

    completed = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "usage", "--json"],
        text=True,
        capture_output=True,
        env=env,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert completed.stdout.count("\n") == 1
    report = json.loads(completed.stdout)
    assert report["session"]["status"] == "not_requested"
    assert report["accounts"]["provider"]["status"] == "not_configured"
