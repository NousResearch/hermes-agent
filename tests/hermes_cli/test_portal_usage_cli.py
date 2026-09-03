"""Tests for the stable, sanitized ``hermes portal usage --json`` contract."""

from __future__ import annotations

import json
from argparse import Namespace

import agent.billing_usage as billing_usage
import hermes_cli.portal_cli as portal_cli


def test_usage_payload_is_versioned_sanitized_and_preserves_known_denominators():
    model = billing_usage.UsageModel(
        available=True,
        status="healthy",
        plan_name="Plus",
        renews_at="2026-08-31T00:00:00Z",
        subscription_remaining_usd=14.0,
        topup_remaining_usd=12.0,
        total_spendable_usd=26.0,
        plan_bar=billing_usage.UsageBar(kind="plan", remaining_usd=14.0, total_usd=20.0, spent_usd=6.0),
        topup_bar=billing_usage.UsageBar(kind="topup", remaining_usd=12.0, total_usd=12.0),
    )

    payload = portal_cli.build_usage_payload(model)

    assert payload == {
        "schema_version": 1,
        "available": True,
        "status": "healthy",
        "plan": "Plus",
        "renews_at": "2026-08-31T00:00:00Z",
        "subscription": {"remaining_usd": 14.0, "monthly_allowance_usd": 20.0, "used_percent": 30},
        "top_up": {"remaining_usd": 12.0},
        "total_usable_usd": 26.0,
    }
    serialized = json.dumps(payload, sort_keys=True)
    assert "token" not in serialized.lower()
    assert "email" not in serialized.lower()
    assert "organisation" not in serialized.lower()


def test_usage_payload_omits_percentage_when_the_portal_did_not_supply_a_denominator():
    payload = portal_cli.build_usage_payload(
        billing_usage.UsageModel(
            available=True,
            status="healthy",
            topup_remaining_usd=4.0,
            total_spendable_usd=4.0,
            topup_bar=billing_usage.UsageBar(kind="topup", remaining_usd=4.0, total_usd=4.0),
        )
    )

    assert payload["subscription"] is None
    assert payload["top_up"] == {"remaining_usd": 4.0}
    assert payload["total_usable_usd"] == 4.0


def test_portal_usage_json_prints_machine_readable_payload(monkeypatch, capsys):
    model = billing_usage.UsageModel(available=True, status="low", plan_name="Plus", total_spendable_usd=3.0)
    monkeypatch.setattr(portal_cli, "build_usage_model", lambda timeout: model)

    assert portal_cli._cmd_usage(Namespace(json=True)) == 0

    assert json.loads(capsys.readouterr().out) == {
        "schema_version": 1,
        "available": True,
        "status": "low",
        "plan": "Plus",
        "renews_at": None,
        "subscription": None,
        "top_up": None,
        "total_usable_usd": 3.0,
    }


def test_portal_usage_json_is_unavailable_when_not_logged_in(monkeypatch, capsys):
    monkeypatch.setattr(portal_cli, "build_usage_model", lambda timeout: billing_usage.UsageModel(available=False))

    assert portal_cli._cmd_usage(Namespace(json=True)) == 1

    assert json.loads(capsys.readouterr().out) == {
        "schema_version": 1,
        "available": False,
        "status": "unavailable",
        "plan": None,
        "renews_at": None,
        "subscription": None,
        "top_up": None,
        "total_usable_usd": None,
    }


def test_portal_parser_registers_usage_json_flag():
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    portal_cli.add_parser(subparsers)

    args = parser.parse_args(["portal", "usage", "--json"])

    assert args.portal_command == "usage"
    assert args.json is True
