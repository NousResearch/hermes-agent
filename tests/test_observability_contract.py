"""Static guards for the observer/observability contract.

The verified improvement plan says observability should come before larger
runtime refactors, but Hermes already has a narrow observer hook contract. These
checks keep that baseline documented and wired without adding speculative core
hooks.
"""

from __future__ import annotations

import pathlib

from hermes_cli.middleware import OBSERVER_SCHEMA_VERSION, observer_payload

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_observer_schema_version_is_stable_and_documented():
    assert OBSERVER_SCHEMA_VERSION == "hermes.observer.v1"
    docs = (REPO_ROOT / "docs" / "observability" / "README.md").read_text(encoding="utf-8")
    assert f'telemetry_schema_version = "{OBSERVER_SCHEMA_VERSION}"' in docs


def test_observer_payload_preserves_existing_schema_version():
    payload = observer_payload(
        telemetry_schema_version="custom.consumer.v1",
        session_id="s1",
    )
    assert payload["telemetry_schema_version"] == "custom.consumer.v1"
    assert payload["session_id"] == "s1"


def test_observer_payload_adds_default_schema_version():
    payload = observer_payload(session_id="s1", turn_id="t1")
    assert payload["telemetry_schema_version"] == OBSERVER_SCHEMA_VERSION
    assert payload["session_id"] == "s1"
    assert payload["turn_id"] == "t1"


def test_observability_docs_define_fail_open_read_only_contract():
    docs = (REPO_ROOT / "docs" / "observability" / "README.md").read_text(encoding="utf-8")
    required_phrases = [
        "read-only telemetry contract",
        "do not replace Hermes' planner",
        "Hook callbacks are fail-open",
        "Plugins should accept",
        "**kwargs",
    ]
    for phrase in required_phrases:
        assert phrase in docs, f"observability docs lost contract phrase: {phrase}"
