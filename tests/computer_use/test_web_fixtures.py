"""Web fixture pack for browser-shaped decide-loop benchmarks."""

from __future__ import annotations

from tools.computer_use.decide_loop import run_decide_loop
from tools.computer_use.web_fixtures import (
    all_web_fixtures,
    flights_search_fixture,
    parse_html_fixture,
    two_step_wizard_fixture,
    write_default_html_fixtures,
)


def test_registry_has_two_fixtures():
    fixtures = all_web_fixtures()
    assert len(fixtures) == 2
    assert {f.name for f in fixtures} == {"two_step_wizard", "flights_search"}


def test_two_step_wizard_completes_via_loop():
    fx = two_step_wizard_fixture()
    script = [
        {"ok": True, "fail_open": False, "decision": {"action": "click", "target_element": 1, "backend": "rules"}},
        {"ok": True, "fail_open": False, "decision": {"action": "click", "target_element": 1, "backend": "rules"}},
        {"ok": True, "fail_open": False, "decision": {"action": "done", "done": True, "backend": "rules"}},
    ]
    handle = fx.make_handle(script)
    result = run_decide_loop(
        "Complete the two-step wizard",
        handle,
        max_steps=6,
        step_pause_s=0,
        external_done=fx.is_done,
    )
    assert result.ok is True
    assert result.status == "completed"
    assert fx.pages[fx._page_idx].name == "complete"


def test_flights_search_reaches_booked():
    fx = flights_search_fixture()
    script = [
        {"ok": True, "fail_open": False, "decision": {"action": "click", "target_element": 3, "backend": "jev"}},
        {"ok": True, "fail_open": False, "decision": {"action": "click", "target_element": 1, "backend": "jev"}},
        {"ok": True, "fail_open": False, "decision": {"action": "done", "done": True, "backend": "jev"}},
    ]
    handle = fx.make_handle(script)
    result = run_decide_loop("Book the cheapest flight", handle, max_steps=6, step_pause_s=0, external_done=fx.is_done)
    assert result.ok is True
    assert fx.is_done()


def test_html_fixture_files_parse_buttons(tmp_path):
    paths = write_default_html_fixtures(tmp_path)
    assert len(paths) == 2
    buttons = parse_html_fixture(paths[0])
    labels = [b[1] for b in buttons]
    assert "Continue" in labels
    assert "Submit" in labels
