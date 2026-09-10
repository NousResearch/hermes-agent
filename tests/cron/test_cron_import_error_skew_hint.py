"""Import-class cron failures name gateway code skew when it exists (#95294).

Field incident: an interrupted `hermes update` (pull done, restart never ran)
left the gateway on stale code for two days; every agent cron job failed with
`ImportError: cannot import name 'user_originated_turn_view'` and the operator
had no way to know the fix was one `hermes gateway restart`. The failure
summarizer runs inside the gateway process, which knows its own boot
fingerprint — when boot SHA != disk HEAD, the delivered error must say so and
name the command.
"""

import cron.scheduler as scheduler
from cron.scheduler import _summarize_cron_failure_for_delivery

IMPORT_ERROR = (
    "ImportError: cannot import name 'user_originated_turn_view' "
    "from 'agent.context_compressor'"
)


def test_import_error_with_skew_names_shas_and_the_restart_command(monkeypatch):
    monkeypatch.setattr(
        scheduler,
        "_detect_gateway_code_skew",
        lambda: ("7e67f64fce", "ec5e369fe6"),
    )
    job = {"name": "morning-brief", "id": "aaa111"}
    msg = _summarize_cron_failure_for_delivery(job, IMPORT_ERROR)
    # The raw error text (with the failing symbol) must survive — the hint
    # is APPENDED, never a replacement.
    assert "cannot import name 'user_originated_turn_view'" in msg
    assert "stale code" in msg
    assert "booted on 7e67f64fce" in msg
    assert "disk is at ec5e369fe6" in msg
    assert "hermes gateway restart" in msg


def test_import_error_without_skew_stays_a_plain_import_message(monkeypatch):
    """No skew (or non-git install): message is byte-identical to today's."""
    monkeypatch.setattr(scheduler, "_detect_gateway_code_skew", lambda: None)
    job = {"name": "morning-brief", "id": "aaa111"}
    msg = _summarize_cron_failure_for_delivery(job, IMPORT_ERROR)
    assert "cannot import name 'user_originated_turn_view'" in msg
    assert "stale code" not in msg
    assert "hermes gateway restart" not in msg


def test_modulenotfound_matches_the_import_class(monkeypatch):
    monkeypatch.setattr(
        scheduler,
        "_detect_gateway_code_skew",
        lambda: ("aaaa111111", "bbbb222222"),
    )
    job = {"name": "nightly-digest", "id": "ccc333"}
    msg = _summarize_cron_failure_for_delivery(
        job, "ModuleNotFoundError: No module named 'agent.turn_context'"
    )
    assert "hermes gateway restart" in msg


def test_no_agent_script_import_error_never_blames_gateway_skew(monkeypatch):
    """A no_agent script runs in a fresh subprocess — its ImportError is the
    script's own problem and must reach the generic cleaner untouched."""
    monkeypatch.setattr(
        scheduler,
        "_detect_gateway_code_skew",
        lambda: ("aaaa111111", "bbbb222222"),
    )
    job = {"name": "disk-watchdog", "id": "ddd444", "no_agent": True}
    msg = _summarize_cron_failure_for_delivery(
        job, "ImportError: cannot import name 'requests'"
    )
    assert "stale code" not in msg
    assert "hermes gateway restart" not in msg


def test_skew_probe_failure_degrades_to_the_plain_message(monkeypatch):
    """The seam swallowing an exception must behave exactly like no-skew."""

    def boom():
        raise RuntimeError("git exploded")

    monkeypatch.setattr(scheduler, "_detect_gateway_code_skew", boom)
    job = {"name": "morning-brief", "id": "aaa111"}
    try:
        msg = _summarize_cron_failure_for_delivery(job, IMPORT_ERROR)
    except RuntimeError:
        raise AssertionError(
            "summarizer must not propagate a skew-probe failure"
        ) from None
    assert "cannot import name" in msg


def test_wrapper_seam_swallows_detector_import_failure(monkeypatch):
    """_detect_gateway_code_skew itself never raises when the gateway module
    is unimportable (e.g. stripped install)."""
    import builtins

    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name.startswith("gateway"):
            raise ImportError("gateway package missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    assert scheduler._detect_gateway_code_skew() is None
