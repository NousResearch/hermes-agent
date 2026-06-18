import json

import pytest


def _valid_artifact(**overrides):
    artifact = {
        "preflight_ok": True,
        "chrome_access_ok": True,
        "x_tab_seen": True,
        "items_scanned": 0,
        "new_items": 0,
        "failure_reason": None,
    }
    artifact.update(overrides)
    return artifact


def test_chrome_x_preflight_matches_private_logged_in_monitor():
    from cron.scheduler import _requires_chrome_x_preflight

    job = {
        "name": "Daily Chrome-only X private tasks window",
        "prompt": "Use existing logged-in Chrome/X tabs only for Twitter bookmarks.",
    }

    assert _requires_chrome_x_preflight(job) is True


def test_chrome_x_preflight_does_not_match_public_x_jobs():
    from cron.scheduler import _requires_chrome_x_preflight

    job = {
        "name": "Public X search monitor",
        "prompt": "Search X via the x_search API; no Chrome session involved.",
    }

    assert _requires_chrome_x_preflight(job) is False


def test_chrome_x_preflight_contract_includes_artifact_path(tmp_path):
    from cron.scheduler import _append_chrome_x_preflight_contract

    artifact_path = tmp_path / "cron_abc.preflight.json"
    prompt = _append_chrome_x_preflight_contract("Do the job", artifact_path)

    assert str(artifact_path) in prompt
    assert "preflight_ok" in prompt
    assert "new_items" in prompt
    assert "You may return exactly `[SILENT]` only after" in prompt


def test_read_chrome_x_preflight_validates_schema(tmp_path):
    from cron.scheduler import _read_chrome_x_preflight_artifact

    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(_valid_artifact(items_scanned=3)), encoding="utf-8")

    artifact, error = _read_chrome_x_preflight_artifact(path)

    assert error is None
    assert artifact is not None
    assert artifact["items_scanned"] == 3


def test_read_chrome_x_preflight_rejects_missing_keys(tmp_path):
    from cron.scheduler import _read_chrome_x_preflight_artifact

    path = tmp_path / "preflight.json"
    path.write_text(json.dumps({"preflight_ok": True}), encoding="utf-8")

    artifact, error = _read_chrome_x_preflight_artifact(path)

    assert artifact is None
    assert error is not None
    assert "missing key" in error


def test_silent_gate_allows_verified_no_new_items():
    from cron.scheduler import _enforce_chrome_x_silent_gate

    final = _enforce_chrome_x_silent_gate(
        final_response="[SILENT]",
        artifact=_valid_artifact(items_scanned=5, new_items=0),
        artifact_error=None,
    )

    assert final == "[SILENT]"


@pytest.mark.parametrize(
    "artifact,error,expected",
    [
        (None, "missing artifact", "missing artifact"),
        (_valid_artifact(preflight_ok=False, failure_reason="AppleScript denied"), None, "failed preflight"),
        (_valid_artifact(chrome_access_ok=False), None, "without verified Chrome access"),
        (_valid_artifact(x_tab_seen=False), None, "without verifying an existing X/Twitter tab"),
        (_valid_artifact(new_items=2), None, "new_items > 0"),
    ],
)
def test_silent_gate_rejects_unverified_silence(artifact, error, expected):
    from cron.scheduler import _enforce_chrome_x_silent_gate

    with pytest.raises(RuntimeError, match=expected):
        _enforce_chrome_x_silent_gate(
            final_response="[SILENT]",
            artifact=artifact,
            artifact_error=error,
        )


def test_silent_gate_allows_non_silent_failed_preflight_report():
    from cron.scheduler import _enforce_chrome_x_silent_gate

    final = _enforce_chrome_x_silent_gate(
        final_response="Blocked: Chrome did not expose an existing X tab.",
        artifact=_valid_artifact(
            preflight_ok=False,
            chrome_access_ok=False,
            x_tab_seen=False,
            failure_reason="Chrome unavailable",
        ),
        artifact_error=None,
    )

    assert final.startswith("Blocked:")


def test_finalize_chrome_x_preflight_allows_novelty_silence_after_verified_scan(tmp_path):
    from cron.scheduler import SILENT_MARKER, _finalize_chrome_x_preflight_response

    artifact_path = tmp_path / "preflight.json"
    artifact_path.write_text(
        json.dumps(_valid_artifact(items_scanned=4, new_items=0)),
        encoding="utf-8",
    )
    ledger_path = tmp_path / "ledger.json"
    job = {"id": "chrome-x-job", "novelty_ledger": True}
    response = "Already-reported X item: https://x.com/example/status/1"

    assert _finalize_chrome_x_preflight_response(
        job,
        response,
        artifact_path,
        ledger_path=ledger_path,
    ) == response
    assert _finalize_chrome_x_preflight_response(
        job,
        response,
        artifact_path,
        ledger_path=ledger_path,
    ) == SILENT_MARKER


def test_finalize_chrome_x_preflight_rejects_novelty_silence_after_failed_preflight(tmp_path):
    from cron.scheduler import _finalize_chrome_x_preflight_response

    artifact_path = tmp_path / "preflight.json"
    artifact_path.write_text(
        json.dumps(
            _valid_artifact(
                preflight_ok=False,
                chrome_access_ok=False,
                x_tab_seen=False,
                failure_reason="AppleScript denied",
            )
        ),
        encoding="utf-8",
    )
    ledger_path = tmp_path / "ledger.json"
    job = {"id": "chrome-x-job", "novelty_ledger": True}
    response = "Blocked: AppleScript denied for https://x.com/example/status/1"

    assert _finalize_chrome_x_preflight_response(
        job,
        response,
        artifact_path,
        ledger_path=ledger_path,
    ) == response
    with pytest.raises(RuntimeError, match="failed preflight"):
        _finalize_chrome_x_preflight_response(
            job,
            response,
            artifact_path,
            ledger_path=ledger_path,
        )
