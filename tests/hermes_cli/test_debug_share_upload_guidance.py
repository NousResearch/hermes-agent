"""Public upload failure guidance for the support workflow in #151408."""

from types import SimpleNamespace

import pytest


def test_public_upload_failure_suggests_private_support_without_uploading(
    tmp_path, monkeypatch, capsys
):
    from hermes_cli import debug
    from hermes_cli import diagnostics_upload

    home = tmp_path / "hermes-home"
    logs = home / "logs"
    logs.mkdir(parents=True)
    (logs / "agent.log").write_text("INFO: support fixture\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    uploads = []

    def fail_public_upload(content, expiry_days=1):
        uploads.append(content)
        raise RuntimeError("public paste services unavailable")

    def unexpected_private_upload(*args, **kwargs):
        pytest.fail("failure guidance must not automatically upload privately")

    monkeypatch.setattr(debug, "upload_to_pastebin", fail_public_upload)
    monkeypatch.setattr(diagnostics_upload, "share_to_nous", unexpected_private_upload)
    args = SimpleNamespace(
        lines=10, expire=1, local=False, nous=False, no_redact=False, yes=True
    )
    with pytest.raises(SystemExit) as exc:
        debug.run_debug_share(args)

    assert exc.value.code == 1
    assert len(uploads) == 1
    assert "support fixture" in uploads[0]
    captured = capsys.readouterr()
    assert "public paste services unavailable" in captured.err
    assert "hermes debug share --nous" in captured.out
    assert "private" in captured.out.lower()
    assert "Nous support" in captured.out
    assert captured.out.rindex("--nous") < captured.out.rindex("--local")
    assert "without uploading" in captured.out


@pytest.mark.parametrize(
    "response",
    [
        {"viewUrl": "https://example.invalid/view/fixture", "expiresAt": "2026-11-24T00:00:00Z"},
        {"id": "fixture"},
    ],
)
def test_private_upload_receipt_names_all_allowed_viewers(
    tmp_path, monkeypatch, capsys, response
):
    """The receipt must describe moderator access, not just the consent notice."""
    import gzip
    import json

    from hermes_cli import debug
    from hermes_cli import diagnostics_upload

    home = tmp_path / "hermes-home"
    logs = home / "logs"
    logs.mkdir(parents=True)
    (logs / "agent.log").write_text("INFO: private support fixture\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    uploads = []

    def upload_private(payload):
        uploads.append(json.loads(gzip.decompress(payload)))
        return response

    monkeypatch.setattr(diagnostics_upload, "share_to_nous", upload_private)
    args = SimpleNamespace(
        lines=10, expire=1, local=False, nous=True, no_redact=False, yes=True
    )
    debug.run_debug_share(args)

    assert len(uploads) == 1
    assert "private support fixture" in str(uploads[0])
    output = capsys.readouterr().out
    receipt = output.split("Debug bundle uploaded to Nous (private):", 1)[1]
    assert "Nous staff and allowlisted Discord mods" in receipt
    assert "Google login" in receipt
    assert "only Nous staff" not in receipt
    assert "14" in receipt
    if "viewUrl" in response:
        assert response["viewUrl"] in receipt
        assert response["expiresAt"] in receipt
    else:
        assert "upload id: fixture" in receipt


def test_debug_usage_names_private_bundle_viewers(capsys):
    """The CLI option summary must not narrow private-bundle access to staff."""
    from hermes_cli import debug

    debug.run_debug(SimpleNamespace(debug_command=None))

    output = capsys.readouterr().out
    assert "Nous staff and allowlisted Discord mods" in output
    assert "staff-only" not in output
