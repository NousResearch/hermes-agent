"""A Feishu upload/send failure must carry the platform's own ``code``/``msg``.

Two defects shared one symptom — the operator saw only ``Feishu file upload missing file_key``
(or a bare override label) while the wire error held the actual diagnosis:

1. ``_response_error_result`` returned early on ``override_error``, dropping ``code``/``msg``
   from both the log and ``SendResult.error``. In the reported case the tenant was missing the
   ``im:resource`` scope, which the adapter never surfaced.
2. ``_build_file_upload_body`` passed a bare file object. The SDK's ``Files.parse_form_data``
   keeps it as-is and ``MultipartEncoder`` only emits ``filename=`` for a ``(name, file)`` tuple,
   so a body over roughly 26MiB produced a nameless part and ``code=9499 Bad Request``.

Both are behavior contracts: the error text must retain the platform diagnosis, and the upload
body must carry a named file part — with and without the SDK bound.
"""

from types import SimpleNamespace

import pytest

import plugins.platforms.feishu.adapter as adapter_mod
from plugins.platforms.feishu.adapter import FeishuAdapter


@pytest.fixture
def captured_logs(monkeypatch):
    """Collect ``logger.error`` calls made by the adapter module."""
    messages: list[str] = []

    def _record(fmt, *args, **kwargs):
        messages.append(fmt % args if args else fmt)

    monkeypatch.setattr(adapter_mod.logger, "error", _record)
    return messages


@pytest.fixture
def adapter():
    # No __init__: both helpers under test are pure formatting/builder functions.
    return FeishuAdapter.__new__(FeishuAdapter)


def test_override_error_keeps_platform_code_and_message(captured_logs, adapter):
    """``override_error`` is a label; the Feishu code/msg must still reach log and error text."""
    response = SimpleNamespace(code=9499, msg="Bad Request")

    result = adapter._response_error_result(
        response, default_message="upload failed", override_error="missing file_key",
    )

    assert result.success is False
    assert result.error == "missing file_key (code=9499 msg=Bad Request)"
    assert any("code=9499" in m and "missing file_key" in m for m in captured_logs), captured_logs


def test_override_error_without_code_still_reports_unknown(captured_logs, adapter):
    """A response carrying no ``code`` must not lose the override label."""
    result = adapter._response_error_result(
        SimpleNamespace(), default_message="upload failed", override_error="missing file_key",
    )

    assert result.error == "missing file_key (code=unknown msg=upload failed)"


def test_plain_error_keeps_bracketed_code(captured_logs, adapter):
    """No override: the pre-existing ``[code] msg`` shape is unchanged."""
    result = adapter._response_error_result(
        SimpleNamespace(code=230001, msg="invalid param"), default_message="send failed",
    )

    assert result.error == "[230001] invalid param"


def test_file_upload_body_without_sdk_names_the_file_part(monkeypatch, adapter):
    """SDK unbound: the fallback request object must still carry a ``(filename, file)`` tuple."""
    monkeypatch.setattr(adapter_mod, "CreateFileRequestBody", None)
    payload = b"x" * 16

    body = adapter._build_file_upload_body(
        file_type="stream", file_name="clip.mp4", file=payload, duration=12,
    )

    assert isinstance(body.file, tuple), f"file part must be (name, file), got {type(body.file)}"
    assert body.file[0] == "clip.mp4"
    assert body.file[1] is payload
    assert body.duration == 12


def test_file_upload_body_with_sdk_names_the_file_part(monkeypatch, adapter):
    """SDK bound: the fields handed to the request builder must carry the same tuple."""
    captured: dict = {}

    def _capture(request_cls, **fields):
        captured.update(fields)
        return SimpleNamespace(**fields)

    monkeypatch.setattr(adapter_mod, "_sdk_build", _capture)
    monkeypatch.setattr(adapter_mod, "CreateFileRequestBody", object())  # non-None: take the SDK path
    payload = b"x" * 16

    adapter._build_file_upload_body(
        file_type="stream", file_name="clip.mp4", file=payload, duration=12,
    )

    file_part = captured.get("file")
    assert isinstance(file_part, tuple), f"file part must be (name, file), got {type(file_part)}"
    assert file_part[0] == "clip.mp4"
    assert file_part[1] is payload
    assert captured.get("duration") == 12
