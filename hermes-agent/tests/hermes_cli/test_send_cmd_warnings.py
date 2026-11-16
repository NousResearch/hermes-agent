"""``hermes send`` without ``--json`` prints the result's warnings, so a failed media upload shows
its cause ("Timed out ...") instead of only the generic "No deliverable" error (#133164)."""

import json

from hermes_cli.send_cmd import _emit_result


def test_plain_output_prints_warnings_before_the_error(capsys):
    payload = {"error": "No deliverable text or media remained after processing MEDIA tags",
               "warnings": ["Failed to send media /x.pdf: Timed out (Telegram may still deliver it; do not resend)"]}
    assert _emit_result(json.dumps(payload), json_mode=False, quiet=False) != 0
    err = capsys.readouterr().err
    assert "warning: Failed to send media /x.pdf: Timed out" in err
    assert "No deliverable" in err
