"""Lease readiness control closes every response without starting backends."""

import io
from unittest.mock import Mock

import pytest


class ProbeComplete(BaseException):
    """Stop after both readiness probes, before the unrelated session battery."""


@pytest.mark.parametrize("retry", [False, True])
def test_control_response_closed(tmp_path, monkeypatch, retry):
    from evals.desktop_bug_campaign import leases_live as probe

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(probe.tempfile, "mkdtemp", lambda **kwargs: str(home))
    monkeypatch.setattr(probe, "fixture", lambda port: Mock())
    monkeypatch.setattr(probe.subprocess, "check_output", lambda *args, **kwargs: "test-head")
    process = Mock()
    process.poll.return_value = None
    monkeypatch.setattr(probe.subprocess, "Popen", Mock(return_value=process))
    monkeypatch.setattr(probe, "Client", Mock(side_effect=ProbeComplete))
    responses = ([io.BytesIO(b"bad json")] if retry else []) + [io.BytesIO(b"{}"), io.BytesIO(b"{}")]
    urlopen = Mock(side_effect=responses)
    monkeypatch.setattr(probe.urllib.request, "urlopen", urlopen)

    with pytest.raises(ProbeComplete):
        probe.main(tmp_path / "receipt")

    assert urlopen.call_count == len(responses)
    assert all(response.closed for response in responses)
    for call in urlopen.call_args_list:
        assert call.args[0].get_method() == "POST"
        assert call.kwargs["timeout"] == 30
