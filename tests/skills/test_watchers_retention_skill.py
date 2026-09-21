"""Watchers expose their existing retention setting independently of output (#76542)."""
import importlib.util
import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "optional-skills/devops/watchers/scripts"


@pytest.fixture(params=["watch_rss", "watch_github", "watch_http_json"])
def watcher(request, monkeypatch, tmp_path):
    monkeypatch.setenv("WATCHER_STATE_DIR", str(tmp_path))
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(request.param, SCRIPTS / f"{request.param}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = [request.param, "--name", "feed"]
    args += ["--repo", "owner/repo"] if request.param == "watch_github" else ["--url", "https://example.test/feed"]
    return module, args, tmp_path / "feed.json"


def payload(module, count):
    if module.__name__ == "watch_rss":
        return ("<rss><channel>" + "".join(f"<item><guid>{i}</guid><title>item-{i}</title></item>" for i in range(count)) + "</channel></rss>").encode()
    return json.dumps([{"id": str(i), "title": f"item-{i}"} for i in range(count)]).encode()


def invoke(module, args, body):
    with patch.object(sys, "argv", args), patch.object(module.urllib.request, "urlopen", return_value=io.BytesIO(body)):
        try:
            return module.main()
        except SystemExit as exc:
            return exc.code


@pytest.mark.parametrize("cap", [None, 2, 600])
def test_retention_reaches_persisted_state_and_keeps_output_cap(watcher, cap, capsys):
    module, args, state = watcher
    options = [] if cap is None else ["--max-seen", str(cap)]
    expected_cap = module.Watermark("control").max_seen if cap is None else cap
    args += options + ["--max", "1"]
    assert invoke(module, args, payload(module, 0)) == 0
    assert capsys.readouterr().out == ""
    count = expected_cap + 2
    assert invoke(module, args, payload(module, count)) == 0
    assert len(json.loads(state.read_text())["seen_ids"]) == expected_cap
    assert capsys.readouterr().out.count("## ") == 1
    before = state.read_bytes()
    with patch.object(sys, "argv", args), patch.object(module.urllib.request, "urlopen", side_effect=OSError("offline")):
        assert module.main() == 2
    assert state.read_bytes() == before


@pytest.mark.parametrize("value", ["0", "-1", "bad"])
def test_invalid_retention_fails_before_fetch_or_state_write(watcher, value):
    module, args, state = watcher
    with patch.object(sys, "argv", args + ["--max-seen", value]), patch.object(module.urllib.request, "urlopen") as fetch:
        with pytest.raises(SystemExit) as exc:
            module.main()
    assert exc.value.code == 2
    fetch.assert_not_called()
    assert not state.exists()
