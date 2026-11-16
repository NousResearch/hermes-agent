"""Gateway ``plugins.manage inspect``: the browser-hosted app's repository check before install.

Drives the real RPC dispatch and a real local clone: the answer carries the probe contract's fields,
a failed read is ``ok: false`` (not an RPC error), and nothing lands in the profile's plugins folder.
"""

import subprocess

from tui_gateway import server


def _manage(**params):
    return server.handle_request({"id": "1", "method": "plugins.manage", "params": params})


def _repo(path, files):
    path.mkdir()
    for rel, text in files.items():
        (path / rel).parent.mkdir(parents=True, exist_ok=True)
        (path / rel).write_text(text, encoding="utf-8")
    for args in (("init", "-q", "-b", "main"), ("add", "-A"),
                 ("-c", "user.email=t@example.invalid", "-c", "user.name=t", "commit", "-q", "-m", "c")):
        subprocess.run(["git", *args], cwd=path, check=True, capture_output=True)
    return path


def test_inspect_reports_halves_and_installs_nothing(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    repo = _repo(tmp_path / "pkg", {"plugin.yaml": "name: pkg\nversion: 1.0.0\n", "__init__.py": "",
                                    "desktop/plugin.js": "export default {}\n"})

    result = _manage(action="inspect", identifier=repo.as_uri())["result"]

    assert {k: result[k] for k in ("ok", "agent", "desktop", "agent_name", "desktop_name", "insecure")} == {
        "ok": True, "agent": True, "desktop": True, "agent_name": "pkg", "desktop_name": "pkg", "insecure": True}
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True,
                          encoding="utf-8").stdout.strip()
    assert result["sha"] == head
    assert not (home / "plugins" / "pkg").exists()
    listed = _manage(action="list")["result"]["plugins"]
    assert all(row["name"] != "pkg" for row in listed)


def test_failed_read_is_an_answer_not_an_rpc_error(tmp_path):
    response = _manage(action="inspect", identifier=(tmp_path / "missing").as_uri())

    assert "error" not in response
    assert response["result"]["ok"] is False
    assert response["result"]["error"]


def test_inspect_requires_a_source():
    response = _manage(action="inspect")

    assert response["error"]["code"] == 4019
