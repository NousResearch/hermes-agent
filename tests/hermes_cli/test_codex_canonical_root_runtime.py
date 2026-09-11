import json

from hermes_cli.auth import read_credential_pool, write_credential_pool
from hermes_constants import get_default_hermes_root


def _store(path, pool):
    path.write_text(json.dumps({"credential_pool": pool}), encoding="utf-8")


def test_codex_profile_shadow_is_ignored_and_writes_use_root(tmp_path, monkeypatch):
    root = tmp_path / "root"
    profile = root / "profiles" / "vector"
    profile.mkdir(parents=True)
    _store(root / "auth.json", {"openai-codex": [{"id": "canonical"}], "openrouter": [{"id": "other"}]})
    _store(profile / "auth.json", {"openai-codex": [{"id": "stale-local"}], "openrouter": [{"id": "local-other"}]})
    monkeypatch.setenv("HERMES_ROOT", str(root))
    monkeypatch.setenv("HERMES_HOME", str(profile))

    assert get_default_hermes_root() == root
    assert [row["id"] for row in read_credential_pool("openai-codex")] == ["canonical"]
    assert [row["id"] for row in read_credential_pool()["openai-codex"]] == ["canonical"]

    write_credential_pool("openai-codex", [{"id": "rotated"}])
    root_rows = json.loads((root / "auth.json").read_text(encoding="utf-8"))["credential_pool"]["openai-codex"]
    assert {row["id"] for row in root_rows} == {"rotated", "canonical"}
    assert json.loads((profile / "auth.json").read_text(encoding="utf-8"))["credential_pool"]["openai-codex"] == [{"id": "stale-local"}]
    assert json.loads((root / "auth.json").read_text(encoding="utf-8"))["credential_pool"]["openrouter"] == [{"id": "other"}]
