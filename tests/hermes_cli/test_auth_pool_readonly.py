"""Diagnostics preserve native profile authority without recovery writes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from hermes_cli import auth, auth_store_readonly


@pytest.mark.parametrize('profile_entries', [[], [{'id':'profile','label':'Profile'}]])
def test_readonly_pool_matches_native_profile_shadowing(tmp_path, monkeypatch, profile_entries):
    profile=tmp_path/'profile.json'; root=tmp_path/'root.json'
    profile.write_text(json.dumps({'credential_pool':{'openai-codex':profile_entries}}))
    root.write_text(json.dumps({'credential_pool':{'openai-codex':[{'id':'root','label':'Root'}]}}))
    monkeypatch.setattr(auth,'_auth_file_path',lambda:profile)
    monkeypatch.setattr(auth,'_global_auth_file_path',lambda:root)
    monkeypatch.setattr(auth_store_readonly,'_auth_store_paths',lambda:(profile,root))
    monkeypatch.setattr(auth,'_global_auth_store_cache',None)
    normal=auth.read_credential_pool('openai-codex')
    assert auth.read_credential_pool('openai-codex',read_only=True)==normal
    root.write_text('{invalid')
    corrupt={p.name:p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    with pytest.raises(ValueError,match='valid JSON'):
        auth.read_credential_pool('openai-codex',read_only=True)
    assert {p.name:p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}==corrupt
    # Default callers retain the existing recovery policy.
    auth._global_auth_store_cache=None
    auth.read_credential_pool('openai-codex')
    assert root.with_suffix('.json.corrupt').read_bytes()==corrupt['root.json']


def test_readonly_profile_corruption_raises_without_recovery_file(tmp_path, monkeypatch):
    profile=tmp_path/'auth.json';profile.write_text('{invalid')
    monkeypatch.setattr(auth,'_auth_file_path',lambda:profile)
    monkeypatch.setattr(auth,'_global_auth_file_path',lambda:None)
    monkeypatch.setattr(auth_store_readonly,'_auth_store_paths',lambda:(profile,None))
    with pytest.raises(ValueError):
        auth.read_credential_pool(read_only=True)
    assert [p for p in tmp_path.iterdir() if p.is_file()]==[profile]


@pytest.mark.parametrize("malformed_pool", [[], None, {"openai-codex": {"id": "broken"}}])
@pytest.mark.parametrize("malformed_source", ["profile", "root"])
def test_readonly_malformed_pool_never_silently_falls_back_or_writes(
    tmp_path, monkeypatch, malformed_pool, malformed_source,
):
    profile = tmp_path / "profile.json"
    root = tmp_path / "root.json"
    profile.write_text(json.dumps({"providers": {}, "credential_pool": {"openai-codex": []}}))
    root_entries = [{"id": "root"}]
    root.write_text(json.dumps({"providers": {}, "credential_pool": {"openai-codex": root_entries}}))
    malformed = profile if malformed_source == "profile" else root
    malformed.write_text(json.dumps({"providers": {}, "credential_pool": malformed_pool}))
    monkeypatch.setattr(auth, "_auth_file_path", lambda: profile)
    monkeypatch.setattr(auth, "_global_auth_file_path", lambda: root)
    monkeypatch.setattr(auth, "_global_auth_store_cache", None)
    monkeypatch.setattr(auth_store_readonly, "_auth_store_paths", lambda: (profile, root))
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}

    with pytest.raises(ValueError, match="credential_pool"):
        auth.read_credential_pool("openai-codex", read_only=True)
    with pytest.raises(ValueError, match="credential_pool"):
        auth_store_readonly.read_credential_pool()
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()} == before
    # Strict diagnostics must not change the operational reader's legacy fallback.
    assert auth.read_credential_pool("openai-codex") == (
        root_entries if malformed_source == "profile" else [])


@pytest.mark.parametrize("store", [{}, {"version": 1}, {"active_provider": "openai-codex"}])
def test_readonly_metadata_only_store_is_empty_without_writes(tmp_path, monkeypatch, store):
    profile = tmp_path / "auth.json"
    profile.write_text(json.dumps(store))
    monkeypatch.setattr(auth_store_readonly, "_auth_store_paths", lambda: (profile, None))
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    assert auth_store_readonly.read_credential_pool() == {}
    assert auth_store_readonly.read_credential_pool("openai-codex") == []
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()} == before


@pytest.mark.parametrize("use_profile", [False, True])
def test_readonly_real_path_resolution_matches_operational_auth(tmp_path, monkeypatch, use_profile):
    root = tmp_path / "custom-root"
    profile = root / "profiles" / "work" if use_profile else root
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    root_entries = [{"id": "root"}]
    (root / "auth.json").write_text(json.dumps({"credential_pool": {"openai-codex": root_entries}}))
    if use_profile:
        (profile / "auth.json").write_text(json.dumps({"credential_pool": {"openai-codex": []}}))
    # Use every real resolver; no patched paths can make this assertion pass by construction.
    assert auth_store_readonly._auth_store_paths() == (
        auth._auth_file_path(), auth._global_auth_file_path())
    assert auth.read_credential_pool("openai-codex", read_only=True) == root_entries
    assert auth.read_credential_pool("openai-codex") == root_entries


def test_fresh_reader_uses_real_profile_paths_without_operational_imports(tmp_path):
    user=tmp_path/'user'; root=user/'.hermes'; profile=root/'profiles/work'
    profile.mkdir(parents=True)
    (root/'auth.json').write_text(json.dumps({'credential_pool':{'openai-codex':[{'id':'root'}]}}))
    (profile/'auth.json').write_text(json.dumps({'credential_pool':{'openai-codex':[{'id':'profile'}]}}))
    source=Path(auth_store_readonly.__file__).resolve().parent.parent
    temp = user / 'temp'; temp.mkdir()
    env={'PATH':os.environ.get('PATH','/usr/bin:/bin'),'HOME':str(user),'HERMES_HOME':str(profile),
         'USERPROFILE':str(user),'LOCALAPPDATA':str(user/'AppData/Local'),
         'APPDATA':str(user/'AppData/Roaming'),'TEMP':str(temp),'TMP':str(temp),'TMPDIR':str(temp)}
    if 'SYSTEMROOT' in os.environ:
        env['SYSTEMROOT'] = os.environ['SYSTEMROOT']
    before={str(p.relative_to(user)):p.read_bytes() for p in user.rglob('*') if p.is_file()}
    probe=subprocess.run([sys.executable,'-I','-B','-c',
        'import sys;sys.path.insert(0,sys.argv[1]);'
        'from hermes_cli.auth_store_readonly import read_credential_pool;'
        'assert read_credential_pool("openai-codex")==[{"id":"profile"}];'
        'assert "hermes_cli.auth" not in sys.modules;'
        'assert "hermes_cli.config" not in sys.modules;'
        'assert "providers" not in sys.modules',str(source)],
        cwd=tmp_path,env=env,capture_output=True,text=True)
    assert probe.returncode==0,probe.stderr
    assert {str(p.relative_to(user)):p.read_bytes() for p in user.rglob('*') if p.is_file()}==before
