"""``load_yaml_file_readonly`` re-parses only when the file signature changes."""
import os


from utils import load_yaml_file_readonly


def test_cache_hit_returns_same_object_and_invalidates_on_rewrite(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("secrets: {a: 1}\n")
    first = load_yaml_file_readonly(path)
    assert first == {"secrets": {"a": 1}}
    assert load_yaml_file_readonly(path) is first

    path.write_text("secrets: {a: 2}\n")
    os.utime(path, ns=(os.stat(path).st_mtime_ns + 1_000_000,) * 2)
    second = load_yaml_file_readonly(path)
    assert second == {"secrets": {"a": 2}}
    assert second is not first


def test_callers_do_not_mutate_the_shared_cached_object(tmp_path, monkeypatch):
    """Both callers only read one section; the cached mapping must survive them byte-for-byte,
    otherwise a later reader of the same file would observe another caller's edits."""
    import copy

    from hermes_cli.env_loader import _load_secrets_config
    from tools.terminal_scope import build_profile_terminal_scope

    home = tmp_path / "profiles" / "work"
    home.mkdir(parents=True)
    path = home / "config.yaml"
    path.write_text(
        "terminal:\n  backend: local\n  cwd: auto\n  timeout: 5\n"
        "secrets:\n  onepassword: {enabled: false}\n"
    )
    monkeypatch.setattr("hermes_cli.env_loader._process_hermes_home", lambda: tmp_path / "other")
    cached = load_yaml_file_readonly(path)
    snapshot = copy.deepcopy(cached)

    scope = build_profile_terminal_scope(home)
    secrets = _load_secrets_config(home)

    assert scope["TERMINAL_ENV"] == "local"
    assert secrets == {"onepassword": {"enabled": False}}
    assert cached == snapshot
    assert load_yaml_file_readonly(path) is cached
