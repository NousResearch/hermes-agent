"""``load_yaml_file_readonly`` re-parses only when the file signature changes."""
import os

import pytest

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


def test_parse_error_propagates_and_is_not_cached(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("secrets: [unclosed\n")
    with pytest.raises(Exception):
        load_yaml_file_readonly(path)
    path.write_text("secrets: {}\n")
    os.utime(path, ns=(os.stat(path).st_mtime_ns + 1_000_000,) * 2)
    assert load_yaml_file_readonly(path) == {"secrets": {}}


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_yaml_file_readonly(tmp_path / "nope.yaml")


def test_atomic_replace_invalidates_even_with_identical_mtime_and_size(tmp_path):
    """``save_config`` writes via ``atomic_yaml_write`` → a new inode; the signature must change even
    when a writer pins mtime and keeps the size (the case mtime+size alone would miss)."""
    from utils import atomic_replace

    path = tmp_path / "config.yaml"
    path.write_text("terminal: {backend: local}\n")
    st = os.stat(path)
    first = load_yaml_file_readonly(path)

    tmp = tmp_path / "config.yaml.tmp"
    tmp.write_text("terminal: {backend: lokal}\n")  # same size, different content
    os.utime(tmp, ns=(st.st_atime_ns, st.st_mtime_ns))
    atomic_replace(tmp, path)
    after = os.stat(path)
    assert (after.st_mtime_ns, after.st_size) == (st.st_mtime_ns, st.st_size)

    assert load_yaml_file_readonly(path) == {"terminal": {"backend": "lokal"}}
    assert load_yaml_file_readonly(path) is not first


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


def test_profile_config_edit_is_visible_on_next_scope_build(tmp_path):
    """A config.yaml edit reaches the next ``build_profile_terminal_scope`` — the cache never pins
    a stale policy across the edit (the "config changes stop applying" upgrade risk)."""
    from tools.terminal_scope import build_profile_terminal_scope

    home = tmp_path / "profiles" / "work"
    home.mkdir(parents=True)
    path = home / "config.yaml"
    path.write_text("terminal:\n  backend: local\n")
    assert build_profile_terminal_scope(home)["TERMINAL_ENV"] == "local"

    path.write_text("terminal:\n  backend: docker\n")
    os.utime(path, ns=(os.stat(path).st_mtime_ns + 1_000_000,) * 2)
    assert build_profile_terminal_scope(home)["TERMINAL_ENV"] == "docker"
