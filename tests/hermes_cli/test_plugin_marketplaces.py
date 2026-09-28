from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import types
import errno
from pathlib import Path

import pytest

from hermes_cli.plugin_marketplaces import (
    MarketplaceError,
    add_marketplace,
    list_marketplaces,
    public_marketplace,
    remove_marketplace,
)


def test_marketplace_clone_uses_git_credential_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import hermes_cli.git_credentials as credentials
    import hermes_cli.plugin_marketplaces as marketplaces

    calls = []

    def fake_git(args, url, **kwargs):
        calls.append((args, url, kwargs))
        return subprocess.CompletedProcess(args, 0, '', '')

    monkeypatch.setattr(credentials, 'run_git_with_credential_fallback', fake_git)
    marketplaces._clone('https://example.com/private.git', tmp_path / 'clone')
    assert calls[0][1] == 'https://example.com/private.git'
    assert calls[0][0][1:3] == ['clone', '--depth']


def _git(repo: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return result.stdout.strip()


def _marketplace_repo(
    tmp_path: Path, *, compatible: bool = True, name: str = "demo"
) -> Path:
    repo = tmp_path / "marketplace"
    plugin = repo / "plugins" / name
    (repo / ".claude-plugin").mkdir(parents=True)
    (plugin / ".claude-plugin").mkdir(parents=True)
    (plugin / "skills" / "demo").mkdir(parents=True)
    (repo / ".claude-plugin" / "marketplace.json").write_text(
        json.dumps({
            "name": "test-marketplace",
            "owner": {"name": "Test Publisher"},
            "plugins": [
                {
                    "name": name,
                    "displayName": f"{name.title()} Plugin",
                    "description": "A private marketplace plugin.",
                    "source": f"./plugins/{name}",
                }
            ],
        }),
        encoding="utf-8",
    )
    (plugin / ".claude-plugin" / "plugin.json").write_text(
        json.dumps({
            "name": name,
            "displayName": f"{name.title()} Plugin",
            "version": "1.0.0",
            "description": "A private marketplace plugin.",
            "author": {"name": "Test Publisher"},
        }),
        encoding="utf-8",
    )
    if compatible:
        (plugin / "plugin.json").write_text(
            json.dumps({
                "$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json",
                "name": name,
                "version": "1.0.0",
                "description": "A private marketplace plugin.",
            }),
            encoding="utf-8",
        )
    (plugin / "skills" / "demo" / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Demo.\n---\n\n# Demo\n",
        encoding="utf-8",
    )
    _git(repo, "init", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "initial")
    return repo


def test_add_lists_and_persists_private_marketplace(tmp_path: Path) -> None:
    repo = _marketplace_repo(tmp_path)

    added = add_marketplace(f"file://{repo}", allow_file=True)
    listed = list_marketplaces()

    assert added["name"] == "Test Marketplace"
    assert len(listed) == 1
    assert listed[0]["id"] == added["id"]
    assert listed[0]["url"] == f"file://{repo}"
    assert listed[0]["entries"][0] == {
        "compatible": True,
        "description": "A private marketplace plugin.",
        "display_name": "Demo Plugin",
        "incompatibility_reason": "",
        "maintainer": "Test Publisher",
        "name": "demo",
        "repo": f"file://{repo}",
        "sha": _git(repo, "rev-parse", "HEAD"),
        "source_id": added["id"],
        "source_name": "Test Marketplace",
        "subdir": "plugins/demo",
        "tree_sha": _git(repo, "rev-parse", "HEAD:plugins/demo"),
        "version": "1.0.0",
    }

    registry = Path(os.environ["HERMES_HOME"]) / "plugin-marketplaces.json"
    saved = json.loads(registry.read_text(encoding="utf-8"))
    assert saved == {
        "marketplaces": [
            {
                "id": added["id"],
                "name": "Test Marketplace",
                "url": f"file://{repo}",
            }
        ],
        "version": 1,
    }

    public = public_marketplace(added)
    assert "url" not in public
    assert "repo" not in public["entries"][0]


def test_add_rejects_embedded_credentials() -> None:
    with pytest.raises(MarketplaceError, match="credentials"):
        add_marketplace("https://user:secret@example.com/private/repo.git")


@pytest.mark.parametrize(
    ("source_id", "url"),
    [
        ("../../outside", "https://github.com/example/repo.git"),
        ("0123456789abcdef", "https://user:secret@example.com/repo.git"),
        ("0123456789abcdef", "https://github.com/example/repo.git"),
    ],
)
def test_registry_rejects_untrusted_source_identity_without_touching_cache(
    source_id: str, url: str
) -> None:
    home = Path(os.environ["HERMES_HOME"])
    outside = home.parent / "outside.json"
    outside.write_text("sentinel", encoding="utf-8")
    (home / "plugin-marketplaces.json").write_text(
        json.dumps({
            "marketplaces": [{"id": source_id, "name": "bad", "url": url}],
            "version": 1,
        }),
        encoding="utf-8",
    )

    with pytest.raises(MarketplaceError, match="invalid source|credentials"):
        list_marketplaces()
    with pytest.raises(MarketplaceError, match="invalid source|credentials"):
        remove_marketplace(source_id)

    assert outside.read_text(encoding="utf-8") == "sentinel"


def test_registry_absolute_id_cannot_delete_outside_cache(tmp_path: Path) -> None:
    home = Path(os.environ["HERMES_HOME"])
    outside = tmp_path / "outside.json"
    outside.write_text("sentinel", encoding="utf-8")
    source_id = str(outside.with_suffix(""))
    (home / "plugin-marketplaces.json").write_text(
        json.dumps({
            "marketplaces": [
                {
                    "id": source_id,
                    "name": "bad",
                    "url": "https://github.com/example/repo.git",
                }
            ],
            "version": 1,
        }),
        encoding="utf-8",
    )

    with pytest.raises(MarketplaceError, match="invalid source ID"):
        remove_marketplace(source_id)

    assert outside.read_text(encoding="utf-8") == "sentinel"


def test_add_rejects_query_and_external_source_forms(tmp_path: Path) -> None:
    with pytest.raises(MarketplaceError, match="query or fragment"):
        add_marketplace("https://github.com/example/repo?token=nope")

    repo = _marketplace_repo(tmp_path)
    manifest = repo / ".claude-plugin" / "marketplace.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["plugins"][0]["source"] = {
        "source": "url",
        "url": "https://example.com/plugin.git",
    }
    manifest.write_text(json.dumps(data), encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "external source")
    with pytest.raises(MarketplaceError, match="unsupported object source"):
        add_marketplace(f"file://{repo}", allow_file=True)


def test_add_rejects_plugin_path_escape(tmp_path: Path) -> None:
    repo = _marketplace_repo(tmp_path)
    manifest = repo / ".claude-plugin" / "marketplace.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["plugins"][0]["source"] = "./../outside"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "escape")

    with pytest.raises(MarketplaceError, match="inside the marketplace"):
        add_marketplace(f"file://{repo}", allow_file=True)

    assert not (Path(os.environ["HERMES_HOME"]) / "plugin-marketplaces.json").exists()


def test_add_rejects_symlinked_plugin_root(tmp_path: Path) -> None:
    repo = _marketplace_repo(tmp_path)
    real = repo / "plugins" / "real"
    (repo / "plugins" / "demo").rename(real)
    (repo / "plugins" / "demo").symlink_to(real.name, target_is_directory=True)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "symlink plugin root")

    with pytest.raises(MarketplaceError, match="symlink"):
        add_marketplace(f"file://{repo}", allow_file=True)


def test_only_plugin_subtree_change_advertises_new_tree(tmp_path: Path) -> None:
    repo = _marketplace_repo(tmp_path)
    source = add_marketplace(f"file://{repo}", allow_file=True)
    first = source["entries"][0]

    (repo / "README.md").write_text("unrelated\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "unrelated")
    unrelated = list_marketplaces(force=True)[0]["entries"][0]

    assert unrelated["sha"] != first["sha"]
    assert unrelated["tree_sha"] == first["tree_sha"]

    skill = repo / "plugins" / "demo" / "skills" / "demo" / "SKILL.md"
    skill.write_text(
        skill.read_text(encoding="utf-8") + "\nChanged.\n", encoding="utf-8"
    )
    _git(repo, "add", "plugins/demo")
    _git(repo, "commit", "-m", "plugin update")
    changed = list_marketplaces(force=True)[0]["entries"][0]

    assert changed["tree_sha"] != first["tree_sha"]


def test_concurrent_refreshes_serialize_cache_commits(
    tmp_path: Path, monkeypatch
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, Lock

    import hermes_cli.plugin_marketplaces as marketplaces

    repo = _marketplace_repo(tmp_path)
    add_marketplace(f"file://{repo}", allow_file=True)
    clone = marketplaces._clone
    first_entered = Event()
    second_entered = Event()
    allow_first = Event()
    calls = 0
    calls_lock = Lock()

    def controlled_clone(url, target):
        nonlocal calls
        with calls_lock:
            calls += 1
            call = calls
        if call == 1:
            first_entered.set()
            assert allow_first.wait(timeout=5)
        else:
            second_entered.set()
        clone(url, target)

    monkeypatch.setattr(marketplaces, "_clone", controlled_clone)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(list_marketplaces, force=True)
        assert first_entered.wait(timeout=5)
        second = pool.submit(list_marketplaces, force=True)
        assert second_entered.wait(timeout=0.2) is False
        allow_first.set()
        first.result(timeout=10)
        second.result(timeout=10)

    assert second_entered.is_set()


def test_refresh_lock_serializes_independent_processes(tmp_path: Path) -> None:
    home = Path(os.environ["HERMES_HOME"])
    ready = tmp_path / "ready"
    acquired = tmp_path / "acquired"
    source_id = "0123456789abcdef"
    first_code = (
        "import time; from pathlib import Path; "
        "from hermes_cli.plugin_marketplaces import _refresh_lock; "
        f"p=Path({str(ready)!r}); "
        f"c=_refresh_lock({source_id!r}); c.__enter__(); "
        "p.write_text('ready'); time.sleep(1.0); c.__exit__(None,None,None)"
    )
    second_code = (
        "from pathlib import Path; "
        "from hermes_cli.plugin_marketplaces import _refresh_lock; "
        f"p=Path({str(acquired)!r}); "
        f"c=_refresh_lock({source_id!r}); c.__enter__(); "
        "p.write_text('acquired'); c.__exit__(None,None,None)"
    )
    env = {**os.environ, "HERMES_HOME": str(home)}
    first = subprocess.Popen([sys.executable, "-c", first_code], env=env)
    second = None
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        second = subprocess.Popen([sys.executable, "-c", second_code], env=env)
        time.sleep(0.2)
        assert not acquired.exists()
        assert first.wait(timeout=5) == 0
        assert second.wait(timeout=5) == 0
        assert acquired.read_text(encoding="utf-8") == "acquired"
    finally:
        if first.poll() is None:
            first.kill()
        if second is not None and second.poll() is None:
            second.kill()


def test_windows_lock_retries_past_msvcrt_ten_second_limit(monkeypatch) -> None:
    import hermes_cli.plugin_marketplace_files as install_state

    calls = 0

    def locking(_fd, _mode, _length):
        nonlocal calls
        calls += 1
        if calls <= 12:
            raise OSError(errno.EACCES, "locked")

    fake_msvcrt = types.SimpleNamespace(
        LK_NBLCK=1,
        LK_UNLCK=2,
        locking=locking,
    )
    with tempfile.TemporaryFile() as handle, monkeypatch.context() as patcher:
        patcher.setattr(install_state.os, "name", "nt")
        patcher.setattr(install_state.time, "sleep", lambda _seconds: None)
        patcher.setitem(sys.modules, "msvcrt", fake_msvcrt)
        install_state._lock_file(handle)

    assert calls == 13


def test_forced_refresh_never_authorizes_stale_cache(
    tmp_path: Path, monkeypatch
) -> None:
    import hermes_cli.plugin_marketplaces as marketplaces

    repo = _marketplace_repo(tmp_path)
    source = add_marketplace(f"file://{repo}", allow_file=True)

    def offline(_source):
        raise MarketplaceError("offline")

    monkeypatch.setattr(marketplaces, "_refresh", offline)

    stale = list_marketplaces(force=True)[0]
    assert stale["available"] is True
    assert stale["stale"] is True
    assert marketplaces.get_marketplace_entry(source["id"], "demo", force=True) is None


@pytest.mark.parametrize("cache_kind", ["malformed", "oversized"])
def test_refresh_recovers_from_non_authoritative_corrupt_cache(
    tmp_path: Path, cache_kind: str
) -> None:
    contents = "{broken" if cache_kind == "malformed" else "x" * (1024 * 1024 + 1)
    repo = _marketplace_repo(tmp_path)
    source = add_marketplace(f"file://{repo}", allow_file=True)
    cache = (
        Path(os.environ["HERMES_HOME"])
        / "cache"
        / "plugin-marketplaces"
        / f"{source['id']}.json"
    )
    cache.write_text(contents, encoding="utf-8")

    refreshed = list_marketplaces(force=True)

    assert refreshed[0]["entries"][0]["name"] == "demo"
    assert json.loads(cache.read_text(encoding="utf-8"))["source"]["id"] == source["id"]


def test_removed_source_cannot_authorize_inflight_refresh(
    tmp_path: Path, monkeypatch
) -> None:
    import hermes_cli.plugin_marketplaces as marketplaces

    repo = _marketplace_repo(tmp_path)
    source = add_marketplace(f"file://{repo}", allow_file=True)
    refresh = marketplaces._refresh

    def remove_during_refresh(saved):
        remove_marketplace(saved["id"])
        return refresh(saved)

    monkeypatch.setattr(marketplaces, "_refresh", remove_during_refresh)

    assert marketplaces.get_marketplace_entry(source["id"], "demo", force=True) is None
    assert list_marketplaces() == []


def test_claude_only_package_is_visible_but_not_installable(tmp_path: Path) -> None:
    repo = _marketplace_repo(tmp_path, compatible=False)

    source = add_marketplace(f"file://{repo}", allow_file=True)

    assert source["entries"][0]["compatible"] is False


def test_remove_marketplace_keeps_other_sources(tmp_path: Path) -> None:
    first = _marketplace_repo(tmp_path / "one")
    second = _marketplace_repo(tmp_path / "two")
    first_source = add_marketplace(f"file://{first}", allow_file=True)
    second_source = add_marketplace(f"file://{second}", allow_file=True)

    assert remove_marketplace(first_source["id"]) is True
    assert [source["id"] for source in list_marketplaces()] == [second_source["id"]]
    assert remove_marketplace(first_source["id"]) is False


def test_concurrent_adds_do_not_lose_registry_entries(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    repos = [
        _marketplace_repo(tmp_path / "one"),
        _marketplace_repo(tmp_path / "two"),
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        sources = list(
            pool.map(
                lambda repo: add_marketplace(f"file://{repo}", allow_file=True),
                repos,
            )
        )

    assert {item["id"] for item in list_marketplaces()} == {
        source["id"] for source in sources
    }


def test_marketplace_registry_and_cache_symlinks_do_not_write_external_files(
    tmp_path: Path,
) -> None:
    import hermes_cli.plugin_marketplaces as marketplaces

    external = tmp_path / "external.json"
    external.write_text("unchanged", encoding="utf-8")
    registry = marketplaces._registry_path()
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.symlink_to(external)
    with pytest.raises(MarketplaceError, match="symlink"):
        marketplaces._write_registry([])
    assert external.read_text(encoding="utf-8") == "unchanged"
    registry.unlink()

    marketplaces._ensure_cache_dir()
    cache_file = marketplaces._cache_path("0" * 16)
    cache_file.symlink_to(external)
    source = {"id": "0" * 16, "name": "test", "url": "https://example.com/x.git"}
    with pytest.raises(MarketplaceError, match="symlink"):
        marketplaces._write_cache(source, [])
    assert external.read_text(encoding="utf-8") == "unchanged"


def test_registry_parent_swap_does_not_write_external_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hermes_cli.plugin_marketplaces as marketplaces
    import utils

    home = Path(os.environ["HERMES_HOME"])
    saved = home.with_name("hermes-saved")
    outside = tmp_path / "outside-home"
    outside.mkdir()
    real_open = utils.os.open
    swapped = False

    def swapping_open(candidate, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if Path(candidate) == home and dir_fd is None and not swapped:
            swapped = True
            home.rename(saved)
            home.symlink_to(outside, target_is_directory=True)
        return real_open(candidate, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(utils.os, "open", swapping_open)
    with pytest.raises(OSError):
        marketplaces._write_registry([])
    assert not (outside / "plugin-marketplaces.json").exists()


def test_marketplace_cache_directory_symlink_is_rejected(tmp_path: Path) -> None:
    import hermes_cli.plugin_marketplaces as marketplaces

    cache = marketplaces._cache_dir()
    cache.parent.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside-cache"
    outside.mkdir()
    cache.symlink_to(outside, target_is_directory=True)

    with pytest.raises(MarketplaceError, match="symlink"):
        marketplaces._ensure_cache_dir()


def test_remove_marketplace_does_not_delete_through_cache_symlink(
    tmp_path: Path,
) -> None:
    import hermes_cli.plugin_marketplaces as marketplaces

    source = {
        "id": marketplaces._source_id("https://example.com/test.git"),
        "name": "test",
        "url": "https://example.com/test.git",
    }
    marketplaces._write_registry([source])
    cache = marketplaces._cache_dir()
    cache.parent.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside-cache"
    outside.mkdir()
    external = outside / f"{source['id']}.json"
    external.write_text("keep", encoding="utf-8")
    cache.symlink_to(outside, target_is_directory=True)

    assert marketplaces.remove_marketplace(source["id"]) is True
    assert external.read_text(encoding="utf-8") == "keep"


def test_marketplace_plugin_installs_separately_at_selected_pin(tmp_path: Path, monkeypatch) -> None:
    from hermes_cli import plugins_cmd, plugins_cmd_catalog

    repo = _marketplace_repo(tmp_path)
    source = add_marketplace(f"file://{repo}", allow_file=True)
    plugins_dir = Path(os.environ["HERMES_HOME"]) / "plugins"
    assert not (plugins_dir / "demo").exists()  # registering is not installing
    monkeypatch.setattr(plugins_cmd_catalog, "raise_if_removed", lambda *args: None)
    from hermes_cli import plugins_transaction
    import shutil

    # PM's separate publication contract is covered by its own tests; exercise the real
    # marketplace Git fetch, manifest, scan, authority check and installer metadata handoff.
    def publish(staged, target, old, new, **kwargs):
        shutil.copytree(staged, target)
        plugins_cmd._write_install_metadata(new)
    monkeypatch.setattr(plugins_transaction, "publish_plugin", publish)

    installed = plugins_cmd.dashboard_install_plugin(
        "", force=False, enable=False, marketplace_id=source["id"], marketplace_plugin_name="demo")
    assert installed["ok"] is True, installed
    assert (plugins_dir / "demo" / "plugin.json").is_file()
    metadata = plugins_cmd._read_install_metadata()["demo"]
    assert metadata["marketplace"]["id"] == source["id"]
    assert metadata["revision"] == _git(repo, "rev-parse", "HEAD")

    assert remove_marketplace(source["id"])
    refused = plugins_cmd.dashboard_install_plugin(
        "", force=True, enable=False, marketplace_id=source["id"], marketplace_plugin_name="demo")
    assert refused["ok"] is False


@pytest.mark.parametrize("name", ["CON", "demo.", ".install-temp", "demo:name", "COM¹"])
def test_marketplace_rejects_cross_platform_plugin_filename_aliases(tmp_path: Path, name: str) -> None:
    repo = _marketplace_repo(tmp_path)
    manifest = repo / ".claude-plugin" / "marketplace.json"
    value = json.loads(manifest.read_text())
    value["plugins"][0]["name"] = name
    manifest.write_text(json.dumps(value))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "unsafe filename")
    with pytest.raises(MarketplaceError, match="Invalid marketplace plugin name"):
        add_marketplace(f"file://{repo}", allow_file=True)
