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
    remove_marketplace,
)


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
    import hermes_cli.plugins_cmd as plugins_cmd

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
        patcher.setattr(plugins_cmd.os, "name", "nt")
        patcher.setattr(plugins_cmd.time, "sleep", lambda _seconds: None)
        patcher.setitem(sys.modules, "msvcrt", fake_msvcrt)
        plugins_cmd._lock_file(handle)

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


def test_dashboard_install_resolves_private_entry_server_side(
    tmp_path: Path, monkeypatch
) -> None:
    import hermes_cli.plugin_catalog as plugin_catalog
    import hermes_cli.plugins_cmd as plugins_cmd
    from hermes_cli.plugin_marketplaces import as_catalog_entry

    target = tmp_path / "installed" / "demo"
    entry = as_catalog_entry({
        "compatible": True,
        "description": "Demo.",
        "display_name": "Demo",
        "maintainer": "Test",
        "name": "demo",
        "repo": "https://github.com/example/private-marketplace",
        "sha": "a" * 40,
        "source_id": "private-source",
        "source_name": "Private Market",
        "subdir": "plugins/demo",
        "tree_sha": "b" * 40,
        "version": "1.0.0",
    })
    calls: list[dict] = []

    monkeypatch.setattr(
        plugins_cmd,
        "_get_live_catalog_entry",
        lambda name, source_id="official", **_kwargs: (
            entry if (name, source_id) == ("demo", "private-source") else None
        ),
    )
    monkeypatch.setattr(
        plugin_catalog,
        "find_removed",
        lambda _name: pytest.fail("private marketplace consulted official removals"),
    )

    def fake_install(identifier, **kwargs):
        calls.append({"identifier": identifier, **kwargs})
        return target, {"name": "demo", "version": "1.0.0"}, "demo"

    monkeypatch.setattr(plugins_cmd, "_install_plugin_core", fake_install)
    monkeypatch.setattr(plugins_cmd, "_plugins_dir", lambda: tmp_path / "installed")
    monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: set())
    monkeypatch.setattr(plugins_cmd, "_get_disabled_set", lambda: set())

    result = plugins_cmd.dashboard_install_plugin(
        "attacker-controlled-display-value",
        force=False,
        enable=False,
        catalog_name="demo",
        catalog_source="private-source",
    )

    assert result["ok"] is True
    assert calls == [
        {
            "enable_on_commit": False,
            "force": False,
            "identifier": "https://github.com/example/private-marketplace#plugins/demo",
            "metadata_extra": {
                "installed_repo_sha": "a" * 40,
                "installed_tree_sha": "b" * 40,
                "marketplace_id": "private-source",
                "marketplace_name": "Private Market",
                "marketplace_plugin_name": "demo",
                "source": "https://github.com/example/private-marketplace",
                "subdir": "plugins/demo",
            },
            "ref": "a" * 40,
        }
    ]


def test_native_manifest_name_cannot_force_overwrite_another_source(
    tmp_path: Path,
) -> None:
    import hermes_cli.plugins_cmd as plugins_cmd

    repo = _marketplace_repo(tmp_path)
    plugin = repo / "plugins" / "demo"
    (plugin / "plugin.json").unlink()
    (plugin / "plugin.yaml").write_text(
        "name: victim\nversion: 1.0.0\ndescription: Wrong target\n",
        encoding="utf-8",
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "mismatched native manifest")

    source = add_marketplace(f"file://{repo}", allow_file=True)
    assert source["entries"][0]["compatible"] is False
    assert "name must match" in source["entries"][0]["incompatibility_reason"]

    victim = Path(os.environ["HERMES_HOME"]) / "plugins" / "victim"
    victim.mkdir(parents=True)
    marker = victim / "keep.txt"
    marker.write_text("original", encoding="utf-8")

    with pytest.raises(plugins_cmd.PluginOperationError, match="another source"):
        plugins_cmd._install_plugin_core(
            f"file://{repo}#plugins/demo",
            force=True,
            ref=_git(repo, "rev-parse", "HEAD"),
            metadata_extra={
                "marketplace_id": source["id"],
                "source": f"file://{repo}",
            },
        )

    assert marker.read_text(encoding="utf-8") == "original"


def test_internal_install_lock_name_cannot_be_installed(tmp_path: Path) -> None:
    import hermes_cli.plugins_cmd as plugins_cmd

    repo = _marketplace_repo(tmp_path)
    plugin = repo / "plugins" / "demo"
    (plugin / "plugin.json").unlink()
    (plugin / "plugin.yaml").write_text(
        "name: .install-metadata.lock\nversion: 1.0.0\ndescription: reserved\n",
        encoding="utf-8",
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "reserved name")
    with plugins_cmd._install_metadata_lock():
        pass

    with pytest.raises(plugins_cmd.PluginOperationError, match="reserved"):
        plugins_cmd._install_plugin_core(
            f"file://{repo}#plugins/demo",
            force=True,
            ref=_git(repo, "rev-parse", "HEAD"),
        )

    lock = Path(os.environ["HERMES_HOME"]) / "plugins" / ".install-metadata.lock"
    assert lock.is_file()


@pytest.mark.parametrize("reserved_name", [".INSTALL-METADATA.LOCK", ".Install-temp"])
def test_reserved_installer_names_are_case_insensitive(
    tmp_path: Path, reserved_name: str
) -> None:
    import hermes_cli.plugins_cmd as plugins_cmd

    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    with pytest.raises(ValueError, match="reserved for installer state"):
        plugins_cmd._sanitize_plugin_name(reserved_name, plugins_dir)


def test_existing_symlink_cannot_alias_plugin_destination(tmp_path: Path) -> None:
    import hermes_cli.plugins_cmd as plugins_cmd

    plugins_dir = tmp_path / "plugins"
    victim = plugins_dir / "victim"
    victim.mkdir(parents=True)
    (plugins_dir / "alias").symlink_to(victim, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink destinations"):
        plugins_cmd._sanitize_plugin_name("alias", plugins_dir)


@pytest.mark.parametrize(
    "name",
    [
        "demo.",
        "demo ",
        "CON",
        "lpt1.txt",
        "COM¹",
        "COM².txt",
        "LPT³",
        "CLOCK$",
        "CONIN$",
        "CONOUT$.txt",
        "demo:name",
    ],
)
def test_cross_platform_filesystem_alias_names_are_rejected(
    tmp_path: Path, name: str
) -> None:
    import hermes_cli.plugins_cmd as plugins_cmd

    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    with pytest.raises(ValueError):
        plugins_cmd._sanitize_plugin_name(name, plugins_dir)


def test_unicode_normalization_alias_is_rejected(tmp_path: Path) -> None:
    import hermes_cli.plugins_cmd as plugins_cmd

    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    (plugins_dir / "caf\N{LATIN SMALL LETTER E WITH ACUTE}").mkdir()

    with pytest.raises(ValueError, match="aliases existing path"):
        plugins_cmd._sanitize_plugin_name("cafe\N{COMBINING ACUTE ACCENT}", plugins_dir)


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


def test_malformed_install_metadata_entry_fails_closed() -> None:
    import hermes_cli.plugins_cmd as plugins_cmd

    path = plugins_cmd._install_metadata_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"bad":"value"}', encoding="utf-8")

    with pytest.raises(plugins_cmd.PluginOperationError, match="map plugin names"):
        plugins_cmd._read_install_metadata()


@pytest.mark.parametrize("kind", ["metadata", "lock", "transaction"])
def test_install_control_symlinks_fail_closed(tmp_path: Path, kind: str) -> None:
    import hermes_cli.plugins_cmd as plugins_cmd

    plugins_dir = Path(os.environ["HERMES_HOME"]) / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside"
    outside.write_text("{}", encoding="utf-8")
    paths = {
        "metadata": plugins_cmd._install_metadata_path(),
        "lock": plugins_cmd._install_metadata_path().with_suffix(".lock"),
        "transaction": plugins_cmd._install_transaction_path(),
    }
    path = paths[kind]
    path.unlink(missing_ok=True)
    path.symlink_to(outside)

    if kind == "metadata":
        with pytest.raises(
            plugins_cmd.PluginOperationError, match="must not be a symlink"
        ):
            plugins_cmd._read_install_metadata()
    else:
        with pytest.raises(
            plugins_cmd.PluginOperationError, match="must not be a symlink"
        ):
            with plugins_cmd._install_metadata_lock():
                pass


def test_lock_parent_swap_to_symlink_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hermes_cli.plugins_cmd as plugins_cmd
    import utils

    plugins_dir = Path(os.environ["HERMES_HOME"]) / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    saved = plugins_dir.with_name("plugins-saved")
    outside = tmp_path / "outside"
    outside.mkdir()
    real_open = utils.os.open
    swapped = False

    def swapping_open(candidate, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if candidate == "plugins" and dir_fd is not None and not swapped:
            swapped = True
            plugins_dir.rename(saved)
            plugins_dir.symlink_to(outside, target_is_directory=True)
        return real_open(candidate, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(utils.os, "open", swapping_open)
    with pytest.raises(plugins_cmd.PluginOperationError):
        with plugins_cmd._install_metadata_lock():
            pass
    assert not (outside / ".install-metadata.lock").exists()


def test_interrupted_new_install_is_rolled_back_without_losing_other_state() -> None:
    import hermes_cli.plugins_cmd as plugins_cmd
    from hermes_cli.config import load_config, save_config

    plugins_dir = Path(os.environ["HERMES_HOME"]) / "plugins"
    target = plugins_dir / "demo"
    target.mkdir(parents=True)
    (target / "new.txt").write_text("new", encoding="utf-8")
    transaction = plugins_dir / ".install-test-new"
    transaction.mkdir()
    plugins_cmd._write_install_metadata({
        "demo": {"source": "new"},
        "other": {"source": "keep"},
    })
    save_config(
        {"plugins": {"enabled": ["demo", "other"], "disabled": ["blocked"]}},
        preserve_plugin_state=False,
        preserve_platform_toolsets=False,
    )
    plugins_cmd._write_install_transaction({
        "version": 1,
        "plugin_name": "demo",
        "transaction_dir": transaction.name,
        "replaced_existing": False,
        "old_metadata": {"other": {"source": "keep"}},
        "was_enabled": False,
        "was_disabled": False,
    })

    with plugins_cmd._install_metadata_lock():
        pass

    config = load_config()
    assert not target.exists()
    assert plugins_cmd._read_install_metadata() == {"other": {"source": "keep"}}
    assert config["plugins"]["enabled"] == ["other"]
    assert config["plugins"]["disabled"] == ["blocked"]
    assert not plugins_cmd._install_transaction_path().exists()


def test_interrupted_forced_install_restores_previous_artifact() -> None:
    import hermes_cli.plugins_cmd as plugins_cmd
    from hermes_cli.config import save_config

    plugins_dir = Path(os.environ["HERMES_HOME"]) / "plugins"
    target = plugins_dir / "demo"
    target.mkdir(parents=True)
    (target / "marker.txt").write_text("new", encoding="utf-8")
    transaction = plugins_dir / ".install-test-replace"
    backup = transaction / "previous-plugin"
    backup.mkdir(parents=True)
    (backup / "marker.txt").write_text("old", encoding="utf-8")
    plugins_cmd._write_install_metadata({"demo": {"source": "new"}})
    save_config(
        {"plugins": {"enabled": ["demo"], "disabled": []}},
        preserve_plugin_state=False,
        preserve_platform_toolsets=False,
    )
    plugins_cmd._write_install_transaction({
        "version": 1,
        "plugin_name": "demo",
        "transaction_dir": transaction.name,
        "replaced_existing": True,
        "old_metadata": {"demo": {"source": "old"}},
        "was_enabled": True,
        "was_disabled": False,
    })

    with plugins_cmd._install_metadata_lock():
        pass

    assert (target / "marker.txt").read_text(encoding="utf-8") == "old"
    assert plugins_cmd._read_install_metadata() == {"demo": {"source": "old"}}


def test_removed_marketplace_cannot_commit_install_after_resolution(
    tmp_path: Path, monkeypatch
) -> None:
    import hermes_cli.plugins_cmd as plugins_cmd

    repo = _marketplace_repo(tmp_path)
    source = add_marketplace(f"file://{repo}", allow_file=True)
    scan = plugins_cmd._scan_plugin_tree

    def remove_after_resolution(*args, **kwargs):
        scan(*args, **kwargs)
        assert remove_marketplace(source["id"]) is True

    monkeypatch.setattr(plugins_cmd, "_scan_plugin_tree", remove_after_resolution)

    result = plugins_cmd.dashboard_install_plugin(
        "",
        force=False,
        enable=False,
        catalog_name="demo",
        catalog_source=source["id"],
    )

    assert result["ok"] is False
    assert "removed" in result["error"]
    assert not (Path(os.environ["HERMES_HOME"]) / "plugins" / "demo").exists()


def test_concurrent_marketplace_installs_preserve_both_provenance_entries(
    tmp_path: Path,
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    import hermes_cli.plugins_cmd as plugins_cmd

    sources = []
    for name in ("alpha", "beta"):
        repo = _marketplace_repo(tmp_path / name, name=name)
        sources.append(add_marketplace(f"file://{repo}", allow_file=True))

    def install(source):
        name = source["entries"][0]["name"]
        return plugins_cmd.dashboard_install_plugin(
            "",
            force=False,
            enable=False,
            catalog_name=name,
            catalog_source=source["id"],
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(install, sources))

    assert all(result["ok"] is True for result in results)
    metadata = plugins_cmd._read_install_metadata()
    assert {name: value["marketplace_id"] for name, value in metadata.items()} == {
        source["entries"][0]["name"]: source["id"] for source in sources
    }


def test_concurrent_enabled_installs_preserve_both_config_entries(
    tmp_path: Path,
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    import hermes_cli.plugins_cmd as plugins_cmd

    sources = []
    for name in ("alpha", "beta"):
        repo = _marketplace_repo(tmp_path / name, name=name)
        sources.append(add_marketplace(f"file://{repo}", allow_file=True))

    def install(source):
        name = source["entries"][0]["name"]
        return plugins_cmd.dashboard_install_plugin(
            "",
            force=False,
            enable=True,
            catalog_name=name,
            catalog_source=source["id"],
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(install, sources))

    assert all(result["ok"] is True for result in results)
    assert plugins_cmd._get_enabled_set() == {"alpha", "beta"}


def test_unrelated_stale_config_save_preserves_plugin_state() -> None:
    from hermes_cli.config import load_config, save_config

    save_config(
        {
            "plugins": {"enabled": [], "disabled": []},
            "platform_toolsets": {"cli": ["file"]},
        },
        preserve_plugin_state=False,
        preserve_platform_toolsets=False,
    )
    stale = load_config()
    current = load_config()
    current["plugins"]["enabled"] = ["demo"]
    current["platform_toolsets"]["cli"].append("demo-tools")
    save_config(
        current,
        preserve_plugin_state=False,
        preserve_platform_toolsets=False,
    )
    stale.setdefault("display", {})["tool_progress"] = "off"
    save_config(stale)

    persisted = load_config()
    assert persisted["plugins"]["enabled"] == ["demo"]
    assert persisted["platform_toolsets"]["cli"] == ["file", "demo-tools"]


def test_unrelated_atomic_writer_preserves_plugin_runtime_state() -> None:
    from hermes_cli.config import (
        atomic_config_write,
        get_config_path,
        load_config,
        save_config,
    )

    save_config(
        {
            "plugins": {"enabled": [], "disabled": []},
            "platform_toolsets": {"cli": ["file"]},
        },
        preserve_plugin_state=False,
        preserve_platform_toolsets=False,
    )
    stale = load_config()
    current = load_config()
    current["plugins"]["enabled"] = ["demo"]
    current["platform_toolsets"]["cli"].append("demo-tools")
    save_config(
        current,
        preserve_plugin_state=False,
        preserve_platform_toolsets=False,
    )
    stale["display"] = {"tool_progress": "compact"}

    atomic_config_write(get_config_path(), stale, sort_keys=False)

    persisted = load_config()
    assert persisted["plugins"]["enabled"] == ["demo"]
    assert persisted["platform_toolsets"]["cli"] == ["file", "demo-tools"]


def test_unrelated_migration_preserves_plugin_runtime_state() -> None:
    from hermes_cli.config import _persist_migration, load_config, save_config

    save_config(
        {
            "plugins": {"enabled": [], "disabled": []},
            "platform_toolsets": {"cli": ["file"]},
        },
        preserve_plugin_state=False,
        preserve_platform_toolsets=False,
    )
    stale = load_config()
    current = load_config()
    current["plugins"]["enabled"] = ["demo"]
    current["platform_toolsets"]["cli"].append("demo-tools")
    save_config(
        current,
        preserve_plugin_state=False,
        preserve_platform_toolsets=False,
    )
    stale["_config_version"] = 999

    _persist_migration(stale)

    persisted = load_config()
    assert persisted["plugins"]["enabled"] == ["demo"]
    assert persisted["platform_toolsets"]["cli"] == ["file", "demo-tools"]


def test_cross_process_stale_config_save_preserves_plugin_state(tmp_path: Path) -> None:
    import hermes_cli.plugins_cmd as plugins_cmd
    from hermes_cli.config import load_config, save_config

    save_config(
        {
            "plugins": {"enabled": [], "disabled": []},
            "platform_toolsets": {"cli": ["file"]},
        },
        preserve_plugin_state=False,
        preserve_platform_toolsets=False,
    )
    ready = tmp_path / "ready"
    proceed = tmp_path / "proceed"
    code = f"""import time
from pathlib import Path
from hermes_cli.config import load_config, save_config
c = load_config()
Path({str(ready)!r}).write_text('ready')
p = Path({str(proceed)!r})
deadline = time.monotonic() + 5
while not p.exists() and time.monotonic() < deadline:
    time.sleep(0.02)
c.setdefault('display', {{}})['tool_progress'] = 'compact'
save_config(c)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        env={**os.environ, "HERMES_HOME": os.environ["HERMES_HOME"]},
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        current = load_config()
        current["plugins"]["enabled"] = ["demo"]
        current["platform_toolsets"]["cli"].append("demo-tools")
        save_config(
            current,
            preserve_plugin_state=False,
            preserve_platform_toolsets=False,
        )
        proceed.write_text("go", encoding="utf-8")
        assert process.wait(timeout=5) == 0
    finally:
        if process.poll() is None:
            process.kill()

    assert plugins_cmd._get_enabled_set() == {"demo"}
    assert load_config()["platform_toolsets"]["cli"] == ["file", "demo-tools"]


def test_install_rolls_back_config_after_write_then_raise(
    tmp_path: Path, monkeypatch
) -> None:
    import hermes_cli.config as config_module
    import hermes_cli.plugins_cmd as plugins_cmd

    repo = _marketplace_repo(tmp_path)
    real_save = config_module.save_config
    calls = 0

    def write_then_raise(*args, **kwargs):
        nonlocal calls
        calls += 1
        real_save(*args, **kwargs)
        if calls == 1:
            raise OSError("injected post-write failure")

    monkeypatch.setattr(config_module, "save_config", write_then_raise)

    with pytest.raises(OSError, match="post-write failure"):
        plugins_cmd._install_plugin_core(
            f"file://{repo}#plugins/demo",
            force=False,
            ref=_git(repo, "rev-parse", "HEAD"),
            enable_on_commit=True,
        )

    assert not (Path(os.environ["HERMES_HOME"]) / "plugins" / "demo").exists()
    assert "demo" not in plugins_cmd._get_enabled_set()
    assert plugins_cmd._read_install_metadata() == {}


def test_remove_cleans_enabled_state_in_same_transaction(tmp_path: Path) -> None:
    import hermes_cli.plugins_cmd as plugins_cmd

    repo = _marketplace_repo(tmp_path)
    source = add_marketplace(f"file://{repo}", allow_file=True)
    result = plugins_cmd.dashboard_install_plugin(
        "",
        force=False,
        enable=True,
        catalog_name="demo",
        catalog_source=source["id"],
    )
    assert result["ok"] is True
    target = Path(os.environ["HERMES_HOME"]) / "plugins" / "demo"
    assert "demo" in plugins_cmd._get_enabled_set()

    plugins_cmd._remove_plugin_core(target)

    assert not target.exists()
    assert "demo" not in plugins_cmd._get_enabled_set()


def test_concurrent_remove_and_install_preserve_new_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    import hermes_cli.plugins_cmd as plugins_cmd

    sources = {}
    for name in ("alpha", "beta", "gamma"):
        repo = _marketplace_repo(tmp_path / name, name=name)
        sources[name] = add_marketplace(f"file://{repo}", allow_file=True)
    for name in ("alpha", "beta"):
        result = plugins_cmd.dashboard_install_plugin(
            "",
            force=False,
            enable=False,
            catalog_name=name,
            catalog_source=sources[name]["id"],
        )
        assert result["ok"] is True

    remove_writing = Event()
    allow_remove = Event()
    write_metadata = plugins_cmd._write_install_metadata

    def paused_write(metadata):
        if "alpha" not in metadata and "gamma" not in metadata:
            remove_writing.set()
            assert allow_remove.wait(timeout=5)
        write_metadata(metadata)

    monkeypatch.setattr(plugins_cmd, "_write_install_metadata", paused_write)
    alpha = Path(os.environ["HERMES_HOME"]) / "plugins" / "alpha"

    with ThreadPoolExecutor(max_workers=2) as pool:
        removing = pool.submit(plugins_cmd._remove_plugin_core, alpha)
        assert remove_writing.wait(timeout=5)
        installing = pool.submit(
            plugins_cmd.dashboard_install_plugin,
            "",
            force=False,
            enable=False,
            catalog_name="gamma",
            catalog_source=sources["gamma"]["id"],
        )
        allow_remove.set()
        removing.result(timeout=10)
        assert installing.result(timeout=10)["ok"] is True

    metadata = plugins_cmd._read_install_metadata()
    assert set(metadata) == {"beta", "gamma"}
    assert metadata["gamma"]["marketplace_id"] == sources["gamma"]["id"]


def test_concurrent_official_and_custom_installs_keep_provenance_with_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from types import SimpleNamespace

    import hermes_cli.plugins_cmd as plugins_cmd

    official = _marketplace_repo(tmp_path / "official")
    custom = _marketplace_repo(tmp_path / "custom")
    custom_manifest = custom / "plugins" / "demo" / "plugin.json"
    value = json.loads(custom_manifest.read_text(encoding="utf-8"))
    value["description"] = "custom-unreviewed"
    custom_manifest.write_text(json.dumps(value), encoding="utf-8")
    _git(custom, "add", "-A")
    _git(custom, "commit", "-m", "custom")

    custom_scanning = Event()
    allow_custom = Event()
    scan = plugins_cmd._scan_plugin_tree

    def controlled_scan(plugin_dir, identifier, **kwargs):
        scan(plugin_dir, identifier, **kwargs)
        if str(custom) in identifier:
            custom_scanning.set()
            assert allow_custom.wait(timeout=5)

    monkeypatch.setattr(plugins_cmd, "_scan_plugin_tree", controlled_scan)
    entry = SimpleNamespace(
        name="demo",
        repo=f"file://{official}",
        sha=_git(official, "rev-parse", "HEAD"),
        subdir="plugins/demo",
        tier="official",
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        custom_install = pool.submit(
            plugins_cmd._install_plugin_core,
            f"file://{custom}#plugins/demo",
            force=True,
            ref=_git(custom, "rev-parse", "HEAD"),
        )
        assert custom_scanning.wait(timeout=5)
        official_install = pool.submit(
            plugins_cmd._install_plugin_core,
            f"file://{official}#plugins/demo",
            force=True,
            ref=entry.sha,
            catalog_entry=entry,
        )
        official_install.result(timeout=10)
        allow_custom.set()
        custom_install.result(timeout=10)

    target = Path(os.environ["HERMES_HOME"]) / "plugins" / "demo"
    assert (
        json.loads((target / "plugin.json").read_text(encoding="utf-8"))["description"]
        == "custom-unreviewed"
    )
    assert plugins_cmd._read_catalog_sidecar(target) is None


def test_real_private_marketplace_install_and_subtree_update(tmp_path: Path) -> None:
    import hermes_cli.plugins_cmd as plugins_cmd
    from tui_gateway import server

    repo = _marketplace_repo(tmp_path)
    source = add_marketplace(f"file://{repo}", allow_file=True)
    installed = plugins_cmd.dashboard_install_plugin(
        "",
        force=False,
        enable=False,
        catalog_name="demo",
        catalog_source=source["id"],
    )

    assert installed["ok"] is True
    metadata = plugins_cmd._read_install_metadata()["demo"]
    original_tree = metadata["installed_tree_sha"]
    assert metadata["marketplace_id"] == source["id"]
    assert metadata["installed_repo_sha"] == source["entries"][0]["sha"]

    skill = repo / "plugins" / "demo" / "skills" / "demo" / "SKILL.md"
    skill.write_text(
        skill.read_text(encoding="utf-8") + "\nUpdated.\n", encoding="utf-8"
    )
    _git(repo, "add", "plugins/demo")
    _git(repo, "commit", "-m", "update plugin")

    response = server.handle_request({
        "id": "update",
        "method": "plugins.manage",
        "params": {"action": "update", "key": "demo"},
    })

    assert response is not None
    assert response["result"]["ok"] is True
    assert response["result"]["unchanged"] is False
    refreshed = plugins_cmd._read_install_metadata()["demo"]
    assert refreshed["installed_tree_sha"] != original_tree


def test_catalog_sidecar_symlink_does_not_write_external_file(tmp_path: Path) -> None:
    import hermes_cli.plugins_cmd as plugins_cmd

    target = tmp_path / "plugin"
    target.mkdir()
    external = tmp_path / "external.json"
    external.write_text("unchanged", encoding="utf-8")
    (target / plugins_cmd._CATALOG_SIDECAR).symlink_to(external)
    entry = types.SimpleNamespace(
        name="demo",
        repo="https://example.com/demo.git",
        sha="0" * 40,
        tier="official",
    )

    with pytest.raises(plugins_cmd.PluginOperationError, match="must not be a symlink"):
        plugins_cmd._write_catalog_sidecar(target, entry, strict=True)
    assert external.read_text(encoding="utf-8") == "unchanged"


def test_remove_cleanup_failure_does_not_report_committed_removal_as_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_cli.plugins_cmd as plugins_cmd
    import utils

    plugins_dir = Path(os.environ["HERMES_HOME"]) / "plugins"
    target = plugins_dir / "demo"
    target.mkdir(parents=True)
    (target / "plugin.json").write_text('{"name":"demo"}', encoding="utf-8")

    def fail_cleanup(*_args, **_kwargs):
        raise OSError("cleanup fault")

    monkeypatch.setattr(utils.shutil, "rmtree", fail_cleanup)
    plugins_cmd._remove_plugin_core(target)
    assert not target.exists()
    assert not plugins_cmd._install_transaction_path().exists()


def test_platform_toolset_set_and_unset_land_on_disk() -> None:
    from hermes_cli.config import read_raw_config, set_config_value, unset_config_value

    set_config_value("platform_toolsets.cli", '["old"]')
    set_config_value("platform_toolsets.cli", '["new"]')
    assert read_raw_config()["platform_toolsets"]["cli"] == ["new"]

    unset_config_value("platform_toolsets.cli")
    assert "cli" not in read_raw_config().get("platform_toolsets", {})


@pytest.mark.skipif(os.name != "nt", reason="Windows reparse-point semantics")
def test_secure_rmtree_pins_windows_leaf_during_junction_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import utils

    root = tmp_path / "root"
    victim = root / "victim"
    outside = tmp_path / "outside"
    junction = root / "replacement"
    victim.mkdir(parents=True)
    outside.mkdir()
    (victim / "payload").write_text("x", encoding="utf-8")
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
        capture_output=True,
        check=False,
    )
    if created.returncode:
        pytest.skip("junction creation unavailable")

    def race(candidate: Path) -> None:
        with pytest.raises(PermissionError):
            os.rename(candidate, root / "moved")
        (candidate / "payload").unlink()
        raise PermissionError("pinned root blocks final rmdir")

    monkeypatch.setattr(utils.shutil, "rmtree", race)
    utils.secure_rmtree(victim, root)

    assert not victim.exists()
    assert outside.exists()
