"""Tests for DIRECT_ALIASES hot-reload (model.aliases takes effect without restart).

Regression coverage for the load-once bug where `/model <alias>` resolved a stale
alias table because `_ensure_direct_aliases()` populated `DIRECT_ALIASES` only on
first use and never refreshed after a `model.aliases` config edit.
"""
import logging

import pytest

from hermes_cli import model_switch as ms


@pytest.fixture(autouse=True)
def _reset_direct_aliases():
    """Save/restore the module-global DIRECT_ALIASES and the degraded latch
    around each test so state doesn't leak between tests.

    Rebinds (never mutates) — matching the production atomic-swap contract.
    """
    saved = ms.DIRECT_ALIASES
    saved_degraded = ms._DIRECT_ALIASES_DEGRADED
    ms.DIRECT_ALIASES = {}
    ms._DIRECT_ALIASES_DEGRADED = False
    yield
    ms.DIRECT_ALIASES = saved
    ms._DIRECT_ALIASES_DEGRADED = saved_degraded


def _fake_load(monkeypatch, aliases_map):
    """Point _load_direct_aliases at a synthetic config (builtins + given user aliases)."""
    def _loader():
        merged = dict(ms._BUILTIN_DIRECT_ALIASES)
        for k, v in aliases_map.items():
            prov, model = v.split("/", 1)
            merged[k] = ms.DirectAlias(model=model, provider=prov, base_url="")
        return merged, True
    monkeypatch.setattr(ms, "_load_direct_aliases", _loader)


def test_reload_on_config_change(monkeypatch):
    """The core bug: after first population, a changed config value is reflected
    on the NEXT call — with no manual clear. RED on the load-once implementation."""
    _fake_load(monkeypatch, {"zed": "claude-apr/claude-x"})
    ms._ensure_direct_aliases()
    assert ms.DIRECT_ALIASES["zed"] == ms.DirectAlias("claude-x", "claude-apr", "")

    # Config edited: zed now points somewhere else.
    _fake_load(monkeypatch, {"zed": "claude-apr/claude-y"})
    ms._ensure_direct_aliases()
    assert ms.DIRECT_ALIASES["zed"] == ms.DirectAlias("claude-y", "claude-apr", ""), (
        "alias table did not hot-reload after config change (load-once bug)"
    )


def test_resolve_alias_reflects_reload(monkeypatch):
    """End-to-end via resolve_alias(): a config edit changes what /model <alias> resolves to."""
    _fake_load(monkeypatch, {"opustest": "claude-apr/claude-opus-4-8"})
    ms._ensure_direct_aliases()
    assert ms.resolve_alias("opustest", "claude-apr") == (
        "claude-apr", "claude-opus-4-8", "opustest",
    )
    # Re-point the alias to a different provider; must follow.
    _fake_load(monkeypatch, {"opustest": "openai-codex/gpt-5.6-terra"})
    assert ms.resolve_alias("opustest", "claude-apr") == (
        "openai-codex", "gpt-5.6-terra", "opustest",
    )


def test_removed_alias_disappears(monkeypatch):
    """An alias deleted from config is pruned on the next refresh."""
    _fake_load(monkeypatch, {"gone": "claude-apr/claude-x"})
    ms._ensure_direct_aliases()
    assert "gone" in ms.DIRECT_ALIASES
    _fake_load(monkeypatch, {})  # user removed it
    ms._ensure_direct_aliases()
    assert "gone" not in ms.DIRECT_ALIASES
    # Any built-in direct aliases survive the prune (fork ships some; upstream may not).
    for builtin_key in ms._BUILTIN_DIRECT_ALIASES:
        assert builtin_key in ms.DIRECT_ALIASES


def test_degraded_read_retains_last_known_good(monkeypatch, caplog):
    """RC-A: a transient config-read failure must NOT prune user aliases back to
    builtins — the exact wrong-provider symptom this feature fixes."""
    _fake_load(monkeypatch, {"zed": "claude-apr/claude-x"})
    ms._ensure_direct_aliases()
    assert "zed" in ms.DIRECT_ALIASES

    # Next refresh: config read fails (ok=False, builtins only).
    monkeypatch.setattr(
        ms, "_load_direct_aliases", lambda: (dict(ms._BUILTIN_DIRECT_ALIASES), False)
    )
    with caplog.at_level(logging.WARNING, logger=ms.logger.name):
        ms._ensure_direct_aliases()
        # RC-B / Greptile P2: further degraded calls must NOT re-flood the log.
        ms._ensure_direct_aliases()
        ms._ensure_direct_aliases()

    assert ms.DIRECT_ALIASES.get("zed") == ms.DirectAlias("claude-x", "claude-apr", ""), (
        "user alias was pruned to builtins on a degraded config read (MB-2 regression)"
    )
    assert ms.resolve_alias("zed", "claude-apr") == ("claude-apr", "claude-x", "zed")
    # RC-B: the degraded path emits exactly ONE warning across repeated calls.
    warnings = [r for r in caplog.records if "retaining" in r.message]
    assert len(warnings) == 1, (
        f"degraded-retain path logged {len(warnings)} warnings, expected exactly 1 (RC-B/Greptile P2)"
    )


def test_degraded_then_healthy_rearms_warning(monkeypatch, caplog):
    """After recovery, a subsequent degraded read warns again (latch re-arms)."""
    _fake_load(monkeypatch, {"zed": "claude-apr/claude-x"})
    ms._ensure_direct_aliases()

    degraded = lambda: (dict(ms._BUILTIN_DIRECT_ALIASES), False)
    with caplog.at_level(logging.WARNING, logger=ms.logger.name):
        monkeypatch.setattr(ms, "_load_direct_aliases", degraded)
        ms._ensure_direct_aliases()          # warn #1
        _fake_load(monkeypatch, {"zed": "claude-apr/claude-x"})
        ms._ensure_direct_aliases()          # healthy: re-arm
        monkeypatch.setattr(ms, "_load_direct_aliases", degraded)
        ms._ensure_direct_aliases()          # warn #2
    warnings = [r for r in caplog.records if "retaining" in r.message]
    assert len(warnings) == 2


def test_atomic_swap_and_never_empty(monkeypatch):
    """INV-2 (revised for #67007): reload publishes by ATOMIC REPLACEMENT and a
    reader never observes an empty or half-built table.

    The original invariant was "mutate in place, never rebind" (#16767), which
    existed to stop a `from hermes_cli.model_switch import DIRECT_ALIASES`
    consumer being left pointing at a stale empty dict.  Review of #67007
    (teknium1) showed in-place mutation is unsafe under the gateway's
    ``asyncio.to_thread`` /model path: the prune could invalidate a concurrent
    ``resolve_alias()`` iterator.  The stronger, thread-safe invariant is the
    inverse — build a NEW dict and rebind atomically, so a reader holding the
    previous generation iterates an object that is never written to again.
    ``test_no_from_import_of_direct_aliases`` keeps the original hazard closed.
    """
    _fake_load(monkeypatch, {"a": "p/m1"})
    ms._ensure_direct_aliases()
    old_dict = ms.DIRECT_ALIASES
    old_id = id(old_dict)
    old_snapshot = dict(old_dict)

    # Observe the table during the reload: _load_direct_aliases is called first,
    # so snapshot the live dict's length at that moment — it must never be empty.
    seen_lengths = []

    def _observing_loader():
        seen_lengths.append(len(ms.DIRECT_ALIASES))
        merged = dict(ms._BUILTIN_DIRECT_ALIASES)
        merged["a"] = ms.DirectAlias("m2", "p", "")
        merged["b"] = ms.DirectAlias("m3", "p", "")
        return merged, True

    monkeypatch.setattr(ms, "_load_direct_aliases", _observing_loader)
    ms._ensure_direct_aliases()

    # A NEW generation was published (atomic swap, not in-place mutation).
    assert id(ms.DIRECT_ALIASES) != old_id, (
        "DIRECT_ALIASES was mutated in place — a concurrent reverse-lookup "
        "iterator can be invalidated (see #67007 review)"
    )
    # The dict a concurrent reader might still be holding is UNCHANGED, so its
    # iterator stays valid for the whole scan.
    assert old_dict == old_snapshot, (
        "the previously published dict was mutated after publication — "
        "concurrent readers holding it can raise "
        "'dictionary changed size during iteration'"
    )
    # The table was never empty at any observable point.
    assert seen_lengths and all(n > 0 for n in seen_lengths), (
        "DIRECT_ALIASES was empty during reload (concurrency hazard)"
    )
    assert ms.DIRECT_ALIASES["a"] == ms.DirectAlias("m2", "p", "")
    assert "b" in ms.DIRECT_ALIASES


def test_no_from_import_of_direct_aliases():
    """Guards the ORIGINAL #16767 hazard under the new atomic-swap contract.

    Because ``_ensure_direct_aliases()`` now rebinds the module attribute, any
    consumer doing ``from hermes_cli.model_switch import DIRECT_ALIASES`` would
    hold a permanently stale dict.  Consumers must go through the module
    attribute (``model_switch.DIRECT_ALIASES``), as ``hermes_cli/oneshot.py``
    does.  This is a source contract asserted over the real tree via AST, so
    prose mentioning the pattern (like this docstring) is not a false positive.
    """
    import ast
    from pathlib import Path

    root = Path(ms.__file__).resolve().parent.parent
    offenders = []
    scanned = 0
    for path in root.rglob("*.py"):
        if any(p in {"venv", ".venv", "node_modules", "build", "dist"} for p in path.parts):
            continue
        try:
            src = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        scanned += 1
        if "DIRECT_ALIASES" not in src:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if not (node.module or "").endswith("model_switch"):
                continue
            if any(a.name == "DIRECT_ALIASES" for a in node.names):
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")

    assert scanned > 50, f"source scan found only {scanned} files — glob is wrong"
    assert not offenders, (
        "DIRECT_ALIASES is rebound on refresh, so a from-import captures a "
        f"permanently stale table. Use `model_switch.DIRECT_ALIASES`. Offenders: {offenders}"
    )


# ---------------------------------------------------------------------------
# Integration: the REAL mtime-cached load_config() path, no mocks.
#
# Review finding (teknium1 on #67007): every test above monkeypatches
# _load_direct_aliases(), so none of them exercise the real
# hermes_cli.config.load_config() mtime cache that makes per-call refresh
# affordable.  These tests write an actual config.yaml under a temp
# HERMES_HOME and assert resolve_alias() follows edits with no process restart.
# ---------------------------------------------------------------------------

@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """A real, isolated HERMES_HOME with a real config.yaml on disk.

    Clears hermes_cli.config's module caches so the (mtime_ns, size) key can't
    carry over from another test's config path.
    """
    import hermes_cli.config as hcfg

    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    monkeypatch.delenv("HERMES_CONFIG", raising=False)

    for cache_name in ("_LOAD_CONFIG_CACHE", "_READ_RAW_CONFIG_CACHE"):
        cache = getattr(hcfg, cache_name, None)
        if isinstance(cache, dict):
            cache.clear()

    cfg_path = hcfg.get_config_path()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    assert str(home) in str(cfg_path), (
        f"config path {cfg_path} did not land under the temp HERMES_HOME {home} "
        "— the test is not hermetic"
    )
    yield cfg_path
    for cache_name in ("_LOAD_CONFIG_CACHE", "_READ_RAW_CONFIG_CACHE"):
        cache = getattr(hcfg, cache_name, None)
        if isinstance(cache, dict):
            cache.clear()


def _write_config(cfg_path, aliases: dict) -> None:
    """Write a real config.yaml with a ``model.aliases`` map, bumping mtime.

    The loader's cache key is (mtime_ns, size); a rewrite inside the same
    filesystem-timestamp granularity could otherwise reuse the cached parse, so
    stamp the mtime forward explicitly (this is what a real editor/`hermes
    config set` does over human timescales).
    """
    import os
    import time

    import yaml

    body = {"model": {"provider": "openrouter", "aliases": dict(aliases)}}
    cfg_path.write_text(yaml.safe_dump(body), encoding="utf-8")
    bumped = time.time() + _write_config.counter
    _write_config.counter += 1
    os.utime(cfg_path, (bumped, bumped))


_write_config.counter = 1


def test_integration_real_config_hot_reload(hermes_home):
    """END-TO-END, no mocks: editing a real config.yaml changes what
    resolve_alias() returns, in the same process, with no restart.

    Exercises the real hermes_cli.config.load_config() mtime cache — the path
    the mocked tests above never touch.
    """
    cfg_path = hermes_home

    # 1. Initial config on disk.
    _write_config(cfg_path, {"zed": "claude-apr/claude-x"})
    assert ms.resolve_alias("zed", "openrouter") == ("claude-apr", "claude-x", "zed")

    # 2. Repeat lookups are served from the mtime cache and stay correct.
    for _ in range(3):
        assert ms.resolve_alias("zed", "openrouter") == ("claude-apr", "claude-x", "zed")

    # 3. The user edits the file — re-pointing the alias at another provider.
    _write_config(cfg_path, {"zed": "openai-codex/gpt-5.6-terra"})
    assert ms.resolve_alias("zed", "openrouter") == (
        "openai-codex", "gpt-5.6-terra", "zed",
    ), "resolve_alias did not follow a real config.yaml edit (hot-reload broken)"

    # 4. The user adds a second alias; both resolve.
    _write_config(cfg_path, {
        "zed": "openai-codex/gpt-5.6-terra",
        "zzalias": "moonshot/zz-model-9",
    })
    assert ms.resolve_alias("zzalias", "openrouter") == ("moonshot", "zz-model-9", "zzalias")
    assert ms.resolve_alias("zed", "openrouter") == (
        "openai-codex", "gpt-5.6-terra", "zed",
    )

    # 5. Reverse lookup by full model id also follows the real file.
    assert ms.resolve_alias("zz-model-9", "openrouter") == (
        "moonshot", "zz-model-9", "zzalias",
    )

    # 6. The user deletes an alias — it stops resolving (pruned, not sticky).
    #    resolve_alias() triggers the refresh, so drive it through the public API.
    _write_config(cfg_path, {"zed": "openai-codex/gpt-5.6-terra"})
    assert ms.resolve_alias("zzalias", "openrouter") is None, (
        "a deleted alias still resolved after a real-config refresh"
    )
    assert "zzalias" not in ms.DIRECT_ALIASES, (
        "a deleted alias survived a real-config refresh"
    )
    # ...and the surviving alias is untouched by the prune.
    assert ms.resolve_alias("zed", "openrouter") == (
        "openai-codex", "gpt-5.6-terra", "zed",
    )


def test_integration_real_loader_is_actually_exercised(hermes_home, monkeypatch):
    """Proves these integration tests run the REAL loader, not a mock.

    Two independent proofs, both of which a mocked ``_load_direct_aliases()``
    would make impossible:

    1. The alias that ``resolve_alias()`` returns is the one *written to disk* —
       change the bytes in the file, the answer changes.
    2. Breaking the real ``hermes_cli.config.load_config()`` (the mtime-cached
       function under test) makes the loader report ``ok=False``; the refresh
       then retains its last-known-good table instead of pruning to builtins.
       If the loader were mocked out, breaking ``load_config`` would be inert.

    Note ``load_config()`` deliberately tolerates a *corrupt* config.yaml
    (it keeps the previously loaded config and warns), so corrupting the YAML
    is NOT a way to reach the degraded branch — the degraded branch is for a
    read that actually raises.
    """
    import hermes_cli.config as hcfg

    cfg_path = hermes_home

    # --- Proof 1: the answer tracks the real bytes on disk. ---
    _write_config(cfg_path, {"realpath": "claude-apr/claude-x"})
    merged, ok = ms._load_direct_aliases()
    assert ok is True
    assert merged["realpath"] == ms.DirectAlias("claude-x", "claude-apr", ""), (
        "the real loader did not read the alias written to the real config.yaml"
    )
    assert cfg_path.exists() and "realpath" in cfg_path.read_text(encoding="utf-8")

    ms._ensure_direct_aliases()
    assert ms.DIRECT_ALIASES["realpath"] == ms.DirectAlias("claude-x", "claude-apr", "")

    # --- Proof 2: break the REAL load_config() the loader calls. ---
    real_load_config = hcfg.load_config

    def _boom():
        raise OSError("simulated config read failure")

    hcfg.load_config = _boom
    try:
        _merged_bad, ok_bad = ms._load_direct_aliases()
        assert ok_bad is False, (
            "breaking the real hermes_cli.config.load_config() did not degrade "
            "_load_direct_aliases() — this test is not exercising the real path"
        )

        # Degraded read retains last-known-good rather than pruning to builtins.
        ms._ensure_direct_aliases()
        assert ms.DIRECT_ALIASES["realpath"] == ms.DirectAlias("claude-x", "claude-apr", ""), (
            "degraded real-config read pruned user aliases (MB-2 regression)"
        )
    finally:
        hcfg.load_config = real_load_config

    # Restore the real loader: the refresh recovers from the real file on disk.
    merged_ok, ok_again = ms._load_direct_aliases()
    assert ok_again is True
    assert merged_ok["realpath"] == ms.DirectAlias("claude-x", "claude-apr", "")
