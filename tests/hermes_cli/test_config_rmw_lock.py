"""Concurrent config writers must not drop each other's mutations.

Regression for the settings-pipeline audit Bug 4: `/api/model/set`,
`/api/model/moa`, the custom-endpoint handlers, and the memory-provider
saves ran their load→mutate→save cycles WITHOUT `_CONFIG_MUTATION_LOCK`
(only `PUT /api/config` took it). They execute in worker threads (sync-def
FastAPI endpoints / asyncio.to_thread), so a model assignment racing the
desktop's debounced whole-record autosave could interleave:

    T1 load (model=A)      T2 load (model=A)
    T1 mutate model=B      T2 mutate moa=X
    T1 save (model=B)      T2 save (model=A + moa=X)   ← T1's write erased

The fix wraps every RMW span in the RLock. This test provokes the exact
interleaving with a slowed save_config and asserts both mutations survive.
"""

import importlib
import sys
import threading
import time

import pytest


@pytest.fixture()
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n  provider: openrouter\n  default: some/model\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    for mod in ("hermes_cli.config",):
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])
    return home


def test_model_set_and_moa_save_do_not_drop_each_other(isolated_home, monkeypatch):
    import hermes_cli.web_server as ws
    from hermes_cli import config as cfg_mod

    real_save = cfg_mod.save_config
    real_load = cfg_mod.load_config

    # Slow the save so an unlocked interleaving would reliably lose a write.
    def slow_save(cfg, **kwargs):
        time.sleep(0.15)
        return real_save(cfg, **kwargs)

    monkeypatch.setattr(ws, "save_config", slow_save)
    monkeypatch.setattr(ws, "load_config", real_load)

    errors: list[BaseException] = []

    def assign_model():
        try:
            ws._apply_model_assignment_sync("main", "openrouter", "anthropic/claude-fable-5", "", "")
        except BaseException as exc:  # noqa: BLE001 - captured for the assertion
            errors.append(exc)

    def write_other_key():
        try:
            # Same RMW shape as the moa/memory handlers: load, mutate an
            # unrelated top-level key, save — through the same lock.
            with ws._CONFIG_MUTATION_LOCK:
                cfg = real_load()
                cfg["display"] = {**(cfg.get("display") or {}), "personality": "canary"}
                slow_save(cfg)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    t1 = threading.Thread(target=assign_model)
    t2 = threading.Thread(target=write_other_key)
    t1.start()
    t2.start()
    t1.join(timeout=30)
    t2.join(timeout=30)

    assert not errors, errors

    text = (isolated_home / "config.yaml").read_text(encoding="utf-8")
    # BOTH mutations must survive — an unlocked interleave drops one.
    assert "anthropic/claude-fable-5" in text
    assert "personality: canary" in text


def test_rmw_lock_covers_the_flagged_handlers():
    """Source contract: every audited RMW handler holds the span lock.

    Cheap tripwire so a refactor can't silently drop the lock from one of
    the six sites the audit flagged (model.set wrapper, moa, custom-endpoint
    upsert/activate/delete, memory-provider saves, profile-dir model write).
    """
    import inspect

    import hermes_cli.web_server as ws

    for fn in (
        ws._apply_model_assignment_sync,
        ws.set_moa_models,
        ws.upsert_custom_endpoint,
        ws.activate_custom_endpoint,
        ws.delete_custom_endpoint,
        ws._update_memory_provider_config,
    ):
        source = inspect.getsource(fn)
        assert "_CONFIG_MUTATION_LOCK" in source, f"{fn.__name__} lost the RMW lock"
