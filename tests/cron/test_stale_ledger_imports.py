"""Ledger consumers must survive a checkout update in a long-running scheduler."""

import builtins
import importlib
import sys
from types import ModuleType

import pytest


LEDGER_HELPERS = ("ledger_transaction", "open_ledger", "prepare_ledger")


@pytest.fixture
def ledger_cache():
    import cron

    executions = importlib.import_module("cron.executions")
    snapshot = dict(sys.modules)
    package_snapshot = dict(vars(cron))
    try:
        yield cron, executions
    finally:
        # Restore both caches so later imports and existing monkeypatch targets agree.
        sys.modules.update(snapshot)
        for name in list(sys.modules):
            if name not in snapshot:
                del sys.modules[name]
        vars(cron).clear()
        vars(cron).update(package_snapshot)


@pytest.mark.parametrize("consumer", ["incidents", "notepad"])
@pytest.mark.parametrize(
    "missing_helpers",
    [LEDGER_HELPERS, ("open_ledger",), ("prepare_ledger",), ()],
    ids=["pre-refactor", "missing-open", "missing-prepare", "current-cache"],
)
def test_ledger_operations_survive_cached_executions(
    consumer, missing_helpers, ledger_cache, tmp_path, monkeypatch,
):
    cron, executions = ledger_cache
    cached = executions
    if missing_helpers:
        cached = ModuleType("cron.executions")
        cached.EXECUTIONS_FILE = None
        for name in LEDGER_HELPERS:
            if name not in missing_helpers:
                setattr(cached, name, getattr(executions, name))
    sys.modules["cron.executions"] = cached
    cron.executions = cached
    sys.modules.pop(f"cron.{consumer}", None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    module = importlib.import_module(f"cron.{consumer}")
    current = importlib.import_module("cron.executions")
    if missing_helpers:
        assert current is not cached
    else:
        assert current is cached
    for name in LEDGER_HELPERS:
        assert getattr(module, name) is getattr(current, name)

    if consumer == "incidents":
        # The incident store must follow the refreshed ledger's path override too.
        path = tmp_path / "redirected" / "executions.db"
        monkeypatch.setattr(current, "EXECUTIONS_FILE", path)
        incident_id, is_new = module.upsert_incident("job-1", "script failed")
        assert is_new
        assert module.get_incident(incident_id)["error"] == "script failed"
        assert path.is_file()
    else:
        module.set_note("job-1", "cursor", "page=7")
        assert module.get_note("job-1", "cursor") == "page=7"
        assert (tmp_path / "cron" / "notepad.db").is_file()


@pytest.mark.parametrize("consumer", ["incidents", "notepad"])
def test_failed_ledger_retry_surfaces_original_import_error(
    consumer, ledger_cache, monkeypatch,
):
    cron, _ = ledger_cache
    stale = ModuleType("cron.executions")
    sys.modules["cron.executions"] = stale
    cron.executions = stale
    sys.modules.pop(f"cron.{consumer}", None)
    original_error = ImportError("cached executions has no ledger_transaction")
    real_import = builtins.__import__
    attempts = []

    def failing_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cron.executions" and "ledger_transaction" in fromlist:
            attempts.append(sys.modules.get(name))
            if len(attempts) == 1:
                raise original_error
            raise ImportError("on-disk ledger import also failed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    with pytest.raises(ImportError) as caught:
        importlib.import_module(f"cron.{consumer}")

    assert caught.value is original_error
    assert attempts == [stale, None]
    assert f"cron.{consumer}" not in sys.modules
