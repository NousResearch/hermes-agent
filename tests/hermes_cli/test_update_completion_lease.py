"""L3: the Windows bootstrap completion child joins the checkout lock (with its own R5b lease)
before it writes the checkout."""

import os
import sys
from pathlib import Path

from hermes_cli import update_completion


class _Win32Sys:
    """``sys`` as update_completion sees it on Windows; everything but ``platform`` is real."""

    platform = "win32"

    def __getattr__(self, name):
        return getattr(sys, name)


def _fake_windows_prepare(tmp_path, monkeypatch, acquire):
    """Run the bootstrap's ``_prepare`` as on Windows, with PM, venv_sync and the --prepared
    child replaced by recorders; returns the ordered events."""
    from contextlib import nullcontext
    import pm
    import pm.client
    import pm.environments
    import pm.receipt
    from hermes_cli import update_lock, venv_sync

    events = []
    monkeypatch.setattr(update_completion, "sys", _Win32Sys())
    monkeypatch.setattr(update_lock, "_acquire_checkout", lambda root: acquire(events, Path(root)))
    for name in ("arm_completion", "collect_superseded_generations"):
        monkeypatch.setattr(venv_sync, name, lambda root, _name=name: events.append(_name))
    monkeypatch.setattr(pm.client, "ensure_tools_for_sync", lambda: events.append("tools"))
    monkeypatch.setattr(pm, "sync_venv", lambda **kw: events.append("sync_venv"))
    monkeypatch.setattr(pm.receipt, "worker_context", lambda update_id: nullcontext())
    monkeypatch.setattr(pm.receipt, "last_for_update", lambda update_id: None)
    monkeypatch.setattr(pm.environments, "project_python", lambda root: Path(sys.executable))
    monkeypatch.setattr(pm.environments, "activation_environment", lambda root: dict(os.environ))
    result_path = tmp_path / "result.json"

    def prepared_child(*args, **kwargs):  # a successful --prepared child writes its result
        events.append("prepared")
        result_path.write_text("{}", encoding="utf-8")
        return 0

    monkeypatch.setattr(update_completion.subprocess, "call", prepared_child)
    root = tmp_path / "checkout"
    root.mkdir()
    request = {"source": str(root), "receipt": {"update_id": "u-l3"}, "bytecode_cache": str(tmp_path / "bc")}
    try:
        update_completion._prepare(request, tmp_path / "request.json", result_path)
    except RuntimeError as exc:
        events.append(f"raised: {exc}")
    return root, events


def test_windows_bootstrap_holds_a_checkout_lease_before_it_writes_the_checkout(tmp_path, monkeypatch):
    """L3: a bootstrap child the job refused runs outside the kill-on-close job and outlives a
    killed owner; arm_completion, the sync's uv children and the generation collector must
    not run before it joined the checkout lock with a lease of its own (R5b)."""
    root, events = _fake_windows_prepare(
        tmp_path, monkeypatch, lambda events, root: events.append(("lease", root)))
    assert events[0] == ("lease", root), events
    assert events[1:] == ["arm_completion", "tools", "sync_venv", "collect_superseded_generations",
                          "prepared"]


def test_windows_bootstrap_that_cannot_join_the_checkout_lock_writes_nothing(tmp_path, monkeypatch):
    """L3: refused the join (a foreign update owns the checkout), the bootstrap stops before
    arming the completion or touching the venv, never an unfenced writer."""
    from hermes_cli.update_lock import UpdateHolder

    _root, events = _fake_windows_prepare(
        tmp_path, monkeypatch, lambda events, root: UpdateHolder(pid=4242, age_seconds=0.0))
    assert events == ["raised: could not join the update's checkout lock (held by process 4242)"]
