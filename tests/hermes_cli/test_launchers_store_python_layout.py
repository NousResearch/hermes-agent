"""``_store_python`` must find the managed interpreter in a nested Termux payload.

A Termux/Android runtime payload unpacks a whole ``$PREFIX`` tree inside the
entry directory, so the interpreter sits at
``<entry>/data/data/com.termux/files/usr/bin/python3`` rather than
``<entry>/bin/python3``. When the flat path is the only candidate the store
reads as having no managed Python and the launcher falls back to composing the
store's ``site-packages`` onto the host interpreter, which surfaces as
``did not return an extension module`` for every C extension.
"""
import json

from hermes_cli import _launchers


def _write_facts(runtime, entry):
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "facts.json").write_text(
        json.dumps({"packages": {"python": {"entry": entry}}}), encoding="utf-8"
    )


def test_store_python_finds_nested_prefix_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(_launchers, "_is_windows", lambda: False)
    runtime = tmp_path / "tools"
    entry = "python-3.14.7-linux-arm64-bionic"
    exe = runtime / entry / "data/data/com.termux/files/usr/bin/python3"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    _write_facts(runtime, entry)

    assert _launchers._store_python(runtime) == exe


def test_store_python_prefers_flat_layout_when_present(tmp_path, monkeypatch):
    monkeypatch.setattr(_launchers, "_is_windows", lambda: False)
    runtime = tmp_path / "tools"
    entry = "python-3.14.7-linux-x86_64"
    flat = runtime / entry / "bin/python3"
    flat.parent.mkdir(parents=True)
    flat.write_text("")
    nested = runtime / entry / "data/data/com.termux/files/usr/bin/python3"
    nested.parent.mkdir(parents=True)
    nested.write_text("")
    _write_facts(runtime, entry)

    assert _launchers._store_python(runtime) == flat


def test_store_python_returns_none_without_any_interpreter(tmp_path, monkeypatch):
    monkeypatch.setattr(_launchers, "_is_windows", lambda: False)
    runtime = tmp_path / "tools"
    entry = "python-3.14.7-linux-arm64-bionic"
    (runtime / entry).mkdir(parents=True)
    _write_facts(runtime, entry)

    assert _launchers._store_python(runtime) is None
