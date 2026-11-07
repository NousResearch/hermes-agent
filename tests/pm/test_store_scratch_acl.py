"""A staging directory must stay usable by the invoking user after an elevated install.

``tempfile.mkdtemp`` asks for mode 0o700; on Windows that is a protected DACL naming only
SYSTEM, Administrators and OWNER RIGHTS. Under an elevated installer the owner is
BUILTIN\\Administrators, so the same user's filtered token matches nothing and everything
published out of that staging tree is unreachable unelevated (pm/store.py::scratch).
"""

from __future__ import annotations

import os

from pm.store import Store


def test_scratch_dir_is_kept_usable(tmp_path, monkeypatch):
    calls = []

    import hermes_dir_acl

    def fake_make_user_dir(path, **kwargs):
        calls.append((path, kwargs))
        return path

    monkeypatch.setattr(hermes_dir_acl, "make_user_dir", fake_make_user_dir)

    store = Store(tmp_path / "store")
    with store.scratch() as scratch:
        assert scratch.is_dir()
        staged = scratch

    if os.name == "nt":
        # Granted once, on the freshly created directory, as an existing leaf.
        assert calls == [(staged, {"exist_ok": True})]
    else:
        assert calls == []
    # Staging cleanup is unchanged.
    assert not staged.exists()
