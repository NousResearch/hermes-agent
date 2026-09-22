"""The post-swap child must start in the checkout, not the caller's cwd.

A stale setuptools finder cannot import a new top-level package. ``python -m``
only adds the process cwd to ``sys.path``, so a hand-off spawned from outside
the checkout dies in ``import hermes_cli.main`` before the dependency sync can
refresh the map.
"""

import sys

import hermes_cli.update_handoff as handoff


class _Child:
    def wait(self, timeout=None):
        return 0


def test_both_handoff_launches_use_the_checkout_cwd(monkeypatch, tmp_path):
    recorded = []
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    assert handoff._post_swap_cwd() == str(handoff.Path(handoff.__file__).resolve().parent.parent)

    monkeypatch.setattr(handoff, "write_handoff", lambda payload: tmp_path / "handoff.json")
    monkeypatch.setattr(handoff, "post_swap_python", lambda: sys.executable)
    monkeypatch.setattr(handoff, "post_swap_child_env", lambda: {"HERMES_UPDATE_POST_SWAP": "1"})
    monkeypatch.setattr(handoff, "detached_shim_child_env", lambda env: env)
    monkeypatch.setattr(handoff, "_post_swap_cwd", lambda: str(checkout))
    monkeypatch.setattr(
        handoff.subprocess,
        "Popen",
        lambda cmd, **kwargs: recorded.append(kwargs.get("cwd")) or _Child(),
    )

    monkeypatch.setattr(handoff, "_running_from_windows_shim", lambda: False)
    assert handoff.continue_update_in_fresh_interpreter({}, argv_tail=["--yes"]) == 0

    monkeypatch.setattr(handoff, "_running_from_windows_shim", lambda: True)
    assert handoff.continue_update_in_fresh_interpreter({}, argv_tail=["--yes"]) == 0

    assert recorded == [str(checkout), str(checkout)]
