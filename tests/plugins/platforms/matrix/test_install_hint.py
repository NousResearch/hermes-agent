"""Matrix remediation hints must be actionable in a PM-managed install.

PM owns Hermes environments: a ``pip install`` hint tells the user to mutate an
environment PM will flag or overwrite on the next sync, and the mautrix
``[encryption]`` extra builds python-olm from source against the libolm C
library — the hint must point at ``hermes setup``, the supported install
channel. Regression for #146281.
"""
from __future__ import annotations


def test_registry_install_hint_points_at_hermes_setup():
    captured: dict = {}

    class _Ctx:
        def register_platform(self, **kwargs):
            captured.update(kwargs)

    from plugins.platforms.matrix.adapter import register
    register(_Ctx())
    hint = captured["install_hint"]
    assert "hermes setup" in hint
    assert "pip install" not in hint
    assert "mautrix[encryption]" in hint


def test_e2ee_hint_points_at_hermes_setup():
    from plugins.platforms.matrix.adapter import _E2EE_INSTALL_HINT
    assert "hermes setup" in _E2EE_INSTALL_HINT
    assert "pip install" not in _E2EE_INSTALL_HINT