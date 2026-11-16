"""Built-in ``cua`` computer-use provider: cua-driver over MCP (:mod:`tools.computer_use.cua_backend`)."""

from __future__ import annotations

import sys

from tools.computer_use.backend import ComputerUseBackend, ComputerUseProvider


class CuaDriverProvider(ComputerUseProvider):
    name = "cua"
    display_name = "cua-driver (background)"

    def create_backend(self, *, permission_mode: str) -> ComputerUseBackend:
        from tools.computer_use.cua_backend import CuaDriverBackend
        return CuaDriverBackend(permission_mode=permission_mode)

    def is_available(self) -> bool:
        """macOS/Windows/Linux + cua-driver binary (or env override). `hermes computer-use doctor` names blocked checks."""
        if sys.platform not in ("darwin", "win32", "linux"):
            return False
        from tools.computer_use.cua_backend_driver import cua_driver_binary_available
        if cua_driver_binary_available():
            return True
        # No host driver: the tool is still real when the desktop is placed inside a terminal backend whose image
        # carries cua-driver (nousresearch/hermes-sandbox:desktop). Placement is config; the binary is probed lazily
        # at first use, so this stays a cheap check_fn.
        from tools.bot_desktop import placement
        return placement.resolve().where == placement.TERMINAL


def register(ctx):
    ctx.register_computer_use_provider(CuaDriverProvider())
