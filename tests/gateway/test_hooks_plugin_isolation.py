"""Gateway HOOK.yaml handlers load the same way under every ``plugins.isolation`` mode.

The hook loader has never checked ``plugins.isolation``: a profile's ``hooks/<name>/handler.py`` is
imported into the gateway process under ``host`` exactly as under ``in_process``.
"""

import sys
from unittest.mock import patch

import hermes_yaml as yaml
import pytest

from gateway.hooks import HookRegistry
from hermes_constants import get_hermes_home

MODULE = "hermes_hook_isoprobe"
# Module body records whether it ran registered in sys.modules (forward references need that).
HANDLER = (
    "import sys\n"
    "from pathlib import Path\n"
    "Path(__file__).with_name('imported').write_text(str(sys.modules.get(__name__) is not None))\n"
    "def handle(event_type, context):\n"
    "    return 'user-hook'\n"
)


@pytest.mark.asyncio
@pytest.mark.parametrize("isolation", ["host", "in_process"])
async def test_user_hook_handler_loads_in_the_gateway_under_every_isolation_mode(tmp_path, isolation):
    (get_hermes_home() / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"isolation": isolation}}), encoding="utf-8")
    hook_dir = tmp_path / "hooks" / "isoprobe"
    hook_dir.mkdir(parents=True)
    (hook_dir / "HOOK.yaml").write_text("name: isoprobe\nevents: [\"agent:start\"]\n", encoding="utf-8")
    (hook_dir / "handler.py").write_text(HANDLER, encoding="utf-8")
    sys.modules.pop(MODULE, None)
    try:
        reg = HookRegistry()
        with patch("gateway.hooks.HOOKS_DIR", tmp_path / "hooks"):
            reg.discover_and_load()

        assert (hook_dir / "imported").read_text() == "True"
        assert sys.modules[MODULE].__file__ == str(hook_dir / "handler.py")
        assert [h["name"] for h in reg.loaded_hooks] == ["isoprobe"]
        assert await reg.emit_collect("agent:start", {}) == ["user-hook"]
    finally:
        sys.modules.pop(MODULE, None)
