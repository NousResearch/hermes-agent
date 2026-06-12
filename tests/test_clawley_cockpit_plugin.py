from __future__ import annotations

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_plugin_api():
    path = ROOT / "plugins" / "clawley-cockpit" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location("clawley_cockpit_plugin_api", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_clawley_cockpit_plugin_status_is_read_only() -> None:
    module = _load_plugin_api()

    status = module.build_status_snapshot()

    assert status["schema"] == "clawley_cockpit_status.v1"
    assert status["write_performed"] is False
    assert status["safety_flags"]["read_only"] is True
    assert status["sections"]["quantos"]["status"] in {"not_configured", "available"}
