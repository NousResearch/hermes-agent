"""hongxing-enhancements plugin shim for project-local loading."""

import importlib.util
import os
import sys
from typing import Callable


_SRC_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_module(filename: str):
    path = os.path.join(_SRC_DIR, filename)
    if not os.path.isfile(path):
        return None
    mod_name = "hongxing_" + filename.replace(".py", "")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _dynamic_hook(filename: str, attr_name: str) -> Callable:
    def _hook(*args, **kwargs):
        mod = _load_module(filename)
        if mod is None:
            return None
        callback = getattr(mod, attr_name, None)
        if callback is None:
            return None
        return callback(*args, **kwargs)

    _hook.__name__ = f"dynamic_{filename.replace('.py', '')}_{attr_name}"
    return _hook


def register(ctx):
    """Called by PluginManager to register hooks."""
    for filename in ("permission_engine.py", "plan_mode_hook.py"):
        if os.path.isfile(os.path.join(_SRC_DIR, filename)):
            ctx.register_hook("pre_tool_call", _dynamic_hook(filename, "pre_tool_call"))

    mod = _load_module("smart_recall.py")
    if mod is None:
        return

    recall = mod.SmartRecall()
    ctx.smart_recall = recall
