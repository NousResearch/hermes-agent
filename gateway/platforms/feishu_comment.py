"""Backward-compatibility shim for the pre-Phase-7 path.

Phase 7 (Task 7) physically moved this module to
``gateway/platforms/feishu/comments.py``. The shim keeps the original
``gateway.platforms.feishu_comment`` import path alive so external callers
(and any docs that reference the legacy path) continue to work transparently.
All names re-export verbatim from the new module location.
"""
from gateway.platforms.feishu.comments import *  # noqa: F401, F403
from gateway.platforms.feishu import comments as _comments_mod  # noqa: F401

# Re-export the module's named attributes (including underscored helpers) for
# import compatibility. Patch the canonical module path
# ``gateway.platforms.feishu.comments.<name>`` when a test needs to affect
# functions called inside that module.
import sys as _sys

_target = _sys.modules["gateway.platforms.feishu.comments"]
for _name in dir(_target):
    if _name.startswith("__"):
        continue
    globals().setdefault(_name, getattr(_target, _name))
del _sys, _target, _name
