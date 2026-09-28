"""Re-export shim: the canonical taste tool lives in ``tools/taste_tool.py``.

``agent/tools/`` does not exist as a tool-discovery directory (the registry
auto-discovers ``tools/*.py``); this shim exists so the spec path
``agent.tools.taste_tool`` stays importable and exposes the same surface.
"""

from tools.taste_tool import *  # noqa: F401,F403
from tools.taste_tool import (  # noqa: F401
    taste_tool,
    taste_learn,
    taste_forget,
    taste_summary,
    taste_write,
    write_taste_md,
    get_engine,
    _save_state,
    resolve_taste_dir,
    TASTE_SCHEMA,
    DEFAULT_TASTE_CFG,
)
