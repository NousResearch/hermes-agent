"""Path bootstrap for content_engine tests.

content_engine/ is a standalone package tree whose modules use flat imports
(``import approval_state``, ``from blog.blog_generator import ...``) resolved
from the content_engine root. This conftest puts that root on sys.path so the
suite runs identically under any runner (pytest from anywhere, run_tests.sh,
IDE test discovery) without PYTHONPATH ceremony.
"""

from __future__ import annotations

import sys
from pathlib import Path

_CE_ROOT = Path(__file__).resolve().parent.parent
if str(_CE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CE_ROOT))
