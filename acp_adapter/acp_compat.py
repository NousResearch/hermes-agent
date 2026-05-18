"""ACP import shim.

Prefer the external ``agent-client-protocol`` package.  Lean source/test
environments may not install that optional dependency, so fall back to the
private in-repo implementation without exposing a top-level ``acp`` package.
"""

from __future__ import annotations

try:
    import acp as acp  # type: ignore[import-not-found]
    from acp import *  # type: ignore[import-not-found,import-not-found]
    from acp.exceptions import RequestError  # type: ignore[import-not-found]
    from acp.schema import *  # type: ignore[import-not-found,import-not-found]
    try:
        from acp.schema import AuthMethodAgent  # type: ignore[import-not-found]
    except ImportError:
        from acp.schema import AuthMethod as AuthMethodAgent  # type: ignore[attr-defined,import-not-found]
except ImportError:
    import _acp_fallback as acp  # type: ignore[no-redef]
    from _acp_fallback import *  # type: ignore[import-not-found,import-not-found]
    from _acp_fallback.exceptions import RequestError
    from _acp_fallback.schema import *  # type: ignore[import-not-found,import-not-found]
    from _acp_fallback.schema import AuthMethodAgent
