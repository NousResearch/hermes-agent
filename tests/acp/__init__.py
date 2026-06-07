"""ACP test package marker without shadowing ``agent-client-protocol``.

Pytest can collect this directory either as ``tests.acp`` or, depending on
import path ordering, as top-level ``acp``. The latter shadows the real ACP
package from ``agent-client-protocol`` and breaks imports such as
``from acp.schema import ...``. If that happens, execute the installed ACP
package's ``__init__`` in this module namespace so tests still see the real
public API.
"""

if __name__ == "acp":
    import sys
    from pathlib import Path

    _here = Path(__file__).resolve()
    _real_init: Path | None = None
    for _entry in sys.path:
        if not _entry:
            continue
        _candidate = Path(_entry).resolve() / "acp" / "__init__.py"
        if _candidate.exists() and _candidate.resolve() != _here:
            _real_init = _candidate.resolve()
            break

    if _real_init is None:  # pragma: no cover - dependency/setup failure
        raise ImportError("agent-client-protocol package is required for ACP tests")

    __file__ = str(_real_init)
    __path__ = [str(_real_init.parent)]  # type: ignore[name-defined]
    exec(compile(_real_init.read_text(encoding="utf-8"), str(_real_init), "exec"), globals())
