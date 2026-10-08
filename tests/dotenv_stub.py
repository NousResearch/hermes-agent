"""Package-shaped ``dotenv`` stub shared by the gateway/provider tests.

``gateway.run`` calls ``load_hermes_dotenv(project_env=<repo>/.env)`` at import
time, so the tests that import it patch ``sys.modules["dotenv"]`` to keep the
checkout's launch env out of the test process. A bare
``types.ModuleType("dotenv")`` exposing only ``load_dotenv`` is not enough when
the checkout has a project ``.env``: ``hermes_cli.env_loader``'s
``_load_dotenv_with_fallback`` re-parses that file with
``from dotenv.main import DotEnv`` and ``from dotenv.variables import
parse_variables``, and a non-package stub raises ``ModuleNotFoundError: No
module named 'dotenv.main'; 'dotenv' is not a package``. A fresh clone has no
``.env``, so the fallback is never reached there and CI never saw this.

``install()`` keeps the original no-op ``load_dotenv`` and adds package-shaped
``dotenv.main`` / ``dotenv.variables`` submodules whose parser yields no
assignments: the fallback path stays inert and nothing is loaded into
``os.environ``.
"""

from __future__ import annotations

import sys
import types

__all__ = ["install"]


class _NoopDotEnv:
    """Stand-in for ``dotenv.main.DotEnv`` that parses nothing."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def parse(self):
        return iter(())


def _build_stub() -> types.ModuleType:
    fake = types.ModuleType("dotenv")
    fake.__path__ = []  # package-shaped: ``import dotenv.main`` must resolve
    setattr(fake, "load_dotenv", lambda *args, **kwargs: None)
    main = types.ModuleType("dotenv.main")
    setattr(main, "DotEnv", _NoopDotEnv)
    variables = types.ModuleType("dotenv.variables")
    setattr(variables, "parse_variables", lambda value: [])
    setattr(fake, "main", main)
    setattr(fake, "variables", variables)
    return fake


def install(monkeypatch=None) -> types.ModuleType:
    """Register the stub in ``sys.modules``, overriding any real dotenv.

    Pass the test's ``monkeypatch`` fixture to have the stub undone at teardown;
    without it the stub stays registered for the rest of the process (what the
    module-level provider tests did before this helper existed).
    """
    fake = _build_stub()
    modules = {
        "dotenv": fake,
        "dotenv.main": fake.main,
        "dotenv.variables": fake.variables,
    }
    if monkeypatch is None:
        sys.modules.update(modules)
    else:
        for name, module in modules.items():
            monkeypatch.setitem(sys.modules, name, module)
    return fake
