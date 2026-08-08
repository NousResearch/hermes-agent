"""Provider discovery must execute only package entrypoints and requested imports."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import pytest

from plugins.context_engine import _load_engine_from_dir
from plugins.cron_providers import _load_provider_from_dir as _load_cron_provider
from plugins.memory import _load_provider_from_dir as _load_memory_provider


@pytest.mark.parametrize(
    ("loader", "module_root", "register_method"),
    [
        (_load_memory_provider, "_hermes_user_memory", "register_memory_provider"),
        (_load_engine_from_dir, "plugins.context_engine", "register_context_engine"),
        (_load_cron_provider, "_hermes_user_cron", "register_cron_scheduler"),
    ],
    ids=("memory", "context-engine", "cron"),
)
def test_loader_executes_entrypoint_and_requested_relative_import_only(
    tmp_path: Path,
    loader: Callable[[Path], object | None],
    module_root: str,
    register_method: str,
) -> None:
    provider_name = f"entrypoint_safety_{register_method}"
    provider_dir = tmp_path / provider_name
    provider_dir.mkdir()
    marker = tmp_path / f"{provider_name}-build-executed"
    module_name = f"{module_root}.{provider_name}"

    (provider_dir / "helper.py").write_text(
        "class RelativeProvider:\n"
        "    loaded_via = 'relative import'\n",
        encoding="utf-8",
    )
    (provider_dir / "build.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n",
        encoding="utf-8",
    )
    (provider_dir / "__init__.py").write_text(
        "from .helper import RelativeProvider\n\n"
        "def register(ctx):\n"
        f"    ctx.{register_method}(RelativeProvider())\n",
        encoding="utf-8",
    )

    try:
        provider = loader(provider_dir)

        assert provider is not None
        assert getattr(provider, "loaded_via") == "relative import"
        assert f"{module_name}.helper" in sys.modules
        assert f"{module_name}.build" not in sys.modules
        assert not marker.exists()
    finally:
        for loaded_name in list(sys.modules):
            if loaded_name == module_name or loaded_name.startswith(f"{module_name}."):
                sys.modules.pop(loaded_name, None)
        parent = sys.modules.get(module_root)
        if parent is not None and getattr(parent, provider_name, None) is not None:
            delattr(parent, provider_name)
