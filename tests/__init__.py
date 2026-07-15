import importlib.resources
import os.path
import sys
from pathlib import Path
from types import ModuleType


def load_module_from_file(rel_path: Path) -> ModuleType:
    anchor = str(rel_path.parent).replace(os.sep, ".")
    abs_module_file = str(importlib.resources.files(anchor).joinpath(rel_path.name))

    assert os.path.exists(abs_module_file), f"Module file missing: {rel_path}"

    spec = importlib.util.spec_from_file_location(
        f"hermes_{anchor.replace('.', '_')}", abs_module_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod
