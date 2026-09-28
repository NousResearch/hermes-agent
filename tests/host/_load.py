"""Load the host tooling scripts (some have no .py suffix) as modules."""

from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HOST = REPO / "deploy" / "host"


def load_script(name: str, filename: str):
    path = HOST / filename
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module
