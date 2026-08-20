"""Load Hermes' top-level utilities without colliding with third-party ``utils``."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_UTILS_PATH = Path(__file__).resolve().parent.parent / "utils.py"
_SPEC = spec_from_file_location("_hermes_utils_impl", _UTILS_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - broken install
    raise ImportError(f"Unable to load Hermes utilities from {_UTILS_PATH}")
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

atomic_replace = _MODULE.atomic_replace
atomic_write_text = _MODULE.atomic_write_text

__all__ = ["atomic_replace", "atomic_write_text"]