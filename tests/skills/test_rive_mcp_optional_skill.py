"""Exercise the portable optional Rive scripts, including disposable HTTP peers.

Keep one shared stdlib unittest suite so installed skill copies can run the same
checks without importing Hermes. No source-text assertions or live editor access.
"""
import importlib.util
from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[2] / "optional-skills" / "creative" / "rive-mcp" / "scripts"
spec = importlib.util.spec_from_file_location("rive_portable_tests", SCRIPTS / "test_rive_mcp.py")
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(SCRIPTS))
try:
    spec.loader.exec_module(module)
finally:
    sys.path.remove(str(SCRIPTS))

CliTests = module.CliTests
SdkBoundaryTests = module.SdkBoundaryTests
DoctorTests = module.DoctorTests
