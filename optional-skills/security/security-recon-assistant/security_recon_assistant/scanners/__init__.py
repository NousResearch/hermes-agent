from importlib import import_module
from pathlib import Path

def discover():
    scanners = {}
    here = Path(__file__).parent
    for file in here.glob("*_scanner.py"):
        mod_name = file.stem
        mod = import_module(f"security_recon_assistant.scanners.{mod_name}")
        cls = getattr(mod, "Scanner", None)
        if cls:
            scanners[mod_name.replace("_scanner", "")] = cls
    return scanners
