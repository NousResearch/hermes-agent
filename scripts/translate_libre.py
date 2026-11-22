import json
import os
import sys
from pathlib import Path

# Locate the local Argos / LibreTranslate models in the active Hermes cache.
# HERMES_HOME wins; otherwise follow the platform default so the script works
# on any machine instead of the one it was authored on.
_hermes_home = os.environ.get("HERMES_HOME")
if not _hermes_home:
    if os.name == "nt":
        local_appdata = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        _hermes_home = str(Path(local_appdata) / "hermes")
    else:
        data_home = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
        _hermes_home = str(Path(data_home) / "hermes")
_cache_root = Path(_hermes_home) / "cache"
os.environ.setdefault("XDG_DATA_HOME", str(_cache_root / "data"))
os.environ.setdefault("XDG_CONFIG_HOME", str(_cache_root / "config"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root / "cache"))

try:
    import argostranslate.translate
except Exception as e:
    sys.stderr.write(f"ArgosTranslate import error: {e}\n")
    sys.exit(1)

def translate_text(text: str, from_code: str = "en", to_code: str = "ro") -> str:
    if not text or not text.strip():
        return text
    try:
        return argostranslate.translate.translate(text, from_code, to_code)
    except Exception as e:
        sys.stderr.write(f"Translation error: {e}\n")
        return text

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        payload = json.load(sys.stdin)
        results = [translate_text(item) for item in payload]
        json.dump(results, sys.stdout, ensure_ascii=False)
    elif len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        print(translate_text(text))
    else:
        text = sys.stdin.read()
        print(translate_text(text), end="")
