"""Historical instance #1 — env_loader primary path (pre-#65124).

load_dotenv(..., encoding="utf-8") on user .env: a UTF-8 BOM sticks to the
first key as U+FEFF and silently drops it from os.environ under its
canonical name. See #65123.
"""
from pathlib import Path
from dotenv import load_dotenv


def _load_dotenv_with_fallback(path: Path, *, override: bool) -> None:
    try:
        load_dotenv(dotenv_path=path, override=override, encoding="utf-8")
    except UnicodeDecodeError:
        pass
