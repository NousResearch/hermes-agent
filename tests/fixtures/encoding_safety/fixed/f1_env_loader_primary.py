"""Fixed form of H1 — primary path uses utf-8-sig (#65124)."""
from pathlib import Path
from dotenv import load_dotenv


def _load_dotenv_with_fallback(path: Path, *, override: bool) -> None:
    try:
        # utf-8-sig strips a leading UTF-8 BOM if present.
        load_dotenv(dotenv_path=path, override=override, encoding="utf-8-sig")
    except UnicodeDecodeError:
        pass
