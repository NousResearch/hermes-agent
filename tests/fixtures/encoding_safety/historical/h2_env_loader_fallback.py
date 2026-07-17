"""Historical instance #2 — env_loader latin-1 fallback (pre-#65124).

UnicodeDecodeError → load_dotenv(..., encoding="latin-1") without stripping
codecs.BOM_UTF8. A UTF-8 BOM survives as U+FEFF on the first key.
"""
from pathlib import Path
from dotenv import load_dotenv


def _load_dotenv_with_fallback(path: Path, *, override: bool) -> None:
    try:
        # Primary intentionally uses utf-8-sig so this fixture isolates R2.
        load_dotenv(dotenv_path=path, override=override, encoding="utf-8-sig")
    except UnicodeDecodeError:
        load_dotenv(dotenv_path=path, override=override, encoding="latin-1")
