"""Fixed form of H2 — latin-1 fallback strips BOM_UTF8 first (#65124)."""
import codecs
import io
from pathlib import Path
from dotenv import load_dotenv


def _load_dotenv_with_fallback(path: Path, *, override: bool) -> None:
    try:
        load_dotenv(dotenv_path=path, override=override, encoding="utf-8-sig")
    except UnicodeDecodeError:
        raw = path.read_bytes()
        if raw.startswith(codecs.BOM_UTF8):
            raw = raw[len(codecs.BOM_UTF8) :]
        load_dotenv(stream=io.StringIO(raw.decode("latin-1")), override=override)
