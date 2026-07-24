"""Suppression handling — intentional plain utf-8 must be silenceable."""
from pathlib import Path
from dotenv import load_dotenv


def load_test_env(path: Path) -> None:
    # Test fixture env files are authored as BOM-less UTF-8 by the suite.
    load_dotenv(dotenv_path=path, encoding="utf-8")  # encoding-safety: ok — test fixture authored BOM-less
