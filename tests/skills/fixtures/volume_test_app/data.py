"""Deterministic synthetic content and a Python oracle independent of catalog SQL."""

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("volume_support", Path(__file__).resolve().parents[4] / "optional-skills/software-development/volume-test/scripts/volume_support.py")
SUPPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUPPORT)
FIELDS = ("id", "name", "price", "category")


def record(seed, index):
    value = SUPPORT.stable_integer(seed, index, "product")
    bucket = value % 100
    category = index if index < 8 else (0 if bucket < 60 else 1 if bucket < 80 else 2 + value % 6)
    return {"id": index, "name": ["Café", "東京", "Málaga", "Crème", "Δelta", "Música"][value % 6] + f" item {index % 29:02}",
            "price": 100 + (value % 37) ** 2, "category": category}


def reference(rows, q="", category="", sort="name"):
    matched = [row for row in rows if q in row["name"] and (category == "" or row["category"] == int(category))]
    return sorted(matched, key=lambda row: (row[sort], row["id"]))
