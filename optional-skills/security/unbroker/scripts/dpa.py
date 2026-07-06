"""Load and query the DPA (data-protection authority) adapter registry.

Each national DPA is one JSON file under references/dpa/ for clean diffs/PRs. Files
beginning with `_` are ignored (reserved for notes/scratch). The shape mirrors
scripts/brokers.py exactly so a contributor who knows one loader knows the other.

An adapter records:
  id                  short slug (e.g. "garante", "cnil") — matches dossier.legal_framework
  name                full official name
  country             ISO 3166-1 alpha-2
  language            primary complaint language
  web_form_url        URL of the DPA's online complaint form (browser-form automation target)
  complaint_template  path to the templates/dpa-complaints/<id>.txt file
  expected_days       statutory response window per Art. 78(2) where applicable
  email               direct email (when web form is unavailable)
  phone               for voice complaints where accepted
  notes               free-text quirks (e.g. "requires PEC for Italian residents")
"""
from __future__ import annotations

import json
from pathlib import Path

import paths


def _load_curated(directory: Path | None = None) -> list[dict]:
    directory = directory or paths.dpa_dir()
    out: list[dict] = []
    if not directory.exists():
        return out
    for fp in sorted(directory.glob("*.json")):
        if fp.name.startswith("_"):
            continue
        out.append(json.loads(fp.read_text(encoding="utf-8")))
    return out


def load_all(directory: Path | None = None) -> list[dict]:
    """Return all DPA adapters, sorted by country code then id."""
    out = list(_load_curated(directory))
    out.sort(key=lambda d: (d.get("country", ""), d.get("id", "")))
    return out


def get(dpa_id: str, directory: Path | None = None) -> dict | None:
    """Look up a single DPA by its short id (case-insensitive)."""
    target = dpa_id.lower()
    for d in load_all(directory):
        if d.get("id", "").lower() == target:
            return d
    return None


def for_residency(residency: str, directory: Path | None = None) -> dict | None:
    """Resolve a DPA adapter from a residency code using the dossier.legal_framework table.

    Returns None for residencies that have no mapped DPA (the subject files directly
    with a controller or uses the generic fallback).
    """
    # Import here to avoid a circular import: dossier.py defines RESIDENCY_LEGAL_FRAMEWORK,
    # but it's the canonical source of the residency->dpa mapping.
    import dossier
    meta = dossier.legal_framework(residency)
    dpa_id = meta.get("dpa")
    if not dpa_id:
        return None
    return get(dpa_id, directory)