"""Optional per-pet ``stateRows`` + ``behaviors`` from ``pet.json`` (user guide: Pets → Custom rows and
behaviors). Pets without them render exactly as before; nothing here ever raises on bad input."""

from __future__ import annotations

import json
import logging
import os
import stat
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from agent.pet.constants import FRAME_H, PetState, state_row_index, state_rows_for_grid
from utils import file_cache_version, file_signature

logger = logging.getLogger(__name__)

_KNOWN_STATES = frozenset(s.value for s in PetState)
_MAX_ROWS = 64
_MAX_STATE_ROWS = 16  # rows one state may cycle through
_SLEEP_SECONDS = (10, 24 * 3600)  # rest before ``sleep``
_HOLD_SECONDS = (1, 3600)  # how long a cycled row holds (desktop default 30 s)


@dataclass(frozen=True)
class PetSpec:
    rows: tuple[str, ...]
    overrides: dict[str, tuple[str, ...]] = field(default_factory=dict)  # state -> row names (rotated)
    sleep_after: float = 0.0  # seconds at rest before ``sleep``; 0 = never
    hold: dict[str, float] = field(default_factory=dict)  # state -> seconds each cycled row holds
    warnings: tuple[str, ...] = ()  # what cleaning dropped or clamped, for ``hermes pets doctor``

    def row_index(self, state: PetState | str) -> int:
        """Row for *state*: its first declared override (terminals don't cycle), else the standard mapping."""
        value = state.value if isinstance(state, PetState) else str(state)
        override = self.overrides.get(value)
        return self.rows.index(override[0]) if override else state_row_index(value, rows=self.rows)

    def payload(self) -> dict:
        """JSON-safe shape shipped to the desktop renderer (``pet.info.behaviors``)."""
        out: dict = {}
        if self.overrides:
            out["rows"] = {k: list(v) for k, v in self.overrides.items()}
        if self.sleep_after:
            out["sleepAfterSeconds"] = self.sleep_after
        if self.hold:
            out["holdSeconds"] = dict(self.hold)
        return out


def _name(value) -> str:
    """A pet-author-supplied name, quoted and capped for a one-line warning."""
    return repr(str(value)[:40])


def _read_meta(directory: Path, warnings: list[str]) -> dict:
    """``pet.json`` beside the sheet (regular files only: no FIFO / device reads); ``{}`` plus a warning if unreadable."""
    path = directory / "pet.json"
    try:
        meta = json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}
    except (OSError, ValueError, RecursionError) as exc:  # RecursionError: hostile deep nesting
        warnings.append(f"unreadable ({type(exc).__name__}: {str(exc)[:80]})")
        return {}
    return meta if isinstance(meta, dict) else {}


def _clean_rows(raw, row_count: int, warnings: list[str]) -> tuple[str, ...] | None:
    """Declared row names, or None (with a warning when rejected) to infer them from the sheet."""
    if raw is None:
        return None
    names = [str(x).strip() for x in raw] if isinstance(raw, list) else []
    if not names:
        problem = "not a non-empty list"
    elif len(names) > min(row_count, _MAX_ROWS):
        problem = f"names {len(names)} rows but the sheet has {row_count}"
    elif any(not n or len(n) > 40 for n in names) or len(set(names)) != len(names):
        problem = "blank, duplicate or over-40-character names"
    else:
        return tuple(names)
    warnings.append(f"stateRows: {problem}; rows inferred from the sheet")
    return None


def _clean_seconds(spec: dict, key: str, bounds: tuple[float, float], label: str, notes: list[str]) -> float:
    """``spec[key]`` clamped to *bounds*; 0 (with a note) when unusable."""
    try:
        secs = float(spec.get(key, 0))
    except (TypeError, ValueError, OverflowError):  # OverflowError: huge JSON integers
        secs = 0.0
    if not secs > 0:  # also rejects NaN
        notes.append(f"{label} needs a positive {key}")
        return 0.0
    clamped = max(bounds[0], min(bounds[1], secs))
    if clamped != secs:
        notes.append(f"{label} {key} {secs:g} clamped to {clamped:g}")
    return clamped


def _clean_behaviors(raw, rows: tuple[str, ...], warnings: list[str]) -> PetSpec:
    """Spec for *rows* with the usable state overrides, sleep delay and per-state hold in *raw*; a one-line
    summary of what was ignored or clamped is appended to *warnings*."""
    if not isinstance(raw, dict):
        if raw is not None:
            warnings.append("behaviors: not an object; ignored")
        return PetSpec(rows, warnings=tuple(warnings))
    overrides: dict[str, tuple[str, ...]] = {}
    hold: dict[str, float] = {}
    sleep_after = 0.0
    notes: list[str] = []
    for state, spec in raw.items():
        if state not in _KNOWN_STATES:
            notes.append(f"unknown state {_name(state)}")
            continue
        if not isinstance(spec, dict):
            notes.append(f"{_name(state)} is not an object")
            continue
        listed = spec.get("rows") if isinstance(spec.get("rows"), list) else [spec.get("row")]
        if len(listed) > _MAX_STATE_ROWS:
            notes.append(f"{_name(state)} lists {len(listed)} rows; first {_MAX_STATE_ROWS} used")
        names = [str(x or "").strip() for x in listed[:_MAX_STATE_ROWS]]
        notes.extend(f"unknown row {_name(r)}" for r in dict.fromkeys(names) if r and r not in rows)
        picked = tuple(dict.fromkeys(r for r in names if r in rows))
        if not picked:
            if not any(names):
                notes.append(f"{_name(state)} names no row")
            continue
        if state == PetState.SLEEP.value and not (sleep_after := _clean_seconds(spec, "afterSeconds", _SLEEP_SECONDS, "sleep", notes)):
            continue
        overrides[state] = picked
        if "holdSeconds" in spec and (secs := _clean_seconds(spec, "holdSeconds", _HOLD_SECONDS, state, notes)):
            hold[state] = secs
    if notes:
        warnings.append(f"behaviors: {len(overrides)} ok, {len(raw) - len(overrides)} ignored ({'; '.join(dict.fromkeys(notes))})")
    return PetSpec(rows, overrides, sleep_after, hold, tuple(warnings))


def _file_version(path: str | Path) -> tuple:
    """:func:`utils.file_cache_version` (catches same-tick rewrites) for regular files, stat-only otherwise (its
    digest would open a FIFO / device); ``()`` when the file is missing."""
    try:
        st = os.stat(path)
        return file_cache_version(path, st) if stat.S_ISREG(st.st_mode) else file_signature(st)
    except (OSError, ValueError):
        return ()


def pet_revision(sheet_path: str | Path) -> str:
    """Change key for everything drawn from a pet: its sheet's and its ``pet.json``'s file versions. Never raises."""
    sheet = Path(str(sheet_path))
    return "/".join("-".join(map(str, _file_version(p))) or "0" for p in (sheet, sheet.parent / "pet.json"))


@lru_cache(maxsize=32)
def _spec_cached(sheet_path: str, row_count: int, _json_version: tuple) -> PetSpec:
    warnings: list[str] = []
    meta = _read_meta(Path(sheet_path).parent, warnings)
    rows = _clean_rows(meta.get("stateRows"), row_count, warnings) or tuple(state_rows_for_grid(row_count))
    return _clean_behaviors(meta.get("behaviors"), rows, warnings)


def pet_spec(sheet_path: str | Path, row_count: int | None = None) -> PetSpec:
    """Spec for the sheet at *sheet_path* (``pet.json`` beside it). Never raises."""
    path = str(sheet_path)
    if row_count is None:
        row_count = _row_count(path)  # never raises either
    try:
        return _spec_cached(path, int(row_count), _file_version(Path(path).parent / "pet.json"))
    except (OSError, ValueError, TypeError, ArithmeticError, RecursionError) as exc:  # a mascot must never break a turn
        logger.debug("pet spec failed for %s: %s", path, exc)
        return PetSpec(tuple(state_rows_for_grid(row_count)))


def _row_count(sheet_path: str) -> int:
    """Rows in the sheet (0 if unreadable); cached per file version, so per-frame callers stay cheap."""
    version = _file_version(sheet_path)
    return _row_count_cached(sheet_path, version) if version else 0


@lru_cache(maxsize=32)
def _row_count_cached(sheet_path: str, _version: tuple) -> int:
    try:
        from PIL import Image

        with Image.open(sheet_path) as im:
            return max(1, im.height // FRAME_H)
    except Exception:  # untrusted third-party image: Pillow raises many types (e.g. DecompressionBombError)
        logger.debug("unreadable pet sheet %s", sheet_path, exc_info=True)
        return 0
