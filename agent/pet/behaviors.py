"""Optional per-pet ``stateRows`` + ``behaviors`` from ``pet.json`` (user guide: Pets → Custom rows and
behaviors). Pets without them render exactly as before; nothing here ever raises on bad input."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from agent.pet.constants import FRAME_H, PetState, state_row_index, state_rows_for_grid
from agent.pet.store import _read_pet_json

logger = logging.getLogger(__name__)

_KNOWN_STATES = frozenset(s.value for s in PetState)
_MAX_ROWS = 64
_MAX_SLEEP_SECONDS = 24 * 3600
_MIN_SLEEP_SECONDS = 10


@dataclass(frozen=True)
class PetSpec:
    rows: tuple[str, ...]
    overrides: dict[str, tuple[str, ...]] = field(default_factory=dict)  # state -> row names (rotated)
    sleep_after: float = 0.0  # seconds at rest before ``sleep``; 0 = never

    def row_index(self, state: PetState | str) -> int:
        """Row for *state*: its first declared override (terminals don't cycle), else the standard mapping."""
        value = state.value if isinstance(state, PetState) else str(state)
        override = self.overrides.get(value)
        return self.rows.index(override[0]) if override else state_row_index(value, rows=self.rows)

    def payload(self) -> dict:
        """JSON-safe shape shipped to the desktop renderer (``pet.info.behaviors``)."""
        out: dict = {}
        if self.overrides:
            out["rows"] = {k: (v[0] if len(v) == 1 else list(v)) for k, v in self.overrides.items()}
        if self.sleep_after:
            out["sleepAfterSeconds"] = self.sleep_after
        return out


def _clean_rows(raw, row_count: int) -> tuple[str, ...] | None:
    if not isinstance(raw, list) or not raw or len(raw) > min(row_count, _MAX_ROWS):
        return None
    names = [str(x).strip() for x in raw]
    if any(not n or len(n) > 40 for n in names) or len(set(names)) != len(names):
        return None
    return tuple(names)


def _clean_behaviors(raw, rows: tuple[str, ...]) -> tuple[dict[str, tuple[str, ...]], float]:
    overrides: dict[str, tuple[str, ...]] = {}
    sleep_after = 0.0
    if not isinstance(raw, dict):
        return overrides, sleep_after
    for state, spec in raw.items():
        if state not in _KNOWN_STATES or not isinstance(spec, dict):
            continue
        listed = spec.get("rows") if isinstance(spec.get("rows"), list) else [spec.get("row")]
        picked = tuple(dict.fromkeys(r for r in (str(x or "").strip() for x in listed[:16]) if r in rows))
        if not picked:
            continue
        if state == PetState.SLEEP.value:
            try:
                secs = float(spec.get("afterSeconds", 0))
            except (TypeError, ValueError, OverflowError):  # OverflowError: huge JSON integers
                continue
            if secs <= 0:
                continue
            sleep_after = max(_MIN_SLEEP_SECONDS, min(_MAX_SLEEP_SECONDS, secs))
        overrides[state] = picked
    return overrides, sleep_after


@lru_cache(maxsize=32)
def _spec_cached(sheet_path: str, row_count: int, _json_mtime: int) -> PetSpec:
    inferred = tuple(state_rows_for_grid(row_count))
    meta = _read_pet_json(Path(sheet_path).parent)  # regular files only (no FIFO / device reads)
    if not isinstance(meta, dict):
        meta = {}
    rows = _clean_rows(meta.get("stateRows"), row_count) or inferred
    overrides, sleep_after = _clean_behaviors(meta.get("behaviors"), rows)
    return PetSpec(rows, overrides, sleep_after)


def pet_spec(sheet_path: str | Path, row_count: int | None = None) -> PetSpec:
    """Spec for the sheet at *sheet_path* (``pet.json`` beside it). Never raises."""
    path = str(sheet_path)
    if row_count is None:
        row_count = _row_count(path)  # never raises either
    try:
        mtime = os.stat(Path(path).parent / "pet.json").st_mtime_ns
    except OSError:
        mtime = 0
    try:
        return _spec_cached(path, int(row_count), mtime)
    except (OSError, ValueError, TypeError, ArithmeticError, RecursionError) as exc:  # a mascot must never break a turn
        logger.debug("pet spec failed for %s: %s", path, exc)
        return PetSpec(tuple(state_rows_for_grid(row_count)))


def _row_count(sheet_path: str) -> int:
    """Rows in the sheet (0 if unreadable); cached per file version, so per-frame callers stay cheap."""
    try:
        return _row_count_cached(sheet_path, os.stat(sheet_path).st_mtime_ns)
    except OSError:
        return 0


@lru_cache(maxsize=32)
def _row_count_cached(sheet_path: str, _mtime: int) -> int:
    try:
        from PIL import Image

        with Image.open(sheet_path) as im:
            return max(1, im.height // FRAME_H)
    except Exception:  # untrusted third-party image: Pillow raises many types (e.g. DecompressionBombError)
        logger.debug("unreadable pet sheet %s", sheet_path, exc_info=True)
        return 0
