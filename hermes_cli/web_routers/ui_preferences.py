"""``/api/ui-preferences``: the Webapp's door to this machine's UI preferences store (#145213).

Thin HTTP over :mod:`hermes_cli.ui_preferences`. The store belongs to the machine serving the
Webapp, so every browser and origin that opens it reads and writes the same document native
Desktop on this machine uses (through its preload capability, not this route). Mounted only on
surfaces whose policy names ``ui_preferences`` (404 elsewhere) and, like every ``/api/`` route,
only past the session-token / sign-in gate.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from hermes_cli import ui_preferences

router = APIRouter()


class PreferencesPatch(BaseModel):
    #: Per-key changes; ``null`` deletes the key (the renderer's "back to default").
    values: dict[str, Optional[str]] = Field(default_factory=dict)
    #: This browser's first contact with the store generation: adds only the keys the store
    #: lacks, never overwriting or deleting one.
    seed: bool = False


@router.get("/api/ui-preferences")
def get_ui_preferences() -> dict:
    try:
        return ui_preferences.read_preferences()
    except ui_preferences.PreferencesBusy as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.patch("/api/ui-preferences")
def patch_ui_preferences(body: PreferencesPatch) -> dict:
    try:
        return ui_preferences.write_preferences(body.values, seed=body.seed)
    except ui_preferences.PreferencesError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ui_preferences.PreferencesBusy as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
