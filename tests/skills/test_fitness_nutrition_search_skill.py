"""Offline USDA search response contracts for the nutrition CLI."""
import io
import json
from pathlib import Path
import runpy
import sys
import urllib.request

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "optional-skills/health/fitness-nutrition/scripts/nutrition_search.py"


def nutrient(nid, name, unit, value):
    return {"nutrientId": nid, "nutrientName": name, "unitName": unit, "value": value}


@pytest.mark.parametrize("reverse", [False, True])
def test_search_cli_keeps_kcal_and_macros_independent_of_names_and_order(monkeypatch, capsys, reverse):
    nutrients = [
        nutrient(1008, "Energy", "KCAL", 172),
        nutrient(1062, "Energy", "kJ", 720),
        nutrient(1003, "Protein", "G", 21),
        nutrient(1004, "Total lipid (fat)", "G", 9),
        nutrient(1005, "Carbohydrate, by difference", "G", 2),
        nutrient(1079, "Fiber, total dietary", "G", 1),
        nutrient(2000, "Total Sugars", "G", 0),
    ]
    if reverse:
        nutrients.reverse()
    payload = {"foods": [{"description": "Test food", "fdcId": 123, "foodNutrients": nutrients}]}
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(json.dumps(payload).encode()))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "test food"])
    monkeypatch.setenv("USDA_API_KEY", "offline-test")
    runpy.run_path(str(SCRIPT), run_name="__main__")
    output = capsys.readouterr().out
    for expected in ["Calories : 172 kcal", "Protein  : 21g", "Fat      : 9g", "Carbs    : 2g (fiber: 1g, sugar: 0g)", "FDC ID   : 123"]:
        assert expected in output


@pytest.mark.parametrize("energies,expected", [
    ([(2048, "KCAL", 112), (2047, "KCAL", 106)], 106),
    ([(2047, "KCAL", 106), (2048, "KCAL", 112), (1008, "KCAL", 105)], 105),
    ([(2048, "kcal", 112)], 112),
    ([(1008, "KCAL", None), (2047, "KCAL", 0)], 0),
    ([(1008, "kJ", 720), (1062, "kJ", 720)], "?"),
    ([], "?"),
])
def test_energy_fallback_requires_kcal_and_preserves_zero(energies, expected, capsys):
    display = runpy.run_path(str(SCRIPT))["display"]
    display({"foodNutrients": [nutrient(nid, "Energy" if nid in (1008, 1062) else "Energy (Atwater factors)", unit, value) for nid, unit, value in energies]})
    output = capsys.readouterr().out
    assert f"Calories : {expected} kcal" in output
    assert "Protein  : ?g" in output
