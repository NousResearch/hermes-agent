"""Offline USDA search response contracts for the fitness-nutrition skill.

Regression tests for #152535: USDA nutrients must be matched by stable nutrient
ID and unit rather than display name, because SR Legacy foods carry duplicate
"Energy" entries (kcal and kJ) and Foundation foods use different names/IDs for
Atwater energy and macros.
"""

import io
import json
from pathlib import Path
import runpy
import sys
import urllib.request

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "health"
    / "fitness-nutrition"
    / "scripts"
    / "nutrition_search.py"
)


def _nutrient(nid, name, unit, value):
    return {
        "nutrientId": nid,
        "nutrientName": name,
        "unitName": unit,
        "value": value,
    }


@pytest.mark.parametrize("reverse", [False, True])
def test_search_cli_selects_nutrients_by_id_and_unit_regardless_of_order(
    monkeypatch, capsys, reverse
):
    """SR Legacy kcal/kJ Energy collision and Total Sugars resolve identically in any order."""
    nutrients = [
        _nutrient(1008, "Energy", "KCAL", 172),
        _nutrient(1062, "Energy", "kJ", 720),
        _nutrient(1003, "Protein", "G", 20.8),
        _nutrient(1004, "Total lipid (fat)", "G", 9.25),
        _nutrient(1005, "Carbohydrate, by difference", "G", 0.0),
        _nutrient(1079, "Fiber, total dietary", "G", 0.0),
        _nutrient(2000, "Total Sugars", "G", 0.0),
    ]
    if reverse:
        nutrients.reverse()

    payload = {
        "foods": [
            {
                "description": "Chicken, broilers or fryers, breast, meat and skin, raw",
                "fdcId": 171474,
                "foodNutrients": nutrients,
            }
        ]
    }
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **kw: io.BytesIO(json.dumps(payload).encode()),
    )
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "chicken breast"])
    monkeypatch.setenv("USDA_API_KEY", "offline-test")

    runpy.run_path(str(SCRIPT), run_name="__main__")
    output = capsys.readouterr().out

    for expected in (
        "Calories : 172 kcal",
        "Protein  : 20.8g",
        "Fat      : 9.25g",
        "Carbs    : 0.0g (fiber: 0.0g, sugar: 0.0g)",
        "FDC ID   : 171474",
    ):
        assert expected in output


@pytest.mark.parametrize(
    ("food_nutrients", "expected_lines"),
    [
        pytest.param(
            [
                _nutrient(2048, "Energy (Atwater Specific Factors)", "KCAL", 112),
                _nutrient(2047, "Energy (Atwater General Factors)", "KCAL", 106),
                _nutrient(1063, "Sugars, Total", "G", 15.8),
            ],
            ("Calories : 106 kcal", "Carbs    : ?g (fiber: ?g, sugar: 15.8g)"),
            id="foundation-atwater-general-and-sugars-total",
        ),
        pytest.param(
            [
                _nutrient(2047, "Energy (Atwater General Factors)", "KCAL", 106),
                _nutrient(2048, "Energy (Atwater Specific Factors)", "KCAL", 112),
                _nutrient(1008, "Energy", "KCAL", 105),
            ],
            ("Calories : 105 kcal",),
            id="energy-prefers-1008-over-atwater",
        ),
        pytest.param(
            [_nutrient(2048, "Energy (Atwater Specific Factors)", "kcal", 112)],
            ("Calories : 112 kcal",),
            id="atwater-specific-lowercase-unit",
        ),
        pytest.param(
            [
                _nutrient(1008, "Energy", "KCAL", None),
                _nutrient(2047, "Energy (Atwater General Factors)", "KCAL", 0),
                _nutrient(2000, "Total Sugars", "G", 0),
            ],
            ("Calories : 0 kcal", "sugar: 0g"),
            id="skips-none-and-preserves-zero",
        ),
        pytest.param(
            [
                _nutrient(1008, "Energy", "kJ", 720),
                _nutrient(1062, "Energy", "kJ", 720),
            ],
            ("Calories : ? kcal",),
            id="rejects-kj-even-for-1008",
        ),
        pytest.param(
            [
                _nutrient(1085, "Total fat (NLEA)", "G", 8.65),
                _nutrient(1050, "Carbohydrate, by summation", "G", 18.0),
                _nutrient(2033, "Total dietary fiber (AOAC 2011.25)", "G", 2.02),
                _nutrient(1063, "Sugars, Total", "G", 1.2),
                {"nutrientName": "Malformed entry without id or unit"},
            ],
            (
                "Calories : ? kcal",
                "Protein  : ?g",
                "Fat      : 8.65g",
                "Carbs    : 18.0g (fiber: 2.02g, sugar: 1.2g)",
            ),
            id="foundation-macro-fallbacks-and-malformed-entry",
        ),
    ],
)
def test_display_energy_and_macro_fallbacks_preserve_zero_and_reject_non_kcal(
    food_nutrients, expected_lines, capsys
):
    """Energy requires kcal in 1008 > 2047 > 2048 order; macros fall back across Foundation IDs."""
    display = runpy.run_path(str(SCRIPT))["display"]
    display({"description": "Fixture food", "fdcId": 42, "foodNutrients": food_nutrients})
    output = capsys.readouterr().out
    for line in expected_lines:
        assert line in output
