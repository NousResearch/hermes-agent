"""Behavior tests for the optional fitness-nutrition skill."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from unittest import mock

import pytest


SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "health"
    / "fitness-nutrition"
)
BODY_CALC = SKILL_DIR / "scripts" / "body_calc.py"
EXERCISE_SEARCH = SKILL_DIR / "scripts" / "exercise_search.py"
SKILL_MD = SKILL_DIR / "SKILL.md"
FORMULAS_MD = SKILL_DIR / "references" / "FORMULAS.md"


def run_body_calc(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(BODY_CALC), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_bodyfat_converts_male_centimetres_for_navy_equation() -> None:
    result = run_body_calc("bodyfat", "M", "38", "85", "180")

    assert result.returncode == 0
    assert "Estimated body fat: 16.2%" in result.stdout


def test_bodyfat_converts_female_centimetres_for_navy_equation() -> None:
    result = run_body_calc("bodyfat", "F", "34", "75", "100", "165")

    assert result.returncode == 0
    assert "Estimated body fat: 29.2%" in result.stdout


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("bmi", "-80", "180"), "weight must be greater than zero"),
        (("bmi", "80", "0"), "height must be greater than zero"),
        (("tdee", "80", "180", "0", "M", "3"), "age must be greater than zero"),
        (("tdee", "80", "180", "35", "X", "3"), "sex must be M or F"),
        (("tdee", "80", "180", "35", "M", "9"), "activity must be between 1 and 5"),
        (("1rm", "0", "5"), "weight must be greater than zero"),
        (("1rm", "100", "11"), "reps must be between 1 and 10"),
        (["macros", "2500", "recomp"], "goal must be cut, maintain, or bulk"),
        (["bodyfat", "X", "34", "75", "100", "165"], "sex must be M or F"),
        (["bodyfat", "M", "38", "85", "0"], "height must be greater than zero"),
        (
            ["bodyfat", "M", "40", "41", "200"],
            "measurements produce an invalid body-fat estimate",
        ),
    ],
)
def test_invalid_calculator_inputs_fail_closed(
    args: tuple[str, ...], message: str
) -> None:
    result = run_body_calc(*args)

    assert result.returncode == 2
    assert result.stdout == ""
    assert message in result.stderr


def load_exercise_search():
    spec = importlib.util.spec_from_file_location("exercise_search", EXERCISE_SEARCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exercise_record(exercise_id: int, name: str) -> dict:
    return {
        "id": exercise_id,
        "category": {},
        "muscles": [],
        "muscles_secondary": [],
        "equipment": [],
        "translations": [{"language": 2, "name": name, "description": ""}],
        "images": [],
    }


def test_exercise_search_cli_uses_supported_wger_search(capsys) -> None:
    exercise_search = load_exercise_search()
    payload = {
        "count": 2,
        "results": [
            {
                "id": 289,
                "category": {"name": "Shoulders"},
                "muscles": [{"name_en": "Shoulders"}],
                "muscles_secondary": [],
                "equipment": [],
                "translations": [
                    {
                        "language": 2,
                        "name": "High Pull",
                        "description": "Explosive barbell pull.",
                    }
                ],
                "images": [],
            },
            {
                "id": 475,
                "category": {"name": "Back"},
                "muscles": [{"name_en": "Lats"}],
                "muscles_secondary": [{"name_en": "Biceps"}],
                "equipment": [{"name": "Pull-up bar"}],
                "translations": [
                    {
                        "language": 2,
                        "name": "Pull-ups",
                        "description": "<p>Pull under control.</p>",
                    }
                ],
                "images": [{"image": "https://wger.de/pull-up.jpg"}],
            },
        ],
    }
    response = mock.MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = json.dumps(payload).encode()

    with mock.patch.object(
        exercise_search.urllib.request, "urlopen", return_value=response
    ) as urlopen:
        exit_code = exercise_search.main(["pull up", "--limit", "1"])

    assert exit_code == 0
    request = urlopen.call_args.args[0]
    assert "name__search=pull+up" in request.full_url
    assert "language__code=en" in request.full_url
    assert "limit=100" in request.full_url
    result = json.loads(capsys.readouterr().out)
    assert result == {
        "query": "pull up",
        "count": 2,
        "results": [
            {
                "id": 475,
                "name": "Pull-ups",
                "category": "Back",
                "primary_muscles": ["Lats"],
                "secondary_muscles": ["Biceps"],
                "equipment": ["Pull-up bar"],
                "description": "Pull under control.",
                "image": "https://wger.de/pull-up.jpg",
            }
        ],
    }


def test_exercise_search_cli_filters_and_prints_english_names(capsys) -> None:
    exercise_search = load_exercise_search()
    payload = {
        "count": 1,
        "results": [exercise_record(475, "Pull-ups")],
    }
    response = mock.MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = json.dumps(payload).encode()

    with mock.patch.object(
        exercise_search.urllib.request, "urlopen", return_value=response
    ) as urlopen:
        exit_code = exercise_search.main([
            "--muscle",
            "4",
            "--equipment",
            "1",
            "--limit",
            "10",
        ])

    assert exit_code == 0
    request = urlopen.call_args.args[0]
    assert "muscles=4" in request.full_url
    assert "equipment=1" in request.full_url
    result = json.loads(capsys.readouterr().out)
    assert result["filters"] == {"muscles": 4, "equipment": 1}
    assert result["results"][0]["name"] == "Pull-ups"


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("pullup", "Pull-ups"),
        ("pullups", "Pull-ups"),
        ("squat", "Squats"),
        ("deadlift", "Deadlifts"),
        ("chinup", "Chin Up"),
    ],
)
def test_exercise_search_cli_reranks_common_spellings(capsys, query, expected) -> None:
    exercise_search = load_exercise_search()
    names = [
        "Pullback",
        "Pullover",
        "Pull-ups",
        "Pullup on fingerboard",
        "Belt Squat",
        "Squats",
        "Romanian Deadlift",
        "Deadlifts",
        "Chin tuck",
        "Chin Up",
    ]
    payload = {
        "count": len(names),
        "results": [exercise_record(index, name) for index, name in enumerate(names)],
    }
    response = mock.MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = json.dumps(payload).encode()

    with mock.patch.object(
        exercise_search.urllib.request, "urlopen", return_value=response
    ):
        exit_code = exercise_search.main([query, "--limit", "1"])

    assert exit_code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["results"][0]["name"] == expected


@pytest.mark.parametrize("query", ["   ", "!!!"])
def test_exercise_search_cli_rejects_unsearchable_query(capsys, query) -> None:
    exercise_search = load_exercise_search()

    with mock.patch.object(exercise_search.urllib.request, "urlopen") as urlopen:
        exit_code = exercise_search.main([query])

    assert exit_code == 2
    assert capsys.readouterr().err == "Error: query must contain letters or numbers\n"
    urlopen.assert_not_called()


def test_skill_contract_routes_through_corrected_helpers() -> None:
    skill = SKILL_MD.read_text(encoding="utf-8")
    formulas = FORMULAS_MD.read_text(encoding="utf-8")

    assert "scripts/exercise_search.py" in skill
    assert "--muscle" in skill
    assert "--category" in skill
    assert "--equipment" in skill
    assert "name__search" in skill
    assert "/api/v2/exercise/search/" not in skill
    assert "/api/v2/exercise/?" not in skill
    assert "status=2" not in skill
    assert "product label" in skill.lower()
    assert (
        "do not use this as the primary skill for progressive workout programming"
        in skill.lower()
    )
    assert "convert centimetres to inches" in formulas.lower()
