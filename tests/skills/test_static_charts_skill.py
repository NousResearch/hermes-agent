"""Tests for the bundled static-charts skill without importing plotting packages."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / "skills" / "data-science" / "static-charts"
SCRIPT_PATH = SKILL_DIR / "scripts" / "render_chart.py"
SKILL_MD = SKILL_DIR / "SKILL.md"


def load_module():
    spec = importlib.util.spec_from_file_location("static_charts_renderer", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def frontmatter() -> dict:
    source = SKILL_MD.read_text(encoding="utf-8")
    match = re.search(r"^---\n(.*?)\n---", source, re.DOTALL)
    assert match, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(match.group(1))


def test_skill_frontmatter_matches_hardline_requirements(frontmatter: dict) -> None:
    assert frontmatter["name"] == "static-charts"
    assert len(frontmatter["description"]) <= 60
    assert frontmatter["description"].endswith(".")
    assert set(frontmatter["platforms"]) == {"linux", "macos", "windows"}


def test_skill_documentation_describes_media_delivery() -> None:
    source = SKILL_MD.read_text(encoding="utf-8")
    assert "MEDIA:/absolute/path.png" in source
    assert "$HERMES_HOME/cache/images/" in source
    assert "IPA Font License v1.0" in source
    assert "## Pitfalls" in source
    assert "## Verification" in source


def test_requirements_include_a_bundled_japanese_font() -> None:
    requirements = (SKILL_DIR / "requirements.txt").read_text(encoding="utf-8")
    assert "japanize-matplotlib>=1.1.3,<1.2" in requirements
    assert "setuptools>=77,<83" in requirements


def test_line_chart_spec_is_normalized() -> None:
    renderer = load_module()
    result = renderer.validate_spec(
        {
            "type": "line",
            "title": "Monthly sales",
            "x_label": "Month",
            "y_label": "Sales",
            "series": [{"name": "Sales", "x": ["Jan", "Feb"], "y": [10, 12]}],
        }
    )
    assert result["type"] == "line"
    assert result["series"][0]["y"] == [10.0, 12.0]


def test_bar_chart_rejects_misaligned_categories() -> None:
    renderer = load_module()
    with pytest.raises(renderer.ChartSpecError, match="same x labels"):
        renderer.validate_spec(
            {
                "type": "bar",
                "title": "Comparison",
                "series": [
                    {"name": "A", "x": ["One", "Two"], "y": [1, 2]},
                    {"name": "B", "x": ["One", "Three"], "y": [3, 4]},
                ],
            }
        )


def test_histogram_and_pie_specs_have_type_specific_validation() -> None:
    renderer = load_module()
    histogram = renderer.validate_spec(
        {
            "type": "histogram",
            "title": "Latency",
            "bins": 8,
            "series": [{"name": "Requests", "values": [1.2, 1.5, 2.0]}],
        }
    )
    assert histogram["bins"] == 8

    with pytest.raises(renderer.ChartSpecError, match="add up to more than zero"):
        renderer.validate_spec(
            {
                "type": "pie",
                "title": "Breakdown",
                "series": [{"labels": ["A", "B"], "values": [0, 0]}],
            }
        )


def test_histogram_renderer_uses_one_color_per_series() -> None:
    renderer = load_module()

    class FakeAxis:
        def hist(self, values, **kwargs):
            self.values = values
            self.kwargs = kwargs

    axis = FakeAxis()
    renderer._render_histogram(
        axis,
        [{"name": "Requests", "values": [1.2, 1.5, 2.0]}],
        ["blue", "orange", "green"],
        8,
    )
    assert axis.values == [[1.2, 1.5, 2.0]]
    assert axis.kwargs["color"] == ["blue"]


def test_output_path_is_limited_to_hermes_image_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    renderer = load_module()
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    output = renderer.resolve_output_path(str(hermes_home / "cache" / "images" / "chart.png"))
    assert output == hermes_home / "cache" / "images" / "chart.png"

    with pytest.raises(renderer.ChartSpecError, match="inside"):
        renderer.resolve_output_path(str(tmp_path / "chart.png"))


def test_main_prints_gateway_media_directive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    renderer = load_module()
    hermes_home = tmp_path / "hermes-home"
    output = hermes_home / "cache" / "images" / "chart.png"
    spec_path = tmp_path / "chart.json"
    spec_path.write_text(
        json.dumps(
            {
                "type": "scatter",
                "title": "Relationship",
                "series": [{"name": "Values", "x": [1, 2], "y": [3, 5]}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    with patch.object(renderer, "render_chart", return_value="Noto Sans CJK JP") as render_chart:
        assert renderer.main([str(spec_path), "--output", str(output)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    assert payload["file_path"] == str(output)
    assert payload["media"] == f"MEDIA:{output}"
    assert payload["font"] == "Noto Sans CJK JP"
    render_chart.assert_called_once()
