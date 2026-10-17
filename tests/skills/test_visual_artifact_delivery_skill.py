"""The visual-artifact-delivery skill ships a geometry checker that behaves as documented."""
import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = ROOT / "optional-skills" / "creative" / "visual-artifact-delivery"
SCRIPT = SKILL_DIR / "scripts" / "svg_geometry_check.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("svg_geometry_check_under_test", SCRIPT)
    if spec is None or spec.loader is None:  # pragma: no cover - the file ships with the skill
        raise AssertionError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves string annotations through sys.modules, so register first.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _svg(*body: str, view_box: str = "0 0 400 200", attrs: str = "") -> str:
    inner = "\n  ".join(body)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_box}" {attrs}>\n'
            f"  {inner}\n</svg>\n")


def _cli(*args: str, stdin_text: str | None = None):
    return subprocess.run([sys.executable, str(SCRIPT), *args], input=stdin_text,
                          capture_output=True, text=True, check=False)


def _run(tmp_path: Path, text: str, *args: str):
    target = tmp_path / "artifact.svg"
    target.write_text(text, encoding="utf-8")
    return _cli(str(target), *args)


# --- the skill package itself -----------------------------------------------------------

def test_frontmatter_declares_the_required_fields():
    text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    head = text.split("---", 2)[1]
    fields = {}
    for line in head.splitlines():
        if ":" in line and not line.startswith(" "):
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    assert fields["name"] == "visual-artifact-delivery"
    for required in ("name", "description", "version", "author", "license", "platforms"):
        assert fields.get(required), f"frontmatter is missing {required}"
    description = fields["description"].strip('"')
    assert len(description) <= 60, description
    assert description.endswith(".")
    assert "[" in fields["platforms"] and "]" in fields["platforms"]


def test_sections_follow_the_documented_order():
    text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    headings = [line.strip() for line in text.splitlines() if line.startswith("## ")]
    expected = ["## When to Use", "## Prerequisites", "## How to Run", "## Quick Reference",
                "## Procedure", "## Pitfalls", "## Verification"]
    positions = [headings.index(name) for name in expected]
    assert positions == sorted(positions), headings


def test_related_skills_all_exist_in_the_shipped_trees():
    text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    head = text.split("---", 2)[1]
    related = next(line for line in head.splitlines() if "related_skills" in line)
    names = [name.strip() for name in related.split("[", 1)[1].split("]", 1)[0].split(",")]
    available = {path.name for tree in ("skills", "optional-skills")
                 for path in (ROOT / tree).glob("*/*")}
    missing = [name for name in names if name and name not in available]
    assert not missing, missing


def test_reference_and_script_are_present():
    assert (SKILL_DIR / "references" / "rendering-and-verification.md").is_file()
    assert SCRIPT.is_file()


def test_the_checker_imports_nothing_outside_the_standard_library():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    assert roots <= set(sys.stdlib_module_names), sorted(roots)


# --- parsing ----------------------------------------------------------------------------

def test_extract_svg_pulls_the_first_block_out_of_an_html_page():
    module = _load_module()
    page = "<!doctype html><html><body><div>" + _svg() + "</div></body></html>"
    assert module.extract_svg(page).startswith("<svg")
    assert module.extract_svg("\n" + _svg()).startswith("<svg")
    assert module.extract_svg("<html><p>no drawing here</p></html>") is None


def test_collect_measures_boxes_circles_and_ellipses():
    module = _load_module()
    svg = _svg(
        '<rect id="r" x="10" y="10" width="100" height="50" fill="#eee"/>',
        '<circle id="c" cx="200" cy="100" r="20" fill="#ddd"/>',
        '<ellipse id="e" cx="300" cy="100" rx="30" ry="10" fill="#ccc"/>',
    )
    report = module.check_text(svg)
    labels = sorted(box.label() for box in report.shapes)
    assert len(labels) == 3
    assert "circle#c (180,80 40x40)" in labels
    assert "ellipse#e (270,90 60x20)" in labels


def test_unfilled_shapes_are_skipped_and_named():
    module = _load_module()
    svg = _svg('<rect id="guide" x="0" y="0" width="40" height="40" fill="none"/>')
    report = module.check_text(svg)
    assert report.shapes == []
    assert any("rect#guide" in note for note in report.notes)


def test_hidden_elements_are_skipped_including_inside_a_hidden_group():
    module = _load_module()
    svg = _svg(
        '<rect id="shown" x="10" y="10" width="100" height="50" fill="#eee"/>',
        '<rect id="ghost" x="10" y="10" width="100" height="50" fill="#eee" display="none"/>',
        '<g id="layer" display="none">',
        '  <rect id="nested" x="10" y="10" width="100" height="50" fill="#eee"/>',
        "</g>",
    )
    report = module.check_text(svg)
    assert [box.elem_id for box in report.shapes] == ["rect#shown"]
    assert sum("hidden" in note for note in report.notes) == 2


def test_bad_input_raises_a_value_error():
    module = _load_module()
    for text in ("<html><body>nothing</body></html>", _svg()[:-4]):
        try:
            module.check_text(text)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {text[:24]!r}")


# --- findings ---------------------------------------------------------------------------

def test_clean_layout_needs_no_fix():
    module = _load_module()
    report = module.check_text(_svg(
        '<rect id="left" x="10" y="10" width="100" height="50" fill="#eee"/>',
        '<rect id="right" x="200" y="10" width="100" height="50" fill="#eee"/>',
    ))
    assert report.findings == []


def test_partial_overlap_is_a_collision():
    module = _load_module()
    report = module.check_text(_svg(
        '<rect id="a" x="10" y="10" width="100" height="50" fill="#eee"/>',
        '<rect id="b" x="80" y="30" width="100" height="50" fill="#ddd"/>',
    ))
    assert [f["kind"] for f in report.findings] == ["collision"]
    assert report.findings[0]["severity"] == "fail"
    assert report.findings[0]["overlap_px"] == 900


def test_the_same_box_twice_is_a_duplicate_not_a_collision():
    module = _load_module()
    report = module.check_text(_svg(
        '<rect id="one" x="10" y="10" width="100" height="50" fill="#eee"/>',
        '<rect id="two" x="10" y="10" width="100" height="50" fill="#eee"/>',
    ))
    assert [f["kind"] for f in report.findings] == ["duplicate"]


def test_nesting_is_reported_as_containment_and_a_toolbar_over_content_as_stacked():
    module = _load_module()
    report = module.check_text(_svg(
        '<rect id="panel" x="0" y="0" width="400" height="200" fill="#fff"/>',
        '<rect id="inner" x="20" y="20" width="100" height="60" fill="#eee"/>',
    ))
    assert [(f["kind"], f["severity"]) for f in report.findings] == [("contained", "info")]

    stacked = module.check_text(_svg(
        '<rect id="outer" x="10" y="10" width="100" height="50" fill="#eee"/>',
        '<rect id="inner" x="15" y="15" width="80" height="40" fill="#ddd"/>',
    ))
    assert [(f["kind"], f["severity"]) for f in stacked.findings] == [("stacked", "warn")]


def test_a_dashed_frame_over_content_is_a_boundary_not_a_collision():
    module = _load_module()
    svg = _svg(
        '<rect id="frame" x="0" y="0" width="200" height="100" fill="none" stroke="#999" '
        'stroke-dasharray="6 4"/>',
        '<rect id="content" x="150" y="50" width="100" height="50" fill="#eee"/>',
    )
    report = module.check_text(svg)
    assert [(f["kind"], f["severity"]) for f in report.findings] == [("boundary", "warn")]


def test_a_shape_past_the_viewbox_is_outside_canvas():
    module = _load_module()
    report = module.check_text(_svg(
        '<rect id="spills" x="380" y="150" width="100" height="60" fill="#ddd"/>',
    ))
    assert [f["kind"] for f in report.findings] == ["outside-canvas"]
    assert "viewBox" in report.findings[0]["detail"]


def test_translate_offsets_are_applied_to_group_children():
    module = _load_module()
    svg = _svg(
        '<rect id="fixed" x="10" y="10" width="100" height="50" fill="#eee"/>',
        '<g transform="translate(60, 20)">',
        '  <rect id="moved" x="40" y="10" width="100" height="50" fill="#ddd"/>',
        "</g>",
    )
    report = module.check_text(svg)
    moved = next(box for box in report.shapes if box.elem_id == "rect#moved")
    assert (moved.x, moved.y) == (100.0, 30.0)
    assert [f["kind"] for f in report.findings] == ["collision"]


def test_a_rotated_group_is_reported_as_unverifiable_rather_than_guessed():
    module = _load_module()
    report = module.check_text(_svg(
        '<rect id="flat" x="10" y="10" width="100" height="50" fill="#eee"/>',
        '<g transform="rotate(15 200 100)">',
        '  <rect id="tilted" x="180" y="90" width="60" height="30" fill="#ddd"/>',
        "</g>",
    ))
    assert [box.elem_id for box in report.shapes] == ["rect#flat"]
    assert any("unverifiable" in note and "rotate" in note for note in report.notes)


def test_overlaps_below_min_overlap_are_ignored():
    module = _load_module()
    svg = _svg(
        '<rect id="a" x="10" y="10" width="100" height="50" fill="#eee"/>',
        '<rect id="b" x="105" y="10" width="100" height="50" fill="#ddd"/>',
    )
    assert [f["kind"] for f in module.check_text(svg).findings] == ["collision"]
    assert module.check_text(svg, min_overlap=1000.0).findings == []


def test_a_two_pixel_seam_is_only_flagged_at_zero_tolerance():
    module = _load_module()
    svg = _svg(
        '<rect id="a" x="10" y="10" width="100" height="50" fill="#eee"/>',
        '<rect id="b" x="108" y="10" width="100" height="50" fill="#ddd"/>',
    )
    assert module.check_text(svg, tolerance=2.0).findings == []
    assert [f["kind"] for f in module.check_text(svg, tolerance=0.0).findings] == ["collision"]


# --- CLI --------------------------------------------------------------------------------

def test_cli_exit_codes_and_json_report(tmp_path):
    clean = _run(tmp_path, _svg(
        '<rect id="only" x="10" y="10" width="100" height="50" fill="#eee"/>'))
    assert clean.returncode == 0, clean.stdout + clean.stderr
    assert "0 to fix" in clean.stdout

    collide = _run(tmp_path, _svg(
        '<rect id="a" x="10" y="10" width="100" height="50" fill="#eee"/>',
        '<rect id="b" x="80" y="30" width="100" height="50" fill="#ddd"/>', ), "--json")
    assert collide.returncode == 1
    payload = json.loads(collide.stdout)
    assert payload["failures"] == 1
    assert payload["canvas"]["source"] == "viewBox"
    assert {f["kind"] for f in payload["findings"]} == {"collision"}
    assert set(payload) >= {"input", "canvas", "shapes_checked", "notes", "findings", "failures"}


def test_cli_strict_boundaries_promotes_the_warning(tmp_path):
    svg = _svg(
        '<rect id="frame" x="0" y="0" width="200" height="100" fill="none" stroke="#999" '
        'stroke-dasharray="6 4"/>',
        '<rect id="content" x="150" y="50" width="100" height="50" fill="#eee"/>',
    )
    assert _run(tmp_path, svg).returncode == 0
    strict = _run(tmp_path, svg, "--strict-boundaries")
    assert strict.returncode == 1
    assert "boundary" in strict.stdout


def test_cli_reads_an_html_page_and_stdin(tmp_path):
    page = "<!doctype html><html><body>" + _svg(
        '<rect id="a" x="10" y="10" width="100" height="50" fill="#eee"/>') + "</body></html>"
    target = tmp_path / "page.html"
    target.write_text(page, encoding="utf-8")
    proc = _cli(str(target))
    assert proc.returncode == 0, proc.stderr
    assert "1 shape(s) checked" in proc.stdout

    piped = _cli("-", stdin_text=page)
    assert piped.returncode == 0, piped.stderr


def test_cli_rejects_input_without_an_svg(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_text("just prose", encoding="utf-8")
    proc = _cli(str(target))
    assert proc.returncode == 2
    assert "no <svg> element found" in proc.stderr
    assert proc.stdout == ""


def test_cli_reports_malformed_xml_as_unusable(tmp_path):
    proc = _run(tmp_path, '<svg xmlns="http://www.w3.org/2000/svg"><rect width="10"</svg>')
    assert proc.returncode == 2
    assert "well-formed" in proc.stderr


def test_cli_help_mentions_the_exit_contract():
    proc = _cli("--help")
    assert proc.returncode == 0
    assert "svg" in proc.stdout.lower()


# --- discoverability --------------------------------------------------------------------

def test_the_skill_is_discoverable_and_loadable_from_the_optional_catalog(tmp_path, monkeypatch):
    from tools.skills_hub_official import OptionalSkillSource

    source = OptionalSkillSource()
    source._optional_dir = ROOT / "optional-skills"
    matches = [m for m in source.list_local() if m.name == "visual-artifact-delivery"]
    assert matches, "the skill must be listed in the optional catalog"
    bundle = source.fetch(matches[0].identifier)
    assert bundle is not None
    assert "references/rendering-and-verification.md" in bundle.files
    assert "scripts/svg_geometry_check.py" in bundle.files

    from tools.skills_tool import skill_view

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    destination = tmp_path / "skills" / bundle.name
    destination.mkdir(parents=True)
    for name, data in bundle.files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
    loaded = json.loads(skill_view(bundle.name))
    assert loaded["success"], loaded
    assert "svg_geometry_check.py" in loaded["content"]
    # the copied script still runs from its installed location
    copied = destination / "scripts" / "svg_geometry_check.py"
    installed = subprocess.run([sys.executable, str(copied), "--help"], capture_output=True,
                               text=True, check=False)
    assert installed.returncode == 0


def test_no_machine_local_paths_in_the_shipped_text():
    """Authoring standard: no absolute paths from the author's machine, no host names."""
    for name in ("SKILL.md", "references/rendering-and-verification.md"):
        text = (SKILL_DIR / name).read_text(encoding="utf-8")
        for needle in ("/opt/", "/home/", "/Users/", "C:\\", "/tmp/"):
            assert needle not in text, f"{name} contains {needle}"
