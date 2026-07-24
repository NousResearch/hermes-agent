"""Tests for the wallpaper-engine optional skill.

Structural + unit tests for the wallpaper generation, desktop setting, and
preference-tracking scripts.  Stdlib + pytest only; no network calls.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "creative"
    / "wallpaper-engine"
)
SKILL_MD = SKILL_DIR / "SKILL.md"
SET_WALLPAPER_PY = SKILL_DIR / "scripts" / "set_wallpaper.py"
HISTORY_PY = SKILL_DIR / "scripts" / "wallpaper_history.py"
WORKFLOW_JSON = SKILL_DIR / "workflows" / "wallpaper_txt2img.json"
PROMPT_LIBRARY = SKILL_DIR / "references" / "prompt-library.md"


@pytest.fixture(scope="module")
def skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def set_wallpaper_mod():
    """Import set_wallpaper as a module for unit testing."""
    sys.path.insert(0, str(SET_WALLPAPER_PY.parent))
    import set_wallpaper as mod
    return mod


@pytest.fixture(scope="module")
def history_mod():
    """Import wallpaper_history as a module for unit testing."""
    sys.path.insert(0, str(HISTORY_PY.parent))
    import wallpaper_history as mod
    return mod


# ===========================================================================
# Structural tests — SKILL.md
# ===========================================================================


def test_skill_file_exists():
    assert SKILL_MD.is_file(), f"missing {SKILL_MD}"


def test_frontmatter_present(skill_text: str):
    assert skill_text.startswith("---\n"), "SKILL.md must open with YAML frontmatter"
    assert skill_text.count("---") >= 2, "frontmatter must be delimited by two '---'"


def test_name_in_frontmatter(skill_text: str):
    m = re.search(r"^name: (.*)$", skill_text, re.MULTILINE)
    assert m, "missing 'name:' in frontmatter"
    assert m.group(1).strip() == "wallpaper-engine"


def test_description_under_sixty_chars(skill_text: str):
    m = re.search(r"^description: (.*)$", skill_text, re.MULTILINE)
    assert m, "no description field"
    desc = m.group(1).strip()
    assert len(desc) <= 60, f"description is {len(desc)} chars (>60): {desc!r}"
    assert desc.endswith("."), "description should end with a period"


def test_platforms_declared(skill_text: str):
    m = re.search(r"^platforms: (.*)$", skill_text, re.MULTILINE)
    assert m, "platforms field required"
    for os_name in ("linux", "macos", "windows"):
        assert os_name in m.group(1), f"platforms must include {os_name}"


def test_required_sections_present(skill_text: str):
    for heading in (
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ):
        assert heading in skill_text, f"missing section: {heading}"


def test_comfyui_skill_referenced(skill_text: str):
    """The wallpaper engine delegates generation to the comfyui skill."""
    assert "comfyui" in skill_text.lower()
    assert "run_workflow.py" in skill_text


def test_memory_integration_documented(skill_text: str):
    """Preference learning must reference Hermes memory."""
    assert "memory" in skill_text.lower()
    assert "memory(action=add)" in skill_text or "memory(action=read)" in skill_text


def test_cron_scheduling_documented(skill_text: str):
    """Scheduling must reference Hermes cron."""
    assert "cron" in skill_text.lower()
    assert "hermes cron" in skill_text


def test_set_wallpaper_referenced(skill_text: str):
    assert "set_wallpaper.py" in skill_text


def test_wallpaper_history_referenced(skill_text: str):
    assert "wallpaper_history.py" in skill_text


def test_related_skills_declared(skill_text: str):
    assert "related_skills:" in skill_text
    assert "comfyui" in skill_text


# ===========================================================================
# Supporting files existence
# ===========================================================================


def test_set_wallpaper_script_exists():
    assert SET_WALLPAPER_PY.is_file(), f"missing {SET_WALLPAPER_PY}"


def test_wallpaper_history_script_exists():
    assert HISTORY_PY.is_file(), f"missing {HISTORY_PY}"


def test_workflow_json_exists():
    assert WORKFLOW_JSON.is_file(), f"missing {WORKFLOW_JSON}"


def test_prompt_library_exists():
    assert PROMPT_LIBRARY.is_file(), f"missing {PROMPT_LIBRARY}"


def test_workflow_json_is_valid_api_format():
    """Workflow must have only node objects at top level (no _comment, strings, etc.)."""
    wf = json.loads(WORKFLOW_JSON.read_text(encoding="utf-8"))
    assert isinstance(wf, dict), "workflow must be a JSON object"
    for key, value in wf.items():
        assert isinstance(value, dict), (
            f"top-level key '{key}' is a {type(value).__name__}, not a dict. "
            f"Non-node keys like _comment cause ComfyUI validate_prompt() to reject the payload."
        )
        assert "class_type" in value, (
            f"node '{key}' missing 'class_type' field"
        )
    # Must have the core nodes for a txt2img workflow
    class_types = {v["class_type"] for v in wf.values()}
    for required in ("KSampler", "CheckpointLoaderSimple", "CLIPTextEncode",
                     "VAEDecode", "SaveImage", "EmptyLatentImage"):
        assert required in class_types, f"workflow missing required node: {required}"


def test_workflow_has_wallpaper_resolution():
    wf = json.loads(WORKFLOW_JSON.read_text(encoding="utf-8"))
    for node in wf.values():
        if node.get("class_type") == "EmptyLatentImage":
            inputs = node.get("inputs", {})
            assert inputs.get("width") == 1920, "wallpaper workflow should default to 1920 width"
            assert inputs.get("height") == 1080, "wallpaper workflow should default to 1080 height"
            return
    pytest.fail("No EmptyLatentImage node found in workflow")


def test_workflow_save_prefix_is_wallpaper():
    wf = json.loads(WORKFLOW_JSON.read_text(encoding="utf-8"))
    for node in wf.values():
        if node.get("class_type") == "SaveImage":
            prefix = node.get("inputs", {}).get("filename_prefix", "")
            assert "wallpaper" in prefix, (
                f"SaveImage filename_prefix should contain 'wallpaper', got {prefix!r}"
            )
            return
    pytest.fail("No SaveImage node found in workflow")


# ===========================================================================
# set_wallpaper.py unit tests
# ===========================================================================


class TestSetWallpaper:
    """Unit tests for set_wallpaper.py — mock subprocess to avoid real desktop calls."""

    def test_file_not_found(self, set_wallpaper_mod):
        result = set_wallpaper_mod.set_wallpaper("/nonexistent/path/image.png")
        assert result["status"] == "error"
        assert "File not found" in result["error"]

    def test_invalid_fit_mode(self, set_wallpaper_mod, tmp_path):
        img = tmp_path / "test.png"
        img.write_text("fake image")
        result = set_wallpaper_mod.set_wallpaper(str(img), mode="bogus")
        assert result["status"] == "error"
        assert "Unknown fit mode" in result["error"]

    def test_all_fit_modes_are_valid(self, set_wallpaper_mod):
        for mode in ("fill", "center", "stretch", "fit", "tile"):
            assert mode in set_wallpaper_mod.FIT_MODES, f"mode {mode!r} missing from FIT_MODES"

    def test_gnome_success(self, set_wallpaper_mod, tmp_path, monkeypatch):
        """GNOME gsettings returns code 0 → wallpaper set successfully."""
        img = tmp_path / "test.png"
        img.write_text("fake image")

        fake_run = mock.Mock(return_value=mock.Mock(returncode=0))
        monkeypatch.setattr(set_wallpaper_mod, "_run", fake_run)
        monkeypatch.setattr(set_wallpaper_mod, "_which", lambda x: f"/usr/bin/{x}")
        # Force Linux chain
        monkeypatch.setattr(set_wallpaper_mod.sys, "platform", "linux")

        result = set_wallpaper_mod.set_wallpaper(str(img))
        assert result["status"] == "ok"
        assert result["method"] == "gnome"

    def test_macos_success(self, set_wallpaper_mod, tmp_path, monkeypatch):
        """macOS osascript returns code 0 → wallpaper set."""
        img = tmp_path / "test.png"
        img.write_text("fake image")

        fake_run = mock.Mock(return_value=mock.Mock(returncode=0))
        monkeypatch.setattr(set_wallpaper_mod, "_run", fake_run)
        monkeypatch.setattr(set_wallpaper_mod, "_which", lambda x: f"/usr/bin/{x}")
        monkeypatch.setattr(set_wallpaper_mod.sys, "platform", "darwin")

        result = set_wallpaper_mod.set_wallpaper(str(img))
        assert result["status"] == "ok"
        assert result["method"] == "osascript"

    def test_windows_success(self, set_wallpaper_mod, tmp_path, monkeypatch):
        """Windows SystemParametersInfoW succeeds."""
        img = tmp_path / "test.png"
        img.write_text("fake image")

        # ctypes.windll only exists on Windows — test _try_windows directly
        monkeypatch.setattr(set_wallpaper_mod.ctypes, "windll",
                           mock.MagicMock(user32=mock.MagicMock(
                               SystemParametersInfoW=mock.Mock(return_value=True))),
                           raising=False)
        assert set_wallpaper_mod._try_windows(str(img), "fill") is True

    def test_all_setters_fail(self, set_wallpaper_mod, tmp_path, monkeypatch):
        """When no desktop environment is found, return error with useful message."""
        img = tmp_path / "test.png"
        img.write_text("fake image")

        # which returns None (no tools found) → every setter returns False
        monkeypatch.setattr(set_wallpaper_mod, "_which", lambda x: None)
        monkeypatch.setattr(set_wallpaper_mod.sys, "platform", "linux")

        result = set_wallpaper_mod.set_wallpaper(str(img))
        assert result["status"] == "error"
        assert "No supported desktop environment" in result["error"]

    def test_image_uri_formats_correctly(self, set_wallpaper_mod, tmp_path):
        img = tmp_path / "my image.png"
        img.write_text("fake")
        uri = set_wallpaper_mod._image_uri(str(img))
        assert uri.startswith("file://")
        assert "my%20image.png" not in uri or "my image.png" in uri

    def test_feh_flag_mapping(self, set_wallpaper_mod):
        assert set_wallpaper_mod.FIT_MODES["fill"]["feh"] == "--bg-fill"
        assert set_wallpaper_mod.FIT_MODES["center"]["feh"] == "--bg-center"
        assert set_wallpaper_mod.FIT_MODES["tile"]["feh"] == "--bg-tile"

    def test_mode_values_match_across_backends(self, set_wallpaper_mod):
        """Every backend defined in FIT_MODES has an entry for every mode."""
        backends = {"gsettings", "feh", "sway", "kde", "xfce"}
        for mode, backends_dict in set_wallpaper_mod.FIT_MODES.items():
            for backend in backends:
                assert backend in backends_dict, (
                    f"mode {mode!r} missing backend {backend!r}"
                )


# ===========================================================================
# wallpaper_history.py unit tests
# ===========================================================================


class TestWallpaperHistory:
    """Unit tests for wallpaper_history.py — uses temp JSON files, no real data."""

    @pytest.fixture(autouse=True)
    def isolated_history(self, history_mod, tmp_path, monkeypatch):
        """Redirect history storage to a temp directory for each test."""
        data_dir = tmp_path / "wallpaper-engine"
        data_dir.mkdir()
        monkeypatch.setattr(history_mod, "_data_dir", lambda: data_dir)
        # Reset module-level state if any
        return data_dir

    def _add(self, history_mod, image_path, prompt, workflow, meta=None):
        """Helper: call cmd_add programmatically."""
        args = [image_path, prompt, workflow]
        if meta:
            args.extend(["--meta", json.dumps(meta)])
        history_mod.cmd_add(args)

    def _feedback(self, history_mod, entry_id, rating, tags=()):
        args = [entry_id, str(rating), *tags]
        history_mod.cmd_feedback(args)

    def _list(self, history_mod, *extra):
        history_mod.cmd_list(list(extra))

    def _stats(self, history_mod):
        history_mod.cmd_stats([])

    def test_add_creates_record(self, history_mod, tmp_path):
        img = tmp_path / "wallpaper.png"
        img.write_text("fake")
        self._add(history_mod, str(img), "a beautiful sunset", "test_workflow.json")

        history = history_mod._load()
        assert len(history) == 1
        entry = history[0]
        assert entry["prompt"] == "a beautiful sunset"
        assert entry["workflow"] == "test_workflow.json"
        assert entry["rating"] is None
        assert entry["tags"] == []
        assert "id" in entry
        assert "timestamp" in entry

    def test_add_with_metadata(self, history_mod, tmp_path):
        img = tmp_path / "wp.png"
        img.write_text("fake")
        self._add(history_mod, str(img), "test", "wf.json",
                  meta={"resolution": "1920x1080", "seed": 42})

        history = history_mod._load()
        assert history[0]["meta"] == {"resolution": "1920x1080", "seed": 42}

    def test_add_invalid_meta_falls_back_to_empty(self, history_mod, tmp_path):
        img = tmp_path / "wp.png"
        img.write_text("fake")
        # Force JSON parse error by passing invalid JSON
        history_mod.cmd_add([str(img), "test", "wf.json", "--meta", "not-json}"])

        history = history_mod._load()
        assert history[0]["meta"] == {}

    def test_feedback_updates_entry(self, history_mod, tmp_path):
        img = tmp_path / "wp.png"
        img.write_text("fake")
        self._add(history_mod, str(img), "test", "wf.json")
        entry_id = history_mod._load()[0]["id"]

        self._feedback(history_mod, entry_id, 5, ("dark", "moody"))

        history = history_mod._load()
        assert history[0]["rating"] == 5
        assert history[0]["tags"] == ["dark", "moody"]
        assert history[0]["rated_at"] is not None

    def test_feedback_nonexistent_id(self, history_mod):
        with pytest.raises(SystemExit) as exc_info:
            self._feedback(history_mod, "nonexistent", 3)
        assert exc_info.value.code == 1

    def test_list_returns_newest_first(self, history_mod, tmp_path):
        img = tmp_path / "wp.png"
        img.write_text("fake")
        self._add(history_mod, str(img), "first", "wf.json")
        self._add(history_mod, str(img), "second", "wf.json")
        self._add(history_mod, str(img), "third", "wf.json")

        # Capture stdout from cmd_list
        import io
        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            self._list(history_mod)

        output = json.loads(buf.getvalue())
        assert output["status"] == "ok"
        assert output["count"] == 3
        # Newest first
        prompts = [e["prompt"] for e in output["entries"]]
        assert prompts == ["third", "second", "first"]

    def test_list_rated_only(self, history_mod, tmp_path):
        img = tmp_path / "wp.png"
        img.write_text("fake")
        self._add(history_mod, str(img), "unrated", "wf.json")
        self._add(history_mod, str(img), "rated", "wf.json")
        entry_id = history_mod._load()[1]["id"]
        self._feedback(history_mod, entry_id, 4, ("cool",))

        import io
        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            self._list(history_mod, "--rated-only")

        output = json.loads(buf.getvalue())
        assert output["count"] == 1
        assert output["entries"][0]["prompt"] == "rated"

    def test_list_limit(self, history_mod, tmp_path):
        img = tmp_path / "wp.png"
        img.write_text("fake")
        for i in range(10):
            self._add(history_mod, str(img), f"prompt-{i}", "wf.json")

        import io
        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            self._list(history_mod, "--limit", "3")

        output = json.loads(buf.getvalue())
        assert output["count"] == 3

    def test_get_retrieves_full_entry(self, history_mod, tmp_path):
        img = tmp_path / "wp.png"
        img.write_text("fake")
        self._add(history_mod, str(img), "my prompt text", "my_workflow.json")
        entry_id = history_mod._load()[0]["id"]

        import io
        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            history_mod.cmd_get([entry_id])

        output = json.loads(buf.getvalue())
        assert output["status"] == "ok"
        assert output["entry"]["prompt"] == "my prompt text"
        assert "image_path" in output["entry"]

    def test_get_nonexistent_id(self, history_mod):
        with pytest.raises(SystemExit) as exc_info:
            history_mod.cmd_get(["nonexistent"])
        assert exc_info.value.code == 1

    def test_stats_empty_history(self, history_mod):
        import io
        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            self._stats(history_mod)

        output = json.loads(buf.getvalue())
        assert output["status"] == "ok"
        assert output["total"] == 0
        assert output["rated"] == 0

    def test_stats_with_ratings(self, history_mod, tmp_path):
        img = tmp_path / "wp.png"
        img.write_text("fake")

        # Add 5 entries with varied ratings
        self._add(history_mod, str(img), "p1", "wf.json")
        self._add(history_mod, str(img), "p2", "wf.json")
        self._add(history_mod, str(img), "p3", "wf.json")
        self._add(history_mod, str(img), "p4", "wf.json")
        self._add(history_mod, str(img), "p5", "wf.json")

        history = history_mod._load()
        self._feedback(history_mod, history[0]["id"], 5, ("dark", "moody", "mountains"))
        self._feedback(history_mod, history[1]["id"], 4, ("dark", "space"))
        self._feedback(history_mod, history[2]["id"], 3, ("bright",))
        self._feedback(history_mod, history[3]["id"], 2, ("bright", "abstract"))
        self._feedback(history_mod, history[4]["id"], 1, ("bright", "noisy"))

        import io
        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            self._stats(history_mod)

        output = json.loads(buf.getvalue())
        assert output["total_generated"] == 5
        assert output["total_rated"] == 5
        assert output["average_rating"] == 3.0  # (5+4+3+2+1)/5
        # "dark" appears in 2 loved (4-5), "bright" in 2 disliked (1-2)
        loved = dict(output["loved_tags"])
        disliked = dict(output["disliked_tags"])
        assert loved.get("dark", 0) >= 1
        assert disliked.get("bright", 0) >= 1
        # Top tags overall
        assert len(output["top_tags"]) > 0

    def test_stats_only_rated(self, history_mod, tmp_path):
        """Unrated entries don't affect stats."""
        img = tmp_path / "wp.png"
        img.write_text("fake")
        self._add(history_mod, str(img), "unrated", "wf.json")
        self._add(history_mod, str(img), "rated", "wf.json")
        history = history_mod._load()
        self._feedback(history_mod, history[1]["id"], 5, ("awesome",))

        import io
        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            self._stats(history_mod)

        output = json.loads(buf.getvalue())
        assert output["total_generated"] == 2
        assert output["total_rated"] == 1
        assert output["average_rating"] == 5.0

    def test_history_directory_created_on_save(self, history_mod, tmp_path):
        """_save creates the parent directory if it doesn't exist."""
        img = tmp_path / "wp.png"
        img.write_text("fake")
        self._add(history_mod, str(img), "test", "wf.json")
        assert history_mod._data_dir().is_dir()
        assert history_mod._history_path().is_file()

    def test_load_corrupted_file_returns_empty(self, history_mod):
        """Corrupted JSON returns an empty list rather than crashing."""
        history_mod._history_path().write_text("not valid json {{{")
        assert history_mod._load() == []

    def test_load_non_list_returns_empty(self, history_mod):
        """A JSON object (not a list) at the history path returns empty list."""
        history_mod._history_path().write_text('{"not": "a list"}')
        assert history_mod._load() == []

    def test_add_preserves_existing_entries(self, history_mod, tmp_path):
        img = tmp_path / "wp.png"
        img.write_text("fake")
        self._add(history_mod, str(img), "first", "wf.json")
        self._add(history_mod, str(img), "second", "wf.json")
        assert len(history_mod._load()) == 2

    def test_load_creates_no_file_when_missing(self, history_mod):
        """_load() should not create a file when none exists."""
        hp = history_mod._history_path()
        assert not hp.exists()
        result = history_mod._load()
        assert result == []
        assert not hp.exists()  # still shouldn't exist


# ===========================================================================
# Prompt library tests
# ===========================================================================


def test_prompt_library_has_categories():
    text = PROMPT_LIBRARY.read_text(encoding="utf-8")
    for category in ("Nature", "Abstract", "Dark", "Sci-Fi", "Seasonal"):
        assert category in text, f"prompt library missing category: {category}"


def test_prompt_library_has_usage_notes():
    text = PROMPT_LIBRARY.read_text(encoding="utf-8")
    assert "## Usage Notes" in text
    assert "seed" in text.lower()
