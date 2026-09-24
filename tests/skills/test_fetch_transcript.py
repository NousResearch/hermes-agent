"""Tests for skills/media/youtube-content/scripts/fetch_transcript.py (issue #22243)."""

import re
import shutil
import subprocess
import sys
from pathlib import Path


from agent.skill_preprocessing import preprocess_skill_content

SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / "media" / "youtube-content"
SCRIPTS_DIR = SKILL_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import fetch_transcript


class TestExtractVideoId:
    def test_standard_watch_url(self):
        assert fetch_transcript.extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_short_url(self):
        assert fetch_transcript.extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


    def test_shorts_url(self):
        assert fetch_transcript.extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


    def test_with_extra_params(self):
        assert fetch_transcript.extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42") == "dQw4w9WgXcQ"


class TestSkillInvocation:
    def test_preprocessed_script_path_runs_from_another_directory(self, tmp_path):
        copied_skill = tmp_path / "youtube skill"
        copied_scripts = copied_skill / "scripts"
        copied_scripts.mkdir(parents=True)
        copied_script = copied_scripts / "fetch_transcript.py"
        shutil.copy2(SCRIPTS_DIR / "fetch_transcript.py", copied_script)

        content = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
        rendered = preprocess_skill_content(content, copied_skill, skills_cfg={})
        command = next(
            line for line in rendered.splitlines()
            if line.startswith("uv run --no-project --with youtube-transcript-api==1.2.4 python")
        )
        match = re.fullmatch(r'uv run .+ python "(.+)" "https://youtube\.com/watch\?v=VIDEO_ID"', command)
        assert match is not None
        rendered_script = Path(match.group(1))
        assert rendered_script == copied_script

        other_cwd = tmp_path / "unrelated cwd"
        other_cwd.mkdir()
        result = subprocess.run(
            [sys.executable, str(rendered_script), "--help"],
            cwd=other_cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


class TestMissingDependency:
    def test_error_identifies_the_interpreter_without_suggesting_bare_uv_install(self, monkeypatch, capsys):
        real_import = __import__

        def import_without_youtube(name, *args, **kwargs):
            if name == "youtube_transcript_api":
                raise ImportError
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", import_without_youtube)

        with pytest.raises(SystemExit, match="1"):
            fetch_transcript.fetch_transcript("dQw4w9WgXcQ")

        error = capsys.readouterr().err
        assert sys.executable in error
        assert "uv pip install" not in error


class TestFormatTimestamp:
    def test_seconds_only(self):
        assert fetch_transcript.format_timestamp(90) == "1:30"


    def test_zero(self):
        assert fetch_transcript.format_timestamp(0) == "0:00"

    def test_minutes_only(self):
        assert fetch_transcript.format_timestamp(600) == "10:00"





