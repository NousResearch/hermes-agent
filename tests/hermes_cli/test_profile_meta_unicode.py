"""write_profile_meta must emit real UTF-8 for emoji descriptions.

Regression sibling of GitHub #51356 / PR #51676: without
allow_unicode=True, yaml.safe_dump escapes astral-plane characters
as \\UXXXXXXXX sequences that break under non-PyYAML parsers and
hand-edits.
"""

import yaml

from hermes_cli.profiles import write_profile_meta, read_profile_meta


class TestWriteProfileMetaUnicode:

    def test_emoji_description_written_as_utf8(self, tmp_path):
        profile_dir = tmp_path / "test-profile"
        profile_dir.mkdir()

        write_profile_meta(profile_dir, description="Code wizard 🧙‍♂️✨🔥")

        raw = (profile_dir / "profile.yaml").read_text(encoding="utf-8")
        assert "\\U" not in raw
        assert "\\u" not in raw
        assert "🧙" in raw
        assert "🔥" in raw

    def test_emoji_description_round_trips(self, tmp_path):
        profile_dir = tmp_path / "test-profile"
        profile_dir.mkdir()
        desc = "AI assistant 🤖 — fast & accurate ⚡"

        write_profile_meta(profile_dir, description=desc)
        meta = read_profile_meta(profile_dir)

        assert meta["description"] == desc

    def test_ascii_description_unaffected(self, tmp_path):
        profile_dir = tmp_path / "test-profile"
        profile_dir.mkdir()

        write_profile_meta(profile_dir, description="Plain text description")

        meta = read_profile_meta(profile_dir)
        assert meta["description"] == "Plain text description"
