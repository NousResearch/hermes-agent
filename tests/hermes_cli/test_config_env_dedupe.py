"""save_env_value collapses duplicate ``KEY=`` lines so the saved value is the one that loads."""

import os
from unittest.mock import patch

from hermes_cli.config import load_env, save_env_value


class TestSaveEnvValueDeduplication:
    """#8270: the loader is last-assignment-wins, so a save that rewrites only the first ``KEY=`` line
    leaves a stale duplicate below it that silently shadows the new value."""

    def test_save_drops_stale_duplicates_and_loader_sees_saved_value(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "FOO=top\n"
            "OPENROUTER_API_KEY=first\n"
            "BAR=middle\n"
            "export OPENROUTER_API_KEY=stale-bottom\n"
        )
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            save_env_value("OPENROUTER_API_KEY", "winner")

            lines = [ln for ln in env_file.read_text().splitlines() if ln.strip()]
            assert lines == ["FOO=top", "OPENROUTER_API_KEY=winner", "BAR=middle"]
            assert load_env()["OPENROUTER_API_KEY"] == "winner"

    def test_dedupe_retains_quoted_serialization_and_round_trips(self, tmp_path):
        # The retained line must go through the same quoting as the append path, or a value needing
        # dotenv quoting would be corrupted on load while the stale duplicate is dropped.
        needs_quoting = "sk live#not-a-comment with spaces"
        env_file = tmp_path / ".env"
        env_file.write_text("OPENROUTER_API_KEY=old\nKEEP=x\nOPENROUTER_API_KEY=stale\n")
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            save_env_value("OPENROUTER_API_KEY", needs_quoting)

            content = env_file.read_text()
            assert sum(ln.startswith("OPENROUTER_API_KEY=") for ln in content.splitlines()) == 1
            assert "OPENROUTER_API_KEY=stale" not in content
            assert load_env()["OPENROUTER_API_KEY"] == needs_quoting
