"""Tests for pipeline config (Phase A.6)."""
import pytest
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import yaml

# Add the project root to Python path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class TestPipelineConfigDefault:
    """Test that DEFAULT_CONFIG has the pipeline section."""

    def test_default_config_has_pipeline(self):
        """DEFAULT_CONFIG should include a 'pipeline' section."""
        from hermes_cli.config import DEFAULT_CONFIG
        assert "pipeline" in DEFAULT_CONFIG
        pipeline = DEFAULT_CONFIG["pipeline"]
        assert isinstance(pipeline, dict)

    def test_default_config_pipeline_keys(self):
        """Pipeline defaults should have all required keys."""
        from hermes_cli.config import DEFAULT_CONFIG
        pipeline = DEFAULT_CONFIG["pipeline"]
        assert "max_revise_loops" in pipeline
        assert "token_cap" in pipeline
        assert "express_enabled" in pipeline
        assert "artifact_dir" in pipeline
        assert "stage_owners" in pipeline

    def test_default_pipeline_max_revise_loops(self):
        """Default max_revise_loops should be 4."""
        from hermes_cli.config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["pipeline"]["max_revise_loops"] == 4

    def test_default_pipeline_token_cap_is_none(self):
        """Default token_cap should be None (no cap)."""
        from hermes_cli.config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["pipeline"]["token_cap"] is None

    def test_default_pipeline_express_enabled(self):
        """Default express_enabled should be True."""
        from hermes_cli.config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["pipeline"]["express_enabled"] is True

    def test_default_pipeline_artifact_dir(self):
        """Default artifact_dir should be 'feature-artifacts'."""
        from hermes_cli.config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["pipeline"]["artifact_dir"] == "feature-artifacts"

    def test_default_pipeline_stage_owners(self):
        """Default stage_owners should have research, prd, spec, council."""
        from hermes_cli.config import DEFAULT_CONFIG
        owners = DEFAULT_CONFIG["pipeline"]["stage_owners"]
        assert isinstance(owners, dict)
        assert "research" in owners
        assert "prd" in owners
        assert "spec" in owners
        assert "council" in owners


class TestGetPipelineConfig:
    """Test the get_pipeline_config() function."""

    @patch("hermes_cli.config.load_config_readonly")
    def test_returns_defaults_when_empty(self, mock_load):
        """Should return merged defaults when config has no pipeline section."""
        mock_load.return_value = {}
        from hermes_cli.config import get_pipeline_config
        result = get_pipeline_config()
        assert result["max_revise_loops"] == 4
        assert result["token_cap"] is None

    @patch("hermes_cli.config.load_config_readonly")
    def test_overrides_defaults(self, mock_load):
        """Should override defaults with user config values."""
        mock_load.return_value = {
            "pipeline": {
                "max_revise_loops": 8,
                "token_cap": 100000,
            }
        }
        from hermes_cli.config import get_pipeline_config
        result = get_pipeline_config()
        assert result["max_revise_loops"] == 8
        assert result["token_cap"] == 100000
        # Defaults should still be present
        assert result["express_enabled"] is True

    @patch("hermes_cli.config.load_config_readonly")
    def test_deep_merges_stage_owners(self, mock_load):
        """Should deep-merge stage_owners, not replace entirely."""
        mock_load.return_value = {
            "pipeline": {
                "stage_owners": {
                    "research": "custom-research-profile",
                }
            }
        }
        from hermes_cli.config import get_pipeline_config
        result = get_pipeline_config()
        # research should be overridden
        assert result["stage_owners"]["research"] == "custom-research-profile"
        # Other defaults should still be present (prd maps to the spawnable
        # kensei-review sub-profile, NOT the nonspawnable root 'kensei' —
        # a nonspawnable owner silently starves the PRD stage)
        assert result["stage_owners"]["prd"] == "kensei-review"
        assert result["stage_owners"]["spec"] == "octacon"


class TestPipelineValidation:
    """Test pipeline config validation in validate_config.py."""

    def test_valid_config_passes(self):
        """Valid pipeline config should produce no warnings.

        The spawnability gate (added 2026-08-15) flags stage_owners values
        that have no profile dir under HERMES_HOME. Create a 'remii' profile
        dir so the owner is legitimately spawnable in this temp home.
        """
        from hermes_cli.validate_config import run_validate_config
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "profiles" / "remii").mkdir(parents=True)
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(yaml.dump({
                "pipeline": {
                    "max_revise_loops": 4,
                    "token_cap": None,
                    "stage_owners": {"research": "remii"},
                }
            }))
            with patch.dict(os.environ, {"HERMES_HOME": tmpdir}):
                result = run_validate_config(None)
                assert result == 0

    def test_nonspawnable_stage_owner_errors(self):
        """A stage_owners value in nonspawnable_profiles must error."""
        from hermes_cli.validate_config import run_validate_config
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(yaml.dump({
                "kanban": {"nonspawnable_profiles": ["kensei"]},
                "pipeline": {
                    "stage_owners": {"prd": "kensei"},
                }
            }))
            with patch.dict(os.environ, {"HERMES_HOME": tmpdir}):
                result = run_validate_config(None)
                assert result == 1

    def test_missing_profile_dir_stage_owner_errors(self):
        """A stage_owners value with no profile dir must error."""
        from hermes_cli.validate_config import run_validate_config
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(yaml.dump({
                "pipeline": {
                    "stage_owners": {"prd": "ghost-profile"},
                }
            }))
            with patch.dict(os.environ, {"HERMES_HOME": tmpdir}):
                result = run_validate_config(None)
                assert result == 1

    def test_invalid_max_revise_loops_warns(self):
        """Non-positive max_revise_loops should produce warning."""
        from hermes_cli.validate_config import run_validate_config
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(yaml.dump({
                "pipeline": {
                    "max_revise_loops": -1,
                }
            }))
            with patch.dict(os.environ, {"HERMES_HOME": tmpdir}):
                result = run_validate_config(None)
                assert result == 1

    def test_invalid_token_cap_warns(self):
        """Non-positive token_cap should produce warning."""
        from hermes_cli.validate_config import run_validate_config
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(yaml.dump({
                "pipeline": {
                    "token_cap": 0,
                }
            }))
            with patch.dict(os.environ, {"HERMES_HOME": tmpdir}):
                result = run_validate_config(None)
                assert result == 1

    def test_invalid_stage_owners_type_warns(self):
        """Non-dict stage_owners should produce warning."""
        from hermes_cli.validate_config import run_validate_config
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(yaml.dump({
                "pipeline": {
                    "stage_owners": "not-a-dict",
                }
            }))
            with patch.dict(os.environ, {"HERMES_HOME": tmpdir}):
                result = run_validate_config(None)
                assert result == 1

    def test_missing_pipeline_section_passes(self):
        """Missing pipeline section should pass (defaults apply)."""
        from hermes_cli.validate_config import run_validate_config
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(yaml.dump({}))
            with patch.dict(os.environ, {"HERMES_HOME": tmpdir}):
                result = run_validate_config(None)
                assert result == 0
