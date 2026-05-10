"""Tests for ``hermes migrate xai`` — apply path with ruamel round-trip."""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli.xai_retirement import (
    RetirementIssue,
    apply_migration,
    find_retired_xai_refs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def trap_config(tmp_path: Path) -> Path:
    """A config.yaml with retired models AND comments to verify round-trip."""
    p = tmp_path / "config.yaml"
    p.write_text(
        "# Hermes config (sample)\n"
        "principal:\n"
        "  provider: xai             # the main model\n"
        "  model: grok-4-1-fast-non-reasoning  # retiring May 15\n"
        "  temperature: 0.5\n"
        "auxiliary:\n"
        "  vision:\n"
        "    provider: xai\n"
        "    model: grok-4-fast       # retiring\n"
        "  compression:\n"
        "    provider: openai         # not affected\n"
        "    model: gpt-4o-mini\n"
        "delegation:\n"
        "  model: grok-code-fast-1    # retiring\n"
        "plugins:\n"
        "  image_gen:\n"
        "    xai:\n"
        "      model: grok-imagine-image-pro  # retiring\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def clean_config(tmp_path: Path) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(
        "principal:\n"
        "  provider: xai\n"
        "  model: grok-4.3\n",
        encoding="utf-8",
    )
    return p


def _parse(path: Path) -> dict:
    """Load with ruamel for assertion convenience."""
    from ruamel.yaml import YAML
    yaml = YAML(typ="rt")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.load(fh)


# ---------------------------------------------------------------------------
# Dry-run / no-op
# ---------------------------------------------------------------------------

class TestNoOpPaths:
    def test_clean_config_returns_unchanged_result(self, clean_config: Path):
        issues = find_retired_xai_refs(_parse(clean_config))
        assert issues == []
        result = apply_migration(clean_config, issues)
        assert result.config_changed is False
        assert result.backup_path is None
        # File untouched
        assert "grok-4.3" in clean_config.read_text(encoding="utf-8")

    def test_empty_issues_list_is_noop(self, trap_config: Path):
        original = trap_config.read_text(encoding="utf-8")
        result = apply_migration(trap_config, issues=[])
        assert result.config_changed is False
        assert trap_config.read_text(encoding="utf-8") == original

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            apply_migration(tmp_path / "absent.yaml", issues=[
                RetirementIssue(
                    config_path="principal.model",
                    current_model="grok-4",
                    replacement="grok-4.3",
                )
            ])


# ---------------------------------------------------------------------------
# Apply: surgical replacement
# ---------------------------------------------------------------------------

class TestApplyReplacement:
    def test_replaces_principal_model(self, trap_config: Path):
        issues = find_retired_xai_refs(_parse(trap_config))
        result = apply_migration(trap_config, issues)
        assert result.config_changed is True
        cfg = _parse(trap_config)
        assert cfg["principal"]["model"] == "grok-4.3"

    def test_adds_reasoning_effort_for_non_reasoning_variant(self, trap_config: Path):
        issues = find_retired_xai_refs(_parse(trap_config))
        apply_migration(trap_config, issues)
        cfg = _parse(trap_config)
        # Principal was grok-4-1-fast-non-reasoning → reasoning_effort: "none"
        assert cfg["principal"]["reasoning_effort"] == "none"

    def test_replaces_auxiliary_vision(self, trap_config: Path):
        issues = find_retired_xai_refs(_parse(trap_config))
        apply_migration(trap_config, issues)
        cfg = _parse(trap_config)
        assert cfg["auxiliary"]["vision"]["model"] == "grok-4.3"

    def test_replaces_delegation(self, trap_config: Path):
        issues = find_retired_xai_refs(_parse(trap_config))
        apply_migration(trap_config, issues)
        cfg = _parse(trap_config)
        assert cfg["delegation"]["model"] == "grok-4.3"

    def test_replaces_image_gen_plugin(self, trap_config: Path):
        issues = find_retired_xai_refs(_parse(trap_config))
        apply_migration(trap_config, issues)
        cfg = _parse(trap_config)
        assert cfg["plugins"]["image_gen"]["xai"]["model"] == "grok-imagine-image-quality"

    def test_does_not_touch_unrelated_slots(self, trap_config: Path):
        issues = find_retired_xai_refs(_parse(trap_config))
        apply_migration(trap_config, issues)
        cfg = _parse(trap_config)
        # auxiliary.compression was never xAI, must remain untouched
        assert cfg["auxiliary"]["compression"]["model"] == "gpt-4o-mini"
        assert cfg["auxiliary"]["compression"]["provider"] == "openai"
        # principal.temperature must survive
        assert cfg["principal"]["temperature"] == 0.5


# ---------------------------------------------------------------------------
# Round-trip preservation (the hard part)
# ---------------------------------------------------------------------------

class TestRoundTripPreservation:
    def test_preserves_top_of_file_comment(self, trap_config: Path):
        issues = find_retired_xai_refs(_parse(trap_config))
        apply_migration(trap_config, issues)
        text = trap_config.read_text(encoding="utf-8")
        assert "# Hermes config (sample)" in text

    def test_preserves_inline_comments_on_unmodified_lines(self, trap_config: Path):
        issues = find_retired_xai_refs(_parse(trap_config))
        apply_migration(trap_config, issues)
        text = trap_config.read_text(encoding="utf-8")
        assert "# the main model" in text
        assert "# not affected" in text

    def test_preserves_top_level_key_order(self, trap_config: Path):
        issues = find_retired_xai_refs(_parse(trap_config))
        apply_migration(trap_config, issues)
        text = trap_config.read_text(encoding="utf-8")
        order = [
            text.index("principal:"),
            text.index("auxiliary:"),
            text.index("delegation:"),
            text.index("plugins:"),
        ]
        assert order == sorted(order)


# ---------------------------------------------------------------------------
# Backup behaviour
# ---------------------------------------------------------------------------

class TestBackup:
    def test_backup_is_written_by_default(self, trap_config: Path):
        issues = find_retired_xai_refs(_parse(trap_config))
        original = trap_config.read_text(encoding="utf-8")
        result = apply_migration(trap_config, issues)
        assert result.backup_path is not None
        assert result.backup_path.exists()
        assert result.backup_path.read_text(encoding="utf-8") == original

    def test_backup_filename_prefixed(self, trap_config: Path):
        issues = find_retired_xai_refs(_parse(trap_config))
        result = apply_migration(trap_config, issues)
        assert result.backup_path is not None
        assert result.backup_path.name.startswith("config.yaml.bak-pre-migrate-xai-")

    def test_no_backup_when_disabled(self, trap_config: Path):
        issues = find_retired_xai_refs(_parse(trap_config))
        result = apply_migration(trap_config, issues, backup=False)
        assert result.backup_path is None
        # No bak file in the directory
        assert not list(trap_config.parent.glob("*.bak-pre-migrate-xai-*"))

    def test_no_backup_when_no_changes(self, clean_config: Path):
        issues = find_retired_xai_refs(_parse(clean_config))
        result = apply_migration(clean_config, issues, backup=True)
        assert result.backup_path is None  # nothing to back up
        assert not list(clean_config.parent.glob("*.bak-pre-migrate-xai-*"))


# ---------------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------------

class TestIdempotence:
    def test_apply_twice_is_safe(self, trap_config: Path):
        # First pass: replace
        issues_1 = find_retired_xai_refs(_parse(trap_config))
        apply_migration(trap_config, issues_1)
        # Second pass: nothing to do
        issues_2 = find_retired_xai_refs(_parse(trap_config))
        assert issues_2 == []
        result_2 = apply_migration(trap_config, issues_2)
        assert result_2.config_changed is False
