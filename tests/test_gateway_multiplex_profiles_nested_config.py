"""Regression test for #52562 — nested gateway.multiplex_profiles config must be loaded."""
from pathlib import Path
from unittest.mock import patch, MagicMock


def _load_with_yaml_dict(yaml_dict: dict):
    """Patch filesystem so load_gateway_config() sees *yaml_dict* as config.yaml."""
    from gateway.config import load_gateway_config

    fake_home = Path("/tmp/fake_hermes_home_52562")

    def fake_exists(self):
        return str(self).endswith("config.yaml")

    with patch("gateway.config.get_hermes_home", return_value=fake_home), \
         patch.object(Path, "exists", fake_exists), \
         patch("builtins.open", create=True) as mock_file:
        mock_file.return_value.__enter__ = lambda s: s
        mock_file.return_value.__exit__ = MagicMock(return_value=False)
        with patch("yaml.safe_load", return_value=yaml_dict):
            return load_gateway_config()


class TestMultiplexProfilesNestedConfig:
    def test_top_level_multiplex_profiles(self):
        """Top-level multiplex_profiles key is loaded directly."""
        cfg = _load_with_yaml_dict({"multiplex_profiles": True})
        assert cfg.multiplex_profiles is True

    def test_nested_gateway_multiplex_profiles(self):
        """Regression for #52562: gateway.multiplex_profiles written by
        ``hermes config set gateway.multiplex_profiles true`` must be loaded."""
        cfg = _load_with_yaml_dict({"gateway": {"multiplex_profiles": True}})
        assert cfg.multiplex_profiles is True

    def test_top_level_takes_precedence(self):
        """Top-level key wins over nested gateway section."""
        cfg = _load_with_yaml_dict({
            "multiplex_profiles": False,
            "gateway": {"multiplex_profiles": True},
        })
        assert cfg.multiplex_profiles is False

    def test_default_is_false(self):
        """When neither key is present, default is False."""
        cfg = _load_with_yaml_dict({})
        assert cfg.multiplex_profiles is False

    def test_nested_gateway_section_not_dict(self):
        """When gateway section is not a dict, no crash."""
        cfg = _load_with_yaml_dict({"gateway": "invalid"})
        assert cfg.multiplex_profiles is False
