"""Doctor respects model namespaces declared by external provider profiles (#105968)."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestDoctorProfileModelSlugs(unittest.TestCase):
    def check(self, declared_model):
        with tempfile.TemporaryDirectory() as root, patch.dict("os.environ", {
            "HERMES_HOME": root, "HERMES_BUNDLED_PLUGINS": f"{root}/empty-bundled",
            "HERMES_ENABLE_PROJECT_PLUGINS": "0",
        }):
            from hermes_cli.auth import resolve_provider
            from hermes_cli.doctor_config import _validate_model_config

            home = Path(root)
            plugin = home / "plugins" / "model-providers" / "doctor-model-slugs"
            plugin.mkdir(parents=True)
            (plugin / "plugin.yaml").write_text(
                "name: doctor-model-slugs\nkind: model-provider\nversion: 1.0.0\ndescription: Test provider\n")
            (plugin / "__init__.py").write_text(
                "from providers import register_provider\n"
                "from providers.base import ProviderProfile\n"
                "register_provider(ProviderProfile(name='doctor-model-slugs', aliases=('doctor-slug-alias',), "
                f"auth_type='oauth_external', fallback_models=({declared_model!r},)))\n")
            config = home / "config.yaml"
            config.write_text("model:\n  provider: doctor-slug-alias\n  default: vendor/model-new\n")
            self.assertEqual(resolve_provider("doctor-slug-alias"), "doctor-model-slugs")
            issues = []
            _validate_model_config(config, issues)
            return issues

    def test_declared_slash_namespace_accepts_new_live_model_through_alias(self):
        self.assertFalse(self.check("vendor/model-old"))

    def test_flat_namespace_still_warns_about_vendor_prefixed_model(self):
        self.assertTrue(any("vendor-prefixed" in issue for issue in self.check("model-old")))
