"""Test that hermes_cli.setup exposes shutil for Matrix E2EE auto-install."""
import shutil


class TestSetupShutilImport:
    def test_setup_module_exposes_shutil_runtime(self):
        """setup module should expose the stdlib shutil module at runtime."""
        import hermes_cli.setup as setup_module

        assert setup_module.shutil is shutil
        assert setup_module.shutil.which("python") == shutil.which("python")
