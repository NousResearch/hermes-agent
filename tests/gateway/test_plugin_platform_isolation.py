from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


class PluginPlatformIsolationTests(unittest.TestCase):
    def _run_isolated(self, code: str, *, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
        merged_env = os.environ.copy()
        merged_env.update(env or {})
        proc = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            cwd=str(cwd or REPO),
            env=merged_env,
            text=True,
            capture_output=True,
            timeout=20,
        )
        if proc.returncode != 0:
            self.fail(f"subprocess failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        return proc.stdout

    def test_telegram_only_load_gateway_config_does_not_import_teams(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            (home / "config.yaml").write_text(
                "platforms:\n  telegram:\n    enabled: true\n    token: profile-token\n",
                encoding="utf-8",
            )
            out = self._run_isolated(
                """
                import os, sys
                os.environ['HERMES_HOME'] = r'''%s'''
                os.environ.pop('TEAMS_CLIENT_ID', None)
                os.environ.pop('TEAMS_CLIENT_SECRET', None)
                os.environ.pop('TEAMS_TENANT_ID', None)
                from gateway.config import load_gateway_config
                cfg = load_gateway_config()
                assert 'plugins.platforms.teams.adapter' not in sys.modules
                assert 'microsoft_teams.apps.app' not in sys.modules
                assert 'microsoft_teams.apps' not in sys.modules
                print('ok', [p.value for p in cfg.platforms])
                """ % home
            )
            self.assertIn("ok", out)

    def test_profile_token_fingerprint_unchanged_after_config_load(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            (home / "config.yaml").write_text("platforms:\n  telegram:\n    enabled: true\n", encoding="utf-8")
            out = self._run_isolated(
                """
                import hashlib, os
                os.environ['HERMES_HOME'] = r'''%s'''
                os.environ['TELEGRAM_BOT_TOKEN'] = 'profile-token-sentinel'
                before = hashlib.sha256(os.environ['TELEGRAM_BOT_TOKEN'].encode()).hexdigest()
                from gateway.config import load_gateway_config
                load_gateway_config()
                after = hashlib.sha256(os.environ['TELEGRAM_BOT_TOKEN'].encode()).hexdigest()
                assert before == after
                print(before[:12])
                """ % home
            )
            self.assertTrue(out.strip())

    def test_source_directory_cwd_does_not_pull_root_env_via_teams_import(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as cwd:
            home = Path(d)
            cwd_path = Path(cwd)
            (home / "config.yaml").write_text(
                "platforms:\n  telegram:\n    enabled: true\n    token: profile-token\n",
                encoding="utf-8",
            )
            (cwd_path / ".env").write_text("TEAMS_CLIENT_ID=ROOT_SENTINEL\n", encoding="utf-8")
            out = self._run_isolated(
                """
                import os, sys
                os.environ['HERMES_HOME'] = r'''%s'''
                os.environ.pop('TEAMS_CLIENT_ID', None)
                from gateway.config import load_gateway_config
                load_gateway_config()
                assert os.environ.get('TEAMS_CLIENT_ID') != 'ROOT_SENTINEL'
                assert 'plugins.platforms.teams.adapter' not in sys.modules
                print('root-env-not-loaded')
                """ % home,
                cwd=cwd_path,
            )
            self.assertIn("root-env-not-loaded", out)

    def test_telegram_platform_successfully_resolves(self):
        out = self._run_isolated(
            """
            import sys
            from hermes_cli.plugins import discover_plugins
            from gateway.platform_registry import platform_registry
            discover_plugins()
            entry = platform_registry.get('telegram')
            assert entry is not None, 'telegram entry missing'
            assert entry.name == 'telegram'
            assert 'plugins.platforms.teams.adapter' not in sys.modules
            print(entry.name)
            """
        )
        self.assertIn("telegram", out)

    def test_explicit_lookup_imports_only_requested_platform(self):
        from gateway.platform_registry import PlatformEntry, PlatformRegistry

        registry = PlatformRegistry()
        loaded: list[str] = []

        def loader_for(name: str):
            def _loader() -> None:
                loaded.append(name)
                registry.register(
                    PlatformEntry(
                        name=name,
                        label=name.title(),
                        adapter_factory=lambda cfg: object(),
                        check_fn=lambda: True,
                        source="plugin",
                    )
                )
            return _loader

        registry.register_deferred("teams", loader_for("teams"))
        registry.register_deferred("slack", loader_for("slack"))
        entries = registry.entries_for({"teams"})
        self.assertEqual([e.name for e in entries], ["teams"])
        self.assertEqual(loaded, ["teams"])
        self.assertTrue(registry.is_registered("slack"))

    def test_unknown_platform_lookup_is_controlled(self):
        from gateway.platform_registry import PlatformRegistry

        registry = PlatformRegistry()
        self.assertEqual(registry.entries_for({"unknown-platform"}), [])
        self.assertIsNone(registry.create_adapter("unknown-platform", object()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
