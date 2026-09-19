"""Real CLI-to-provider regression for the existing SEARXNG_URL setting."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

from dotenv import dotenv_values
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_searxng_url_cli_reaches_resolver_in_fresh_process(tmp_path):
    """Persist via argparse, then resolve without inheriting the setter's env."""
    home = tmp_path / "hermes"
    home.mkdir()
    config_path = home / "config.yaml"
    config_path.write_text("display:\n  compact: true\n", encoding="utf-8")
    env = {
        key: os.environ[key]
        for key in ("PATH", "SYSTEMROOT", "TEMP", "TMP", "LANG")
        if key in os.environ
    }
    env.update({
        "HOME": str(tmp_path),
        "USERPROFILE": str(tmp_path),
        "HERMES_HOME": str(home),
        "HERMES_MANAGED_DIR": str(tmp_path / "managed"),
        "PYTHONPATH": str(REPO_ROOT),
        "PYTHONIOENCODING": "utf-8",
    })
    url = "http://localhost:8080/searxng/"
    command = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "config", "set", "SEARXNG_URL", url],
        env=env, cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert command.returncode == 0, command.stdout + command.stderr
    assert dotenv_values(home / ".env")["SEARXNG_URL"] == url
    assert yaml.safe_load(config_path.read_text(encoding="utf-8")) == {
        "display": {"compact": True},
    }

    # No CLI import or dotenv preload: exercise the runtime's config-aware reads.
    resolver = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            import os
            import sys
            from hermes_cli.config import get_env_value
            from plugins.web._common import provider_env
            from plugins.web.searxng.provider import SearXNGWebSearchProvider
            from tools.web_tools import _get_search_backend, check_web_api_key

            assert "SEARXNG_URL" not in os.environ
            assert get_env_value("SEARXNG_URL") == sys.argv[1]
            assert provider_env("SEARXNG_URL") == sys.argv[1]
            assert SearXNGWebSearchProvider().is_available()
            assert _get_search_backend() == "searxng"
            assert check_web_api_key()
        """), url],
        env=env, cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert resolver.returncode == 0, resolver.stdout + resolver.stderr
