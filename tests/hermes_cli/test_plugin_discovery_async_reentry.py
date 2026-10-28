"""An async hook reached from register() must not wait for its own discovery sweep."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.mark.parametrize("load_timeout", [0, 10], ids=["inline", "deadline-worker"])
def test_register_can_invoke_an_async_hook(tmp_path, load_timeout):
    home = tmp_path / "home"
    plugin = home / "plugins" / "reentrant"
    plugin.mkdir(parents=True)
    (home / "config.yaml").write_text(
        f"plugins:\n  enabled: [reentrant]\n  load_timeout_seconds: {load_timeout}\n",
        encoding="utf-8",
    )
    (plugin / "plugin.yaml").write_text('name: reentrant\nversion: "0.1.0"\n', encoding="utf-8")
    (plugin / "__init__.py").write_text(textwrap.dedent('''
        import asyncio
        from hermes_cli.plugins import ainvoke_hook

        def register(ctx):
            ctx.register_hook("on_session_start", lambda **kw: "hook-completed")
            assert asyncio.run(ainvoke_hook("on_session_start", session_id="audit")) == ["hook-completed"]
            print("REGISTER_FINISHED", flush=True)
    '''), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-c", "from hermes_cli.plugins import discover_plugins; discover_plugins()"],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "HOME": str(tmp_path), "HERMES_HOME": str(home)},
        capture_output=True, text=True, timeout=10,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "REGISTER_FINISHED" in result.stdout, result.stdout + result.stderr
