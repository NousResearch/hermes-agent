"""Plugin publication must run in the stripped PM runtime without the app graph.

``pm/publication.py`` carries no application dependency imports, and the PM
runtime ships no httpx: ``StagedPlugin.publish()`` guards its install metadata
with the stdlib-only ``hermes_cli.file_lock`` leaf, so publishing a plugin in
that runtime never imports ``hermes_cli.auth`` — whose import chain once ran
bundled provider discovery and warned about Solstice on raw stderr (#134107).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_staged_plugin_publication_without_httpx_or_the_app_graph(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    code = textwrap.dedent("""
        import importlib.abc
        import json
        import sys
        from pathlib import Path

        sys.path.insert(0, sys.argv[1])

        class MissingHttpx(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "httpx" or fullname.startswith("httpx."):
                    raise ModuleNotFoundError("No module named 'httpx'", name=fullname)

        assert "httpx" not in sys.modules
        sys.meta_path.insert(0, MissingHttpx())

        from pm.publication import StagedPlugin

        plugins = Path(sys.argv[2]) / "plugins"
        plugins.mkdir(parents=True)
        target = plugins / "example"
        metadata = plugins / ".install-metadata.json"
        metadata.write_text(json.dumps({"example": {"revision": "old"}}))
        staged = Path(sys.argv[3])
        (staged / "plugin.yaml").write_text("name: example\\n")
        (staged / "code.py").write_text("new code")

        plugin = StagedPlugin({
            "target": str(target),
            "staged": str(staged),
            "target_digest": None,
            "old_metadata": {"example": {"revision": "old"}},
            "new_metadata": {"example": {"revision": "new"}},
        })
        plugin.publish(Path(sys.argv[4]))

        assert "hermes_cli.auth" not in sys.modules
        assert "providers" not in sys.modules
        assert (target / "code.py").read_text() == "new code"
        assert json.loads(metadata.read_text()) == {"example": {"revision": "new"}}
    """)
    project = tmp_path / "project"
    project.mkdir()
    staged = tmp_path / "staged"
    staged.mkdir()
    env = dict(os.environ, HOME=str(tmp_path), USERPROFILE=str(tmp_path),
               HERMES_HOME=str(tmp_path / "hermes"))
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(repo),
         str(tmp_path / "hermes"), str(staged), str(project)],
        env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Failed to load bundled provider plugin" not in result.stderr
