"""A relocated ``$HERMES_HOME/plugins`` (symlink / Windows junction) still publishes and recovers.

Regression for #134952: ``StagedPlugin`` and boot recovery required the *resolved* plugin target to
sit under the Hermes root, so a plugins directory linked to another drive failed every install,
update and dependency sync with "plugin publication paths escape or overlap their home". The
install side hands publication a target already resolved through that link
(``_sanitize_plugin_name``, ``update_plugin``), so the target must be judged against the homes'
resolved ``plugins`` directories — while a ``plugins`` dir owned by no Hermes home stays refused.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hermes_cli.plugins_cmd import _sanitize_plugin_name

_REPO = str(Path(__file__).resolve().parents[2])

_PUBLISH = '''
from pathlib import Path
import os,sys
from pm.publication import StagedPlugin
from pm.store import tree_digest
from pm.environments import runtime_facts_path
from pm.lock import Facts
project,staged,target = map(Path,sys.argv[1:4])
change = StagedPlugin({"staged": str(staged), "target": str(target), "target_digest": tree_digest(target),
                       "old_metadata": {"example":{"revision":"old"}},
                       "new_metadata": {"example":{"revision":"new"}}})
if sys.argv[4] == "check":
    sys.exit(0)
change.publish(project)
if sys.argv[4] == "True":
    Facts(runtime_facts_path(project)).record_state("venv","new",[])
os._exit(17)
'''

_RECOVER = '''
from pathlib import Path
import sys
from hermes_cli.runtime_state import runtime_lock,recover_publication
project=Path(sys.argv[1])
with runtime_lock(project):
    recover_publication(project)
'''


def _link_dir(link: Path, real: Path) -> None:
    """The host's own way to relocate a directory: a symlink, or a junction where Windows refuses one."""
    try:
        os.symlink(real, link, target_is_directory=True)
    except OSError:
        if sys.platform != "win32":
            raise
        import _winapi

        _winapi.CreateJunction(str(real), str(link))


def _plugin_home(tmp_path: Path, *, profile: bool = False, relocated: bool = True) -> tuple[Path, Path, Path]:
    """``(HERMES_HOME, real plugins dir, staged clone)`` with an installed ``example`` plugin; the
    home is the Hermes root itself or a named profile under it."""
    root = tmp_path / "root"
    home = root / "profiles" / "work" if profile else root
    home.mkdir(parents=True)
    plugins = tmp_path / "other-drive" / "plugin-store" if relocated else home / "plugins"
    (plugins / "example").mkdir(parents=True)
    if relocated:
        _link_dir(home / "plugins", plugins)
    (plugins / "example" / "__init__.py").write_text("old code", encoding="utf-8")
    (plugins / ".install-metadata.json").write_text('{"example":{"revision":"old"}}\n', encoding="utf-8")
    staged = plugins / ".install-new"  # install stages inside the plugins dir (same volume as the target)
    staged.mkdir()
    (staged / "__init__.py").write_text("new code", encoding="utf-8")
    return home, plugins, staged


def _run(program: str, *args, home: Path, cwd: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "HERMES_HOME": str(home), "PYTHONPATH": _REPO}
    return subprocess.run([sys.executable, "-c", program, *map(str, args)], env=env, cwd=cwd,
                          capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("committed", [False, True])
def test_installed_target_publishes_and_recovers_through_a_relocated_plugins_dir(tmp_path, committed):
    home, plugins, staged = _plugin_home(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    # The target `hermes plugins install` hands publication: already resolved through the link.
    target = _sanitize_plugin_name("example", home / "plugins")

    published = _run(_PUBLISH, project, staged, target, committed, home=home, cwd=tmp_path)
    assert published.returncode == 17, published.stderr
    recovered = _run(_RECOVER, project, home=home, cwd=tmp_path)
    assert recovered.returncode == 0, recovered.stderr

    assert (plugins / "example" / "__init__.py").read_text(encoding="utf-8") == ("new code" if committed else "old code")
    metadata = json.loads((plugins / ".install-metadata.json").read_text(encoding="utf-8"))
    assert metadata["example"]["revision"] == ("new" if committed else "old")
    assert not list(plugins.glob(".previous-*"))


@pytest.mark.parametrize("profile", [False, True], ids=["root-home", "profile-home"])
@pytest.mark.parametrize("shape", ["installed", "literal"])
def test_relocated_plugins_target_is_accepted_in_either_shape(tmp_path, profile, shape):
    home, _plugins, staged = _plugin_home(tmp_path, profile=profile)
    target = (_sanitize_plugin_name("example", home / "plugins") if shape == "installed"
              else home / "plugins" / "example")

    accepted = _run(_PUBLISH, tmp_path, staged, target, "check", home=home, cwd=tmp_path)

    assert accepted.returncode == 0, accepted.stderr


def test_plugins_dir_owned_by_no_hermes_home_is_refused(tmp_path):
    home, _plugins, staged = _plugin_home(tmp_path, relocated=False)
    stray = tmp_path / "stray" / "plugins" / "example"
    stray.mkdir(parents=True)

    refused = _run(_PUBLISH, tmp_path, staged, stray, "check", home=home, cwd=tmp_path)

    assert refused.returncode != 0
    assert "escape or overlap their home" in refused.stderr
