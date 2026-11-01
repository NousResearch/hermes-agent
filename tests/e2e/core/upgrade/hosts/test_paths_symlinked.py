"""Launches through a symlinked ``~/.hermes`` and a per-task home that links back to it.

Failure class: path handling. Users keep ``~/.hermes`` on another disk behind a symlink, and
orchestrators give each task its own HERMES_HOME whose ``tools``/``installs`` link back to the
main home. The PM runtime identity must not depend on which spelling of the path a launch used,
or every launch re-runs the source-update completion.

One real install through HEAD's ``scripts/install.sh`` with ``~/.hermes`` a symlink to a directory
on "another volume" (a path with a space and a non-ASCII char). Plain launches through it must be
current (a precondition), and so must a launch from the per-task home (#123798).
"""

from __future__ import annotations

import os
import shutil

import pytest

from tests.e2e.core.upgrade import _helpers as H
from tests.e2e.core.upgrade import _install_helpers as I
from tests.e2e.core.upgrade.hosts import _hosts as X
from tests.fakes.fake_llm_provider import FakeLLMServer

pytestmark = [
    pytest.mark.platforms("linux"),
    pytest.mark.live_system_guard_bypass,
    pytest.mark.skipif(H.sandbox_required_reason() is not None, reason=str(H.sandbox_required_reason())),
    pytest.mark.skipif(shutil.which("git") is None, reason="git required"),
    pytest.mark.skipif(I.real_uv() is None, reason="uv required"),
]

VOLUME_DIR = "external volume ü"


@pytest.fixture(scope="module")
def provider():
    with FakeLLMServer(default_text="fake reply for the host-paths suite") as srv:
        yield srv


@pytest.fixture(scope="module")
def linked_home(tmp_path_factory, provider):
    root = tmp_path_factory.mktemp("paths-linked")
    origin = X.make_origin(root)
    sb = X.new_sandbox(root / "sb", origin)
    real = root / "sb" / VOLUME_DIR / "hermes-data"
    real.mkdir(parents=True)
    os.symlink(real, sb.hermes_home)
    first = I.run_installer(sb)
    return {"sb": sb, "origin": origin, "root": root, "real": real, "install": first}


def _pending_markers(sb: I.Sandbox) -> list[str]:
    return sorted(str(p) for p in (sb.hermes_home / "installs").glob("*/source-completion-pending"))


def test_per_task_home_sharing_the_tools_store_by_symlink_is_current(linked_home, provider):
    """An orchestrator's per-task HERMES_HOME whose ``tools``/``installs`` link back to the main home.

    Precondition: the main home is itself a symlink, and plain launches through it are current; the
    per-task home's launch must then be current too (#123798).
    """
    sb, first = linked_home["sb"], linked_home["install"]
    assert sb.hermes_home.is_symlink(), "harness: ~/.hermes is not a symlink"
    assert first.returncode == 0, "install.sh failed with ~/.hermes behind a symlink:\n" + I.describe(first)
    assert (linked_home["real"] / "hermes-agent" / ".git").exists(), "install did not land on the symlink's target"
    X.configure(sb, provider)
    alt = linked_home["root"] / "per-task home ü"
    alt.mkdir()
    for name in ("tools", "installs"):
        os.symlink(sb.hermes_home / name, alt / name)
    for name in ("config.yaml", ".env"):
        shutil.copy(sb.hermes_home / name, alt / name)
    launches = [X.turn(sb, provider, f"turn {i} through the symlinked main home") for i in range(2)]
    reruns = [i for i, cp in enumerate(launches) if X.reran_completion(cp)]
    assert not reruns, (f"plain launches {reruns} through a symlinked ~/.hermes re-ran the source-update completion:\n"
                        + I.describe(launches[reruns[0]]))
    assert not _pending_markers(sb), f"a plain launch left source-completion-pending markers: {_pending_markers(sb)}"
    task = X.turn(sb, provider, "turn in a per-task home", env=dict(sb.env, HERMES_HOME=str(alt)))
    assert not X.reran_completion(task), (
        "a per-task HERMES_HOME whose tools/installs symlink to the main home re-ran the source-update "
        "completion on a current install:\n" + I.describe(task))
