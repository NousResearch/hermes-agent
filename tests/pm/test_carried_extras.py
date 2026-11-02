"""An extra carried over from a pre-PM venv that no longer builds never fails the sync that
carries it: it is left out, recorded so the old venv's next scan skips it, and reported with
the command that adds it back. Core and requested extras still fail loudly."""

from __future__ import annotations

import pytest

from pm import extras, install
from pm.environment import BuildFailure
from pm.environments import runtime_facts_path
from pm.lock import Facts
from pm.paths import repo_root


class _Venv:
    """The venv package with a build backend that fails whenever a broken extra is selected."""

    def __init__(self, broken: set[str]):
        self.broken, self.applied = broken, []

    def project_root(self):
        return repo_root()

    def expected_stamp(self, enabled, **_inputs):
        return "stamp:" + ",".join(sorted(enabled))

    def apply(self, enabled, **_inputs):
        self.applied.append(sorted(enabled))
        if self.broken & set(enabled):
            raise BuildFailure("venv", "uv sync exited 1: Failed to build `python-olm==3.2.16`")
        return {}


@pytest.fixture()
def venv(monkeypatch):
    def make(broken):
        package = _Venv(broken)
        monkeypatch.setattr(install, "get_package", lambda name: package)
        return package
    return make


def _recorded():
    return (Facts(runtime_facts_path(repo_root())).get("venv") or {}).get("extras")


def test_a_carried_extra_that_cannot_build_is_left_out_and_not_carried_again(venv, capsys):
    package = venv({"matrix"})

    install.sync_venv(["all", "fal", "matrix"], explicit=True, carried=["fal", "matrix"])

    # The update completes with everything that builds, the carried extra that builds included.
    assert _recorded() == ["all", "fal"]
    assert extras.declined_extras() == {"matrix"}
    assert extras.declined_reasons() == {"matrix": "it no longer builds on this machine"}
    notice = capsys.readouterr().err
    assert "'matrix'" in notice and "hermes pm install --extra matrix" in notice
    # The next update finds the environment current: nothing is rebuilt or re-carried.
    builds = len(package.applied)
    install.sync_venv(None, explicit=True)
    assert len(package.applied) == builds
    # A lazy first use names why the extra is out instead of a `--without` nobody ran.
    with pytest.raises(Exception, match="an update left 'matrix' out"):
        install.sync_venv(["matrix"])


@pytest.mark.parametrize(("broken", "carried"), [({"matrix"}, None), ({"all"}, ["matrix"])])
def test_a_requested_extra_or_core_that_cannot_build_still_fails(venv, broken, carried):
    venv(broken)

    with pytest.raises(BuildFailure):
        install.sync_venv(["all", "matrix"], explicit=True, carried=carried)

    assert _recorded() is None
    assert extras.declined_extras() == frozenset()
