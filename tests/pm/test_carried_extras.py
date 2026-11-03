"""An extra carried over from a pre-PM venv that no longer builds never fails the sync that
carries it: it is left out, recorded so the old venv's next scan skips it, and reported with
the command that adds it back. Core and requested extras still fail loudly."""

from __future__ import annotations

import pytest

from pm import extras, install, paths
from pm.environment import BuildFailure
from pm.environments import install_state_dir, runtime_facts_path
from pm.lock import Facts


class _Venv:
    """The venv package with a build backend that fails whenever a broken extra is selected.

    A build that succeeds commits a generation of this install, as the real one does: a
    stamp only counts as current while the environment the facts name is on disk, so the
    test must never lean on whatever venv the checkout running it happens to have.
    """

    def __init__(self, broken: set[str]):
        self.broken, self.applied = broken, []

    def project_root(self):
        return paths.repo_root()

    def expected_stamp(self, enabled, **_inputs):
        return "stamp:" + ",".join(sorted(enabled))

    def apply(self, enabled, **_inputs):
        self.applied.append(sorted(enabled))
        if self.broken & set(enabled):
            raise BuildFailure("venv", "uv sync exited 1: Failed to build `python-olm==3.2.16`")
        environment = install_state_dir(self.project_root()) / "environments" / str(len(self.applied)) / "venv"
        environment.mkdir(parents=True)
        (environment / "pyvenv.cfg").write_text("home = fixture\n")
        return {"environment": environment}


@pytest.fixture()
def venv(monkeypatch, tmp_path):
    repo = tmp_path / "checkout"
    repo.mkdir()
    monkeypatch.setattr(paths, "repo_root", lambda: repo)

    def make(broken):
        package = _Venv(broken)
        monkeypatch.setattr(install, "get_package", lambda name: package)
        return package
    return make


def _recorded():
    return (Facts(runtime_facts_path(paths.repo_root())).get("venv") or {}).get("extras")


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


def test_a_carried_extra_that_fails_for_the_moment_is_retried_by_the_next_update(venv):
    """A carried extra whose trials fail on a download or tooling problem is not a verdict. Publishing
    without it would write the facts that end the old venv's re-scan, losing the extra with no record;
    the sync fails instead, and the next update carries it again and builds it once the cause clears."""
    from pm.package import InstallError

    package = venv(set())
    # The full sync fails to build matrix; then its source goes missing for every per-extra trial.
    failures = [BuildFailure("venv", "uv sync exited 1: Failed to build `review-optional`")]
    unavailable = InstallError("venv", "uv sync exited 2: `review-optional` does not appear to be a Python project")
    source_down = [True]

    def apply(enabled, **inputs):
        if "matrix" in enabled and source_down[0]:
            package.applied.append(sorted(enabled))
            raise failures.pop() if failures else unavailable
        return _Venv.apply(package, enabled, **inputs)

    package.apply = apply
    with pytest.raises(InstallError, match=r"\['matrix'\].*could not be prepared"):
        install.sync_venv(["all", "fal", "matrix"], explicit=True, carried=["fal", "matrix"])
    assert _recorded() is None
    assert extras.declined_extras() == frozenset()

    source_down[0] = False
    # No facts yet, so the next update scans the old venv and carries the same selection.
    install.sync_venv(["all", "fal", "matrix"], explicit=True, carried=["fal", "matrix"])
    assert _recorded() == ["all", "fal", "matrix"]
