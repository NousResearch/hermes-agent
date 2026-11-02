"""Keep a migration alive when an extra carried over from a pre-PM venv no longer builds.

``legacy_selection`` carries the extras a main-era venv held into PM's first generation, so a
feature that worked keeps working. The user did not ask for them in this sync, and the new
graph builds them for a different interpreter. One that no longer builds (a native sdist whose
toolchain is missing) must not fail the update: it is left out, recorded beside the dependency
facts so a later scan of the old venv does not carry it again, and reported with the command
that adds it back. ``hermes pm install --extra NAME`` is an explicit request: it clears the
record, and if the extra still does not build it fails loudly. Only a carried extra is left
out this way; core and requested extras that fail still fail the sync.
"""
from __future__ import annotations

import sys

from pm.environment import BuildFailure, ResolutionConflict

# Verdicts about the build. A fetch or tooling failure is the moment, not the extra.
BUILD_VERDICTS = (ResolutionConflict, BuildFailure)


def failing_carried(package, failure: BaseException, *, extras: list[str], carried: list[str], fact: dict,
                    shipped, frozen, explicit: bool) -> dict[str, tuple[str, bool]]:
    """``{extra: (reason, is_build_verdict)}`` for the carried extras that keep the environment
    from building; re-raises *failure* when they are not the cause.

    Core plus everything requested must build without the carried extras. Each carried extra is
    then tried on top of core and the carried extras that already fit, without plugins (plugin
    eviction judges those against the selection this leaves).
    """
    from pm.install import _target_selection
    from pm.plugin_eviction import _trial

    optional = [extra for extra in dict.fromkeys(carried) if extra in extras]
    if not optional:
        raise failure
    required = [extra for extra in extras if extra not in optional]
    base = _target_selection(package, fact, extras=required, inputs={"plugin_dirs": []}, repair=False,
                             shipped=shipped, frozen=frozen)[0]
    if _trial(package, base, explicit, []) is not None:
        raise failure
    fitting: list[str] = []
    failing: dict[str, tuple[str, bool]] = {}
    for extra in optional:
        verdict = _trial(package, sorted({*base, *fitting, extra}), explicit, [])
        if verdict is None:
            fitting.append(extra)
        else:
            failing[extra] = verdict
    if not failing:
        raise failure  # each fits on its own and together: the failure was not about them
    return failing


_NO_BUILD = "it no longer builds on this machine"


def record_left_out(package, failing: dict[str, tuple[str, bool]]) -> None:
    """Record the build verdicts so no later scan carries them, then tell the operator.

    A download failure is not recorded: the next legacy carry may try it again.
    """
    from pm import receipt
    from pm.extras import install_hint, record_declined_extras

    built_badly = [extra for extra, (_detail, verdict) in failing.items() if verdict]
    record_declined_extras(add=built_badly, reasons=dict.fromkeys(built_badly, _NO_BUILD),
                           project_root=package.project_root())
    for extra, (detail, verdict) in failing.items():
        why = f"{_NO_BUILD} (the build output is above)" if verdict else "its dependencies could not be downloaded"
        message = (f"Left the optional {extra!r} extra out of this update: Hermes carried it over from the old "
                   f"venv, and {why}. Everything else is installed. Once that is fixed, run "
                   f"`{install_hint(extra)}` to add it back.")
        print(f"⚠ {message}", file=sys.stderr, flush=True)
        receipt.record_warning(f"{message} Detail: {detail[:300]}")
