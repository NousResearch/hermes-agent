"""Regression: the ResolvedPathReport must not read an unset resolver.

Issue #93017: fresh Windows installs died at the ``$script:ResolvedPathReport``
block with ``The variable '$script:LastResolver' cannot be retrieved because
it has not been set`` — before any install stage ran. The mechanism:
``ConvertTo-LongPath`` short-circuits at the top for ordinary long paths
(``-notmatch '~\\d'`` returns early, per the comment "skip every resolver for
ordinary long paths, which is the overwhelmingly common case"), so
``$script:LastResolver`` is only ever assigned when a 8.3 short path actually
needs expansion. The report block read it unconditionally, which is fatal in
a ``Set-StrictMode`` session (the reporter hit it through three different
invocation styles, i.e. their host/session enables strict mode).

The fix initializes ``$script:LastResolver = 'none'`` at script scope before
``Set-LongProfileEnvVars`` can run any resolver — ``'none'`` being the
resolver's own value for "nothing ran".

install.ps1 only runs on Windows, so these tests lock the contract at the
source-text level (same style as test_install_ps1_uv_install_fallback.py).
"""

from __future__ import annotations

import re
from pathlib import Path

_INSTALL_PS1 = Path(__file__).resolve().parents[1] / "scripts" / "install.ps1"


def _ps1() -> str:
    return _INSTALL_PS1.read_text(encoding="utf-8")


def test_last_resolver_initialized_at_script_scope():
    source = _ps1()
    match = re.search(r"^\$script:LastResolver = 'none'\s*$", source, re.MULTILINE)
    assert match is not None, (
        "$script:LastResolver must be initialized at script scope ('none') — "
        "ConvertTo-LongPath only assigns it when a short path needs expanding, "
        "so an ordinary long profile leaves it unset (#93017)"
    )


def test_initialization_precedes_resolver_run_and_report_read():
    source = _ps1()
    init = re.search(r"^\$script:LastResolver = 'none'\s*$", source, re.MULTILINE)
    run = re.search(r"^\$script:NormalizedProfilePaths = Set-LongProfileEnvVars", source, re.MULTILINE)
    read = re.search(r"^\s+resolver\s+= \$script:LastResolver\s*$", source, re.MULTILINE)
    assert init and run and read, "init / Set-LongProfileEnvVars / report read must all exist"
    assert init.start() < run.start() < read.start(), (
        "initialization must come before Set-LongProfileEnvVars can invoke a "
        "resolver, and before the ResolvedPathReport reads the variable"
    )


def test_convert_to_long_path_still_short_circuits_before_assigning():
    """Pin the mechanism that leaves the variable unset on normal paths.

    The early return for paths without an 8.3 alias is what makes the
    initialization load-bearing: if someone removes the short-circuit the
    resolver always assigns, but the guard is a deliberate performance
    choice (skip Add-Type for the overwhelmingly common case) and must
    stay — so the init must stay too.
    """
    source = _ps1()
    start = source.index("function ConvertTo-LongPath")
    body = source[start : start + 2000]
    assert re.search(r"-notmatch '~\\d'", body), (
        "ConvertTo-LongPath must keep its ordinary-long-path short-circuit "
        "(the reason $script:LastResolver can be unset without an init)"
    )
