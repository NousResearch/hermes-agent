"""Regression tests for the ``browser`` install stage in scripts/install.ps1.

Sibling to PR #65701 (``fix(browser): browser tools unusable after Hermes
restart -- zombie daemon holds port``).  #65701 fixes a runtime *symptom* --
a zombie agent-browser daemon holding a TCP port after Hermes crashes/closes
on Windows.  While revalidating that fix on a Windows machine, the previous
session traced the *install-flow* root cause: the desktop Update button drives
``install.ps1`` per-stage via ``install.ps1 -Stage <name> -NonInteractive -Json``
(see ``apps/desktop/electron/bootstrap-runner.ts`` L779), iterating stages
emitted by ``install.ps1 -Manifest``.  Because ``$InstallStages`` had no
``browser`` entry, ``Install-AgentBrowser`` (the only function that runs
``npm install -g --prefix $HERMES_HOME\\node "agent-browser@^0.26.0"`` -- L355)
was never invoked by the desktop-driven flow.  A Windows user installing or
updating Hermes purely via the desktop GUI therefore keeps falling through to
whatever ``agent-browser`` is on their bare PATH (commonly a stale NVM/Node
global install at v0.17.1, the version with the zombie bug).  The Linux/macOS
equivalent (``scripts/install.sh``'s ``ensure_browser()``) IS in the install
flow via ``ensure_mode`` (L2550-2600); the Windows gap was asymmetric.

This PR adds a ``browser`` stage to ``$InstallStages`` and a thin
``Stage-Browser`` worker that delegates to the existing ``Install-AgentBrowser``
function (extend, don't duplicate).

These tests are source-level by design:

* ``install.ps1`` is Windows-only PowerShell; Linux CI cannot execute it.
* The existing ``tests/test_install_ps1_*.py`` family already pins install.ps1
  contracts via source-text parsing (see ``test_install_ps1_node_path_for_npm.py``).
* The structural assertions below let a CI plant that doesn't run PowerShell
  still verify the stage was added with the expected shape, ordering, and
  soft-skip-existence for an idempotent upgrade path.

The Pester smoke suite at ``scripts/tests/test-install-ps1-stage-protocol.ps1``
gains a parallel ``-Manifest`` assertion enquiring that the ``browser`` stage
name is present; that's the runtime-side cross-check on Windows.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"


def _install_ps1() -> str:
    return INSTALL_PS1.read_text(encoding="utf-8")


def test_install_ps1_declares_browser_stage_in_installstages() -> None:
    """``browser`` stage must appear in $InstallStages with the right shape."""
    text = _install_ps1()
    pattern = re.compile(
        r'@{\s*Name\s*=\s*"browser"\s*;\s*'
        r'Title\s*=\s*"Installing agent-browser"\s*;\s*'
        r'Category\s*=\s*"install"\s*;\s*'
        r'NeedsUserInput\s*=\s*\$false\s*;\s*'
        r'Worker\s*=\s*"Stage-Browser"\s*}',
        re.MULTILINE,
    )
    assert pattern.search(text), (
        "$InstallStages must include a 'browser' entry mapping to the "
        "'Stage-Browser' worker (non-interactive, install category)."
    )


def test_install_ps1_browser_stage_needs_user_input_is_false() -> None:
    """The browser stage is non-interactive -- drives are unattended."""
    text = _install_ps1()
    m = re.search(r'@{\s*Name\s*=\s*"browser"[^}]*?}', text, re.DOTALL)
    assert m, "'browser' stage entry not found"
    block = m.group(0)
    assert "NeedsUserInput = $false" in block, (
        "browser stage must declare NeedsUserInput = $false -- desktop Update "
        "(bootstrap-runner.ts L779) drives stages with -NonInteractive."
    )


def test_install_ps1_browser_stage_respects_stages_contract_shape() -> None:
    """Every other stage entry has the same 5 fields; browser must too."""
    text = _install_ps1()
    m = re.search(r'@{\s*Name\s*=\s*"browser"[^}]*?}', text, re.DOTALL)
    assert m, "'browser' stage entry not found"
    block = m.group(0)
    for field in ("Name", "Title", "Category", "NeedsUserInput", "Worker"):
        assert re.search(rf"\b{field}\b\s*=", block), (
            f"'browser' stage must declare the '{field}' field like every "
            f"other stage entry (per the stage-protocol contract at "
            f"install.ps1 L3486-3492). Missing in: {block}"
        )


def test_install_ps1_stage_browser_worker_defined_and_calls_install_agentbrowser() -> None:
    """``Stage-Browser`` worker must exist and delegate to ``Install-AgentBrowser``."""
    text = _install_ps1()
    m = re.search(
        r"function\s+Stage-Browser\s*\{(?P<body>.*?)\n\}",
        text,
        re.DOTALL,
    )
    assert m, "function 'Stage-Browser' not defined in install.ps1"
    body = m.group("body")
    assert re.search(r"\bInstall-AgentBrowser\b", body), (
        "Stage-Browser must call Install-AgentBrowser -- that's the only "
        "function that runs `npm install -g --prefix $HERMES_HOME\\node "
        "\"agent-browser@^0.26.0\"` (L355). Extend, don't duplicate."
    )


def test_install_ps1_stage_browser_soft_skips_on_no_node() -> None:
    """Stage-Browser must NOT abort the install flow when Node is missing.

    Mirrors the documented Stage-Node pattern at L3548-L3558: browser tools
    are optional, so the worker records a soft-skip reason via
    ``$script:_StageSkippedReason`` and returns, instead of throwing.  Invoke-Stage
    (L3609-3616) surfaces the channel as ``skipped: true, ok: true`` in the
    JSON frame, so a GUI driver can distinguish "ready" from "missing".
    """
    text = _install_ps1()
    m = re.search(r"function\s+Stage-Browser\s*\{(?P<body>.*?)\n\}",
                  text, re.DOTALL)
    assert m, "function 'Stage-Browser' not defined"
    body = m.group("body")
    # The soft-skip check: only-if-not-node + $_StageSkippedReason assignment + return.
    assert re.search(r"if\s*\(-not\s+\$script:HasNode\)", body), (
        "Stage-Browser must guard on $script:HasNode -- Test-Node sets it "
        "(mirrors Stage-Node + Invoke-EnsureMode's 'browser' case L3680-3688)."
    )
    assert re.search(r"\$script:_StageSkippedReason\s*=\s*['\"]", body), (
        "Stage-Browser must populate $script:_StageSkippedReason when Node is "
        "unavailable -- that's how the JSON contract surfaces 'skipped: true' "
        "(see Invoke-Stage's soft-skip channel at L3609-3616)."
    )
    assert re.search(r"\breturn\b", body), (
        "Stage-Browser must return (not throw) when Node is unavailable -- "
        "browser tools are optional; the install flow MUST NOT abort."
    )


def test_install_ps1_stage_browser_does_not_terminate_process_pool() -> None:
    """Stage-Browser must NOT have any process-kill or app-dir reaper logic.

    Defense against review concern from PR #65701's reviewer (tonydwb):
    worker code here is a thin install passthrough only.  It must not
    introduce a parallel orphan-reaper path -- the existing one lives in the
    browser-tool core (tests/tools/test_browser_orphan_reaper.py).
    """
    text = _install_ps1()
    m = re.search(r"function\s+Stage-Browser\s*\{(?P<body>.*?)\n\}",
                  text, re.DOTALL)
    assert m, "function 'Stage-Browser' not defined"
    body = m.group("body")
    forbidden = [
        r"Terminate\(", r"Stop-Process", r"taskkill",
        r"\.agent-browser", r"_reap_app_dir_sessions", r"Kill-Process",
    ]
    for pat in forbidden:
        assert not re.search(pat, body), (
            f"Stage-Browser must not invoke '{pat}' -- worker is a thin "
            f"install passthrough, not a reaper pathway. The orphan reaper "
            f"lives in the browser-tool core (see #65701 for that scoping)."
        )


def test_install_ps1_browser_stage_appears_after_node_and_node_deps() -> None:
    """Stage ordering: browser must come AFTER Node is provisioned.

    ``Install-AgentBrowser``'s body (L357-361) calls ``Resolve-NpmCmd`` and
    throws if npm is missing.  In cross-process driver mode each stage runs
    in its own powershell child, so the ``browser`` stage can't inherit any
    ``$script:HasNode`` flag set by ``Stage-Node`` in a sibling -- it must
    re-check via ``Test-Node`` itself, and Test-Node needs the binaries
    Stage-Node provisioned into PATH.  So ``browser`` MUST follow ``node`` AND
    ``node-deps`` in the manifest.
    """
    text = _install_ps1()
    # Find byte offsets of each stage Name in the InstallStages section.
    pattern_names_sec = re.compile(
        r'@{\s*Name\s*=\s*"(?P<n>[a-z][a-z-]*)"\s*;',
        re.MULTILINE,
    )
    seen: list[tuple[int, str]] = []
    for m in pattern_names_sec.finditer(text):
        # 'desktop' and 'browser' (and others) all use this exact shape.
        # Restrict to the InstallStages section to avoid false positives.
        # The block that declares $InstallStages opens before L3501 -- all
        # seven-plus stage entries are inside that block.
        seen.append((m.start(), m.group("n")))
    names_order = [n for _, n in seen]
    # Confirm the three we care about all appear:
    for required in ("node", "node-deps", "browser"):
        assert required in names_order, f"'{required}' stage missing from source"
    assert names_order.index("node") < names_order.index("browser"), (
        "'browser' stage must come after 'node' -- Install-AgentBrowser needs npm."
    )
    assert names_order.index("node-deps") < names_order.index("browser"), (
        "'browser' stage must come after 'node-deps' (per @teknium1's review on "
        "#65701's closer, where they identified this ordering as the upgrade "
        "pattern ensuring the bundled prefix is installed before downstream stages)."
    )


def test_install_ps1_browser_stage_appears_before_post_install_stages() -> None:
    """The browser stage must be in the install/finalize group, not interactive.

    ``configure`` and ``gateway`` are declared NeedsUserInput=$true and are the
    only interactive stages; browser is declared NeedsUserInput=$false (see the
    other test for that assertion).  Cross-check: browser's source offset must
    precede 'configure' so a non-interactive driver (``bootstrap-runner.ts``)
    completes the install before running the wizard.
    """
    text = _install_ps1()
    pattern_names_sec = re.compile(r'@{\s*Name\s*=\s*"(?P<n>[a-z][a-z-]*)"\s*;', re.MULTILINE)
    names_order: list[str] = []
    for m in pattern_names_sec.finditer(text):
        names_order.append(m.group("n"))
    assert "browser" in names_order and "configure" in names_order, (
        "Both 'browser' and 'configure' stages must be in $InstallStages"
    )
    assert names_order.index("browser") < names_order.index("configure"), (
        "'browser' stage must come before 'configure' (install group precedes "
        "post-install interactive group -- see ordering contract at L3486-3492)."
    )


def test_install_ps1_install_agentbrowser_remains_only_caller_via_stage_browser() -> None:
    """The existing ``Install-AgentBrowser`` function must NOT be moved or duplicated.

    The PR is purely stage-wiring (extend, don't duplicate, per AGENTS.md).
    The function still appears exactly twice in the source: its definition
    (L355) and the pre-existing ``Invoke-EnsureMode`` ``"browser"`` case
    (L3683, still routed via -PostInstall / -Ensure).  The new Stage-Browser
    worker adds a *third* callsite dispatching the same function -- but does
    NOT redefine it or move the npm install call elsewhere.
    """
    text = _install_ps1()
    # Definition: exactly one.
    defs = re.findall(r"function\s+Install-AgentBrowser\s*\{", text)
    assert len(defs) == 1, (
        f"Expected exactly 1 'function Install-AgentBrowser' definition, "
        f"found {len(defs)}.  Do not duplicate Install-AgentBrowser."
    )
    # npm install -g --prefix call: exactly one (inside that function).
    npm_calls = re.findall(r'npm install -g --prefix ', text)
    assert len(npm_calls) == 1, (
        f"Expected exactly 1 `npm install -g --prefix` for agent-browser, "
        f"found {len(npm_calls)}.  Do not move the install logic elsewhere."
    )
    # The new Stage-Browser callsite must reference Install-AgentBrowser.
    m = re.search(r"function\s+Stage-Browser\s*\{(?P<body>.*?)\n\}",
                  text, re.DOTALL)
    assert m, "Stage-Browser not defined"
    assert "Install-AgentBrowser" in m.group("body"), (
        "Stage-Browser must delegate to Install-AgentBrowser (not reimplement "
        "the npm install --extend, don't duplicate, per AGENTS.md)."
    )


def test_install_ps1_stage_browser_forwards_skipchromium_flag() -> None:
    """Stage-Browser must forward any -SkipChromium flag to Install-AgentBrowser.

    Install-AgentBrowser's ``-SkipChromium`` parameter (L356) gates the
    bundled ``agent-browser install`` Chromium download step (L383-399).
    The worker must not *swallow* that flag -- a system-browser override at
    L384 internally short-circuits the Chromium step even when the caller
    didn't set -SkipChromium, but the explicit flag still needs to pass
    through.

    Note that install.ps1 has *no* top-level ``-SkipChromium`` parameter
    today, so when the worker references the variable it resolves to ``$null``
    (falsy) by default -- identical semantics to Install-AgentBrowser's own
    ``[switch]$SkipChromium`` defaulting to ``$false``.  The forwarding -
    ``Install-AgentBrowser -SkipChromium:$SkipChromium`` -- is what future-
    proofs adding a top-level param later without a path-only re-touch.
    """
    text = _install_ps1()
    m = re.search(r"function\s+Stage-Browser\s*\{(?P<body>.*?)\n\}",
                  text, re.DOTALL)
    assert m, "Stage-Browser not defined"
    body = m.group("body")
    assert re.search(
        r"Install-AgentBrowser\s+(-SkipChromium:\$SkipChromium|-SkipChromium)",
        body,
    ), (
        "Stage-Browser must forward -SkipChromium to Install-AgentBrowser "
        "(e.g. `Install-AgentBrowser -SkipChromium:$SkipChromium`), so that "
        "a future top-level install.ps1 -SkipChromium param and the existing "
        "system-browser-respecting Install-AgentBrowser body both line up."
    )


def test_install_ps1_browser_stage_keeps_install_ps1_pure_ascii() -> None:
    """The lines added by this PR must keep install.ps1 pure ASCII.

    Regression: issues #66994 / #67000 crashed the Windows installer because a
    non-ASCII byte (em-dash) inside a double-quoted string was misdecoded by
    Windows PowerShell 5.1 reading a BOM-less .ps1 in CP1252.  The existing
    ``tests/test_install_ps1_ascii_only.py`` pins the whole-file invariant; this
    test cross-checks the same invariant for the lines added by the browser-
    stage PR so a regression here would point at the new lines specifically.
    """
    raw = INSTALL_PS1.read_bytes()
    offenders: list[int] = []
    line_no = 1
    for byte in raw:
        if byte == 0x0A:
            line_no += 1
        elif byte >= 0x80:
            offenders.append(line_no)
    assert not offenders, (
        "scripts/install.ps1 must be pure ASCII (issues #66994 / #67000 -- "
        "Windows PowerShell 5.1 misdecodes a BOM-less .ps1 in CP1252 if it "
        "contains non-ASCII bytes). Non-ASCII byte(s) on line(s): "
        f"{sorted(set(offenders))}."
    )
