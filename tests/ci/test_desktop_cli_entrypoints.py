"""Structural guards for the bundled payload's CLI entrypoint wiring.

The rust shim (apps/desktop/shim) is gone: win32 payloads mint distlib
launcher exes, POSIX payloads stage $0-relative bash trampolines
(scripts/desktop-cli/, driven by scripts/build-bundled-desktop.mjs step 5b).
The real generators are exercised by apps/desktop/scripts/cli-launchers.test.mjs
(vitest) and tests/scripts/test_desktop_cli_wrapper.py /
test_mint_launchers.py; what a python CI test can still prove is the
STRUCTURE of the wiring, so a regression back to a compiled shim (or an
alias manifest still assuming one exe + argv[0] dispatch) fails loudly
even when the js/py suites aren't run on the lane.
"""

from __future__ import annotations


from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_BUILD = _REPO / "scripts" / "build-bundled-desktop.mjs"
_BEFORE_BUILD = _REPO / "apps" / "desktop" / "scripts" / "before-build.mjs"



def test_build_script_no_longer_compiles_a_rust_shim():
    text = _BUILD.read_text(encoding="utf-8")
    assert "cargo build" not in text.lower(), "the desktop bundle must not need the rust toolchain"
    assert "'cargo'" not in text and '"cargo"' not in text
    assert "shim-target.txt" not in text, "the sidecar died with the rust shim"


def test_build_script_stages_the_desktop_cli_generators():
    text = _BUILD.read_text(encoding="utf-8")
    for needle in (
        "cli-entrypoints.mjs",
        "posixTrampolineScripts",
        "renderWinWrapper",
        "mint-launchers.py",
        "launcher-wrapper.py",
    ):
        assert needle in text, f"build-bundled-desktop.mjs lost its wiring to {needle}"
    # Minting runs on the payload's own store python (its arch IS the
    # target arch — distlib picks the launcher stub by minting platform).
    assert "pythonEntry" in text and "mint interpreter" in text


def test_before_build_declares_one_alias_extension_per_launcher_exe():
    text = _BEFORE_BUILD.read_text(encoding="utf-8")
    # The minted exes are three REAL executables; an AppExecutionAlias's
    # Executable must name the exact exe that serves the alias, so the
    # builder emits one uap5:Extension per launcher, deriving the list from
    # cli-entrypoints (no hardcoded, drifting names).
    assert "CLI_LAUNCHER_SPECS" in text
    assert "Executable=\"${executable(name)}\"" in text
    assert '<uap5:ExecutionAlias Alias="${name}.exe" />' in text
    # The default IS the full launcher set.
    assert "launchers = CLI_LAUNCHER_SPECS.map((s) => s.name)" in text


def test_the_shim_crate_is_gone():
    assert not (_REPO / "apps" / "desktop" / "shim").exists(), "apps/desktop/shim must be deleted"
