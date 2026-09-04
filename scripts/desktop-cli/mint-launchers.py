"""Mint the bundled payload's win32 CLI launchers with distlib.

Run by scripts/build-bundled-desktop.mjs (step 5b) with the payload's OWN
store python as the minting interpreter — the store python is the same
distribution the launchers will execute, and its architecture is by
construction the target architecture, which is exactly what distlib's
_get_launcher needs (it picks a 32/64-bit, -arm-suffixed launcher stub by
the MINTING interpreter's platform; cross-arch minting with a mismatched
host python produces launchers that cannot run).

distlib availability: the payload python ships pip (python-build-standalone
includes it), and pip vendors distlib at pip._vendor.distlib — so the mint
works with NO extra dependency. A top-level distlib install is preferred
when present. (Self-contained .\_pth interpreters were REJECTED as an
alternative approach: they break uv venv materialization.)

Layout facts arrive via env (build-bundled-desktop forbids whitespace in
win32 argv, so argv is not an option):

  HERMES_MINT_BIN_DIR   absolute output dir (agent-payload/bin)
  HERMES_MINT_SPECS     JSON list of {"name": exe stem, "module": dotted
                        module, "func": entry function} — mirrors
                        [project.scripts] in pyproject.toml
  HERMES_MINT_WRAPPER   path of the RENDERED launcher-wrapper.py for THIS
                        entry (substitution is cli-entrypoints.mjs's job,
                        one implementation, one test)
  HERMES_MINT_PYTHON    bin-relative path of the store python, BACKslashes,
                        e.g. <launcher_dir>\..\tools\<entry>\python.exe —
                        baked into the shebang; the literal <launcher_dir>
                        prefix is resolved by the launcher at run time
                        relative to its own directory (relocatability is
                        live-proved: the placeholder survives minting)

Testable standalone: tests/scripts/test_mint_launchers.py drives main()
against a temp tree with a stub module and RUNS the minted exe.
"""

from __future__ import annotations

import json
import os

try:  # a real distlib install when available
    from distlib.scripts import ScriptMaker
except ImportError:  # the payload python's guaranteed fallback: pip's vendor
    from pip._vendor.distlib.scripts import ScriptMaker


def make_maker(bin_dir, shebang_python, wrapper_text):
    class MintScriptMaker(ScriptMaker):
        # The wrapper text is fully pre-rendered, so ignore the entry the
        # spec carried — the name is all we need from it.
        def __init__(self, wrapper, **kwargs):
            target_dir = kwargs.pop("target_dir")
            super().__init__(None, target_dir, **kwargs)
            self._wrapper_text = wrapper

        def _get_script_text(self, entry):
            return self._wrapper_text

    # The shebang the launcher stub parses: #! + this string. The
    # <launcher_dir> literal is replaced by the launcher with its own
    # directory at run time — that IS the self-relative mechanism.
    maker = MintScriptMaker(
        wrapper_text,
        target_dir=bin_dir,
        add_launchers=True,
        dry_run=False,
    )
    maker.executable = shebang_python
    return maker


def mint_one(bin_dir, shebang_python, wrapper_text, spec):
    """Mint one launcher exe; returns its absolute path. distlib also
    writes a python-versioned twin (hermes-3.11.exe) — the payload ships
    exactly one name per entry, so the twin is removed."""
    maker = make_maker(bin_dir, shebang_python, wrapper_text)
    filenames = maker.make(f"{spec['name']} = {spec['module']}:{spec['func']}")
    plain = os.path.join(bin_dir, f"{spec['name']}.exe")
    if os.path.abspath(plain) not in (os.path.abspath(f) for f in filenames):
        raise SystemExit(f"distlib did not write {plain}: {filenames}")
    for extra in filenames:
        if os.path.abspath(extra) != os.path.abspath(plain):
            os.remove(extra)
    return plain


def main(environ=None):
    environ = os.environ if environ is None else environ
    bin_dir = environ["HERMES_MINT_BIN_DIR"]
    specs = json.loads(environ["HERMES_MINT_SPECS"])
    shebang_python = environ["HERMES_MINT_PYTHON"]
    if not os.path.isabs(bin_dir):
        raise SystemExit("HERMES_MINT_BIN_DIR must be absolute")
    if not shebang_python.startswith("<launcher_dir>\\..\\"):
        raise SystemExit(
            "HERMES_MINT_PYTHON must be <launcher_dir>\\..\\payload-relative "
            f"(backslashes), got: {shebang_python}"
        )
    os.makedirs(bin_dir, exist_ok=True)
    minted = []
    for spec in specs:
        # utf-8-sig: tolerate a BOM the JS renderer might prepend.
        with open(environ["HERMES_MINT_WRAPPER"], encoding="utf-8-sig") as f:
            wrapper_text = f.read()
        minted.append(mint_one(bin_dir, shebang_python, wrapper_text, spec))
    return minted


if __name__ == "__main__":
    print("\n".join(main()))
