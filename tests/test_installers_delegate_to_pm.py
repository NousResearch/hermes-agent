"""Desired behavior: the installers delegate the python/venv/tools install
to pm, and keep the bootstrap config hygiene, instead of hand-rolling a
`uv sync`. The ambient-config-doesn't-affect-uv contract itself is tested
behaviorally at the pm layer (tests/pm/test_venv_sync_ambient_config.py);
these assertions only pin the delegation shape so a future edit can't
silently reintroduce a second uv-sync path.

The installer scripts are fetched standalone (`curl | sh`, `irm | iex`)
before any checkout exists, so this is a static guard on the shipped
bytes — the same approach test_bootstrap_pins_fragment.py takes.
"""

from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"


def test_installers_delegate_to_pm_cli() -> None:
    """The single install authority is `python -m pm.cli install`; the
    installers must forward to it rather than run `uv sync` themselves."""
    for path in (INSTALL_SH, INSTALL_PS1):
        text = path.read_text(encoding="utf-8")
        assert "python -m pm.cli install" in text, f"{path.name}: missing pm delegation"
        # A raw project sync must not reappear as an INVOCATION in the
        # installers — pm owns it. (Comments may mention the words; the
        # command shape is what we ban.)
        for banned in ('uv sync --', 'uv sync -', '"$UV_CMD" sync', '& $uv sync'):
            assert banned not in text, f"{path.name}: hand-rolled uv sync reappeared ({banned!r})"


def test_install_sh_keeps_bootstrap_config_hygiene() -> None:
    """#21269: the bootstrap uv call stays isolated from ambient config."""
    text = INSTALL_SH.read_text(encoding="utf-8")
    assert "export UV_NO_CONFIG=1" in text


@pytest.mark.parametrize("path", [INSTALL_SH, INSTALL_PS1])
def test_installers_still_create_the_venv_stage(path: Path) -> None:
    """pm's sync targets the venv dir the installer creates; the stage must
    survive (Hermes-Setup's manifest lists it)."""
    text = path.read_text(encoding="utf-8")
    assert "venv" in text  # both keep a venv-creating stage / delegate to pm
