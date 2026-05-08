"""Regression coverage for the Termux broad install profile."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def test_pyproject_defines_termux_all_without_known_blockers() -> None:
    text = PYPROJECT.read_text()
    assert "termux-all = [" in text
    assert '"hermes-agent[termux]"' in text
    assert '"hermes-agent[matrix]"' not in text.split("termux-all = [", 1)[1].split("]", 1)[0]
    assert '"hermes-agent[voice]"' not in text.split("termux-all = [", 1)[1].split("]", 1)[0]


def test_install_script_resolves_termux_profile_extras_and_fallbacks() -> None:
    text = INSTALL_SH.read_text()
    assert "resolve_termux_extra()" in text
    assert 'echo "termux-all"' in text
    assert 'echo "termux"' in text
    assert 'echo "termux-minimal"' in text
    assert 'termux_extra="$(resolve_termux_extra)"' in text
    assert 'pip install -e ".[${termux_extra}]" -c constraints-termux.txt' in text
    assert "Termux profile (.[${termux_extra}]) failed, trying minimal Termux profile..." in text
    assert "Termux minimal profile (.[termux-minimal]) failed, trying base install..." in text
