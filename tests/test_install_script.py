from pathlib import Path


def test_install_script_uses_npm_ci_for_reproducible_node_installs():
    script = Path(__file__).resolve().parents[1] / "scripts" / "install.sh"
    content = script.read_text()

    assert content.count('npm ci --silent 2>/dev/null || {') == 2
    assert 'npm install --silent 2>/dev/null || {' not in content
