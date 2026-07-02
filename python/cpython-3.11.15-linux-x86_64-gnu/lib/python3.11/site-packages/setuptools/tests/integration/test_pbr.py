import subprocess

import pytest


@pytest.mark.uses_network
def test_pbr_integration(pbr_package, venv):
    """Ensure pbr packages install."""
    cmd = [
        'python',
        '-m',
        'pip',
        '-v',
        'install',
        '--no-build-isolation',
        pbr_package,
    ]
    venv.run(cmd, stderr=subprocess.STDOUT)
    out = venv.run(["python", "-c", "import mypkg.hello"])
    assert "Hello world!" in out
