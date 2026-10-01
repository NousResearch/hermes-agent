"""Generated bootstrap pins must match the shared pin and mirror authorities."""
import copy
import importlib.util
import subprocess
import sys
from pathlib import Path


def test_fragments_match_the_pin_table():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(root / "scripts/gen-bootstrap-pins.py"), "--check"],
        cwd=root, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_prepared_git_pin_travels_with_standalone_installer():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("bootstrap_pins", root / "scripts/gen-bootstrap-pins.py")
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    from pm.artifact_mirror import github_asset_url, mirror_url
    git = copy.deepcopy(generator._load_git_pin())
    prepared_sha = "a" * 64
    prepared_digest = "b" * 64
    git["prepared"] = {"win32-x64": {"sha256": prepared_sha, "digest": prepared_digest}}
    fragment = generator._ps1_fragment(generator._load_uv_pin(), git)
    assert f'PreparedUrl = "{github_asset_url(prepared_sha)}"' in fragment
    assert f'PreparedMirrorUrl = "{mirror_url(prepared_sha)}"' in fragment
    assert f'PreparedSha256 = "{prepared_sha}"' in fragment
    assert f'PreparedDigest = "{prepared_digest}"' in fragment
    assert 'PreparedSha256 = ""' in fragment
    assert 'PreparedDigest = ""' in fragment
