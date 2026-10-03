"""Source ZIP updates preserve the separately built Webapp renderer."""

from pathlib import Path
import subprocess
import zipfile

import pytest

from hermes_cli import update_cmd_zip
from hermes_cli.webapp import webapp_dist_dir


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                          text=True, check=True).stdout


@pytest.fixture
def clean_install(tmp_path):
    # Keep the real ignore rules and tracked root entries without copying a full checkout
    # or committing fixture source. Git objects are read-only shared with the test checkout.
    root = tmp_path / "install"
    repo = Path(__file__).resolve().parents[2]
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", "--template=",
                    str(repo), str(root)], check=True)
    _git(root, "config", "core.sparseCheckout", "true")
    (root / ".git/info").mkdir(exist_ok=True)
    (root / ".git/info/sparse-checkout").write_text(
        "/.gitignore\n/apps/desktop/package.json\n", encoding="utf-8")
    _git(root, "read-tree", "-mu", "HEAD")
    assert _git(root, "status", "--porcelain", "--untracked-files=all") == ""
    for dist in (root / "apps/desktop/dist", webapp_dist_dir(root)):
        dist.mkdir(parents=True)
        (dist / "index.html").write_bytes(f"retained {dist.name}".encode())
    return root


def test_clean_webapp_output_is_admitted_but_dirty_source_still_refuses(clean_install):
    root = clean_install
    assert _git(root, "status", "--porcelain", "--untracked-files=all") == ""
    assert _git(root, "check-ignore", str(webapp_dist_dir(root) / "index.html")).strip()
    for shipped in (None, {"apps", ".gitignore"}):
        assert update_cmd_zip._zip_overlay_block_reason(root, shipped=shipped) is None
    source = root / "apps/desktop/package.json"
    source.write_bytes(source.read_bytes() + b"\n")
    for shipped in (None, {"apps", ".gitignore"}):
        assert update_cmd_zip._zip_overlay_block_reason(root, shipped=shipped) is not None


def test_local_zip_grafts_both_renderers_and_preswap_guard_keeps_dirty_source(clean_install, tmp_path, monkeypatch):
    from hermes_cli import main

    root = clean_install
    monkeypatch.setattr(main, "PROJECT_ROOT", root)
    source = root / "apps/desktop/package.json"
    original = source.read_bytes()
    outputs = {path: path.read_bytes() for path in (
        root / "apps/desktop/dist/index.html", webapp_dist_dir(root) / "index.html")}
    extracted = tmp_path / "hermes-agent-main"
    new_source = extracted / "apps/desktop/package.json"
    new_source.parent.mkdir(parents=True)
    new_source.write_bytes(b'{"name":"updated-source"}\n')
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.write(new_source, "hermes-agent-main/apps/desktop/package.json")

    # A failed pre-swap check leaves the live source and both outputs untouched.
    source.write_bytes(original + b"\n")
    with pytest.raises(SystemExit) as exc:
        update_cmd_zip._download_and_swap_zip("main", archive.as_uri())
    assert exc.value.code == 1
    assert source.read_bytes() == original + b"\n"
    assert all(path.read_bytes() == data for path, data in outputs.items())
    assert not (root / "apps.hermes-update-staging").exists()
    source.write_bytes(original)

    # Exercise grafting separately so a missing artifact cannot hide behind a guard refusal.
    staged = update_cmd_zip._stage_entries(str(extracted), ["apps"], str(root))
    try:
        for path, data in outputs.items():
            assert (Path(staged[0][0]) / path.relative_to(root / "apps")).read_bytes() == data
    finally:
        update_cmd_zip._discard_staged(staged)
    update_cmd_zip._download_and_swap_zip("main", archive.as_uri())
    assert source.read_bytes() == new_source.read_bytes()
    assert all(path.read_bytes() == data for path, data in outputs.items())
    assert not (root / "apps.hermes-update-staging").exists()
    assert not (root / "apps.hermes-update-old").exists()
