"""A dev checkout linked into ``plugins/`` with its ``plugin.js`` at the root publishes a real copy.

The host pass shares ``<root>/desktop-plugins`` with native Desktop, whose earlier copy of such a
checkout kept the link itself (marker written into the checkout). Neither writer may stamp a folder
reached through a link, and the prune drops that leftover link once the package link is gone.
"""

import json
import shutil
from pathlib import Path

import pytest

from hermes_cli.desktop_plugin_halves import PACKAGE_MARKER, reconcile_desktop_halves

pytestmark = pytest.mark.platforms("posix")


def _checkout(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "hermes"
    checkout = tmp_path / "src" / "radar"
    checkout.mkdir(parents=True)
    (checkout / "plugin.yaml").write_text("name: radar\n", encoding="utf-8")
    (checkout / "plugin.js").write_text("root entry", encoding="utf-8")
    (root / "plugins").mkdir(parents=True)
    link = root / "plugins" / "radar"
    link.symlink_to(checkout, target_is_directory=True)
    return root, root / "desktop-plugins", checkout, link


def test_a_linked_root_layout_checkout_publishes_a_copy_and_leaves_with_its_link(tmp_path):
    root, app_root, checkout, link = _checkout(tmp_path)
    target = app_root / "radar"

    assert reconcile_desktop_halves(root, app_root) == [str(target)]
    assert not target.is_symlink()
    assert (target / "plugin.js").read_text(encoding="utf-8") == "root entry"
    marker = json.loads((target / PACKAGE_MARKER).read_text(encoding="utf-8"))
    assert (marker["package"], marker["source"], marker["linked"]) == ("radar", str(link), True)
    assert not (checkout / PACKAGE_MARKER).exists()
    assert reconcile_desktop_halves(root, app_root) == []

    link.unlink()
    assert reconcile_desktop_halves(root, app_root) == [str(target)]
    assert not target.exists()
    assert sorted(p.name for p in checkout.iterdir()) == ["plugin.js", "plugin.yaml"]


def test_a_copy_that_kept_the_checkout_link_is_replaced_then_pruned_without_touching_the_checkout(tmp_path):
    root, app_root, checkout, link = _checkout(tmp_path)
    target = app_root / "radar"
    ghost = {"linked": True, "package": "radar", "source": str(link), "sourceMtimeMs": 9e15}
    (checkout / PACKAGE_MARKER).write_text(json.dumps(ghost), encoding="utf-8")
    app_root.mkdir()
    target.symlink_to(checkout, target_is_directory=True)

    assert reconcile_desktop_halves(root, app_root) == [str(target)]
    assert not target.is_symlink()
    assert (target / "plugin.js").read_text(encoding="utf-8") == "root entry"

    shutil.rmtree(target)
    target.symlink_to(checkout, target_is_directory=True)
    link.unlink()
    assert reconcile_desktop_halves(root, app_root) == [str(target)]
    assert not target.is_symlink() and not target.exists()
    assert (checkout / "plugin.js").read_text(encoding="utf-8") == "root entry"


def test_a_plugin_the_user_linked_into_the_app_root_is_never_stamped(tmp_path):
    root = tmp_path / "hermes"
    own = tmp_path / "src" / "pkg"
    own.mkdir(parents=True)
    (own / "plugin.js").write_text("same", encoding="utf-8")
    half = root / "plugins" / "pkg" / "desktop"
    half.mkdir(parents=True)
    (half / "plugin.js").write_text("same", encoding="utf-8")
    app_root = root / "desktop-plugins"
    app_root.mkdir()
    (app_root / "pkg").symlink_to(own, target_is_directory=True)

    assert reconcile_desktop_halves(root, app_root) == []
    assert not (own / PACKAGE_MARKER).exists()
    assert (app_root / "pkg").is_symlink()
