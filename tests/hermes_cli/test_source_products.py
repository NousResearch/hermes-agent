"""An update installs CI's build of a frontend only when it certifies this exact checkout.

The real publisher (scripts/ci/publish_products.py) writes the served tree, real freshness.mjs
receipts are read under the host's Node (standing in for PM's), and pm.downloader fetches over a
loopback HTTP server: only the asset host's address is substituted.
"""
from __future__ import annotations

import functools
import hashlib
import http.server
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import threading
from types import SimpleNamespace

import pytest

import pm
from hermes_cli import source_build, source_products
from scripts.ci import publish_products

REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="product receipts are read by Node")


def _checkout(root: Path) -> Path:
    (root / "scripts/build").mkdir(parents=True, exist_ok=True)
    for name in ("freshness.mjs", "frontend-common.mjs"):
        shutil.copy2(REPO / "scripts/build" / name, root / "scripts/build" / name)
    for rel, text in {"package.json": '{"private":true}', "package-lock.json": "{}",
                      "web/src/main.ts": 'export const answer = "published";',
                      "web/public/favicon.ico": "icon"}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    return root


def _build_web(source: Path, html: str = "<html>published</html>") -> None:
    """Stand-in for scripts/build/web.mjs: an output plus the real receipt it records."""
    out = source / "hermes_cli/web_dist"
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(html, encoding="utf-8")
    script = (source / "scripts/build/freshness.mjs").as_uri()
    subprocess.run([shutil.which("node"), "--input-type=module", "-e",
                    f"import {{recordProduct, buildInputs}} from {json.dumps(script)};"
                    "const [source, out] = process.argv.slice(1);"
                    "recordProduct({source, product: 'web', out, inputs: buildInputs(source, 'web', {icons: source + '/web/public'})});",
                    str(source), str(out)], check=True)


@pytest.fixture
def host(tmp_path, monkeypatch):
    """A loopback asset host, plus the host's Node standing in for PM's receipt reader."""
    node, real = Path(shutil.which("node")), pm.installed_package
    monkeypatch.setattr(pm, "installed_package", lambda name, **kwargs: (
        SimpleNamespace(binary=node) if name == "node" else real(name, **kwargs)))
    served = tmp_path / "served"
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(served))
    handler.log_message = lambda *_a: None
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(source_products, "products_base", lambda: f"http://127.0.0.1:{server.server_port}/products")
    try:
        yield served
    finally:
        server.shutdown()
        server.server_close()


def _publish(ci: Path, served: Path) -> str:
    _build_web(ci)
    assert publish_products.main(["--source", str(ci), "--product", "web", "--dry-run", str(served)]) == 0
    return publish_products.receipt_key(ci / "hermes_cli/web_dist", "web")


def _state(root: Path) -> tuple[bool, str | None]:
    return source_build.source_product_state(root, "web", root / "hermes_cli/web_dist")


def test_published_build_installs_only_when_it_certifies_this_checkout(tmp_path, host):
    key = _publish(_checkout(tmp_path / "ci"), host)
    user = _checkout(tmp_path / "user")  # same tree, another directory (and possibly another host)
    out = user / "hermes_cli/web_dist"
    _build_web(user, "<html>previous local build</html>")
    (user / "web/src/main.ts").write_text('export const answer = "published";', encoding="utf-8")
    assert _state(user) == (True, None)  # current: nothing is looked up

    # The local build's receipt no longer matches (stale after a pull); CI published this tree.
    (out / "hermes-build.json").write_text("{}", encoding="utf-8")
    current, user_key = _state(user)
    assert (current, user_key) == (False, key)
    assert source_products.install_prebuilt(user, "web", key)
    assert (out / "index.html").read_text(encoding="utf-8") == "<html>published</html>"
    assert _state(user) == (True, None)
    assert sorted(p.name for p in out.parent.iterdir()) == ["web_dist"]  # the replaced build left nothing behind

    # A tree CI never published has no index: nothing is touched, the caller compiles.
    (user / "web/src/main.ts").write_text('export const answer = "local edit";', encoding="utf-8")
    _, edited = _state(user)
    assert edited and edited != key
    assert not source_products.install_prebuilt(user, "web", edited)
    assert (out / "index.html").read_text(encoding="utf-8") == "<html>published</html>"


def _rewrite_archive(host: Path, key: str, mutate) -> None:
    """Repack the published archive through ``mutate(members)`` and re-pin its index."""
    archive = host / "products/web" / f"{key}.tar.gz"
    with tarfile.open(archive) as tar:
        members = {m.name: (m, tar.extractfile(m).read() if m.isfile() else None) for m in tar.getmembers()}
    mutate(members)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for member, data in members.values():
            if data is not None:
                member.size = len(data)
            tar.addfile(member, io.BytesIO(data) if data is not None else None)
    archive.write_bytes(buffer.getvalue())
    index = host / "products/web" / f"{key}.json"
    index.write_text(json.dumps({**json.loads(index.read_text()), "size": len(buffer.getvalue()),
                                 "sha256": hashlib.sha256(buffer.getvalue()).hexdigest()}), encoding="utf-8")


def _edit_index(**changes):
    def tamper(host: Path, key: str) -> None:
        index = host / "products/web" / f"{key}.json"
        index.write_text(json.dumps({**json.loads(index.read_text()), **changes}), encoding="utf-8")
    return tamper


def _other_node(members) -> None:
    member, data = members["hermes-build.json"]
    members["hermes-build.json"] = (member, json.dumps({**json.loads(data), "node": "0.0.0"}).encode())


def _escaping_member(members) -> None:
    member = tarfile.TarInfo("../../escaped.txt")
    members[member.name] = (member, b"outside")


TAMPERS = {
    "wrong-sha256": _edit_index(sha256="0" * 64),
    "wrong-size": _edit_index(size=1),
    "index-for-another-product": _edit_index(product="tui"),
    "other-node-receipt": lambda host, key: _rewrite_archive(host, key, _other_node),
    "escaping-member": lambda host, key: _rewrite_archive(host, key, _escaping_member),
}


@pytest.mark.parametrize("tamper", sorted(TAMPERS))
def test_a_published_build_that_does_not_verify_never_replaces_the_product(tmp_path, host, tamper):
    """sha256 and size prove the bytes are CI's; only the receipt proves they are THIS tree's build;
    the tar containment policy keeps every member inside the product."""
    key = _publish(_checkout(tmp_path / "ci"), host)
    TAMPERS[tamper](host, key)
    user = _checkout(tmp_path / "user")
    assert not source_products.install_prebuilt(user, "web", key)
    assert not (user / "hermes_cli/web_dist").exists()
    assert not (tmp_path / "escaped.txt").exists() and not (user / "escaped.txt").exists()


@pytest.mark.parametrize("index", [
    {"product": "tui"}, {"key": "f" * 64}, {"archive": "other.tar.gz"}, {"size": 10**12}, {"schema": 2}])
def test_an_index_naming_anything_else_is_not_published(index):
    key = "a" * 64
    good = source_products.make_index("web", key, b"bytes")
    assert source_products.valid_index(good, "web", key)
    assert not source_products.valid_index({**good, **index}, "web", key)


def test_prebuilt_products_can_be_turned_off_in_config(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert source_products.prebuilt_products_enabled()
    (home / "config.yaml").write_text("updates:\n  prebuilt_products: false\n", encoding="utf-8")
    assert not source_products.prebuilt_products_enabled()
