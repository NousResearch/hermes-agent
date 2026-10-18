"""Install the published build of a frontend product instead of compiling it.

A product is keyed by the sha256 of its input tree (``scripts/build/freshness.mjs`` ``sourceHash``),
the same key its freshness receipt records. CI publishes every TUI and web build of ``main``
(``scripts/ci/publish_products.py``) as ``products/<product>/<key>.tar.gz`` plus a small index
``products/<product>/<key>.json``. When the index for this checkout's exact key exists, the update
downloads the archive (size- and sha256-checked), unpacks it beside the destination, and asks
``freshness.mjs`` whether the unpacked receipt certifies this tree with PM's Node before swapping it
in. Anything else (no published key: forks, local edits, unpublished commits; offline; a blocked
CDN; any verification failure) returns False and the caller compiles locally, exactly as before.

Trust: the archive's sha256 comes from the index on the same HTTPS origin as every other Hermes
release artifact (``hermes-assets.nousresearch.com``), so this trusts that origin the way the release
channels and desktop updater already do. ``updates.prebuilt_products: false`` restores the
git-only trust of a pure source build.

There is deliberately no "nearest published build" fallback: a frontend that does not match the
backend it talks to is worse than a short local build.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import tempfile

#: Where each portable product lives in a checkout (freshness.mjs ``portableProducts``).
OUTPUTS = {"tui": "ui-tui/dist", "web": "hermes_cli/web_dist"}
INDEX_SCHEMA = 1
#: Larger than any real product by ~20x (TUI 0.8 MB, web 1.5 MB compressed); bounds a bad index.
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")
_INDEX_LIMIT = 64 * 1024


def products_base() -> str:
    from hermes_cli.source_releases import _PUBLIC_BASE

    return f"{_PUBLIC_BASE}/products"


def object_names(product: str, key: str) -> tuple[str, str]:
    """``(archive, index)`` object paths under the products base, shared by publisher and reader."""
    if product not in OUTPUTS or not _SHA256.fullmatch(key):
        raise ValueError(f"invalid product key: {product}/{key}")
    return f"{product}/{key}.tar.gz", f"{product}/{key}.json"


def make_index(product: str, key: str, archive: bytes) -> dict:
    import hashlib

    return {"schema": INDEX_SCHEMA, "product": product, "key": key,
            "archive": object_names(product, key)[0].rsplit("/", 1)[1],
            "sha256": hashlib.sha256(archive).hexdigest(), "size": len(archive)}


def valid_index(index: object, product: str, key: str) -> bool:
    if not isinstance(index, dict):
        return False
    size, sha256 = index.get("size"), index.get("sha256")
    return (index.get("schema") == INDEX_SCHEMA and index.get("product") == product and index.get("key") == key
            and index.get("archive") == object_names(product, key)[0].rsplit("/", 1)[1]
            and isinstance(sha256, str) and _SHA256.fullmatch(sha256) is not None
            and isinstance(size, int) and not isinstance(size, bool) and 0 < size <= MAX_ARCHIVE_BYTES)


def prebuilt_products_enabled() -> bool:
    """``updates.prebuilt_products`` in config.yaml; on unless set to false."""
    from hermes_cli.config import load_config_readonly

    updates = (load_config_readonly() or {}).get("updates")
    return not isinstance(updates, dict) or updates.get("prebuilt_products", True) is not False


def read_index(product: str, key: str) -> dict | None:
    """The published index for ``key``; None when absent, unreachable or not this product's.

    Same opener as PM downloads (https-only redirects, the partial-chain TLS context) and the
    same bounded retry for a transient CDN failure; a 404 is the common "not published" answer.
    """
    import urllib.request

    from pm.downloader import _OPENER, _UA
    from pm.network import retry_network

    url = f"{products_base()}/{object_names(product, key)[1]}"

    def read() -> bytes:
        request = urllib.request.Request(url, headers={**_UA, "Cache-Control": "no-cache"})
        with _OPENER.open(request, timeout=15) as response:
            return response.read(_INDEX_LIMIT + 1)

    try:
        body = retry_network(read)
        index = json.loads(body) if len(body) <= _INDEX_LIMIT else None
    except Exception:  # health: allow BLE001 -- HTTP/TLS/JSON/IncompleteRead all mean "not published here"
        return None
    return index if valid_index(index, product, key) else None


def install_prebuilt(project_root: Path, product: str, key: str) -> bool:
    """Install the published build of ``product`` for input ``key``; False = build locally.

    The caller computed ``key`` from this checkout (``source_build.source_product_state``).
    """
    from hermes_cli.source_build import source_product_current

    index = read_index(product, key)
    if index is None:
        return False
    out = project_root / OUTPUTS[product]
    out.parent.mkdir(parents=True, exist_ok=True)
    # A killed install can leave its work dir (at worst holding the previous product, with `out`
    # missing); this install replaces `out` either way, so the leftover is only clutter.
    for leftover in out.parent.glob(f".{out.name}-prebuilt-*"):
        shutil.rmtree(leftover, ignore_errors=True)
    # Unpack on the destination's filesystem so the swap is two renames; the whole work dir,
    # the previous product included, is removed on every exit path.
    with tempfile.TemporaryDirectory(prefix=f".{out.name}-prebuilt-", dir=out.parent) as work:
        staged, previous = Path(work) / "product", Path(work) / "previous"
        try:
            from pm.downloader import Download, Source
            from pm.store import extract_tar

            archive = Path(work) / index["archive"]
            Download([Source(f"{products_base()}/{object_names(product, key)[0]}", archive, index["sha256"])],
                     partials_dir=Path(work) / "partials", connections=1).run()
            if archive.stat().st_size != index["size"]:
                raise ValueError("archive size differs from its index")
            extract_tar(archive, staged)
        except Exception as exc:  # health: allow BLE001 -- any download/unpack failure means "compile locally"
            print(f"  ⓘ Published {product} build unavailable ({type(exc).__name__}); building locally")
            return False
        # The sha256 proves these are the bytes CI published; only the receipt proves they are
        # the build of THIS tree with this Node.
        if not source_product_current(project_root, product, staged):
            print(f"  ⓘ Published {product} build does not match this checkout; building locally")
            return False
        had_previous = out.exists()
        if had_previous:
            os.rename(out, previous)
        try:
            os.rename(staged, out)
        except OSError:
            if had_previous:
                os.rename(previous, out)
            raise
    return True

