"""Publish this commit's TUI and web builds for source updates to install (hermes_cli/source_products.py).

Each product is published under the sha256 of its input tree (the key its freshness receipt records):
``products/<product>/<key>.tar.gz`` (the product directory, receipt included) and then the index
``products/<product>/<key>.json``. Both are immutable: a key that is already published is left as is,
and the index is written only after its archive, so a reader never sees an index naming a missing archive.

    python -m scripts.ci.publish_products --source . [--dry-run DIR]
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
from pathlib import Path
import sys
import tarfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hermes_cli.source_products import OUTPUTS, make_index, object_names

_PREFIX = "products"


def receipt_key(out: Path, product: str) -> str:
    """The input key a built product's receipt was written for."""
    receipt = json.loads((out / "hermes-build.json").read_text(encoding="utf-8-sig"))
    if receipt.get("schema") != 1 or receipt.get("product") != product:
        raise ValueError(f"{out}: not a {product} receipt")
    key = receipt["inputs"]["sourceHash"]
    object_names(product, key)  # validates the key
    return key


def pack(out: Path) -> bytes:
    """A byte-reproducible tar.gz of ``out``: sorted members, no owners, no clocks.

    The builders' ownership marker is left out; the installer writes its own.
    """
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(out.rglob("*"), key=lambda p: p.relative_to(out).as_posix()):
            name = path.relative_to(out).as_posix()
            if name == ".hermes-product" or path.is_symlink():
                continue
            info = tar.gettarinfo(str(path), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            info.mode = 0o755 if path.is_dir() or info.mode & 0o111 else 0o644
            if path.is_file():
                with path.open("rb") as handle:
                    tar.addfile(info, handle)
            else:
                tar.addfile(info)
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode="wb", mtime=0, compresslevel=9) as gz:
        gz.write(raw.getvalue())
    return compressed.getvalue()


def objects(source: Path, products=tuple(OUTPUTS)) -> list[tuple[str, bytes, str]]:
    """``(key, body, content type)`` for each product, archive before its index."""
    result = []
    for product in products:
        out = source / OUTPUTS[product]
        key = receipt_key(out, product)
        archive = pack(out)
        archive_name, index_name = object_names(product, key)
        result.append((f"{_PREFIX}/{archive_name}", archive, "application/gzip"))
        result.append((f"{_PREFIX}/{index_name}",
                       json.dumps(make_index(product, key, archive), sort_keys=True).encode() + b"\n",
                       "application/json"))
    return result


def publish(items: list[tuple[str, bytes, str]]) -> None:
    import tempfile

    from scripts.releases import r2

    creds, base, bucket = r2.credentials()
    with tempfile.TemporaryDirectory(prefix="hermes-products-") as work:
        for number, (key, body, content_type) in enumerate(items):
            # A file payload: put_object's immutable-conflict path (an object already published
            # under this key) verifies the remote bytes are ours and reuses it; a bytes payload
            # would fail the rerun instead.
            payload = Path(work) / str(number)
            payload.write_bytes(body)
            r2.put_object(creds, base, bucket, key, str(payload), r2.amz_timestamp(), content_type,
                          conditions={"If-None-Match": "*"})
            print(f"published {key} ({len(body)} bytes)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dry-run", type=Path, help="write the objects under this directory instead of R2")
    parser.add_argument("--product", action="append", choices=sorted(OUTPUTS), help="default: every product")
    args = parser.parse_args(argv)
    items = objects(args.source.resolve(), tuple(args.product or OUTPUTS))
    if args.dry_run:
        for key, body, _ in items:
            target = args.dry_run / key
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
            print(f"wrote {target}")
        return 0
    publish(items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
