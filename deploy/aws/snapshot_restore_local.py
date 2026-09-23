"""Rebuild an EBS snapshot as a local raw disk image, verifying every block.

Used by the restore drill when a test instance cannot (or should not) be launched: the EBS
direct APIs return the snapshot's own blocks, each with a SHA-256 checksum, so the image
written here is the snapshot's content, byte for byte, and a block that does not match its
checksum stops the drill rather than producing a quietly corrupt restore.

    python3 snapshot_restore_local.py <snapshot-id> <output.img> [--region eu-west-2]

Blocks the snapshot does not list were never written and read as zeros; the output file is
sparse, so a 100 GiB volume with 1 GiB in use costs about 1 GiB of disk.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import sys
from concurrent.futures import ThreadPoolExecutor

import boto3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot_id")
    parser.add_argument("output")
    parser.add_argument("--region", default="eu-west-2")
    args = parser.parse_args()

    ebs = boto3.client("ebs", region_name=args.region)
    blocks, token, size, block_size = [], None, 0, 0
    while True:
        page = ebs.list_snapshot_blocks(
            SnapshotId=args.snapshot_id, MaxResults=10000, **({"NextToken": token} if token else {})
        )
        size, block_size = page["VolumeSize"], page["BlockSize"]
        blocks.extend(page["Blocks"])
        token = page.get("NextToken")
        if not token:
            break
    print(f"{args.snapshot_id}: {size} GiB volume, {len(blocks)} blocks of {block_size} bytes "
          f"({len(blocks) * block_size / 2**20:.0f} MiB written)", flush=True)

    with open(args.output, "wb") as out:
        out.truncate(size * 2**30)

    def fetch(block):
        response = ebs.get_snapshot_block(
            SnapshotId=args.snapshot_id, BlockIndex=block["BlockIndex"], BlockToken=block["BlockToken"]
        )
        data = response["BlockData"].read()
        digest = base64.b64encode(hashlib.sha256(data).digest()).decode()
        if response.get("ChecksumAlgorithm") == "SHA256" and digest != response["Checksum"]:
            raise SystemExit(f"block {block['BlockIndex']}: checksum mismatch — snapshot data is corrupt")
        return block["BlockIndex"], data

    done = 0
    with open(args.output, "r+b") as out, ThreadPoolExecutor(max_workers=16) as pool:
        for index, data in pool.map(fetch, blocks):
            out.seek(index * block_size)
            out.write(data)
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(blocks)} blocks verified", flush=True)
    print(f"restored {done} blocks, every checksum verified -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
