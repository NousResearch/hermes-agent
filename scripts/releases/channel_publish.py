"""Read a channel claim's admitted request, digest-pinned, from the public channel origin.

Each candidate leg and smoke of a channel claim retrieves the request it builds with
(``request --out``); channel_claims.py binds that request to the claim.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from hermes_cli.release_channels import ChannelReader, build_prefix, decode_json, require_sha256, validate_request


def read_request(build_id: str, digest: str, public_base: str, repository: str) -> dict:
    require_sha256(digest)
    reader = ChannelReader(public_base, repository=repository)
    body = reader.read_bytes(build_prefix(build_id) + "request.json", digest)
    request = validate_request(decode_json(body), repository=repository, base_url=public_base)
    if request["buildId"] != build_id:
        raise ValueError("Channel request build ID differs from its key")
    return request


def main(argv: list[str] | None = None) -> None:
    from scripts.releases.r2_scope import channel_public_base

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["request"])
    parser.add_argument("--build-id", default=os.environ.get("CHANNEL_BUILD", ""))
    parser.add_argument("--request-sha256", default=os.environ.get("CHANNEL_REQUEST_SHA256", ""))
    parser.add_argument("--public-base", help="Expected public authority")
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    request = read_request(args.build_id, args.request_sha256, channel_public_base(args.public_base), args.repository)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(request, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
