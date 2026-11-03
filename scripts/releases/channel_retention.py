"""Retire immutable preview metadata only when its claim archive also expires."""
from __future__ import annotations

from collections.abc import Callable
import hashlib

from hermes_cli.release_channels import (
    ChannelError, build_prefix, canonical_json, channel_key, validate_manifest, validate_record, validate_request,
)
from hermes_cli.update_channel import channel_timestamp


class PreviewRetention:
    """One authenticated listing/read snapshot; unreadable metadata aborts before any deletion."""

    def __init__(self, read_json: Callable[[str], object]):
        self.read_json = read_json
        self.requests: dict[str, dict] = {}

    def request(self, build_id: str) -> dict:
        if build_id not in self.requests:
            key = build_prefix(build_id) + 'request.json'
            request = self.read_json(key)
            if not isinstance(request, dict):
                raise ChannelError(f'Cannot read request {key}')
            record = validate_record(self.read_json(channel_key(request.get('channel'))))
            validate_request(request, repository=record['repository'], base_url=request.get('publicBase'),
                             policy=record['policy'])
            if request['buildId'] != build_id or request['identity'] != record['identity']:
                raise ChannelError(f'Request identity mismatch: {key}')
            self.requests[build_id] = request
        return self.requests[build_id]

    def live_claims(self, keys: list[str]) -> set[str]:
        claims = set()
        for key in keys:
            if not key.startswith('releases/channels/'):
                continue
            record = validate_record(self.read_json(key))
            if channel_key(record['name']) != key:
                raise ChannelError(f'Channel record key mismatch: {key}')
            # Retirements pin both the old channel head and a particular destination head.
            for field in ('head', 'lastHead', 'destinationHead'):
                head = record.get(field)
                if head is not None:
                    owner = record
                    if field == 'destinationHead':
                        name = record['destination']
                        owner = validate_record(self.read_json(channel_key(name)), name=name,
                                                repository=record['repository'])
                    claims.add(self.head_request(head, owner)['releaseTag'])
        return claims

    def head_request(self, head: dict, owner: dict) -> dict:
        request = self.request(head['buildId'])
        manifest = self.read_json(head['manifestKey'])
        if hashlib.sha256(canonical_json(manifest)).hexdigest() != head['sha256']:
            raise ChannelError('Live head manifest digest mismatch')
        validate_manifest(manifest, {**owner, 'head': head}, request['publicBase'])
        if manifest['request'] != request:
            raise ChannelError('Live head differs from its immutable request')
        return request

    def check_claim_pointers(self, keys: list[str]) -> None:
        for key in keys:
            if not (key.startswith('releases/tag/channel.') and key.endswith('/channel-request.json')):
                continue
            pointer = self.read_json(key)
            if not isinstance(pointer, dict) or set(pointer) != {'schema', 'buildId', 'sha256'} or pointer['schema'] != 1:
                raise ChannelError(f'Cannot read claim pointer {key}')
            request = self.request(pointer['buildId'])
            if (key != f"releases/tag/{request['releaseTag']}/channel-request.json"
                    or hashlib.sha256(canonical_json(request)).hexdigest() != pointer['sha256']):
                raise ChannelError(f'Claim pointer request mismatch: {key}')

    def expired_keys(self, keys: list[str], cutoff: str, heads: set[str]) -> set[str]:
        builds: dict[str, list[str]] = {}
        for key in keys:
            if key.startswith('releases/channel-builds/'):
                build_id = key.split('/')[2]
                build_prefix(build_id)
                builds.setdefault(build_id, []).append(key)
        doomed = set()
        for build_id, objects in builds.items():
            request = self.request(build_id)
            tag = request.get('releaseTag')
            stamp = channel_timestamp(tag)
            # Protected stable/canary requests and all metadata for live claims stay untouched.
            if stamp is not None and stamp[:8] < cutoff and tag not in heads:
                doomed.update(objects)
        return doomed


def channel_retention_plan(read_json: Callable[[str], object], keys: list[str],
                           cutoff: str) -> tuple[set[str], set[str]]:
    try:
        retention = PreviewRetention(read_json)
        heads = retention.live_claims(keys)
        retention.check_claim_pointers(keys)
        return heads, retention.expired_keys(keys, cutoff, heads)
    except (ChannelError, KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f'Cannot establish preview retention: {error}; refusing to prune') from error
