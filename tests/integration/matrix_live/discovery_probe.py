"""Record the directory response and membership reads in the Linux gateway."""

from __future__ import annotations

import json
import time

from gateway.platforms.base import BasePlatformAdapter
from hermes_constants import get_hermes_home
from mautrix import __version__ as mautrix_version
from mautrix.types import Membership, RoomID


def _observe(adapter, home):
    client = adapter._client
    store = client.state_store
    request = client.api.request
    members = adapter._get_room_members
    discover = adapter.discover_matrix
    path = home / "discovery-trace.jsonl"

    def record(operation, **values):
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"operation": operation, "time": time.monotonic(), **values}) + "\n")

    record("client", mautrix_version=mautrix_version, state_store=type(store).__qualname__)

    async def observed_request(method, api_path, *args, **kwargs):
        response = await request(method, api_path, *args, **kwargs)
        if str(api_path).endswith("/user_directory/search"):
            record("directory", query=args[0] if args else kwargs.get("content"), response=response)
        return response

    async def observed_members(room_id, **kwargs):
        room = RoomID(room_id)
        complete = store.full_member_list.get(room, False)
        cached = sorted(
            str(user) for user, member in store.members.get(room, {}).items()
            if member.membership == Membership.JOIN
        )
        result = await members(room_id, **kwargs)
        record(
            "members", room_id=room_id, complete=complete, cached=cached,
            joined=None if result is None else sorted(result),
        )
        return result

    async def observed_discover(kind, room_id, limit, *, requester, search_term=None):
        result = await discover(kind, room_id, limit, requester=requester, search_term=search_term)
        record("discovery", kind=kind, requester=requester, result=result)
        return result

    client.api.request = observed_request
    adapter._get_room_members = observed_members
    adapter.discover_matrix = observed_discover


def register(ctx):
    original_init = BasePlatformAdapter.__init__

    def observed_init(adapter, *args, **kwargs):
        original_init(adapter, *args, **kwargs)
        if adapter.platform.value != "matrix":
            return
        home = get_hermes_home()
        connect = adapter.connect

        async def observed_connect(*args, **kwargs):
            result = await connect(*args, **kwargs)
            if result:
                _observe(adapter, home)
            return result

        adapter.connect = observed_connect

    BasePlatformAdapter.__init__ = observed_init
