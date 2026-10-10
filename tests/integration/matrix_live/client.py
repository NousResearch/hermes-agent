"""Construct the independent Matrix client inside the Linux observer container."""

from __future__ import annotations

import os
from pathlib import Path

from nio import AsyncClient, AsyncClientConfig


def open_encrypted_client() -> AsyncClient:
    store_path = Path(os.environ["NIO_STORE_PATH"])
    store_path.mkdir(parents=True, exist_ok=True)

    user_id = os.environ["NIO_USER_ID"]
    device_id = os.environ["NIO_DEVICE_ID"]
    access_token = os.environ["NIO_ACCESS_TOKEN"]
    client = AsyncClient(
        os.environ["NIO_HOMESERVER"],
        user=user_id,
        device_id=device_id,
        store_path=str(store_path),
        config=AsyncClientConfig(
            encryption_enabled=True,
            store_sync_tokens=True,
            request_timeout=15,
            max_limit_exceeded=0,
            max_timeouts=0,
        ),
    )
    client.restore_login(user_id, device_id, access_token)
    return client
