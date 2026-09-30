"""One Matrix adapter owns a crypto store, including within a single process."""
from gateway.status import _release_file_lock, _try_acquire_file_lock


def claim_store(adapter) -> bool:
    directory = adapter._resolve_store_dir()
    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory / 'adapter.lock').open('a+')
    if not _try_acquire_file_lock(handle):
        handle.close()
        adapter._set_fatal_error(
            'matrix_store_busy',
            'Matrix encryption store is already in use; use the running gateway for delivery.',
            retryable=True,
        )
        return False
    adapter._matrix_store_lock = handle
    identity = adapter._access_token or f'{adapter._homeserver}:{adapter._user_id}'
    if not adapter._acquire_platform_lock('matrix', identity, 'Matrix credential'):
        release_store(adapter)
        return False
    return True


def release_store(adapter) -> None:
    handle = getattr(adapter, '_matrix_store_lock', None)
    if handle is not None:
        adapter._release_platform_lock()
        _release_file_lock(handle)
        handle.close()
        adapter._matrix_store_lock = None
