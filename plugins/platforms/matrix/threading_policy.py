"""Matrix auto-thread creation policy; genuine inbound threads bypass this policy."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from plugins.platforms.matrix.adapter import MatrixAdapter


def parse_room_auto_thread_overrides(value: object) -> dict[str, bool]:
    if not isinstance(value, dict):
        raise ValueError("matrix.room_overrides must be a mapping of room IDs to settings")
    overrides: dict[str, bool] = {}
    for room_id, settings in value.items():
        if not isinstance(room_id, str) or not room_id.startswith("!") or ":" not in room_id:
            raise ValueError("matrix.room_overrides keys must be Matrix room IDs (!id:server)")
        path = f"matrix.room_overrides[{room_id!r}]"
        if not isinstance(settings, dict):
            raise ValueError(f"{path} must be a mapping containing auto_thread: true or false")
        if settings.keys() - {"auto_thread"}:
            raise ValueError(f"{path} only supports the auto_thread setting")
        if "auto_thread" in settings:
            auto_thread = settings["auto_thread"]
            if not isinstance(auto_thread, bool):
                raise ValueError(f"{path}.auto_thread must be a YAML boolean (true or false)")
            overrides[room_id] = auto_thread
    return overrides


def should_create_thread(
    adapter: "MatrixAdapter", room_id: str, is_dm: bool, is_mentioned: bool,
) -> bool:
    override = adapter._room_auto_thread_overrides.get(room_id)
    if override is not None:
        return override
    if is_dm:
        return (adapter._dm_mention_threads and is_mentioned) or adapter._dm_auto_thread
    return adapter._matrix_session_scope == "thread" or (
        adapter._matrix_session_scope != "room" and adapter._auto_thread)
