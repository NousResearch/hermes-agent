from unittest.mock import call, patch

from plugins.platforms.whatsapp import adapter


def test_bridge_spawn_retries_without_breakaway_after_access_denied():
    denied = PermissionError(5, "Access is denied")
    denied.winerror = 5
    process = object()

    with patch.object(adapter, "_IS_WINDOWS", True), \
         patch.object(adapter, "windows_detach_flags", return_value=0x1000), \
         patch.object(adapter, "windows_detach_flags_without_breakaway", return_value=0x0800), \
         patch.object(adapter, "spawn_owned", side_effect=[denied, process]) as spawn:
        result = adapter._spawn_bridge_process(["node", "bridge.js"], None, {})

    assert result is process
    assert spawn.call_args_list == [
        call(
            ["node", "bridge.js"],
            purpose="whatsapp-bridge",
            stdin=adapter.subprocess.DEVNULL,
            stdout=None,
            stderr=None,
            env={},
            creationflags=0x1000,
        ),
        call(
            ["node", "bridge.js"],
            purpose="whatsapp-bridge",
            stdin=adapter.subprocess.DEVNULL,
            stdout=None,
            stderr=None,
            env={},
            creationflags=0x0800,
        ),
    ]
