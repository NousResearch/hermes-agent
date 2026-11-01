from __future__ import annotations

import stat

from plugins.agentops.control.api import request_control_api, request_health
from plugins.agentops.control.daemon import start_daemon_thread


def test_uds_exposes_health_and_no_write_routes(tmp_path, write_config):
    config_path = write_config()
    handle = start_daemon_thread(config_path)
    try:
        health = request_health(handle.socket_path)
        write_status, _ = request_control_api(handle.socket_path, "POST", "/v1/events")
        unknown_status, _ = request_control_api(handle.socket_path, "GET", "/v1/fleet")

        assert health["authority_mode"] == "observe_only"
        assert health["ready"] is True
        assert write_status == 405
        assert unknown_status == 404
        assert stat.S_IMODE(handle.socket_path.stat().st_mode) == 0o600
    finally:
        handle.stop()
