from cli import _normalize_cli_busy_input_mode


def test_hybrid_maps_to_queue_outside_messaging_gateway():
    assert _normalize_cli_busy_input_mode("hybrid") == "queue"
    assert _normalize_cli_busy_input_mode(" HYBRID ") == "queue"


def test_canonical_cli_busy_modes_keep_their_behavior():
    assert _normalize_cli_busy_input_mode("interrupt") == "interrupt"
    assert _normalize_cli_busy_input_mode("queue") == "queue"
    assert _normalize_cli_busy_input_mode("steer") == "steer"
    assert _normalize_cli_busy_input_mode("unknown") == "interrupt"