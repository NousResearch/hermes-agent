from gateway.run import _sanitize_gateway_final_response
from gateway.stream_consumer import GatewayStreamConsumer


def test_final_response_strips_media_and_local_paths_for_telegram():
    text = "Done MEDIA:/Users/client/.hermes/output/report.pdf /home/client/.hermes/output/chart.png attached"

    cleaned = _sanitize_gateway_final_response("telegram", text)

    assert "MEDIA:" not in cleaned
    assert "/Users/client" not in cleaned
    assert "/home/client" not in cleaned
    assert cleaned == "Done attached"


def test_final_response_replaces_tool_xml_and_runtime_errors():
    tool_xml = '<tool_use name="read_file">/Users/client/.ssh/id_rsa</tool_use>'
    none_type = "⚠️ 'NoneType' object is not iterable"

    assert "tool_use" not in _sanitize_gateway_final_response("telegram", tool_xml)
    assert "NoneType" not in _sanitize_gateway_final_response("telegram", none_type)


def test_final_response_removes_system_nudge_boilerplate():
    text = "Observed unresolved work: internal resume marker\n\nClient-facing line"

    assert _sanitize_gateway_final_response("telegram", text) == "Client-facing line"


def test_non_telegram_keeps_existing_behavior():
    text = "MEDIA:/tmp/report.pdf"

    assert _sanitize_gateway_final_response("slack", text) == text


def test_stream_cleaner_strips_internal_artifacts():
    text = 'Here MEDIA:/Users/client/out.pdf <tool_result>{"ok": true}</tool_result> /tmp/chart.png'

    cleaned = GatewayStreamConsumer._clean_for_display(text)

    assert "MEDIA:" not in cleaned
    assert "tool_result" not in cleaned
    assert "/tmp/chart.png" not in cleaned
