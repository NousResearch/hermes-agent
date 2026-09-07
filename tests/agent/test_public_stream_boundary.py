from agent.stream_delivery import StreamDeliveryMixin


def test_public_stream_is_withheld_until_terminal_response_is_accepted():
    delivered = []
    agent = StreamDeliveryMixin.__new__(StreamDeliveryMixin)
    agent.stream_delta_callback = delivered.append
    agent._stream_callback = None
    agent._public_stream_suppressed = True

    assert agent._deliver_to_stream_callbacks("partial ordinary response") is False
    assert delivered == []

    agent._public_stream_suppressed = False
    assert agent._deliver_to_stream_callbacks("MEETING\nShould we continue?") is True
    assert delivered == ["MEETING\nShould we continue?"]
