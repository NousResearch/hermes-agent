from agent.stream_delivery import StreamDeliveryMixin


def _agent():
    agent = StreamDeliveryMixin.__new__(StreamDeliveryMixin)
    agent.stream_delta_callback = None
    agent._stream_callback = None
    return agent


def test_public_stream_is_delivered_before_terminal_disposition():
    delivered = []
    agent = _agent()
    agent.stream_delta_callback = delivered.append
    assert agent._deliver_to_stream_callbacks("partial ordinary response") is True
    assert delivered == ["partial ordinary response"]


def test_interim_commentary_is_suppressed_before_gate():
    agent = _agent()
    delivered = []
    agent.interim_assistant_callback = (
        lambda text, **kwargs: delivered.append((text, kwargs))
    )
    agent._deliver_interim("still working", already_streamed=False, record=["still working"])

    assert delivered == [("still working", {"already_streamed": False})]


def test_reasoning_is_suppressed_when_explicitly_marked_private():
    agent = _agent()
    delivered = []
    agent.reasoning_callback = delivered.append
    agent._public_stream_suppressed = True
    agent._fire_reasoning_delta("private reasoning")

    assert delivered == []
