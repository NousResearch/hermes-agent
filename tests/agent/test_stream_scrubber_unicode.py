"""Tag filtering must preserve Unicode prose regardless of stream chunking."""

import pytest

from agent.memory_manager import StreamingContextScrubber
from agent.stream_delivery import StreamDeliveryMixin
from agent.think_scrubber import StreamingThinkScrubber


@pytest.mark.parametrize("tag", [
    "memory-context", "think", "thinking", "reasoning", "thought", "REASONING_SCRATCHPAD",
])
@pytest.mark.parametrize("prefix, hidden", [
    ("İstanbul\n", "private context"),
    ("Visible\n", "İzmir İstanbul"),
    ("İİİ\n", "İİİ"),
])
def test_unicode_prose_is_independent_of_tag_chunk_boundaries(tag, prefix, hidden):
    factory = StreamingContextScrubber if tag == "memory-context" else StreamingThinkScrubber
    suffix = "Yanıt: doğru."
    text = f"{prefix}<{tag.upper()}>\n{hidden}</{tag.upper()}>{suffix}"
    partitions = [[text], list(text)]
    partitions.extend([text[:cut], text[cut:]] for cut in range(1, len(text)))

    for chunks in partitions:
        scrubber = factory()
        visible = "".join(scrubber.feed(chunk) for chunk in chunks) + scrubber.flush()
        assert visible == prefix + suffix, chunks


@pytest.mark.parametrize("chunk_size", [None, 1, 11])
def test_unicode_stream_reaches_display_and_tts_without_hidden_blocks(chunk_size):
    agent = StreamDeliveryMixin()
    agent._stream_think_scrubber = StreamingThinkScrubber()
    agent._stream_context_scrubber = StreamingContextScrubber()
    display, speech = [], []
    agent.stream_delta_callback = display.append
    agent._stream_callback = speech.append
    agent._current_streamed_assistant_text = ""
    text = (
        "İstanbul\n<THINK>\nİzmir\n</THINK>"
        "<memory-context>\nİzmit\n</memory-context>Yanıt: doğru."
    )
    size = chunk_size or len(text)
    for start in range(0, len(text), size):
        agent._fire_stream_delta(text[start:start + size])
    agent._reset_stream_delivery_tracking()

    assert "".join(display) == "İstanbul\nYanıt: doğru."
    assert "".join(speech) == "".join(display)
