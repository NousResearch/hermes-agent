import pytest

from gateway.media_response_processor import MediaResponseProcessor, MediaStreamBuffer


def test_stream_buffer_never_emits_a_split_media_marker():
    buffer = MediaStreamBuffer()

    assert buffer.feed("Ready\nMED") == "Ready\n"
    assert buffer.feed("IA:/tmp/partial-report") == ""
    assert buffer.feed(".pdf\nDone") == "MEDIA:/tmp/partial-report.pdf\nDone"
    assert buffer.finish("Ready\nMEDIA:/tmp/partial-report.pdf\nDone") == ""


@pytest.mark.asyncio
async def test_processor_resumes_streaming_after_media_replacement():
    paths = []

    async def replace(path):
        paths.append(path)
        return "[rendered]"

    processor = MediaResponseProcessor(replace)

    assert await processor.feed("Before MED") == "Before "
    assert await processor.feed("IA:/tmp/file.png\nAfter") == "[rendered]\nAfter"
    assert paths == ["/tmp/file.png"]


@pytest.mark.asyncio
async def test_processor_preserves_directive_when_provider_declines_it():
    async def replace(_path):
        return None

    processor = MediaResponseProcessor(replace, intercept_stream=False)

    assert await processor.render("MEDIA:/tmp/report.pdf") == "MEDIA:/tmp/report.pdf"
