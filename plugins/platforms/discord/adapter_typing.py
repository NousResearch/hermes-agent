"""Guard Discord typing attempts after SDK rate-limit and connector waits."""


class TypingPaused(Exception):
    """The transport has not sent this typing attempt; its approval surface is paused."""


def typing_http_trace(adapter):
    import aiohttp

    async def before_send(_session, _context, params):
        parts = params.url.path.rsplit('/', 3)
        if (params.method == 'POST' and len(parts) == 4 and parts[-3] == 'channels'
                and parts[-1] == 'typing' and parts[-2] in adapter._typing_paused):
            # Do not cancel the SDK's 429 sleep: a global 429 clears its shared gate
            # without a cancellation finally. Let it reopen the gate, then drop this attempt.
            raise TypingPaused()

    trace = aiohttp.TraceConfig()
    trace.on_request_start.append(before_send)
    trace.on_request_headers_sent.append(before_send)
    return trace
