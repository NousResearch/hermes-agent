## Summary

`hermes proxy` now aborts the downstream client connection when the upstream streaming response breaks mid-body instead of ending the response with a clean EOF.

## Why

The proxy is a credential-attaching forwarder for OpenAI-compatible streaming calls. Before this change, if the upstream provider sent a partial chunked/SSE response and then dropped the connection without the terminating chunk, the proxy caught the `aiohttp.ClientError`, logged `proxy: streaming interrupted`, and still called `write_eof()` on the downstream response.

That converts a broken upstream stream into a clean client-side EOF. For streaming chat/completions or responses, the caller can treat the truncated output as a successful finished stream instead of retrying or surfacing a provider failure.

## Changes

- Track upstream streaming failures while forwarding the response body.
- Re-raise local cancellation instead of swallowing it as a streaming interruption.
- On upstream `aiohttp.ClientError`, release the upstream response/session and close the downstream transport without writing a final chunk.
- Add a regression test with a fake upstream that sends one SSE frame and then closes the transport before the chunked stream terminates.

## Validation

```text
python -m pytest tests/hermes_cli/test_proxy.py -q --timeout-method=thread
40 passed
```
