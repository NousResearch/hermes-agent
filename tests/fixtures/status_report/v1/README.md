# Gateway status report, contract v1

What a gateway sends when its launch environment carries `GATEWAY_STATUS_REPORT_URL`
(`gateway/status_report/`). The tests in `tests/gateway/status_report/` check the real request
against these files, so a receiver can copy this folder as its test vectors.

## Request

`POST <the URL without its credential>` with `Content-Type: application/json`,
`Authorization: Bearer <credential>` and `User-Agent: hermes-gateway-status-report/1`. The
credential is the value of the `t` query parameter of the stamped URL; it is moved into the header
and never sent in the URL. Every report opens its own connection, and a redirect is not followed.

| Field | Type | Meaning |
| --- | --- | --- |
| `v` | `1` | Contract version. |
| `kind` | `"beat"` or `"goodbye"` | `beat` at the report interval (120 s unless the receiver says otherwise, 10% jitter); `goodbye` once when the gateway suspends its own machine, started just before it asks for the suspend and not waited for. |
| `seq` | integer | Starts at 1 and grows with every report of this process. It can skip numbers: a report that was built and then not sent keeps its number. A quick retry re-sends the same body with the same `seq`. Ignore a report whose `seq` is not above the last one stored for the same `boot_id`. |
| `boot_id` | 32 hex characters | Random per gateway process start. |
| `status.gateway_running` | boolean | Always `true`: the gateway process itself sends the report. |
| `status.gateway_state` | string or null | Lifecycle state (`starting`, `running`, `draining`, ...). |
| `status.gateway_exit_reason` | string or null | A bounded code, never free text. |
| `status.gateway_platforms` | object | Platform key to `{"state", "error_code"}`; `error_code` only when one is set. Keys are the ones the public `/api/status` shows: `<platform>`, or `<profile>:<platform>` for a platform served for another profile. |
| `status.nous_session_valid` | `"valid"`, `"terminal"` or `"unknown"` | Optional. |

`beat.json` carries every field, `beat-minimal.json` the smallest body, `goodbye.json` a goodbye.
Nothing else is sent.

## Response

`responses.json` lists each case and what the sender does next (`then`):

| `then` | Sender behaviour |
| --- | --- |
| `next_interval` | The next report goes out at the report interval in force (`interval_s` until a stored reply sets another). |
| `told_interval` | A stored reply set the report interval: the next report, and every later one, goes out after `wait_s` seconds until another stored reply says otherwise. |
| `wait` | The next report goes out after `wait_s` seconds (`Retry-After`, at most `retry_after_max_s`). |
| `backoff` | The credential was rejected: wait `rejected_backoff_s[n]` after the n-th rejection in a row; a stored report resets it. |
| `quick_retry` | One retry after `quick_retry_s`, then the normal interval. A network error or timeout behaves the same. |

The reply body is never read. A stored reply (`2xx`) may carry `Report-Interval: <seconds>`, a positive
whole number: from then on the sender reports at that interval, kept within `interval_min_s` and
`interval_max_s`, with the same jitter. A stored reply without a valid one sets the interval back to
`interval_s`; the header on any other reply is ignored. The interval is held in memory only: every
process starts at `interval_s`, and its first report goes out at once.

A goodbye is sent once, with a short timeout and no retry, and is skipped while a `wait` or a
`backoff` is running. No beat follows it for `goodbye_quiet_s` seconds; a beat after that means the
gateway is still, or again, running. A `wait` or `backoff` asked for in the goodbye's reply holds the
next beat as it would after a beat.
