# Hermes Service v1 Contract

This contract is the reverse bridge: a local jcode client can ask Hermes to run
selected Hermes-owned services without importing Hermes internals.

Contract version: `hermes-service.v1`

Files:

- `service_request.schema.json`: newline-JSON request from jcode to Hermes.
- `service_response.schema.json`: newline-JSON response from Hermes to jcode.

The first intended service surface is small and allowlisted:

- `web_search`
- `web_extract`
- `session_search`
- `memory`

High-side-effect tools such as `send_message` must be explicitly enabled by the
service operator and carry confirmation fields in the request. Hermes remains
the approval/policy boundary; jcode remains the fast local caller.
