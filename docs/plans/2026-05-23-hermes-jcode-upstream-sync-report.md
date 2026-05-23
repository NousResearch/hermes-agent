# Hermes/jcode upstream sync report

Generated: 2026-05-23T07:02:31+00:00

## Repositories

| Repo | Branch | Commit | Dirty |
| --- | --- | --- | --- |
| hermes | main | 729a778af0b3f984b4934361cad3050f6afb79ba | True |
| jcode | master | 7951a2ddb91bad10155b911ccd0971de5baeafc8 | True |

## Graphify

| Repo | Files | Nodes | Edges | Communities | Report |
| --- | ---: | ---: | ---: | ---: | --- |
| hermes | 2212 | 67683 | 223428 | 452 | /Users/aayu/Workspace/developer/hermes/graphify-out/GRAPH_REPORT.md |
| jcode | 872 | 18201 | 69026 | 81 | /Users/aayu/Workspace/developer/hermes/.codex-research/jcode/graphify-out/GRAPH_REPORT.md |

## Bridge Contract

Success: True
Version: jcode-bridge.v1
Schema dir: /Users/aayu/Workspace/developer/hermes/contracts/jcode_bridge/v1
Fixture dir: /Users/aayu/Workspace/developer/hermes/tests/fixtures/jcode_bridge

| Check | OK |
| --- | --- |
| fixture:run_json_success | True |
| fixture:run_ndjson_success | True |
| fixture:debug_response_success | True |
| fixture:debug_response_error | True |
| generated:debug_command_request | True |
| schema:debug_command.schema.json | True |
| schema:debug_response.schema.json | True |
| schema:run_json.schema.json | True |
| schema:run_ndjson_event.schema.json | True |
| schema:run_ndjson_stream.schema.json | True |
| schema:upstream_sync_report.schema.json | True |

## Bridge Smoke

Success: True

| Check | OK |
| --- | --- |
| contract_tool | True |
| safety_blocks_outbound_human_contact | True |
| contract_rejects_bad_json | True |
| ensure_server_path | True |
| hermes_service_contract | True |
| hermes_service_dispatch | True |
| hermes_service_blocks_send_message | True |
| jcode_tool_hermes_client | True |
| hermes_mcp_server | True |
| hermes_mcp_contract | True |
| mother_repo_scaffold | True |
| webhook_preflight_pass | True |
| webhook_preflight_blocks | True |

## Hermes Service Contract

Success: True
Version: hermes-service.v1
Schema dir: /Users/aayu/Workspace/developer/hermes/contracts/hermes_service/v1
Fixture dir: /Users/aayu/Workspace/developer/hermes/tests/fixtures/hermes_service

| Check | OK |
| --- | --- |
| fixture:service_request_web_search | True |
| fixture:service_response_success | True |
| fixture:service_response_error | True |
| schema:service_request.schema.json | True |
| schema:service_response.schema.json | True |

## Hermes MCP Contract

Success: True
Version: hermes-mcp.v1
Schema dir: /Users/aayu/Workspace/developer/hermes/contracts/hermes_mcp/v1
Fixture dir: /Users/aayu/Workspace/developer/hermes/tests/fixtures/hermes_mcp

| Check | OK |
| --- | --- |
| fixture:initialize_response | True |
| fixture:tools_list_response | True |
| fixture:tools_call_response_success | True |
| schema:initialize_response.schema.json | True |
| schema:tools_list_response.schema.json | True |
| schema:tools_call_response.schema.json | True |
| live:mock_mcp_roundtrip | True |

## Bridge Latency

Success: True
Probe: hermes_mcp_persistent_mock
Iterations: 30

| Metric | ms |
| --- | ---: |
| min | 0.02 |
| p50 | 0.022 |
| p95 | 0.057 |
| max | 0.067 |

## Recommendations

- Review dirty worktree entries before pinning hermes.
- Review dirty worktree entries before pinning jcode.
