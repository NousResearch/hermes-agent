# Hermes/jcode upstream sync report

Generated: 2026-05-23T19:54:00+00:00

## Repositories

| Repo | Branch | Commit | Dirty |
| --- | --- | --- | --- |
| hermes | codex/hermes-jcode-bridge | 3048dd9a4c9a0e9d69ecbb0ab2ddf4a37b4ee8ac | True |
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
| jcode_native_registration_patch | True |
| jcode_supertool_registry_smoke | True |
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
| min | 0.035 |
| p50 | 0.036 |
| p95 | 0.076 |
| max | 0.121 |

## jcode Native Hermes Tool

Success: True
Native tool dir: /Users/aayu/Workspace/developer/hermes/bridges/jcode-native-hermes-tool
jcode path: /Users/aayu/Workspace/developer/hermes/.codex-research/jcode

| Check | OK |
| --- | --- |
| native_tool:cargo_toml_exists | True |
| native_tool:lib_rs_exists | True |
| native_tool:uses_jcode_tool_core | True |
| native_tool:implements_tool_trait | True |
| native_tool:defines_hermes_research_tools | True |
| native_tool:defines_hermes_state_tools | True |
| native_tool:exports_default_toolset | True |
| jcode_checkout:exists | True |
| workspace:manifest_exists | True |
| cargo:check | True |

## jcode Native Registration Patch

Success: True
Patch path: /Users/aayu/Workspace/developer/hermes/patches/jcode/register-external-toolset.patch
jcode path: /Users/aayu/Workspace/developer/hermes/.codex-research/jcode

| Check | OK |
| --- | --- |
| jcode_checkout:exists | True |
| patch:exists | True |
| jcode_registry:has_dynamic_register | True |
| patch:adds_register_toolset | True |
| patch:adds_namespace_test | True |
| jcode_tests:expected_anchor_present | True |
| patch:git_apply_check | True |

## jcode Supertool Registry Smoke

Success: True
jcode path: /Users/aayu/Workspace/developer/hermes/.codex-research/jcode
worktree: /var/folders/t1/lv87zx017jl6tsks2dz22mk00000gn/T/jcode-supertool-registry-fjq985c_/jcode

| Check | OK |
| --- | --- |
| jcode_checkout:exists | True |
| patch:exists | True |
| native_tool:exists | True |
| jcode_worktree:prepared | True |
| jcode_patch:applied | True |
| native_tool:copied_into_jcode | True |
| native_tool:exports_toolset | True |
| jcode_test:writes_native_registry_test | True |
| cargo:test_jcode_registry_with_hermes_tools | True |

## Recommendations

- Review dirty worktree entries before pinning hermes.
- Review dirty worktree entries before pinning jcode.
