# Hit Network Fork Architecture

This document summarizes the Hit Network specific changes in this fork versus upstream NousResearch/hermes-agent. Source: hardened upstream diff captured on 2026-05-25.

## Upstream delta (log)

## Upstream delta (diff --stat)
=== git diff --stat upstream/main..HEAD (top 200 lines) ===
 .env.example                                       |     1 -
 .github/actions/hermes-smoke-test/action.yml       |     7 +-
 .github/workflows/contributor-check.yml            |     2 +-
 .github/workflows/deploy-site.yml                  |     4 +-
 .github/workflows/docker-lint.yml                  |    68 -
 .github/workflows/docker-publish.yml               |   275 +-
 .github/workflows/docs-site-checks.yml             |     4 +-
 .github/workflows/history-check.yml                |     2 +-
 .github/workflows/lint.yml                         |     8 +-
 .github/workflows/nix-lockfile-fix.yml             |     4 +-
 .github/workflows/nix.yml                          |     2 +-
 .github/workflows/osv-scanner.yml                  |     2 +-
 .github/workflows/skills-index.yml                 |     8 +-
 .github/workflows/supply-chain-audit.yml           |    20 +-
 .github/workflows/tests.yml                        |   116 +-
 .github/workflows/upload_to_pypi.yml               |     9 +-
 .github/workflows/uv-lockfile-check.yml            |     2 +-
 .gitignore                                         |     2 -
 .hadolint.yaml                                     |    36 -
 AGENTS.md                                          |    58 +-
 CONTRIBUTING.md                                    |     6 +-
 Dockerfile                                         |   143 +-
 README.md                                          |    24 +-
 README.zh-CN.md                                    |    21 -
 RELEASE_v0.14.0.md                                 |    66 +-
 acp_adapter/auth.py                                |    15 +-
 .../docker => acp_adapter/bootstrap}/__init__.py   |     0
 acp_adapter/bootstrap/bootstrap_browser_tools.ps1  |   288 +
 acp_adapter/bootstrap/bootstrap_browser_tools.sh   |   399 +
 acp_adapter/edit_approval.py                       |   286 -
 acp_adapter/entry.py                               |    61 +-
 acp_adapter/events.py                              |    16 +-
 acp_adapter/permissions.py                         |    24 +-
 acp_adapter/server.py                              |   341 +-
 acp_adapter/tools.py                               |   229 +-
 acp_registry/agent.json                            |     4 +-
 agent/agent_init.py                                |  1637 ---
 agent/agent_runtime_helpers.py                     |  2270 ----
 agent/anthropic_adapter.py                         |   675 +-
 agent/auxiliary_client.py                          |   797 +-
 agent/azure_identity_adapter.py                    |   555 -
 agent/background_review.py                         |   593 -
 agent/bedrock_adapter.py                           |    13 -
 agent/browser_provider.py                          |   175 -
 agent/browser_registry.py                          |   223 -
 agent/chat_completion_helpers.py                   |  2311 ----
 agent/codex_responses_adapter.py                   |    39 +-
 agent/codex_runtime.py                             |   454 -
 agent/context_compressor.py                        |   178 +-
 agent/context_engine.py                            |     1 -
 agent/conversation_compression.py                  |   603 -
 agent/conversation_loop.py                         |  4300 ------
 agent/copilot_acp_client.py                        |     5 +-
 agent/credential_persistence.py                    |   174 -
 agent/credential_pool.py                           |   356 +-
 agent/credential_sources.py                        |     2 +-
 agent/curator_backup.py                            |     5 +-
 agent/display.py                                   |    62 +-
 agent/error_classifier.py                          |   111 -
 agent/file_safety.py                               |   360 +-
 agent/google_oauth.py                              |     8 +-
 agent/image_gen_provider.py                        |    82 -
 agent/image_routing.py                             |    96 +-
 agent/insights.py                                  |   336 +-
 agent/iteration_budget.py                          |    62 -
 agent/lsp/client.py                                |     2 +-
 agent/lsp/install.py                               |     2 +-
 agent/lsp/manager.py                               |     2 +-
 agent/lsp/reporter.py                              |     2 +-
 agent/lsp/servers.py                               |     2 +-
 agent/memory_manager.py                            |    64 +-
 agent/message_sanitization.py                      |   444 -
 agent/model_metadata.py                            |     4 +-
 agent/models_dev.py                                |     3 -
 agent/moonshot_schema.py                           |    31 -
 agent/process_bootstrap.py                         |   167 -
 agent/prompt_builder.py                            |    13 +-
 agent/redact.py                                    |   180 +-
 agent/secret_sources/__init__.py                   |    13 -
 agent/secret_sources/bitwarden.py                  |   661 -
 agent/shell_hooks.py                               |    25 +-
 agent/skill_bundles.py                             |   410 -
 agent/skill_commands.py                            |    28 +-
 agent/skill_preprocessing.py                       |     8 -
 agent/skill_utils.py                               |    61 +-
 agent/stream_diag.py                               |   280 -
 agent/system_prompt.py                             |   380 -
 agent/tool_dispatch_helpers.py                     |   350 -
 agent/tool_executor.py                             |   912 --
 agent/tool_guardrails.py                           |    25 +-
 agent/transcription_provider.py                    |   193 -
 agent/transcription_registry.py                    |   122 -
 agent/transports/anthropic.py                      |    12 +-
 agent/transports/chat_completions.py               |    44 +-
 agent/transports/codex.py                          |    28 +-
 agent/transports/codex_app_server.py               |    33 +-
 agent/transports/codex_app_server_session.py       |    55 +-
 agent/tts_provider.py                              |   274 -
 agent/tts_registry.py                              |   133 -
 batch_runner.py                                    |    23 +-
 cli-config.yaml.example                            |    11 +-
 cli.py                                             |  1413 +-
 cron/jobs.py                                       |   240 +-
 cron/scheduler.py                                  |   269 +-
 docker-compose.yml                                 |    15 +-
 docker/cont-init.d/015-supervise-perms             |    90 -
 docker/cont-init.d/02-reconcile-profiles           |    46 -
 docker/entrypoint.sh                               |   180 +-
 docker/main-wrapper.sh                             |    30 -
 docker/s6-rc.d/dashboard/dependencies.d/base       |     0
 docker/s6-rc.d/dashboard/finish                    |    30 -
 docker/s6-rc.d/dashboard/run                       |    40 -
 docker/s6-rc.d/dashboard/type                      |     1 -
 docker/s6-rc.d/main-hermes/dependencies.d/base     |     0
 docker/s6-rc.d/main-hermes/run                     |    27 -
 docker/s6-rc.d/main-hermes/type                    |     1 -
 docker/s6-rc.d/user/contents.d/dashboard           |     0
 docker/s6-rc.d/user/contents.d/main-hermes         |     0
 docker/stage2-hook.sh                              |   142 -
 ...6-05-07-s6-overlay-dynamic-subagent-gateways.md |   434 -
 .../2026-05-15-acp-zed-edit-approval-diffs.md      |   152 -
 gateway/config.py                                  |   275 +-
 gateway/memory_monitor.py                          |   230 -
 gateway/mirror.py                                  |    10 +
 gateway/pairing.py                                 |   207 +-
 gateway/platforms/api_server.py                    |   135 +-
 gateway/platforms/base.py                          |   581 +-
 gateway/platforms/bluebubbles.py                   |    22 +-
 gateway/platforms/dingtalk.py                      |    32 +-
 .../adapter.py => gateway/platforms/discord.py     |   655 +-
 gateway/platforms/feishu.py                        |    82 +-
 gateway/platforms/helpers.py                       |     4 +-
 gateway/platforms/matrix.py                        |   146 +-
 .../adapter.py => gateway/platforms/mattermost.py  |   350 +-
 gateway/platforms/msgraph_webhook.py               |     8 +-
 gateway/platforms/qqbot/adapter.py                 |   147 +-
 gateway/platforms/signal.py                        |    41 +-
 gateway/platforms/slack.py                         |     2 +-
 gateway/platforms/sms.py                           |     2 -
 gateway/platforms/telegram.py                      |  1426 +-
 gateway/platforms/telegram_network.py              |    10 -
 gateway/platforms/webhook.py                       |   150 +-
 gateway/platforms/wecom.py                         |    14 +-
 gateway/platforms/wecom_callback.py                |    38 +-
 gateway/platforms/weixin.py                        |     2 -
 gateway/platforms/yuanbao.py                       |    34 +-
 gateway/run.py                                     |  1998 +--
 gateway/session.py                                 |   104 +-
 gateway/session_context.py                         |    23 -
 gateway/sticker_cache.py                           |    21 +-
 gateway/stream_consumer.py                         |    50 +-
 hermes_cli/_parser.py                              |     9 +-
 hermes_cli/auth.py                                 |  1750 +--
 hermes_cli/auth_commands.py                        |    53 +-
 hermes_cli/azure_detect.py                         |   146 +-
 hermes_cli/backup.py                               |    34 +-
 hermes_cli/browser_connect.py                      |   149 +-
 hermes_cli/bundles.py                              |   229 -
 hermes_cli/callbacks.py                            |     4 +-
 hermes_cli/cli_output.py                           |     5 +-
 hermes_cli/codex_runtime_switch.py                 |     4 +-
 hermes_cli/commands.py                             |   119 +-
 hermes_cli/config.py                               |   317 +-
 hermes_cli/container_boot.py                       |   325 -
 hermes_cli/cron.py                                 |     9 -
 hermes_cli/curses_ui.py                            |     2 +-
 hermes_cli/debug.py                                |    10 +-
 hermes_cli/dep_ensure.py                           |    87 +-
 hermes_cli/doctor.py                               |   575 +-
 hermes_cli/dump.py                                 |     3 -
 hermes_cli/env_loader.py                           |   135 +-
 hermes_cli/fallback_cmd.py                         |    19 +-
 hermes_cli/fallback_config.py                      |    72 -
 hermes_cli/gateway.py                              |   342 +-
 hermes_cli/gateway_windows.py                      |   426 +-
 hermes_cli/goals.py                                |     6 -
 hermes_cli/kanban.py                               |   574 +-
 hermes_cli/kanban_db.py                            |  2133 +--
 hermes_cli/kanban_decompose.py                     |   477 -
 hermes_cli/kanban_diagnostics.py                   |   308 +-
 hermes_cli/kanban_specify.py                       |     9 +-
 hermes_cli/kanban_swarm.py                         |   279 -
 hermes_cli/main.py                                 |  2099 +--
 hermes_cli/memory_setup.py                         |     9 +-
 hermes_cli/migrate.py                              |   115 -
 hermes_cli/model_switch.py                         |    28 +-
 hermes_cli/models.py                               |    19 +-
 hermes_cli/oneshot.py                              |    12 +-
 hermes_cli/plugins.py                              |   246 -
 hermes_cli/plugins_cmd.py                          |   132 +-
 hermes_cli/portal_cli.py                           |   219 -

## Notable patches

- agent/prompt_builder.py: add per-call and per-task model override (commit e570edb73).
- gateway/platforms/telegram.py: SIE routing callbacks and v1.3 handler for approve/veto/override/promote/decline.
- cron/jobs.py and cron/scheduler.py: SIE Phase 5 Part B per-job exception isolation hardened.
- agent/insights.py: consolidated skill-usage telemetry (SIE AC#8).

Each item above is discoverable in the diff, and file-level changes are attributable to Hit Network integration work.
