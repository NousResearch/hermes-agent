# Progress

## 2026-04-10
- Resumed the QQ NapCat group-gating task from the previous interrupted session.
- Loaded process skills and created persistent planning files.
- Next step: inspect current file state before patching implementation.
- Reproduced QQ-targeted red tests on the remote `venv` environment: 4 failures across QQ group mention gating and QQ config/env bridging.
- Confirmed root causes:
  - `gateway/platforms/qq_napcat.py` only supports direct mention/pattern gating and text cleanup partially; it does not track bot replies or follow-up windows.
  - `gateway/config.py` does not yet bridge `require_mention` / `mention_patterns` from `qq_napcat:` YAML, and QQ mention patterns need normalization after YAML/env parsing.
- Patched `gateway/config.py` to bridge QQ `require_mention` and `mention_patterns` from YAML and env, and normalized doubled regex escapes for mention patterns.
- Patched `gateway/platforms/qq_napcat.py` to track recent bot replies in groups, allow same-user follow-up windows, allow replies to recent bot messages, and attach `reply_to_message_id` to normalized events.
- Synced the changed files to `root@888933.xyz`, ran remote targeted tests, and got `40 passed in 1.38s`.
- Updated `/root/.hermes/config.yaml`, saved backup `/root/.hermes/config.yaml.bak-20260409-183221`, restarted `hermes-gateway`, and confirmed `systemctl is-active` returns `active`.
- 用户随后反馈 QQ 私聊报错：`NameError: name 'context' is not defined`。
- 先新增了回归测试，远端红测结果为 `_run_agent()` 不接受 `admin_user_ids` 参数。
- 修复 `gateway/run.py` 后，远端运行：
  - `python -m pytest -n 0 tests/gateway/test_reasoning_command.py::TestReasoningCommand::test_run_agent_reloads_reasoning_config_per_message tests/gateway/test_reasoning_command.py::TestReasoningCommand::test_run_agent_passes_admin_policy_to_gateway_approval tests/gateway/test_approve_deny_commands.py -q`
  - 结果：`28 passed, 1 warning`
- 之后重启了 `hermes-gateway`，重启后的最近日志里没有再出现新的 `context` NameError。
- 用户进一步要求把马噶设成“东北暴躁老哥”人格，并明确说明 `马哥 / 老马 / 马屌 / 马逼 / 小马 / 马户` 都是马噶的别名。
- 为此：
  - 在 `gateway/session.py` 增加会话提示，避免模型机械复述管理员身份块；
  - 在远端 `/root/.hermes/config.yaml` 更新 `qq_napcat.system_prompt` 和 `mention_patterns`；
  - 重启 `hermes-gateway` 并确认配置已被 `load_gateway_config()` / `QqNapCatAdapter` 读入。

## 2026-04-11
- 承接 QQ 可靠性与运行时硬化方案，先补跑远端残留校验并确认新代码已经部署到 `root@888933.xyz`。
- 本地跑了一组覆盖 QQ / 审批 / fallback / 共享群上下文 / 配置 / 识图的回归：
  - `python -m pytest tests/gateway/test_qq_napcat.py tests/gateway/test_busy_input_mode.py tests/gateway/test_model_identity_query.py tests/gateway/test_shared_group_history.py tests/gateway/test_no_reply_marker.py tests/gateway/test_approve_deny_commands.py tests/tools/test_approval.py tests/tools/test_vision_tools.py tests/run_agent/test_fallback_model.py tests/run_agent/test_primary_runtime_restore.py tests/run_agent/test_run_agent.py tests/hermes_cli/test_config.py tests/hermes_cli/test_config_validation.py tests/test_hermes_state.py tests/cron/test_scheduler.py -q`
  - 初次结果：`3 failed, 797 passed`
- 失败点全部来自审批模块测试断言仍然写死英文 `"administrator"` / `"timed out"`，而实现已切成中文文案；只修正测试，不改运行逻辑。
- 复跑同一组回归，结果：`800 passed, 20 warnings in 24.31s`。
- 对三个候选主端点做了真实运行负载排查：
  - `https://pay.kxaug.xyz/v1`：直连 `chat/completions` 和 `responses` 都可返回 `200`，但正文为空；用 Hermes `AIAgent` 实测得到 `'(empty)'`。
  - `https://api.888933.xyz/v1`：裸 `chat/completions` 可返回 `pong`，但 Hermes 全量系统提示下触发 `HTTP 403: Your request was blocked`。
  - `https://wududu.edu.kg/v1` + `glm-5.1`：Hermes `AIAgent` 最小问答实测可正常返回 `pong`。
- 因此把本地与线上 `~/.hermes/config.yaml` 收敛为：
  - 主模型：`glm-5.1 @ https://wududu.edu.kg/v1`
  - 主上下文：`205000`
  - fallback 链：`api.888933 gpt-5.4` → `pay.kxaug gpt-5.4`
  - 文本类辅助模型（compression / approval / flush_memories）切到 `glm-5.1 @ wududu`
  - 识图仍保留 `gpt-5.4 @ pay.kxaug`，因为此前实际跑通过
  - 本地 QQ 唤醒词补齐 `马嘎`
- 远端操作：
  - 备份 `/root/.hermes/config.yaml`
  - 同步本地配置到远端
  - 通过 `python -m hermes_cli.main gateway run --replace` 重启 gateway
  - 重启后远端 `gateway_state.json` 显示 `qq_napcat: connected`
  - 用远端实际运行时再测一次 Hermes 最小问答，返回 `pong`
