# Progress

## 2026-04-13

- 已完成 QQ 群底座增强：
  - 群策略支持 `daily_report_target` / `manual_report_target` / `purge_raw_after_rollup`
  - 归档工具支持 `snapshot_report` / `deliver_report`
  - 调度器支持 QQ 群日报自动投递
- 已验证：
  - 定向测试 `82 passed`
  - 扩展回归 `151 passed`
- 当前进行中：
  - 已完成情报员 assignment 模型
  - 已完成统一控制工具与状态机联动
  - 正在整理部署与剩余边界说明

## 2026-04-13 收口修复

- 已补委派/审批体验修复：
  - 子代理提示词明确禁止删除共享目录、repo cache、workspace root
  - QQ 会话在“危险命令审批挂起”期间收到显式追问时，会先拒绝挂起命令，再切到新消息
  - 长耗时提示会展示 `waiting for approval: ...`，不再只显示 `Still working`
- 已补模型侧 QQ 总控收口：
  - `model_tools.get_tool_definitions()` 在 `qq_control` 可用时，会隐藏 `qq_social_control` / `qq_intel_control` / `qq_group_policy` / `qq_group_archive` / `qq_group_moderation`
  - `agent.prompt_builder.build_platform_tool_guidance()` 已明确提示 QQ 场景优先走 `qq_control`，不要自己写脚本
- 已补统一总控对群文件的覆盖：
  - `qq_control` 新增 QQ 群文件动作映射，支持 `list_files` / `upload_file` / `delete_file` / `find_file` / `get_file_url` / `forward_file` 等入口
  - 模型侧在 `qq_control` 可用时，也会隐藏底层 `qq_group_file`
- 已验证：
  - `tests/tools/test_delegate.py tests/gateway/test_busy_input_mode.py tests/tools/test_qq_control_tool.py tests/test_model_tools.py tests/run_agent/test_run_agent.py tests/tools/test_qq_group_file_tool.py`
  - `369 passed`

## 本轮新增

- 新增 [gateway/qq_intel_assignments.py](/home/dtamade/projects/hermes-agent/gateway/qq_intel_assignments.py)
- 新增 [tools/qq_intel_tool.py](/home/dtamade/projects/hermes-agent/tools/qq_intel_tool.py)
- 新增测试：
  - [tests/gateway/test_qq_intel_assignments.py](/home/dtamade/projects/hermes-agent/tests/gateway/test_qq_intel_assignments.py)
  - [tests/tools/test_qq_intel_tool.py](/home/dtamade/projects/hermes-agent/tests/tools/test_qq_intel_tool.py)

## 验证

- 相关链路回归：`159 passed`
- 全量测试：`37 failed, 9361 passed, 87 skipped, 6 errors`
- 结论：本轮 QQ 情报员系统相关链路是绿的；仓库全量目前不处于可直接宣称“全绿”的状态。

## 2026-04-13 追加盘点

- 已修复：
  - `resume_worker` 恢复暂停任务时，已在群内的 worker 现在会回到 `active_collecting`
  - 从群会话招募 worker 时，`manual_report_target` / `notify_target` 默认优先落到当前管理员私聊
  - scheduler 自动发送情报员日报后，会同步回写 `last_report_at`
- 已验证：
  - `tests/gateway/test_qq_intel_assignments.py tests/tools/test_qq_intel_tool.py`：`7 passed`
  - QQ/intel 回归：`162 passed`
- 当前剩余边界：
  - 还没有主动加群/加好友能力
  - 控制面仍拆成 `qq_social_control` + `qq_intel_control` + `qq_group_policy` + `qq_group_archive` + `qq_group_moderation`

## 2026-04-13 社交入口补齐

- 新增 [gateway/qq_social_requests.py](/home/dtamade/projects/hermes-agent/gateway/qq_social_requests.py)
- 新增 [tools/qq_social_tool.py](/home/dtamade/projects/hermes-agent/tools/qq_social_tool.py)
- NapCat adapter 已接入 `request` 事件持久化
- 已支持：
  - 列出待处理好友/群请求
  - 批准/拒绝请求
  - 查询陌生人资料
  - 查看好友列表
- 已验证：
  - `tests/gateway/test_qq_social_requests.py tests/tools/test_qq_social_tool.py`：`7 passed`
  - 扩展 QQ 回归：`170 passed`

## 2026-04-18 qq_intel / 群控口头误判收口

- 已修复：
  - `qq_intel` worker 名提取新增状态问句边界，`员工钢镚还在吗` / `员工钢镚什么状态` 会正确命中 `get_worker`
  - 裸状态短句不再把 `还在吗` / `什么状态` 当成 worker 名；`那个情报员还在吗` 会回落到前台模型
  - bot 可见别名在非显式 `情报员/员工` 语境下不再被当成已知 worker；`让马哥现在汇报` 会回落到前台
  - QQ 群控在“无群目标 + 明显是 bot/员工指令”时会让路，不再用 `现在汇报` 直接抢走普通对话
- 已新增测试：
  - [tests/gateway/test_qq_intel_control_requests.py](/home/dtamade/projects/hermes-agent/tests/gateway/test_qq_intel_control_requests.py)
  - [tests/gateway/test_auto_background_jobs.py](/home/dtamade/projects/hermes-agent/tests/gateway/test_auto_background_jobs.py)
- 已验证：
  - `tests/gateway/test_qq_intel_control_requests.py tests/gateway/test_auto_background_jobs.py tests/gateway/test_group_control_requests.py tests/gateway/test_group_control_intents.py tests/gateway/test_group_runtime_status_requests.py -q -k 'bot_alias or bare_intel_status_phrase_falls_back_to_agent or admin_dm_can_orally_query_intel_worker_status or group_control or intel_control_request'`：`29 passed`
  - `tests/gateway/test_qq_intel_control_requests.py tests/gateway/test_qq_intel_runtime_service.py tests/gateway/test_group_control_requests.py tests/gateway/test_group_control_runtime_service.py tests/gateway/test_group_control_intents.py tests/gateway/test_group_runtime_status_requests.py tests/gateway/test_group_runtime_status_runtime_service.py tests/gateway/test_auto_background_jobs.py -q -k 'not weixin and (intel or group_control or runtime_status or bare_intel_status_phrase_falls_back_to_agent or bot_alias or admin_dm_can_orally_query_intel_worker_status)'`：`60 passed`
  - 说明：`tests/gateway/test_auto_background_jobs.py` 全文件仍存在 1 条与本轮改动面无关的既有 Weixin 失败：`test_admin_weixin_group_can_orally_enable_collect_only`

## 2026-04-18 第二轮口头控制误判收口

- 已修复：
  - `列一下群里的部署问题` 不再被误判成 `list_joined_groups`
  - 隐式已知 worker 口令现在只接受“短控制句”；长任务句如 `让钢镚现在汇报一下这个页面为什么回退了`、`让钢镚继续监听线上部署日志...` 会回落前台
  - QQ 群控在“无群目标且无明确群指代”时会直接让路，不再因为 `监听/汇报` 抢普通对话
  - 显式员工路由在存在同名 intel worker 时仍可正常派活，不会被 `resume_worker` 偷走
- 已新增测试：
  - [tests/gateway/test_qq_intents.py](/home/dtamade/projects/hermes-agent/tests/gateway/test_qq_intents.py)
  - [tests/gateway/test_qq_intel_control_requests.py](/home/dtamade/projects/hermes-agent/tests/gateway/test_qq_intel_control_requests.py)
  - [tests/gateway/test_auto_background_jobs.py](/home/dtamade/projects/hermes-agent/tests/gateway/test_auto_background_jobs.py)
- 已验证：
  - `tests/gateway/test_qq_intents.py tests/gateway/test_qq_intel_control_requests.py tests/gateway/test_auto_background_jobs.py -q -k 'joined_group_list_query_requires_actual_group_list_intent or verbose or steal_explicit_employee_route_when_message_is_verbose_task or admin_dm_can_orally_route_named_worker_to_background_job or admin_dm_bot_alias_intel_phrase_falls_back_to_agent or bare_intel_status_phrase_falls_back_to_agent'`：`7 passed`
  - `tests/gateway/test_qq_intents.py tests/gateway/test_qq_intel_control_requests.py tests/gateway/test_qq_intel_runtime_service.py tests/gateway/test_group_control_requests.py tests/gateway/test_group_control_runtime_service.py tests/gateway/test_group_control_intents.py tests/gateway/test_group_runtime_status_requests.py tests/gateway/test_group_runtime_status_runtime_service.py tests/gateway/test_auto_background_jobs.py -q -k 'not weixin and (qq_intents or intel or group_control or runtime_status or joined_group_list or verbose or bot_alias or bare_intel_status_phrase_falls_back_to_agent or admin_dm_can_orally_query_intel_worker_status or admin_dm_can_orally_list_joined_groups)'`：`69 passed`

## 2026-04-18 第三轮 runtime/background shortcut 让路收口

- 已修复：
  - 显式情报员状态查询不再被 `runtime_status` / `background_status` shortcut 抢走
  - 会话里即使存在后台任务，`看看情报员钢镚现在什么状态。` 仍会正确走 `qq_control -> get_worker`
- 已新增测试：
  - [tests/gateway/test_qq_intents.py](/home/dtamade/projects/hermes-agent/tests/gateway/test_qq_intents.py)
  - [tests/gateway/test_auto_background_jobs.py](/home/dtamade/projects/hermes-agent/tests/gateway/test_auto_background_jobs.py)
- 已验证：
  - `tests/gateway/test_qq_intents.py tests/gateway/test_auto_background_jobs.py -q -k 'explicit_intel_status_query or background_status_shortcut_does_not_steal_explicit_intel_status_query'`：`3 passed`
  - `tests/gateway/test_qq_intents.py tests/gateway/test_runtime_shortcuts_service.py tests/gateway/test_qq_intel_control_requests.py tests/gateway/test_qq_intel_runtime_service.py tests/gateway/test_auto_background_jobs.py tests/gateway/test_group_control_requests.py tests/gateway/test_group_control_runtime_service.py tests/gateway/test_group_control_intents.py tests/gateway/test_group_runtime_status_requests.py tests/gateway/test_group_runtime_status_runtime_service.py -q -k 'not weixin and (explicit_intel_status_query or intel or runtime_status or background_status or joined_group_list or verbose or bot_alias or bare_intel_status_phrase_falls_back_to_agent or admin_dm_can_orally_query_intel_worker_status or admin_dm_can_orally_list_joined_groups)'`：`54 passed`

## 2026-04-19 第四轮 group runtime shortcut 让路收口

- 已修复：
  - `这个群现在什么状态` 这类“有群指代 + 泛化状态问句”现在会识别为群运行态查询
  - 会话里即使有后台任务，明确群运行态查询也不会再被 runtime/background shortcut 抢走
- 已新增测试：
  - [tests/gateway/test_group_control_intents.py](/home/dtamade/projects/hermes-agent/tests/gateway/test_group_control_intents.py)
  - [tests/gateway/test_qq_intents.py](/home/dtamade/projects/hermes-agent/tests/gateway/test_qq_intents.py)
  - [tests/gateway/test_auto_background_jobs.py](/home/dtamade/projects/hermes-agent/tests/gateway/test_auto_background_jobs.py)
- 已验证：
  - `tests/gateway/test_group_control_intents.py tests/gateway/test_qq_intents.py tests/gateway/test_auto_background_jobs.py -q -k '这个群现在什么状态 or explicit_group_runtime_query_as_session_status or background_status_shortcut_does_not_steal_explicit_group_runtime_status_query or shared_status_asks'`：`3 passed`
  - `tests/gateway/test_group_control_intents.py tests/gateway/test_group_runtime_status_requests.py tests/gateway/test_group_runtime_status_runtime_service.py tests/gateway/test_qq_intents.py tests/gateway/test_runtime_shortcuts_service.py tests/gateway/test_qq_intel_control_requests.py tests/gateway/test_qq_intel_runtime_service.py tests/gateway/test_auto_background_jobs.py -q -k 'not weixin and (group_runtime or runtime_status or background_status or explicit_group_runtime_query or explicit_intel_status_query or intel or joined_group_list or verbose or bot_alias)'`：`55 passed`
