# Weixin/Hermes 上下文连续性与短问误召回修复状态

更新时间：2026-05-01

## 结论

本轮已完成两个代码级防线，并完成聚焦验收：

1. `MEDIA:` 占位路径/不存在本地文件不再进入实际发送链路。
2. “上次任务 / 最近怎么样 / 继续 / 现在呢”等短问在 `session_search` 工具层走最近会话列表，不再被 FTS 相关性排序拉回古老高密度任务。
3. 压缩/上下文断裂现场显示：磁盘代码和配置倾向于 main runtime，但运行中的 gateway 老进程日志曾出现 `Auxiliary compression: using auto` 与后续连接失败。由于本轮边界是不重启 gateway，暂不让运行中进程加载新代码，只记录为待重启后验证项。

## 已修复文件

- `gateway/platforms/base.py`
  - 新增 `_is_deliverable_local_media(path)`。
  - 发送前最终校验本地媒体必须存在且是文件。
  - 跳过空路径、示例占位路径、`/path/...`、`/absolute/path...`、不存在路径。

- `gateway/platforms/weixin.py`
  - Weixin 媒体发送循环接入同一 guard。
  - 防止不存在的 `MEDIA:/tmp/...` 或文档示例路径进入 `_deliver_media`。

- `tests/gateway/test_media_extraction.py`
  - 保留原测试，并追加占位路径/不存在路径/真实临时文件可发送的回归测试。

- `tools/session_search_tool.py`
  - 新增 `_RECENT_INTENT_PATTERNS` 与 `_is_recent_intent_query()`。
  - 对短 follow-up query 使用 `_list_recent_sessions()`，绕开 FTS 相关性排序。
  - 长 query 仍保留正常 FTS 搜索，避免影响具体主题检索。

- `tests/tools/test_session_search.py`
  - 新增短问 recent 优先回归测试。

- skill `wechat-short-followup-interpretation`
  - 补充短问时间优先、旧任务降权规则。

## 已验证

- 媒体 + 短问聚焦测试：通过。
- 完整 `tests/tools/test_session_search.py` + `tests/gateway/test_media_extraction.py`：通过，43 passed。
- 压缩相关测试：通过，90 passed。

## 已确认现场事实

- gateway 进程没有在本轮重启，运行中进程可能仍是 4/29 启动时加载的旧代码/旧配置。
- 磁盘配置中 compression/session_search 指向 main；磁盘代码也会把 `provider=main` alias 映射到 auto→live main runtime。
- 日志曾出现 `Auxiliary compression: using auto` 后连接失败，导致 summary 失败/上下文压缩降级。这是上下文像断裂的重要线索。
- 当前边界：不重启 gateway、不微信实发测试、不删除资产、不动密钥。

## 后续验收条件

如要让运行中 Weixin gateway 真正加载本轮修复，需要在用户确认后执行受控重启，并验收：

1. 启动日志显示加载新代码。
2. 再触发短问，确认 `session_search` 返回 recent 模式。
3. 再触发含不存在 `MEDIA:` 的测试消息，确认只 warning skip，不调用实际发送。
4. 观察压缩日志：compression 应使用 live main runtime 或明确可用辅助后端，不再无声降级导致 summary 丢失。

## 78 个问题任务切换说明

当前任务完成到“不重启边界内可完成”的代码、测试、文档闭环。下一步按用户要求切换到“78 个问题修复”任务：先定位任务清单/上下文，再按高风险报警、失败测试、用户体验阻断优先修复。
