# 故障排查

先运行 `browser-cli doctor`，依据失败检查的具体信息定位。`ok: true` 不等于服务就绪，还需检查 `data.ready_for_browser_actions`。

| 现象 | 处理方式 |
|---|---|
| `configuration_error` | 按[授权说明](authentication.md)完成登录，或确认受管环境提供了两个必需的凭据变量。 |
| `authentication_error` | 凭据被拒绝；按授权流程处理，必要时退出本地授权后重新登录。 |
| `conflict` | 读写 Context 已被占用。等待当前会话结束、换用专用 Context，或在任务允许时使用只读模式。确认原会话已结束后才考虑强制释放。 |
| `timeout` | 检查会话状态、网络和页面是否满足等待条件。有依据时调整超时再试；不要不断扩大超时或盲目重复有副作用的动作。 |
| `cdp_error` | 先检查 JS 语法、选择器和页面状态；再确认会话处于 active，检查 `session targets` 并获取 snapshot。 |
| `Uncaught` | 此版本错误信息可能缺少具体 JS 异常。检查是否存在顶层 `return`、空元素或未选中项；拆成只读表达式定位，不直接归因于网络或浏览器不稳定。 |
| 下拉框命令成功但未选中 | 读回 `value`、`selectedIndex` 和标签，区分选项显示文字与实际 value。参见[表单示例](commands.md)。 |
| 截图参数不识别 | 完整页面参数是 `--full-page`，不是 `--full`。 |

技能位置不明确时，使用当前宿主提供的位置：Codex 技能元信息中的绝对来源路径、Claude Code 的 `${CLAUDE_SKILL_DIR}`，或 WorkBuddy / CodeBuddy 的 `${CODEBUDDY_SKILL_DIR}`。OpenClaw 使用已加载技能的来源路径或 `{baseDir}`；Hermes 使用 `skill_view` 返回的 `skill_dir`。不要从工作目录猜测，也不要扫描用户主目录寻找替代包。

安装后出现 command not found 时，macOS arm64 / Linux x86_64 使用 `"<skill-root>/bin/browser-cli"`；Windows PowerShell 使用 `& "<skill-root>\bin\browser-cli.exe"`。无需改 PATH 或重启。

同一失败在有依据的修正后仍重复出现时停止重试，报告失败步骤和诊断结果。不要重新执行结果未知的提交等有副作用动作。

放弃失败任务时关闭本次新建的临时会话；用户正在接管操作时保留会话并等待。回复说明失败步骤和已完成内容，不把失败包装成成功。
