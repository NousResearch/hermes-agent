"""CLI 会话启动时随机显示的小贴士，帮助用户发现功能。"""

import random


# ---------------------------------------------------------------------------
# 贴士库 — 涵盖斜杠命令、CLI 标志、配置、
# 快捷键、工具、网关、技能、配置文件和工作流技巧的单行贴士。
# ---------------------------------------------------------------------------

TIPS = [
    # --- 斜杠命令 ---
    "/background <prompt>（别名 /bg 或 /btw）在单独会话中运行任务，当前会话保持空闲。",
    "/branch 分叉当前会话，让你探索不同方向而不丢失进度。",
    "/compress 手动压缩对话上下文，当内容太长时使用。",
    "/rollback 列出文件系统检查点 — 将代理修改过的文件恢复到之前任意状态。",
    "/rollback diff 2 预览自检查点 2 以来的变化，不恢复任何内容。",
    "/rollback 2 src/file.py 从特定检查点恢复单个文件。",
    '/title "my project" 为会话命名 — 稍后使用 /resume 或 hermes -c 恢复。',
    "/resume 恢复之前命名的会话，从上次中断处继续。",
    "/queue <prompt> 将消息排入下一轮队列，不中断当前轮次。",
    "/undo 删除对话中最后一条用户/助手交流记录。",
    "/retry 重新发送你的上一条消息 — 当代理的回复不太对时很有用。",
    "/verbose 切换工具进度显示：关闭 → 新 → 全部 → 详细。",
    "/reasoning high 增加模型的思考深度。/reasoning show 显示推理过程。",
    "/fast 切换优先级处理，获得更快的 API 响应（取决于提供商）。",
    "/yolo 在当前会话剩余时间内跳过所有危险命令批准提示。",
    "/model 让你在会话中切换模型 — 试试 /model sonnet 或 /model gpt-5。",
    "/model --global 永久更改你的默认模型。",
    "/personality pirate 设置有趣的个性 — 从可爱到莎士比亚共 14 种内置选项。",
    "/skin 更改 CLI 主题 — 试试 ares、mono、slate、poseidon 或 charizard。",
    "/statusbar 切换持久状态栏，显示模型、令牌数、上下文填充百分比、成本和耗时。",
    "/tools disable browser 临时移除当前会话的浏览器工具。",
    "/browser connect 通过 CDP 将浏览器工具附加到你正在运行的 Chromium 系列浏览器。",
    "/plugins 列出已安装的插件及其状态。",
    "/cron 管理定时任务 — 设置定期提示，可投递到任何平台。",
    "/reload-mcp 热重载 MCP 服务器配置，无需重启。",
    "/usage 显示令牌使用量、成本明细和会话时长。",
    "/insights 显示过去 30 天的使用分析。",
    "/paste 检查剪贴板中的图片并将其附加到下一条消息。",
    "/profile 显示当前激活的配置及其主目录。",
    "/config 一目了然地显示当前配置。",
    "/stop 杀死代理生成的所有正在运行的后台进程。",

    # --- @ 上下文引用 ---
    "@file:path/to/file.py 将文件内容直接注入到你的消息中。",
    "@file:main.py:10-50 只注入文件的第 10-50 行。",
    "@folder:src/ 注入目录树列表。",
    "@diff 将你未暂存的 Git 更改注入消息。",
    "@staged 注入你已暂存的 Git 更改（git diff --staged）。",
    "@git:5 注入最近 5 次提交及其完整补丁。",
    "@url:https://example.com 获取并注入网页内容。",
    "输入 @ 会触发文件系统路径补全 — 可交互式导航到任何文件。",
    "组合多个引用：\"Review @file:main.py and @file:test.py for consistency.\"",

    # --- 快捷键 ---
    "Alt+Enter 插入换行符以进行多行输入。（Windows 终端会拦截 Alt+Enter — 请改用 Ctrl+Enter。）",
    "Ctrl+C 中断代理。2 秒内连续按两次强制退出。",
    "Ctrl+Z 将 Hermes 挂起到后台 — 在 shell 中运行 fg 恢复。",
    "Tab 接受自动建议的幽灵文本或自动补全斜杠命令。",
    "在代理工作时输入新消息可中断并重定向它。",
    "Alt+V 从剪贴板粘贴图片到对话中。",
    "粘贴 5 行以上时会自动保存到文件并插入紧凑引用。",

    # --- CLI 标志 ---
    "hermes -c 恢复最近的 CLI 会话。hermes -c \"project name\" 按标题恢复。",
    "hermes -w 创建隔离的 Git 工作树 — 非常适合并行代理工作流。",
    "hermes -w -q \"Fix issue #42\" 结合了工作树隔离和一次性查询。",
    "hermes chat -t web,terminal 只启用特定工具集以进行聚焦会话。",
    "hermes chat -s github-pr-workflow 启动时预加载技能。",
    "hermes chat -q \"query\" 运行一次非交互式查询后退出。",
    "hermes chat --max-turns 200 覆盖每轮默认的 90 次迭代限制。",
    "hermes chat --checkpoints 在每次破坏性文件更改前启用文件系统快照。",
    "hermes --yolo 在整个会话中跳过所有危险命令批准提示。",
    "hermes chat --source telegram 标记会话以便在 hermes sessions list 中过滤。",
    "hermes -p work chat 在特定配置下运行，不更改默认配置。",

    # --- CLI 子命令 ---
    "hermes doctor --fix 诊断并自动修复配置和依赖问题。",
    "hermes dump 输出紧凑的设置摘要 — 非常适合错误报告。",
    "hermes config set KEY VALUE 自动将密钥路由到 .env，其余路由到 config.yaml。",
    "hermes config edit 在默认编辑器中打开 config.yaml。",
    "hermes config check 扫描缺失或过期的配置选项。",
    "hermes sessions browse 打开交互式会话选择器，支持搜索。",
    "hermes sessions stats 显示按平台统计的会话数量和数据库大小。",
    "hermes sessions prune --older-than 30 清理旧会话。",
    "hermes skills search react --source skills-sh 搜索 skills.sh 公共目录。",
    "hermes skills check 扫描已安装的中心技能以获取上游更新。",
    "hermes skills tap add myorg/skills-repo 添加自定义 GitHub 技能源。",
    "hermes skills snapshot export setup.json 导出技能配置用于备份或共享。",
    "hermes mcp add github --command npx 从命令行添加 MCP 服务器。",
    "hermes mcp serve 将 Hermes 自身作为 MCP 服务器运行，供其他代理使用。",
    "hermes auth add 让你添加多个 API 密钥以实现凭据池轮换。",
    "hermes completion bash >> ~/.bashrc 为所有命令和配置启用 Tab 补全。",
    "hermes logs -f 实时跟踪 agent.log。--level WARNING --since 1h 过滤输出。",
    "hermes backup 创建整个 Hermes 主目录的 Zip 备份。",
    "hermes profile create coder 创建隔离的配置，该配置将成为独立的命令。",
    "hermes profile create work --clone 将当前配置和密钥复制到新配置。",
    "hermes update 自动将所有新增的捆绑技能同步到所有配置。",
    "hermes gateway install 将 Hermes 设置为系统服务（systemd/launchd）。",
    "hermes memory setup 让你配置外部记忆提供商（Honcho、Mem0 等）。",
    "hermes webhook subscribe 创建事件驱动的 Webhook 路由，支持 HMAC 验证。",
    "节省费用：hermes tools 禁用不用的工具，hermes skills config 精简技能。",
    "/reasoning low 或 /reasoning minimal 降低默认（中等）思考深度 — 更快、更便宜的响应。",
    "hermes models routes 将视觉、压缩和辅助任务路由到更便宜的模型 — 将后台令牌成本降低 85% 以上，同时不降低主聊天模型的质量。",

    # --- 配置 ---
    "在 config.yaml 中设置 display.bell_on_complete: true，长任务完成时听到提示音。",
    "设置 display.streaming: true 可实时看到令牌逐字出现。",
    "设置 display.show_reasoning: true 可观察模型的思维链推理过程。",
    "设置 display.compact: true 可减少输出空白，信息更密集。",
    "设置 display.busy_input_mode: queue 可将消息排队而不是中断代理，或设为 steer 通过 /steer 在运行中注入。",
    "设置 display.resume_display: minimal 可在恢复会话时跳过完整的对话回顾。",
    "设置 compression.threshold: 0.50 控制自动压缩触发的时机（默认：上下文的 50%）。",
    "设置 agent.max_turns: 200 可让代理每轮执行更多工具调用步骤。",
    "设置 file_read_max_chars: 200000 增加每次 read_file 调用的最大内容量。",
    "设置 approvals.mode: smart 可让 LLM 自动批准安全命令并自动拒绝危险命令。",
    "在 config.yaml 中设置 fallback_model，自动故障转移到备份提供商。",
    "设置 privacy.redact_pii: true 可在发送给 LLM 之前对用户 ID 和电话号码进行哈希处理。",
    "设置 browser.record_sessions: true 可自动将浏览器会话录制为 WebM 视频。",
    "在 config.yaml 中设置 worktree: true 可始终创建 Git 工作树（等同于 hermes -w）。",
    "设置 security.website_blocklist.enabled: true 可阻止特定域名被 Web 工具访问。",
    "设置 cron.wrap_response: false 可投递原始代理输出，不带 cron 头部/尾部。",
    "HERMES_TIMEZONE 可用任何 IANA 时区字符串覆盖服务器时区。",
    "config.yaml 支持环境变量替换：使用 ${VAR_NAME} 语法。",
    "config.yaml 中的快速命令可零令牌消耗立即执行 shell 命令。",
    "自定义个性可在 config.yaml 的 agent.personalities 下定义。",
    "provider_routing 控制 OpenRouter 的提供商排序、白名单和黑名单。",

    # --- 工具与能力 ---
    "execute_code 运行可编程调用 Hermes 工具的 Python 脚本 — 结果不占据上下文。",
    "delegate_task 默认最多生成 3 个并发子代理（delegation.max_concurrent_children），使用隔离上下文进行并行工作。",
    "web_extract 支持 PDF 链接 — 传入任何 PDF 链接即可转换为 Markdown。",
    "search_files 基于 ripgrep，比 grep 更快 — 用它代替终端中的 grep。",
    "patch 使用 9 种模糊匹配策略，轻微的空白差异不会破坏编辑。",
    "patch 支持 V4A 格式，一次调用即可批量编辑多个文件。",
    "read_file 在未找到文件时会建议相似文件名。",
    "read_file 自动去重 — 重新读取未更改的文件会返回轻量存根。",
    "browser_vision 截取屏幕截图并用 AI 分析 — 适用于验证码和视觉内容。",
    "browser_console 可在页面上下文中执行 JavaScript 表达式。",
    "image_generate 使用 FLUX 2 Pro 创建图像并自动 2 倍放大。",
    "text_to_speech 将文本转换为音频 — 在 Telegram 上以语音气泡形式播放。",
    "send_message 可在会话中向任何已连接的消息平台发送消息。",
    "todo 工具帮助代理在会话期间跟踪复杂的多步骤任务。",
    "session_search 在所有历史对话中进行全文搜索。",
    "代理会自动将偏好、纠正和环境事实保存到记忆中。",
    "mixture_of_agents 通过 4 个前沿 LLM 协作处理难题。",
    "终端命令支持带 notify_on_complete 的后台模式，用于长时间运行的任务。",
    "终端后台进程支持 watch_patterns，可在特定输出行时发出警报。",
    "terminal 工具支持 6 种后端：本地、Docker、SSH、Modal、Daytona 和 Singularity。",

    # --- 配置文件 ---
    "每个配置文件拥有独立的配置、API 密钥、记忆、会话、技能和定时任务。",
    "配置文件名成为 shell 命令 — 'hermes profile create coder' 创建 'coder' 命令。",
    "hermes profile export coder -o backup.tar.gz 创建可移植的配置文件归档。",
    "如果两个配置文件意外共享同一个机器人令牌，第二个网关将被阻止并显示清晰的错误信息。",

    # --- 会话 ---
    "会话在第一次交流后自动生成描述性标题 — 无需手动命名。",
    "会话标题支持谱系：\"my project\" → \"my project #2\" → \"my project #3\"。",
    "退出时，Hermes 会打印包含会话 ID 和统计信息的恢复命令。",
    "hermes sessions export backup.jsonl 导出所有会话用于备份或分析。",
    "hermes -r SESSION_ID 按 ID 恢复任意历史会话。",

    # --- 记忆 ---
    "记忆是冻结的快照 — 更改仅在下次会话开始时出现在系统提示中。",
    "记忆条目会自动扫描提示注入和数据泄露模式。",
    "代理有两个记忆存储：个人笔记（约 2200 字符）和用户画像（约 1375 字符）。",
    "你给代理的纠正（\"不行，换种方式做\"）通常会自动保存到记忆。",

    # --- 技能 ---
    "超过 80 个捆绑技能，涵盖 GitHub、创意、MLOps、生产力、研究等。",
    "每个安装的技能会自动成为斜杠命令 — 输入 / 即可查看全部。",
    "hermes skills install official/security/1password 从仓库安装可选技能。",
    "技能可以限制到特定操作系统平台 — 有些只在 macOS 或 Linux 上加载。",
    "config.yaml 中的 skills.external_dirs 可让你从自定义目录加载技能。",
    "代理可以使用 skill_manage 将其自身创建为程序性记忆技能。",
    "plan 技能将 Markdown 计划保存到活动工作区的 .hermes/plans/ 下。",

    # --- 定时任务与调度 ---
    "定时任务可以附加技能：hermes cron add --skill blogwatcher \"Check for new posts\"。",
    "定时任务投递目标包括 Telegram、Discord、Slack、电子邮件、短信及 12 个以上平台。",
    "如果定时任务响应以 [SILENT] 开头，投递将被抑制 — 适用于仅监控的任务。",
    "Cron 支持相对延迟（30m）、间隔（every 2h）、Cron 表达式和 ISO 时间戳。",
    "定时任务在全新的代理会话中运行 — 提示必须自包含。",

    # --- 语音 ---
    "如果安装了 faster-whisper（免费本地语音转文字），语音模式无需任何 API 密钥即可工作。",
    "提供五种 TTS 提供商：Edge TTS（免费）、ElevenLabs、OpenAI、NeuTTS（免费本地）、MiniMax。",
    "/voice on 在 CLI 中启用语音模式。Ctrl+B 切换一键通录音。",
    "流式 TTS 在生成时逐句播放 — 无需等待完整响应。",
    "Telegram、Discord、WhatsApp 和 Slack 上的语音消息会自动转录。",

    # --- 网关与消息 ---
    "Hermes 运行在 21 个消息平台上：Telegram、Discord、Slack、WhatsApp、Signal、Matrix、IRC、Microsoft Teams、电子邮件等。",
    "hermes gateway install 将其设置为开机自启的系统服务。",
    "钉钉使用流模式 — 无需 Webhook 或公共 URL。",
    "BlueBubbles 通过本地 macOS 服务器将 iMessage 接入 Hermes。",
    "Webhook 路由支持 HMAC 验证、速率限制和事件过滤。",
    "API 服务器暴露与 OpenAI 兼容的端点，兼容 Open WebUI 和 LibreChat。",
    "Discord 语音频道模式：机器人加入语音频道，转录语音并回话。",
    "group_sessions_per_user: true 让每个用户在群聊中拥有自己的会话。",
    "/sethome 将聊天标记为定时任务投递的主频道。",
    "网关支持基于非活跃的超时 — 活跃代理可以无限期运行。",

    # --- 安全 ---
    "危险命令批准有 4 个级别：一次、会话、始终（永久白名单）、拒绝。",
    "智能批准模式使用 LLM 自动批准安全命令并标记危险命令。",
    "SSRF 保护阻止私有网络、回环地址、链路本地地址和云元数据地址。",
    "Tirith 执行前扫描检测同形字 URL 欺骗和管道到解释器的模式。",
    "MCP 子进程接收到经过过滤的环境 — 只有安全系统变量通过。",
    "上下文文件（.hermes.md、AGENTS.md）在加载前会进行安全扫描，防止提示注入。",
    "config.yaml 中的 command_allowlist 可永久批准特定的 shell 命令模式。",

    # --- 上下文与压缩 ---
    "上下文达到阈值时自动压缩 — 记忆被刷新，历史被摘要化。",
    "状态栏随着上下文填充而变为黄色、橙色，然后红色。",
    "SOUL.md 是代理的主要身份文件 — 自定义它以塑造行为。",
    "Hermes 从 .hermes.md、AGENTS.md、CLAUDE.md 或 .cursorrules（第一个匹配的）加载项目上下文。",
    "子目录中的 AGENTS.md 文件会在代理进入文件夹时逐步发现。",
    "上下文文件上限为 20000 个字符，支持智能头部/尾部截断。",

    # --- 浏览器 ---
    "五种浏览器提供商：本地 Chromium、Browserbase、Browser Use、Camofox 和 Firecrawl。",
    "Camofox 是反检测浏览器 — Firefox 分支，具有 C++ 指纹欺骗功能。",
    "browser_navigate 自动返回页面快照 — 之后无需再调用 browser_snapshot。",
    "browser_vision 使用 annotate=true 可在交互元素上叠加编号标签。",

    # --- MCP ---
    "hermes mcp 打开交互式选择器，展示 Nous 批准的 MCP，一键安装。",
    "hermes mcp catalog 列出仓库附带的 Nous 批准的 MCP 服务器。",
    "hermes mcp install <name> 安装目录条目，提示输入凭据，并让你选择启用哪些工具。",
    "MCP 服务器在 config.yaml 中配置 — 支持 stdio 和 HTTP 两种传输方式。",
    "每个服务器的工具过滤：tools.include 白名单和 tools.exclude 黑名单特定工具。",
    "MCP 服务器在运行时自动生成工具集 — hermes tools 可按平台切换它们。",
    "MCP OAuth 支持：auth: oauth 启用基于浏览器的 PKCE 授权。",

    # --- 检查点与回滚 ---
    "未修改文件时检查点零开销 — 默认启用。",
    "回滚前会自动保存快照，以便撤销撤销操作。",
    "/rollback 同时撤销对话轮次，因此代理不会记住已回滚的更改。",
    "检查点使用 ~/.hermes/checkpoints/ 下的影子仓库 — 绝不会触及项目的 .git。",

    # --- 批处理与数据 ---
    "batch_runner.py 并行处理数百个提示，用于训练数据生成。",
    "hermes chat -Q 为程序化使用启用安静模式 — 隐藏横幅和旋转动画。",
    "轨迹保存（--save-trajectories）捕获完整的工具使用痕迹，用于模型训练。",

    # --- 插件 ---
    "三种插件类型：通用（工具/钩子）、记忆提供商和上下文引擎。",
    "hermes plugins install owner/repo 直接从 GitHub 安装插件。",
    "提供 8 种外部记忆提供商：Honcho、OpenViking、Mem0、Hindsight 等。",
    "插件钩子包括 pre/post_tool_call、pre/post_llm_call 和用于输出标准化的 transform_terminal_output。",

    # --- 杂项 ---
    "提示缓存（Anthropic）通过重用缓存的系统提示前缀降低成本。",
    "代理在后台线程中自动生成会话标题 — 零延迟影响。",
    "智能模型路由可将简单查询自动路由到更便宜的模型。",
    "斜杠命令支持前缀匹配：/h 解析为 /help，/mod 解析为 /model。",
    "将文件路径拖入终端可自动附加图片或作为上下文发送。",
    "仓库根目录下的 .worktreeinclude 列出被 gitignore 忽略但需要复制到工作树的文件。",
    "hermes acp 将 Hermes 作为 ACP 服务器运行，集成 VS Code、Zed 和 JetBrains。",
    "自定义提供商：在 config.yaml 的 custom_providers 下保存命名端点。",
    "HERMES_EPHEMERAL_SYSTEM_PROMPT 注入永不持久化到历史的系统提示。",
    "credential_pool_strategies 支持 fill_first、round_robin、least_used 和 random 轮换策略。",
    "hermes auth add nous 或 hermes auth add openai-codex 设置基于 OAuth 的提供商。",
    "API 服务器同时支持 Chat Completions 和 Responses API，具有服务端状态。",
    "config 中的 tool_preview_length: 0 在旋转动画的活动反馈中显示完整文件路径。",
    "hermes status --deep 对所有组件运行更深入的诊断检查。",

    # --- 隐藏技巧与高级用户秘籍 ---
    "定时任务可以附加 Python 脚本（--script），其标准输出被注入到提示中作为上下文。",
    "定时脚本位于 ~/.hermes/scripts/，在代理之前运行 — 非常适合数据收集管道。",
    "config.yaml 中的 prefill_messages_file 将少样本示例注入每次 API 调用，永不保存到历史。",
    "SOUL.md 完全取代代理的默认身份 — 重写它让 Hermes 成为你自己的。",
    "SOUL.md 在首次运行时自动填充默认个性。编辑它以进行自定义。",
    "/compress <focus topic> 将约 60-70% 的摘要预算分配给你的主题，并积极修剪其余部分。",
    "第二次及以后的压缩时，压缩器会更新之前的摘要，而不是从头开始。",
    "在网关会话重置之前，Hermes 会在后台自动将重要事实刷新到记忆。",
    "config.yaml 中的 network.force_ipv4: true 修复 IPv6 有问题的服务器挂起 — 猴子补丁 socket。",
    "终端工具注释常见退出码：grep 返回 1 = '未找到匹配项（不是错误）'。",
    "失败的前台终端命令最多自动重试 3 次，间隔呈指数增长（2 秒、4 秒、8 秒）。",
    "裸 sudo 命令会自动重写为从 .env 读取 SUDO_PASSWORD — 无需交互式提示。",
    "execute_code 有内置辅助函数：json_parse() 用于容错解析、shell_quote() 和带退避的 retry()。",
    "execute_code 的 7 个沙箱工具（web_search、terminal、read/write/search/patch）使用 RPC — 从不进入上下文。",
    "读取同一文件区域 3 次以上会触发警告。4 次以上会被硬阻止以防止循环。",
    "write_file 和 patch 检测文件自上次读取后是否被外部修改，并警告过时问题。",
    "V4A 补丁格式支持添加文件、删除文件和移动文件指令 — 不仅仅是更新。",
    "MCP 服务器可以通过采样请求 LLM 补全 — 代理成为服务器的工具。",
    "MCP 服务器发送 notifications/tools/list_changed 以触发自动工具重新注册，无需重启。",
    "带有 acp_command: 'claude' 的 delegate_task 从任何平台生成 Claude Code 作为子代理。",
    "委派有一个心跳线程 — 子活动传播到父代理，防止网关超时。",
    "当提供商返回 HTTP 402（需要付款）时，辅助客户端自动回退到下一个提供商。",
    "agent.tool_use_enforcement 引导描述操作而非调用工具的模型 — 对 GPT/Codex 自动生效。",
    "agent.restart_drain_timeout（默认 60 秒）让正在运行的代理在网关重启生效前完成。",
    "agent.api_max_retries（默认 3）控制代理在显示错误前重试失败 API 调用的次数 — 降低它以获得快速回退。",
    "网关会缓存每个会话的 AIAgent 实例 — 销毁此缓存会破坏 Anthropic 提示缓存。",
    "任何网站都可以通过 /.well-known/skills/index.json 暴露技能 — 技能中心自动发现它们。",
    "技能审计日志位于 ~/.hermes/skills/.hub/audit.log，跟踪每次安装和移除操作。",
    "过时的 Git 工作树自动清理：24-72 小时前且没有未推送提交的工作树在启动时被清理。",
    "每个配置在 HERMES_HOME/home/ 下拥有独立的子进程 HOME — 隔离的 git、ssh、npm、gh 配置。",
    "HERMES_HOME_MODE 环境变量（八进制，例如 0701）为 Web 服务器遍历设置自定义目录权限。",
    "容器模式：在 HERMES_HOME 中放置 .container-mode，主机 CLI 自动进入容器执行。",
    "Ctrl+C 有 5 个优先级层：取消录制 → 取消提示 → 取消选择器 → 中断代理 → 退出。",
    "代理运行期间的每次中断都会记录到 ~/.hermes/interrupt_debug.log，包含时间戳。",
    "BROWSER_CDP_URL 将浏览器工具连接到任何正在运行的 Chromium 系列浏览器 — 接受 WebSocket、HTTP 或 host:port。",
    "BROWSERBASE_ADVANCED_STEALTH=true 启用高级反检测功能，使用自定义 Chromium（Scale 套餐）。",
    "CLI 在宽度小于 80 列的终端中自动切换到紧凑模式。",
    "快速命令支持两种类型：exec（直接运行 shell 命令）和 alias（重定向到另一个命令）。",
    "每任务委派模型：config 中的 delegation.model 和 delegation.provider 将子代理路由到更便宜的模型。",
    "delegation.reasoning_effort 独立控制子代理的思考深度。",
    "config.yaml 中的 display.platforms 允许按平台覆盖显示设置：{telegram: {tool_progress: all}}。",
    "config 中的 human_delay.mode 模拟人类打字速度 — 可配置的 min_ms/max_ms 范围。",
    "配置版本迁移在加载时自动运行 — 新配置键无需手动干预即可出现。",
    "GPT 和 Codex 模型获得特殊的系统提示指导，用于工具纪律和强制工具使用。",
    "Gemini 模型获得针对绝对路径、并行工具调用和非交互式命令的定制指令。",
    "config.yaml 中的 context.engine 可设置为插件名称，用于替代的上下文管理策略。",
    "超过 8000 令牌的浏览器页面在返回给代理之前由辅助 LLM 自动摘要。",
    "压缩器进行廉价预扫描：超过 200 字符的工具输出在 LLM 运行前被替换为占位符。",
    "当压缩失败时，后续尝试暂停 10 分钟以避免 API 轰炸。",
    "长危险命令（>70 字符）在批准提示中提供 '查看' 选项，可先查看完整文本。",
    "音频电平可视化在语音录制期间显示 ▁▂▃▄▅▆▇ 条形图，基于麦克风 RMS 电平。",
    "配置文件名称不能与现有 PATH 二进制文件冲突 — 'hermes profile create ls' 会被拒绝。",
    "hermes profile create backup --clone-all 复制所有内容（配置、密钥、SOUL.md、记忆、技能、会话）。",
    "语音录制键可通过 config.yaml 中的 voice.record_key 配置 — 不仅仅是 Ctrl+B。",
    ".cursorrules 和 .cursor/rules/*.mdc 文件会被自动检测并加载为项目上下文。",
    "上下文文件支持 10 种以上的提示注入模式 — 不可见 Unicode、'忽略指令'、数据泄露尝试。",
    "GPT-5 和 Codex 在消息格式中使用 'developer' 角色而不是 'system'。",
    "每任务辅助覆盖：config.yaml 中的 auxiliary.vision.provider、auxiliary.compression.model 等。",
    "辅助客户端将 'main' 视为提供商别名 — 解析为你的实际主要提供商 + 模型。",
    "hermes claw migrate --dry-run 预览 OpenClaw 迁移而不写入任何内容。",
    "带引号或转义空格的文件路径会被自动处理 — 无需手动清理。",
    "斜杠命令从不触发大粘贴折叠 — 带大参数的 /command 也能正确工作。",
    "在中断模式下，代理执行期间键入的斜杠命令绕过中断逻辑并立即执行。",
    "HERMES_DEV=1 绕过本地开发的容器模式检测。",
    "每个 MCP 服务器拥有独立的工具集（mcp-servername），可通过 hermes tools 独立切换。",
    "config 中的 MCP ${ENV_VAR} 占位符在服务器生成时解析 — 包括来自 ~/.hermes/.env 的变量。",
    "来自可信仓库（NousResearch）的技能获得 'trusted' 安全级别；社区技能获得额外扫描。",
    "技能隔离区位于 ~/.hermes/skills/.hub/quarantine/，存放等待安全审查的技能。",

    # --- 高级斜杠命令 ---
    '/steer <prompt> 在下一次工具调用后注入一条备注 — 在任务进行中调整方向而不中断。',
    '/goal <text> 设置持续的 Ralph 循环目标 — Hermes 自动轮次轮换，直到裁判判定完成。',
    '/snapshot create [label] 保存 Hermes 配置的完整状态快照；/snapshot restore <id> 稍后恢复。',
    '/copy [N] 将最后一条助手响应复制到剪贴板，或指定数字复制倒数第 N 条。',
    '/redraw 强制完全重绘 UI，修复 tmux 调整大小或鼠标选择后的终端漂移。',
    '/agents（别名 /tasks）显示当前会话中的活跃代理和运行中的后台任务。',
    '/footer 切换最终回复上的网关底部信息，显示模型、工具计数和轮次耗时。',
    '/busy queue|steer|interrupt 控制当 Hermes 正在工作时按 Enter 的行为。',
    '/topic 在 Telegram 私聊中启用用户管理的多会话主题模式 — /topic <id> 内联恢复历史会话。',
    '/approve session|always 以你选择的信任范围运行待处理的危险命令；/deny 拒绝它。',
    '/restart 在排空正在运行的代理后优雅重启网关，然后在上线时通知请求者。',
    '/kanban boards switch <slug> 从聊天中切换活动的多项目看板。',
    '/reload 将 ~/.hermes/.env 重新加载到正在运行的会话中 — 无需重启即可获取新的 API 密钥。',

    # --- 定时任务（无代理与脚本） ---
    '设置 no_agent=True 的定时任务按计划运行脚本并直接发送其标准输出 — 零令牌，零 LLM。',
    '空的定时脚本标准输出表示静默滴答 — 不投递任何内容，非常适合阈值看门狗。',
    'HERMES_CRON_MAX_PARALLEL（默认 4）限制每个滴答同时运行的定时任务数量，防止突发流量耗尽密钥。',

    # --- 网关钩子 ---
    '网关钩子位于 ~/.hermes/hooks/<name>/ 下，包含 HOOK.yaml + handler.py — 处理函数必须命名为 `handle`。',
    '钩子事件包括 gateway:startup、session:start、agent:step 和通配符订阅 command:*。',
    '放置 ~/.hermes/BOOT.md 清单，网关启动钩子每次启动时将其作为一次性代理运行。',

    # --- 策展人 ---
    'hermes curator run --dry-run 预览策展人将归档或合并的内容，不进行任何更改。',
    "hermes curator pin <skill> 硬性围栏技能，防止自动归档和代理的 skill_manage 工具操作。",
    'hermes curator rollback 从运行前快照恢复技能 — 备份位于 skills/.curator_backups/ 下。',

    # --- 凭据池与路由 ---
    'hermes auth reset <provider> 清除凭据池上的所有冷却时间和耗尽标志。',
    'credential_pool_strategies.<provider>: round_robin 均匀轮换密钥，而不是默认的 fill_first。',
    '每个工具的 use_gateway: true 将 Web、图像、TTS 或浏览器通过你的 Nous 订阅路由 — 无需额外密钥。',
    'provider_routing.data_collection: deny 在 OpenRouter 上排除存储数据的提供商。',
    'provider_routing.require_parameters: true 只路由到支持你请求中所有参数的提供商。',

    # --- TUI 与仪表盘 ---
    'HERMES_TUI_RESUME=1 在启动时自动重新附加到最近的 TUI 会话 — SSH 断开后很方便。',
    "HERMES_TUI_THEME=light|dark|<hex> 在未设置 COLORFGBG 的终端上强制 TUI 主题。",
    '在 TUI 中按 Ctrl+G 或 Ctrl+X Ctrl+E 在 $EDITOR 中打开输入缓冲区，用于长多行提示。',
    'TUI 内联渲染 LaTeX — $E=mc^2$ 变成 Unicode 数学符号而不是原始 TeX。',
    'hermes dashboard 在 127.0.0.1:9119 启动本地 Web UI — 零数据离开本地主机。',
    'hermes dashboard --tui 通过 xterm.js 和 WebSocket PTY 在浏览器中嵌入完整的 Hermes TUI。',
    '在 ~/.hermes/dashboard-themes/ 中放置包含两个调色板颜色的 YAML 文件，即可重新定制整个仪表盘。',
    '仪表盘插件即插即用：在 ~/.hermes/dashboard-plugins/ 中放置 manifest.json + JS 包 — 无需 npm 构建。',
    '仪表盘主题中的 layoutVariant: cockpit 添加 260px 左侧导轨，插件可通过侧边栏槽位填充。',

    # --- 环境变量与配置开关 ---
    "display.tool_progress_command: true 在消息平台上暴露 /verbose；默认仅在 CLI 中可用。",
    'HERMES_BACKGROUND_NOTIFICATIONS=result 仅在后台任务完成时通知（相对于 all/error/off）。',
    'HERMES_WRITE_SAFE_ROOT 将 write_file 和 patch 限制到目录前缀；外部写入需要批准。',
    'HERMES_IGNORE_RULES 跳过 AGENTS.md、SOUL.md、.cursorrules、记忆和预加载技能的自动注入。',
    'HERMES_ACCEPT_HOOKS 自动批准 config.yaml 中声明的未见过的 shell 钩子，无需 TTY 提示。',
    'auxiliary.goal_judge.model 将 /goal 裁判路由到便宜的快速模型，使循环成本接近零。',
    '检查点跳过包含超过 50000 个文件的目录，避免大型单体仓库上的慢速 Git 操作。',

    # --- TTS ---
    'tts.provider: piper 在 CPU 上运行 44 种语言的本地 TTS — 语音包自动下载到 ~/.hermes/cache/piper-voices/。',
    'tts.providers.<name>.type: command 使用 {input_path} 和 {output_path} 占位符连接任何 CLI TTS 引擎。',

    # --- API 服务器与代理 ---
    'API_SERVER_ENABLED=true 在网关旁边运行 OpenAI 兼容端点，用于 Open WebUI 和 LibreChat。',
    'GATEWAY_PROXY_URL 运行分离设置：平台 I/O 在本地，代理工作委派到远程 API 服务器。',

    # --- 平台特定 ---
    'MATRIX_DEVICE_ID 固定稳定的设备 ID 用于端到端加密 — 没有它密钥每次启动都会轮换，历史解密会失效。',
    'TELEGRAM_WEBHOOK_SECRET 在设置 TELEGRAM_WEBHOOK_URL 时必须提供 — 使用 openssl rand -hex 32 生成。',

    # --- 批处理 ---
    "batch_runner.py --resume 通过文本内容匹配已完成的提示，因此数据集重排序不会重新运行已完成的工作。",

    # --- 较少人知的斜杠命令 ---
    '/new 在原地开始新会话（别名 /reset）— 新会话 ID，清空历史，CLI 保持打开。',
    '/clear 清空终端屏幕并开始新会话 — 一个快捷键完成视觉重置。',
    '/history 在不离开 CLI 的情况下内联打印当前对话 — 适合快速回顾。',
    '/save 将当前对话保存到磁盘而不结束会话。',
    '/status 一目了然地显示会话信息：ID、标题、模型、令牌使用量和已用时间。',
    '/image <path> 附加本地图片文件用于下一个提示，无需粘贴或拖放。',
    '/platforms 从聊天内部直接显示网关和消息平台连接状态。',
    '/commands 分页显示完整斜杠命令和已安装技能列表 — 在没有 Tab 补全的平台上很有用。',
    '/toolsets 列出每个可用的工具集，让你知道 -t/--toolsets 接受什么。',
    '/gquota 在 Gemini Code Assist 提供商激活时显示配额使用情况和进度条。',
    '/voice tts 切换仅 TTS 模式 — 代理语音回复，但你仍然键入提示。',
    '/reload-skills 重新扫描 ~/.hermes/skills/，使新增的技能无需重启会话即可生效。',
    '/indicator kaomoji|emoji|unicode|ascii 选择代理运行期间 TUI 忙碌指示器的样式。',
    '/debug 上传支持包（系统信息 + 日志）并返回可共享链接 — 在聊天中也可用。',

    # --- CLI 子命令与标志 ---
    'hermes -z "<prompt>" 是最纯粹的一次性查询：标准输出返回最终答案，别无其他 — 适合在脚本中管道使用。',
    'hermes chat --pass-session-id 将会话 ID 注入系统提示，使代理可以自引用。',
    'hermes chat --image path/to/pic.png 将本地图片附加到单个 -q 查询，无需单独的上传步骤。',
    'hermes chat --ignore-user-config 跳过活跃用户配置 — 可重现的错误报告和 CI 运行。',
    "hermes chat --source tool 标记程序化聊天，使其不杂乱的 hermes sessions 列表。",
    'hermes dump --show-keys 包含脱敏的 API 密钥指纹，用于更深入的支持调试。',
    'hermes sessions rename <ID> "new title" 重命名任何历史会话；hermes sessions delete <ID> 删除一个。',
    'hermes import 恢复由 sessions export 或 profile export 生成的会话导出或配置文件归档。',
    'hermes fallback 交互式管理 fallback_model 链 — 无需手动编辑 config.yaml。',
    'hermes pairing 轮换私聊配对令牌 — 轮换后第一个发消息者获得机器人访问权。',
    'hermes setup 引导首次用户通过一个交互式流程完成提供商、密钥和平台配置。',
    'hermes status --deep 对每个组件运行完整的健康检查；普通 hermes status 是快速视图。',

    # --- 代理行为环境变量 ---
    'HERMES_AGENT_TIMEOUT=0 禁用运行中代理的网关非活跃杀死 — 用于长时间研究运行。',
    'HERMES_ENABLE_PROJECT_PLUGINS=1 自动加载仓库本地插件 ./.hermes/plugins/ — 设计上需要信任授权。',
    "HERMES_DISABLE_FILE_STATE_GUARD=1 关闭 patch 和 write_file 上的'自读取后文件已更改'保护。",
    'HERMES_ALLOW_PRIVATE_URLS=true 让 Web 工具访问 localhost 和私有网络 — 网关模式下默认关闭。',
    'HERMES_OPTIONAL_SKILLS=name1,name2 在每个配置首次运行时自动安装额外的可选目录技能。',
    'HERMES_BUNDLED_SKILLS 指向自定义捆绑技能树 — 由 Homebrew 和 Nix 打包使用。',
    'HERMES_DUMP_REQUEST_STDOUT=1 将每次 API 请求负载转储到标准输出而不是日志文件。',
    'HERMES_OAUTH_TRACE=1 记录脱敏的 OAuth 令牌交换和刷新尝试，用于调试提供商认证。',
    'HERMES_STREAM_RETRIES（默认 3）控制临时网络错误时的流中重连尝试次数。',

    # --- 网关行为环境变量 ---
    'HERMES_GATEWAY_BUSY_ACK_ENABLED=false 当用户向忙碌的代理发送消息时，静默 ⚡/⏳/⏩ 确认消息。',
    'HERMES_AGENT_NOTIFY_INTERVAL（默认 180 秒）设置网关在长轮次中发送进度通知的频率。',
    'HERMES_RESTART_DRAIN_TIMEOUT（默认 900 秒）限制 /restart 等待进行中运行完成的时间，超时则强制。',
    'HERMES_CHECKPOINT_TIMEOUT（默认 30 秒）限制文件系统检查点创建时间 — 在大型单体仓库上增加此值。',

    # --- 辅助任务与图像生成 ---
    'config.yaml 中的 image_gen.model 选择 FAL 模型：flux-2/klein、gpt-image-2、nano-banana-pro 等。',
    'image_gen.provider 通过插件（OpenAI Images、Codex、FAL）路由图像生成，而不是默认方式。',
    'AUXILIARY_VISION_BASE_URL + AUXILIARY_VISION_API_KEY 将视觉分析指向任何 OpenAI 兼容端点。',

    # --- 安全 ---
    'security.tirith_fail_open: false 使 Hermes 在 tirith 扫描器自身出错时阻止命令。',
    'TIRITH_FAIL_OPEN 环境变量覆盖 tirith_fail_open 配置 — 无需编辑 config.yaml 即可快速切换。',

    # --- 会话与源标记 ---
    '--source tool 聊天默认从 hermes sessions list 中排除 — 设置 --source 明确显示它们。',
    '会话 ID 以时间戳为前缀（20250305_091523_abcd），因此在 ls 和 jq 中排序自然有效。',

    # --- 杂项 ---
    'API_SERVER_MODEL_NAME 自定义 /v1/models 上的模型名称 — 对多配置 Open WebUI 设置至关重要。',
    '仪表盘插件从 /dashboard-plugins/<name>/ 提供 — 将文件放入 ~/.hermes/dashboard-plugins/。',
]


def get_random_tip(exclude_recent: int = 0) -> str:
    """返回随机贴士字符串。

    Args:
        exclude_recent: 当前未使用；保留用于未来
            跨会话去重。
    """
    return random.choice(TIPS)
