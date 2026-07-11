# Hermes Agent Desktop 产品完整度审计

> 审计日期：2026-07-10  
> 审计对象：`saas1` 分支的 Hermes Desktop、内置浏览器、AI 员工、MoneyPrinterTurbo 与视频素材库  
> 判断标准：只有代码、测试和隔离运行环境都能证明的行为才标记为“已完成”。

## 1. 总体结论

当前版本已经具备 Hermes Desktop 产品外壳、独立聊天表面、右栏浏览器、Profile 型 AI 员工列表、MoneyPrinterTurbo Video Studio 和视频素材库第一阶段。它还不是可直接交付给普通用户的完整获客系统。

最大的产品缺口不是页面数量，而是四条尚未闭环的业务链：

1. Agent 已能在当前会话的同一个内置浏览器 DOM 中安全读取和执行受控动作，但尚未完成登录测试账号上的抖音互动 E2E、持久审计和频控。
2. 抖音商家、线索、评论和私信尚未进入统一 CRM 数据库并稳定导出到 Obsidian。
3. AI 员工尚缺创建、Skill/MCP/API 能力清单和主 Agent 可视化派工闭环。
4. 视频素材库已经能切分和打技术标签，但还没有 AI 语义标签、文案到镜头匹配和 timeline 到最终成片的一键渲染闭环。

## 2. 功能完整度

| 模块 | 当前完整度 | 已验证完成 | 仍未完成 |
| --- | ---: | --- | --- |
| Desktop 外壳与聊天 | 65% | Electron + React 聊天、路由、右栏、独立 `HERMES_HOME`；开发模式可运行 | 全新安装会因固定提交不在官方远端而 404；macOS 包未签名、未公证；UI 全量测试和 lint 未通过 |
| 内置浏览器 | 72% | 每聊天独立持久 partition；地址导航；前进、后退、刷新；同 webview 快照、fingerprint 定位、点击/输入、结果与新快照回传；过期 URL/歧义目标拒绝 | 清理 partition 数据的显式操作；真实站点长时间稳定性和发布级 E2E |
| 抖音获客自动化 | 32% | 未登录 feed 只读分析；右栏可由用户登录；抖音按钮点击与 Enter 提交逐次审批 | 登录测试账号上的点赞/评论/私信验证；评论区捕捉、持久审计、频控和失败恢复未形成产品闭环 |
| 商家数据库与 Obsidian | 15% | 内置 Obsidian Skill 已在隔离 Vault 完成 Markdown、front matter、wikilink 写入 | Desktop Vault 配置、typed merchant/lead schema、去重、增量更新、隐私脱敏、稳定导出和 RAG 未完成 |
| AI 员工 | 35% | Profile 列表、选择、训练 `SOUL.md`、打开独立聊天 | 创建员工、能力清单、Skill/MCP/API 配置、主 Agent 派工面板和协作状态未完成 |
| MoneyPrinterTurbo | 55% | Desktop 表单、配置、素材、音频、字幕、任务、MiniMax、MCP；使用 MP3 自定义音频和合格本地素材已真实生成 1920x1080 H.264/AAC 成片 | 预览/下载因 token 绕过代理返回 401；默认窗口布局重叠；WAV 等已接受格式会在任务中失败；错误原因、配置热加载、无 BGM 选项和发布级 E2E 未闭环 |
| 视频素材库 | 55% | 本地视频导入、SHA-256 去重、FFmpeg 探测/切分、关键帧、技术标签、查询、timeline、片段加入混剪 | AI 语义标签、ASR、embedding、文案镜头匹配、人工标签编辑 UI、timeline 渲染闭环未完成 |

完整度是工程估算，不是发布承诺；上表的“已完成”只表示当前仓库和隔离开发环境已有可验证实现。

## 3. 本轮发现并处理的漏洞

### P0：跨浏览器 DOM 索引重放

旧逻辑会先在 Agent 的 Camofox 浏览器执行 `browser_click/type/press/scroll`，再把 `@eN` 索引重放到用户已登录的 Desktop webview。两边的 DOM、Cookie 和页面时序不同，同一个索引可能指向不同按钮。

处理结果：跨运行时 DOM 动作重放已禁用。Desktop 网关现在把现有 `browser_*` 工具直接路由到当前会话的同一个 webview；模型只接收由当前快照生成的 `@eN`，执行时按完整 fingerprint 唯一匹配并校验快照 URL。URL 已变化、目标消失或出现多个相同目标时均失败关闭。

### P0：内置浏览器表单值和敏感输入暴露

旧快照会读取 input/textarea 当前值，密码或用户已填写的私密信息可能进入模型上下文。

处理结果：快照和动作 target 均不再返回表单当前值，只保留标签、placeholder、role、href 等定位字段；所有浏览器结果仍经过 Hermes 强制脱敏边界。

### P0：MoneyPrinter sidecar 可被本机其他进程直接调用

旧 sidecar 的视频、任务和媒体接口没有统一 managed token；Hermes 还使用公开 `/docs` 判断端口上的服务身份，任意占用 8080 的 HTTP 服务都可能被误判。

处理结果：managed 模式下 `/api/v1` 和 `/tasks` 统一要求 `X-Hermes-MoneyPrinter-Token`；新增带身份标记的鉴权健康接口；适配器会拒绝无关端口进程并对媒体代理带上 token。

### P0：MoneyPrinter 运行时缺失时仍可能被当作可启动

处理结果：启动前检查 sidecar Python、`fastapi`、`uvicorn`、`moviepy`、`imageio_ffmpeg` 与 FFmpeg 二进制；缺失时返回可诊断的 503，而不是启动后静默失败。

### P1：重新分析失败会破坏旧片段文件

旧流程在新 FFmpeg 片段全部生成前就删除旧目录；中途失败时数据库仍保留旧 clip，但文件已经丢失。

处理结果：新片段先写入 staging 目录，全部成功后再交换目录；clips、tags、asset metadata 和 analysis job 在同一 SQLite 事务提交，失败时恢复旧目录和旧数据库记录。

## 4. 当前安全边界

- 抖音页面读取、公开资料分析和草稿生成可以自动运行。
- 点赞、提交评论、发送私信、发布和删除属于外部副作用。当前桥会对抖音非链接控件、已知写操作文案和 Enter 提交逐次显式批准；持久审计和更细动作分类仍待补。
- 批量互动必须有账号级速率限制、每日上限、去重和熔断；不得绕过验证码、登录墙或平台风控。
- SQLite 是商家和线索事务数据源；Obsidian 只做脱敏后的知识导出，不能保存 Cookie、token、验证码或原始私信全文。
- MoneyPrinterTurbo 和视频素材库是 capability/sidecar，不进入 Hermes Agent Core 工具常驻面。

## 5. 下一步开发任务

### P0：形成可交付闭环

1. 在登录测试账号完成 Desktop-owned browser bridge 的单动作 E2E：只读快照、打开评论区、点赞、评论草稿输入和批准后提交；验证码或风控立即停止。
2. 将现有逐次审批扩展为持久副作用审计：账号、URL、动作、目标、预览文本、批准人、有效期和结果绑定。
3. 将 MoneyPrinter sidecar Python 依赖纳入桌面安装/升级流程，不依赖开发机手工 `.venv`。
4. 增加视频 AI 分析：ASR、视觉描述、人物/场景/商品/动作/情绪标签和 embedding 搜索。
5. 实现“文案核心 -> 标签 -> 镜头匹配 -> timeline -> MoneyPrinter 渲染 -> 成片”的单任务流水线。

### P1：业务系统

1. 把已有 `saas` 分支 Merchant Workbench 以人工解决冲突的方式移植到 `saas1`。
2. 建立 merchant/lead/post/comment/conversation/action/audit SQLite schema 和去重规则。
3. 实现 Obsidian 增量导出、front matter 稳定 ID、脱敏和导出检查点。
4. 完成 AI 员工创建和 capability manifest：toolsets、Skills、MCP、API 引用、知识库和审批上限。
5. 复用 Hermes delegation/Kanban，实现主 Agent 派工、员工状态、产物和验收面板。

### P2：抖音获客工作流

1. 公开商家资料和评论线索捕捉，进入 CRM 待处理队列。
2. 评论/私信草稿生成、人工批准、同 webview 提交和可追溯结果。
3. 账号级速率限制、每日额度、重复触达防护、异常熔断和登录失效恢复。

## 6. 本轮验证基线

| 范围 | 结果 |
| --- | --- |
| 视频素材库、MoneyPrinter adapter 与 Web routes | 40 tests passed |
| Desktop 浏览器、会话来源与路由 | 37 affected UI tests passed |
| Desktop 浏览器协议与脱敏 | 6 tests passed |
| TUI gateway | 315 tests passed；1 个既有 macOS Chrome 手动启动提示断言单独失败并排除 |
| Agent core `test_run_agent.py` | 415 tests passed |
| MoneyPrinter sidecar、MiniMax | 29 tests passed |
| Desktop TypeScript typecheck | passed |
| Desktop production build | passed；保留既有 CSS 与大 chunk 警告 |
| Python `py_compile` | passed |
| Desktop 全量 UI 基线 | 1122 passed / 20 failed；另有 build/release 内复制测试被 Vitest 误收集，本轮受影响文件均已单独通过 |
| 隔离运行 smoke | Renderer `127.0.0.1:5174`、随机本地 backend 均监听；backend 根路径 HTTP 200；16 sessions / 7794 messages 保留 |

后续每个模块必须在此基线上增加行为测试和隔离运行 smoke test，不以页面可见或文档存在代替功能完成。

## 7. 商业发布候选烟测（2026-07-10）

### 发布结论

当前构建可作为内部开发预览，不能作为完整商业应用交付。阻断项是全新安装失败、视频预览/下载失败、默认窗口布局不可用、视频输入契约与实际引擎不一致，以及测试基线未全绿。

### 视频页面真实链路

- Desktop 隔离后端和 MoneyPrinter sidecar 启动成功；安装、运行时、存储写入检查通过。
- 文案生成在模型配置有效且 sidecar 重启后成功；关键词生成请求 2 条并返回 2 条。
- MP3 自定义音频任务、Edge TTS 字幕任务均完成；SRT 非空。
- 合格本地素材 + MP3 自定义音频完成真实成片，任务 `a9f50ea0-c340-4f66-b2b8-24f6d9c35c73` 输出 4 秒、1920x1080、H.264/AAC 视频。
- 视频素材库导入、SHA-256 去重、FFmpeg 切分、技术标签、加入混剪通过；当前标签仍是横屏/短镜头等技术标签，不是 AI 语义标签。
- 预览和下载 URL 被 renderer 改写为 sidecar 直连地址，HTML media/link 无法携带 managed token，实测均返回 401，播放器 `readyState=0`。
- 默认 1220x800 窗口低于 `xl` 断点，三段内容退化为单列但子区没有正常高度/滚动约束，表单、任务和预览互相覆盖。
- UI 接受 WAV/FLAC/M4A 等格式，但引擎时长函数只接受 `.mp3`；2 秒 WAV 上传成功后任务失败，转成 MP3 后同链路成功。
- 配置保存只落盘，不使正在运行的 sidecar 热加载；无效模型错误被当作文案成功写入文本框。
- “无 BGM”发送空字符串后会被 adapter 恢复为 `random`，选项语义不成立。

### Obsidian 结论

基础文件能力已完成：新 `HERMES_HOME` 能安装 bundled Obsidian Skill，并通过 Hermes `write_file` 在隔离 Vault 创建带 front matter 和 wikilink 的 Markdown。业务知识库沉淀尚未完成：没有 Desktop Vault 配置、商家/线索/评论 schema、稳定 ID 增量导出、去重检查点、私信脱敏和语义检索。

### 测试与打包证据

| 检查 | 结果 |
| --- | --- |
| Python 全量 | 1836 文件；37684 passed / 47 failed；另有 9 个 ACP 文件因缺少 `acp` 依赖未收集；639.7 秒 |
| Desktop UI | 1122 passed / 20 failed；54 个 Vitest 文件失败（含被 Vitest 误收集的 `node:test` 文件） |
| Desktop platform | 288 passed / 1 skipped / 0 failed |
| Desktop typecheck | passed |
| Desktop lint | 6 errors / 27 warnings |
| TUI | 1108 passed / 4 failed / 1 skipped；typecheck、build 通过；lint 1 error / 14 warnings |
| MoneyPrinter sidecar | 68 passed / 1 failed / 3 skipped；失败为 Gemini TTS legacy temp 目录问题 |
| Node 生产依赖审计 | 0 vulnerabilities；Python `pip-audit` 未安装，不能声明 Python 供应链已审计 |
| macOS 打包 | `.app` 产出成功，但未签名、未公证 |
| 打包后全新安装 | 失败；bootstrap 固定到本地提交 `50b18eb1a72d`，官方远端不存在该提交，下载 `scripts/install.sh` 返回 404 |

全量 Python 失败中有一部分由 macOS `/tmp` -> `/private/tmp`、缺少 Linux `systemctl`、代理 DNS 映射和 ACP 可选依赖造成；同时也有浏览器路由、配置隔离、插件发现等真实契约失败。测试运行还会读取用户已有永久审批状态，说明测试入口没有完整隔离 `HERMES_HOME`，发布 CI 必须改成干净配置运行并按平台拆分。
