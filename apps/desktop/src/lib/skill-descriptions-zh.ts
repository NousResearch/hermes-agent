// Simplified-Chinese descriptions for the built-in skills shown on the 技能
// (Skills) page. ApexNodes is a China-first product, so when the UI locale is
// `zh` we render these instead of the English `skill.description` that the
// runtime returns.
//
// Display-only, and OURS: the upstream SKILL.md frontmatter and the runtime
// (`tools/skills_tool.py` / `hermes_cli/web_server.py`) are untouched. This map
// is applied at render time in app/skills/index.tsx; a missing key simply falls
// back to the original English description, so nothing breaks if the skill set
// drifts.
//
// Keyed by `SkillInfo.name` (the frontmatter `name`, e.g. "airtable",
// "google-workspace") — the SAME field the page keys rows/toggles on. Product
// and tool proper nouns (Airtable, Notion, gh, vLLM, ComfyUI, …) stay in
// English on purpose; only the surrounding description is translated.
//
// To translate a newly-added skill: add an entry keyed by its frontmatter name.
export const SKILL_DESCRIPTIONS_ZH: Readonly<Record<string, string>> = {
  // ── apple ──────────────────────────────────────────────────────────────
  'apple-notes': '通过 memo CLI 管理 Apple Notes:新建、搜索、编辑备忘录。',
  'apple-reminders': '通过 remindctl 操作 Apple Reminders:添加、列出、完成提醒事项。',
  findmy: '在 macOS 上通过 FindMy.app 追踪 Apple 设备与 AirTag。',
  imessage: '在 macOS 上通过 imsg CLI 收发 iMessage 与短信。',
  'macos-computer-use':
    '在后台操控 macOS 桌面——截图、鼠标、键盘、滚动、拖拽——且不抢占用户的光标、键盘焦点或桌面空间。适配任意支持工具调用的模型。只要 `computer_use` 工具可用即加载此技能。',

  // ── autonomous-ai-agents ───────────────────────────────────────────────
  'claude-code': '把编码任务委派给 Claude Code CLI(实现功能、提交 PR)。',
  codex: '把编码任务委派给 OpenAI Codex CLI(实现功能、提交 PR)。',
  'hermes-agent': '配置、扩展或为 Hermes Agent 贡献代码。',
  opencode: '把编码任务委派给 OpenCode CLI(实现功能、评审 PR)。',

  // ── creative ───────────────────────────────────────────────────────────
  'architecture-diagram': '以 HTML 生成深色主题的 SVG 架构 / 云 / 基础设施图。',
  'ascii-art': 'ASCII 字符画:pyfiglet、cowsay、boxes、图片转 ASCII。',
  'ascii-video': 'ASCII 视频:将视频 / 音频转为彩色 ASCII 的 MP4 / GIF。',
  'baoyu-infographic': '信息图:21 种布局 × 21 种风格(信息图、可视化)。',
  'claude-design': '设计一次性的 HTML 作品(落地页、演示稿、原型)。',
  comfyui:
    '用 ComfyUI 生成图像、视频与音频——安装、启动、管理节点 / 模型,并以参数注入方式运行工作流。生命周期用官方 comfy-cli,执行走 REST / WebSocket API。',
  'design-md': '编写 / 校验 / 导出 Google 的 DESIGN.md 设计令牌规范文件。',
  excalidraw: '生成手绘风格的 Excalidraw JSON 图(架构、流程、时序)。',
  humanizer: '润色文本:去除 AI 腔,注入真实的表达语气。',
  'manim-video': 'Manim CE 动画:3Blue1Brown 风格的数学 / 算法视频。',
  p5js: 'p5.js 创作:生成艺术、着色器、交互、3D。',
  'popular-web-designs': '以 HTML/CSS 呈现 54 套真实设计体系(Stripe、Linear、Vercel)。',
  pretext:
    '用 @chenglou/pretext 构建创意浏览器演示——无需 DOM 的文本排版,适用于 ASCII 字符画、绕障文字流、文本即几何的游戏、动态排版,以及以文字驱动的生成艺术。默认输出单文件 HTML 演示。',
  sketch: '一次性的 HTML 原型:生成 2-3 个设计变体供对比。',
  'songwriting-and-ai-music': '作词技巧与 Suno AI 音乐提示词。',
  'touchdesigner-mcp':
    '通过 twozero MCP 控制运行中的 TouchDesigner 实例——创建算子、设置参数、连线、执行 Python、构建实时视觉。内置 36 个原生工具。',

  // ── data-science ───────────────────────────────────────────────────────
  'jupyter-live-kernel': '通过实时 Jupyter 内核(hamelnb)迭代式运行 Python。',

  // ── devops ─────────────────────────────────────────────────────────────
  'kanban-orchestrator':
    '面向编排者角色的任务拆解手册与「防手痒」准则,用于通过 Kanban 派发工作。「别自己动手做」的原则和基本生命周期已自动注入每个 Kanban worker 的系统提示词;当你专门扮演编排者角色时,加载此技能获取更深入的手册。',
  'kanban-worker':
    'Hermes Kanban worker 的常见坑、示例与边界情况。生命周期本身已作为 KANBAN_GUIDANCE 自动注入每个 worker 的系统提示词(来自 agent/prompt_builder.py);此技能用于需要深入了解具体场景时加载。',

  // ── email ──────────────────────────────────────────────────────────────
  himalaya: 'Himalaya CLI:在终端里通过 IMAP/SMTP 收发邮件。',

  // ── github ─────────────────────────────────────────────────────────────
  'codebase-inspection': '用 pygount 巡检代码库:代码行数、语言构成、占比。',
  'github-auth': 'GitHub 认证配置:HTTPS 令牌、SSH 密钥、gh CLI 登录。',
  'github-code-review': '评审 PR:查看差异,通过 gh 或 REST 添加行内评论。',
  'github-issues': '通过 gh 或 REST 创建、分诊、打标签、指派 GitHub issue。',
  'github-pr-workflow': 'GitHub PR 全流程:建分支、提交、开 PR、跑 CI、合并。',
  'github-repo-management': '克隆 / 创建 / fork 仓库;管理远端与 release。',

  // ── media ──────────────────────────────────────────────────────────────
  'gif-search': '通过 curl + jq 从 Tenor 搜索并下载 GIF。',
  heartmula: 'HeartMuLa:类 Suno 的歌曲生成,由歌词 + 标签生成。',
  songsee: '通过 CLI 生成音频频谱图 / 特征(mel、chroma、MFCC)。',
  'youtube-content': '将 YouTube 字幕转写为摘要、推文串、博客。',

  // ── mlops ──────────────────────────────────────────────────────────────
  'audiocraft-audio-generation': 'AudioCraft:MusicGen 文本生音乐,AudioGen 文本生音效。',
  'evaluating-llms-harness': 'lm-eval-harness:对 LLM 跑基准测试(MMLU、GSM8K 等)。',
  'huggingface-hub': 'HuggingFace hf CLI:搜索 / 下载 / 上传模型与数据集。',
  'llama-cpp': 'llama.cpp 本地 GGUF 推理 + HF Hub 模型发现。',
  'segment-anything-model': 'SAM:通过点、框、掩码实现零样本图像分割。',
  'serving-llms-vllm': 'vLLM:高吞吐 LLM 服务、OpenAI 兼容 API、量化。',
  'weights-and-biases': 'W&B:记录 ML 实验、超参搜索、模型注册表、看板。',

  // ── note-taking ────────────────────────────────────────────────────────
  obsidian: '在 Obsidian 仓库中读取、搜索、新建、编辑笔记。',

  // ── productivity ───────────────────────────────────────────────────────
  airtable: '通过 curl 调用 Airtable REST API。记录增删改查、筛选、upsert。',
  'google-workspace': '通过 gws CLI 或 Python 操作 Gmail、Calendar、Drive、Docs、Sheets。',
  maps: '基于 OpenStreetMap/OSRM 做地理编码、POI、路线、时区查询。',
  'nano-pdf': '通过 nano-pdf CLI 编辑 PDF 文本 / 错别字 / 标题(自然语言指令)。',
  notion: 'Notion API + ntn CLI:页面、数据库、Markdown、Workers。',
  'ocr-and-documents': '从 PDF / 扫描件提取文本(pymupdf、marker-pdf)。',
  powerpoint: '创建、读取、编辑 .pptx 演示稿、幻灯片、备注、模板。',
  'teams-meeting-pipeline':
    '通过 Hermes CLI 操作 Teams 会议纪要流水线——生成会议摘要、查看流水线状态、重放任务、管理 Microsoft Graph 订阅。',

  // ── research ───────────────────────────────────────────────────────────
  arxiv: '按关键词、作者、分类或 ID 搜索 arXiv 论文。',
  blogwatcher: '通过 blogwatcher-cli 监控博客与 RSS/Atom 订阅源。',
  'llm-wiki': 'Karpathy 的 LLM Wiki:构建 / 查询相互链接的 Markdown 知识库。',
  polymarket: '查询 Polymarket:市场、价格、订单簿、历史。',
  'research-paper-writing': '撰写面向 NeurIPS/ICML/ICLR 的 ML 论文:从选题设计到投稿。',

  // ── smart-home ─────────────────────────────────────────────────────────
  openhue: '通过 OpenHue CLI 控制 Philips Hue 灯光、场景、房间。',

  // ── social-media ───────────────────────────────────────────────────────
  xurl: '通过 xurl CLI 使用 X/Twitter:发帖、搜索、私信、媒体、v2 API。',

  // ── software-development ────────────────────────────────────────────────
  'hermes-agent-skill-authoring': '编写仓库内 SKILL.md:frontmatter、校验器、结构。',
  'node-inspect-debugger': '通过 --inspect + Chrome DevTools 协议在命令行调试 Node.js。',
  plan: '规划模式:把可执行的 Markdown 计划写入 .hermes/plans/,不执行。任务拆到最小、路径精确、代码完整。',
  'python-debugpy': '调试 Python:pdb REPL + debugpy 远程调试(DAP)。',
  'requesting-code-review': '提交前评审:安全扫描、质量门禁、自动修复。',
  'simplify-code': '用 3 个并行 agent 清理近期代码改动。',
  spike: '一次性的探索实验,在正式动工前验证想法。',
  'systematic-debugging': '四阶段根因调试:先弄清 bug 再动手修。',
  'test-driven-development': 'TDD:严格执行「红-绿-重构」,先写测试再写代码。',

  // ── general (SKILL.md at skills/<name>/, no category folder) ─────────────
  dogfood: '对 Web 应用做探索式 QA:发现 bug、留存证据、产出报告。',
  yuanbao: '元宝群聊:@提及用户,查询信息 / 成员。'
}

/**
 * Chinese description for a skill, keyed by `SkillInfo.name`. Falls back to the
 * provided (English) description when we don't have a translation — so an
 * unmapped or newly-added skill still renders something sensible.
 */
export function zhSkillDescription(name: string, fallback: string): string {
  return SKILL_DESCRIPTIONS_ZH[name] ?? fallback
}
