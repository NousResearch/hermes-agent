---
name: hermes-obsidian-notes
description: 雷老板的 Obsidian 知识仓库体系 v2.0「阅读健身房」—— 从碎片捕获到费曼检验，把知识仓库从储藏室变成健身房。AI 训练你读，不是 AI 替你读。
author: 雷老板
version: "2.0"
platforms: [macos]
tags: [obsidian, knowledge-management, reading-gym, distillation, feynman, chinese]
---
> 📢 **公开发布版** — 原作为雷老板的「阅读健身房」知识仓库体系。本文已脱敏，完整私有版请联系原作者。


# 知识仓库「阅读健身房」v2.0

> **核心理念**：知识仓库不是储藏室，是健身房。AI 训练你读，不是 AI 替你读。
> **作者**：雷老板
> **适用**：macOS + Obsidian + Hermes Agent

## 仓库架构总览

```
📥 收件箱/          → 蜜蜂通道：所有碎片先扔这里
🗂️ 项目/            → 蚂蚁主场：每个项目文档汇总
📚 知识库/          → 蝴蝶之翼：11 个分类 + 双向链接知识图谱
📅 日记/            → 瓢虫主场：每日三只青蛙 + 精力管理 + 复盘
🍎 Apple Notes      → 外部蜜蜂通道：memo CLI 跨设备速记
⏰ Apple Reminders  → 萤火虫通道：remindctl 定时提醒
🧠 Hermes Memory    → 持久偏好存储（不存进度，只存规则）
```

### 11 个知识分类

| 分类 | 路径 | emoji |
|:--|:--|:--|
| AI 与科技 | `知识库/🤖 AI与科技/` | 🤖 |
| 商业与管理 | `知识库/💼 商业与管理/` | 💼 |
| 创业与投资 | `知识库/🚀 创业与投资/` | 🚀 |
| 产品与设计 | `知识库/🎨 产品与设计/` | 🎨 |
| 效率与工具 | `知识库/⚡ 效率与工具/` | ⚡ |
| 个人成长 | `知识库/🌱 个人成长/` | 🌱 |
| 历史与人文 | `知识库/📜 历史与人文/` | 📜 |
| 行业洞察 | `知识库/🏭 行业洞察/` | 🏭 |
| 技术架构 | `知识库/🔧 技术架构/` | 🔧 |
| 方法论 | `知识库/🧠 方法论/` | 🧠 |
| 蒸馏报告 | `知识库/蒸馏报告/` | 📊 |

## 知识卡片模板

每张卡片含 **16 个 YAML frontmatter 字段** + 正文区块：

```yaml
---
title: ""
created: YYYY-MM-DD
source: ""
source_url: ""
medium: ""           # 🎬视频 / 📄文章 / 📕书籍 / 🔊播客 / 📋PDF
domain: ""           # 11 个分类之一
depth: ""            # 🌊深度 / 🏄浏览 / 💎精华
tags: []
why_useful: ""       # 为什么对你有用
distillation_layer: ""  # L1-信息 / L2-方法 / L3-行动
batch: ""            # 蒸馏批次号
status: 📌待整理
review_questions: ""  # 关联追问答案
feynman_result: ""    # 费曼检验结果
comment_quality: ""   # 评论区评级 ⭐
comment_highlights: 0
pages: null           # PDF 专属：页数
extraction_method: ""  # PDF 专属：pymupdf / marker
is_scanned: false     # PDF 专属
source_file: ""       # PDF 专属：本地缓存路径
---
```

正文区块：

```markdown
## 📝 摘要

## 💡 关键观点

## 🤔 关联追问
① 这篇让你想起了之前收藏过的哪篇？为什么？
② 和你的哪个项目/工作最相关？
③ 如果只能带走一个洞察，你选哪个？

## 💬 评论区精华

## 🎓 费曼检验
| 日期 | 结果 | 笔记 |
|:--|:--|:--|
```

## 场景一：知识捕获（5 步管道）

```
飞书链接 → 1.URL 识别 → 2.内容提取 → 3.AI 分析 → 4.双通道入库 → 5.索引更新
```

### 1. URL 类型识别

| URL 模式 | 类型 | 提取工具 |
|:--|:--|:--|
| `youtube.com/watch` / `youtu.be` | YouTube 视频 | `youtube-content` skill → transcript |
| `mp.weixin.qq.com/s/` | 微信公众号 | Jina AI + 降级方案（CAPTCHA） |
| `weixin.qq.com/sph/` | 微信视频号 | Jina AI（✅ 验证通过） |
| `bilibili.com/video/` | B站 | Jina AI |
| `zhihu.com/` | 知乎 | Jina AI |
| 通用网页 | 其他 | `https://r.jina.ai/<URL>` |

### 2. 内容提取

**Jina AI Reader**（免费，无需 API Key）：
```bash
curl -sS -m 20 -H "Accept: text/markdown" "https://r.jina.ai/<URL>"
```

**YouTube 字幕**：`youtube-content` skill 的 `fetch_transcript.py`

**微信文章降级方案**（CAPTCHA 拦截时）：
```bash
# ① 仅提取标题
curl -sS -m 10 \
  -H "User-Agent: Mozilla/5.0 ..." \
  "<URL>" | grep -oi 'og:title"[^>]*'
# ② 创建骨架卡片，标注"待补全"
# ③ 请求用户粘贴全文 → AI 补全 → patch 更新
```

### 3. AI 分析四维度

1. **摘要**（3-5 句中文）
2. **关键观点**（3-5 条）
3. **自动打标签**：领域标签 + 深度标签（🌊/🏄/💎）+ 媒介标签
4. **主分类判定**：从 11 个分类中选最优

### 4. 双通道入库

**通道一：Obsidian** — `write_file` 创建 `知识库/<分类>/YYYY-MM-DD <标题>.md`

**通道二：Apple Notes** — `memo` CLI（需授权 Automation）：
```bash
printf '# <标题>\n\n📎 <链接>\n\n📝 <摘要>\n\n💡 <关键观点>' \
  | EDITOR=tee memo notes -a -f "知识捕获"
```

### 5. 索引更新

每次捕获后**必须**用 `patch` 更新 `📚 知识库索引.md` 的笔记计数。

---

## 场景二：关联追问（每次捕获触发）

> 「阅读健身房」核心训练：训练联想肌肉，不训练收藏肌肉。

生成知识卡片后，**立即在飞书追问 3 个问题**：

1. 🤔 **这篇让你想起了之前收藏过的哪篇？为什么？**
2. 🤔 **和你的哪个项目/工作最相关？**
3. 🤔 **如果只能带走一个洞察，你选哪个？**

用户回答后 → `patch` 追记到卡片底部的「🤔 关联追问」区块 → 更新双向链接。

---

## 场景三：收件箱整理

1. `search_files` 列出 `收件箱/` 下所有 `📌待整理` 笔记
2. 逐条判断归属：项目 → `项目/`，知识 → `知识库/<分类>/`
3. 移动文件，补全 YAML frontmatter
4. 添加 `[[双向链接]]`
5. 更新 `🗂️ 项目索引.md` 或 `📚 知识库索引.md`

---

## 场景四：项目文档管理

### 4a 创建项目

**独立项目（单文件）**：`项目/<项目名>.md`
```yaml
---
tags: [项目, 活跃]
created: YYYY-MM-DD
status: 进行中
---
```

**项目组（文件夹）**：`项目/<项目组>/🐔 <项目组名>.md` + 子项目 `.md` 文件
- 子项目用 `[[🐔 父项目组]]` 链接回项目组索引
- 项目组索引含简介 + 子项目列表 + 干系人

### 4b 附件数据提取（项目分析）

当用户上传 docx/xlsx/pdf/pptx 附件：

1. 缓存到 `~/.hermes/cache/documents/`
2. 用 `parse_documents.py` 提取结构化摘要（zipfile+xml.etree，零依赖）
3. 在 `execute_code` 中完成数据汇总 + Markdown 排版
4. 用 `patch` 追加到项目文件（⛔ 绝不用 `write_file` 重写整个文件！）
5. 建文件清单表：

```markdown
## 📁 关联文件
| 文件 | 类型 | 核心内容 |
|:--|:--|:--|
| 报价单.xlsx | 预算参考 | xxx 万，7 大子系统 |
```

### 4c 工作占比看板（比例模型）

**重要**：用精力占比，不用时间模型！

| 步骤 | 操作 |
|:--|:--|
| 1 | 拆解工作项 → 标类型（🔁重复/🧠判断/💬沟通） |
| 2 | 标精力档位（🔴高×3/🟡中×2/🟢低×1） |
| 3 | 加权算占比：精力占比 = 该类型加权分 / 总分 |
| 4 | 双输出：Markdown（Mermaid 饼图）+ HTML（SVG 环形图） |

### 4d 生成 PPTX 汇报

用户说「做成 PPT」时：
1. 从项目 `.md` 提取 KPIs、清单表、时间线、结论金句
2. 用 `python-pptx` 生成暗色主题幻灯片（PREP 结构）
3. 输出到 `项目/<项目名> <标题>.pptx`
4. 飞书发送 `MEDIA:` 路径

---

## 场景五：日记管理与复盘

### 晨间规划

创建 `日记/YYYY-MM-DD.md`：
- 🐸 三只青蛙（最重要 3 件事）
- ⚡ 精力曲线（早/中/晚）
- 📋 任务池
- 时间块规划

### 晚间复盘

- 逐只青蛙确认状态（✅/🔄/❌）
- 统计任务池完成率
- 填写实际精力曲线
- 提取明日调整建议

### Apple Reminders 集成

```bash
remindctl today                              # 今日提醒
remindctl add --title "..." --due "..."       # 新建
remindctl complete <id>                      # 完成
```

---

## 场景六：知识蒸馏（每周六）

### 五步蒸馏法

| 步骤 | 内容 | 产出 |
|:--|:--|:--|
| ① | 扫描收件箱中 7 天内的笔记 | 待蒸馏清单 |
| ② | 补全 YAML（domain / depth / why_useful / layer） | 完整卡片 |
| ③ | 按领域分组，≤30 篇一批 | 批次分组 |
| ④ | 三层汇总 | L1/L2/L3 报告 |
| ⑤ | 移动笔记到 `知识库/<domain>/` | 归档 |

### 三层汇总体系

| 层次 | 名称 | 内容 |
|:--|:--|:--|
| **L1 信息层** | 主题深度报告 | 共性观点、高频概念、关键论据 |
| **L2 方法层** | 道法术图谱 | 道（底层逻辑）→ 法（方法论框架）→ 术（操作技巧） |
| **L3 行动层** | 个人行动指南 | 待办清单、决策原则、检查清单 |

### 🫵 用户洞察区

L3 末尾固定追加：

```markdown
## 🫵 用户洞察区
### ❓ 三问
1. 本周最有价值的 3 篇是哪三篇？
2. 哪个观点/方法论你想深入跟进？
3. AI 助手有没有漏掉什么连接或洞察？
### ✍️ 用户补充
```

**AI 写 80%，用户补 20% 独家洞察**——不是 AI 替你读，是 AI 训练你读。

---

## 场景七：费曼检验（每周日）

> 费曼学习法——能用人话说清楚，才是真懂了。

1. 从本周笔记中随机抽一篇
2. 出题让用户用白话解释（不准用原文，不准用术语）
3. AI 基于解释回答相关问题
4. 能力 → ✅ 通过；不能 → ❌ 指出漏洞，重新解释
5. 结果记录到 `feynman_result` 字段 + 「🎓 费曼检验」表格

---

## 场景八：检索与维护

### 三级检索导航

```
顶层（分类）→ 中层（标签）→ 底层（单篇卡片）
例：🤖 AI与科技 → #知识管理 → 蒸馏方法论
```

### 索引文件

- `📚 知识库索引.md` — 11 分类计数 + 检索入口
- `🗂️ 项目索引.md` — 活跃项目列表
- 每次增删笔记后**即时更新**索引——用 `patch`，不用 `write_file`

### 三通道连通性测试

```
🧠 Hermes Memory  → memory(add → remove) 验证
📂 Obsidian       → 写入收件箱/测试.md 验证
🍎 Apple Notes    → memo notes 验证
```

---

## 自动化 Cron

| Job ID | 名称 | 调度 | 加载 Skill |
|:--|:--|:--|:--|
| `<your-cron-job-id>` | 每周知识蒸馏 | `0 9 * * 6` (周六) | `leizong-knowledge-hub` + `obsidian` |
| `<your-feynman-cron-id>` | 每周费曼检验 | `0 9 * * 0` (周日) | `leizong-knowledge-hub` + `obsidian` |
| `<your-cleanup-cron-id>` | 收件箱周清理 | `0 6 * * 6` (周六) | `leizong-knowledge-hub` + `obsidian` |

**每周节奏**：
```
周六 06:00  收件箱自动清理
周六 09:00  蒸馏报告（L1+L2+L3 + 🫵 洞察区）
周日 09:00  费曼检验（抽考一周所学）
每次收藏    关联追问（3 题训练联想肌肉）
```

---

## 工具依赖

| 工具 | 用途 | 安装 |
|:--|:--|:--|
| Obsidian | 知识库本体 | 官网安装 |
| memo CLI | Apple Notes 操作 | `brew install memo` |
| remindctl | Apple Reminders | 内置 |
| pymupdf | PDF 文本提取 | `pip3 install --target ... pymupdf` |
| python-pptx | PPTX 生成 | `pip3 install --target ... python-pptx` |
| Jina AI Reader | 网页内容提取 | 免费，无需 API Key |
| parse_documents.py | docx/xlsx 解析 | 本 skill `scripts/` |

### memo 授权（常见阻断点）

```bash
# 若 memo 挂住不动：
# System Settings → Privacy & Security → Automation
# → 找到 memo → 打开 Notes 开关
```

---

## 评论区补充（人机协作）

> 视频号评论区是微信私域，外部无法程序化获取。

**替代方案**：
- 用户截屏 → 飞书发给 AI → OCR + 提取 → 补入 `💬 评论区精华` 区块
- 复制粘贴 → 直接贴飞书 → AI 追加到卡片
- 卡片 `comment_quality` 字段标记是否需要二次提取

---

## 设计原则

1. **追问优先于总结** — 先让用户思考，再给 AI 答案
2. **协作优于自动化** — AI 写 80%，用户补 20%
3. **检验优于阅读** — 费曼学习法是真懂的检验
4. **连接优于收集** — 用户手动建立连接，训练联想肌肉
5. **索引优于搜索** — 三级导航 > 全文搜索
6. **比例优于时间** — 精力占比 > 掐表计时

---

## 安装与配置

1. 将此 skill 放入 Hermes Agent 的 `skills/note-taking/` 目录
2. 设置环境变量 `OBSIDIAN_VAULT_PATH` 指向你的 Obsidian vault
3. 根据操作系统选择可选模块：
   - **macOS**：可选装 `memo` CLI（`brew install memo`）连接 Apple Notes
   - **跨平台**：核心功能仅需 Obsidian，无需额外依赖
4. 配置 cron job 时，将 `<your-*-cron-id>` 替换为你的实际 job ID

> 📢 **本文件为公开发布版** — 作者 [雷老板](https://github.com/nousresearch/hermes-agent)。私人 cron ID、本地路径已脱敏处理。
> 许可证：MIT
