# AI Agent 领域知识报告

**日期**：2026-06-20
**主题**：AI Agent 领域最新动态与趋势

---

## 一、本周核心事件

### 1. Anthropic 发布新一代旗舰模型 Claude Fable 5 / Mythos 5（6/9）

Anthropic 于 6 月 9 日正式发布 **Claude Fable 5** 与 **Claude Mythos 5** 两款新模型，主打更强的 agentic 任务能力与长程一致性。这是 Opus 4.8（5/28）之后又一次重大迭代。

- **Artificial Analysis Intelligence Index（综合智能）** 排名：Fable 5（with fallback）位列第一，得分约 60；GPT-5.5（xhigh）位列第二；GLM-5.2（max）位列第三。
- **Vals Index（行业加权）**：Fable 5 75.1%，Opus 4.8 70.4%，GPT 5.5 68.0%。
- 6 月 12 日美国政府对 Fable 5 与 Mythos 5 实施出口管制暂停访问。

来源：Anthropic Newsroom · Artificial Analysis

### 2. 长horizon Agent 基准成为新焦点：AA-Briefcase 上线

Artificial Analysis 推出全新私有基准 **AA-Briefcase**，专门评估"长horizon 知识工作"（long-horizon knowledge work）——即让 agent 在长时窗口内自主完成多步骤、跨工具、跨应用的任务。

Intelligence Index v4.1 同时更新了：
- **GDPval-AA V2**：评估 agent 在真实经济任务中的表现
- **𝜏³-Banking**：金融场景多轮 agent 评估
- **Terminal-Bench v2.1**：终端/命令行 agent 评估

行业趋势：从"短任务能不能完成"转向"长任务是否可靠、自主、可监督"。

### 3. Vals AI 行业 Benchmark 排名（6/17 更新）

- **Legal（Harvey 法律 agent）**：Fable 5 11.3%，Opus 4.8 9.6%，GLM 5.2 7.1%
- **Finance Agent v2（新增）**：Gemini 3.5 Flash 57.9%，Fable 5 56.3%，Opus 4.8 53.9%
- **MedScribe（医疗行政）**：Fable 5 88.5%，GPT 5.1 88.1%，MiniMax-M3 87.3%
- **ProofBench（数学证明）**：Fable 5 77.0%，Opus 4.8 69.0%，GPT 5.4（xhigh）56.0%
- **LegalBench**：Fable 5 88.6%，Gemini 3.1 Pro Preview 87.4%，Gemini 3 Pro 87.0%

---

## 二、研究论文亮点

### 4. Project Fetch: Phase two（Anthropic, 6/18）

Anthropic 发布的"Project Fetch"第二阶段研究——让 Claude agent 在受限环境中自主完成复杂的多步任务。研究聚焦于多 agent 协作、长时间自主运行下的可靠性与对齐问题。是公司"agentic misalignment"（5/8）研究系列的延续。

### 5. Project Deal（Anthropic, 4/24）

在旧金山办公室内建立真实交易市场，让多个 Claude agent 互相进行买卖议价（buying / selling / negotiating）。用于研究多 agent 议价、市场均衡、欺骗行为与价格发现。这是少有的"真实市场环境下的多 agent 经济学实验"。

### 6. Project Vend: Phase two（Anthropic, 12/18/2025）

让 Claude 经营旧金山办公室的小卖部——自由形式实验，评估 agent 在真实、长期、多步骤商业任务中的表现。第二阶段报告了 agent 在长程任务中遭遇的失败模式与改进方向。

### 7. Mapping AI-enabled Cyber Threats（Anthropic, 6/3）

Anthropic Frontier Red Team 公布 **LLM ATT&CK Navigator**——将 AI 驱动的网络威胁映射到 MITRE ATT&CK 框架，分析 LLM 在 N-day 漏洞利用（6/8）、新 exploit 开发（5/22）中的能力。是 agent 安全研究的重要里程碑。

### 8. Agents in Biology & Chemistry（Anthropic, 6/5–6/8）

- **Making Claude a chemist（6/5）**：评估 Claude 在化学研究（实验设计、合成路线、文献综述）中的 agent 能力。
- **Paving the way for agents in biology（6/8）**：将 agent 能力扩展到生物学研究流程。

方向：科学发现 agent（scientific discovery agent）成为新热点。

---

## 三、产品与生态动态

### 9. Claude Cowork（Anthropic, 1/2026）

桌面优先（desktop-first）的 agent 产品：在沙箱 Linux VM 中执行任务，可访问本地文件、Excel 等常见文件类型，通过 computer use / MCP / 浏览器扩展连接外部工具。定价 $20–200/月。

Artificial Analysis 的"General Work Agents"对比表将其列为首选——与 ChatGPT Agent、Microsoft Copilot、Google Workspace Studio 并列。

### 10. ChatGPT Agent（OpenAI, 7/2025 → 持续更新）

由 GPT-5.4 Thinking / Pro 驱动，运行在云 VM 上，原生具备 computer-use 能力，支持最长约 30 分钟的自主任务执行。是 Operator 的继任者，融合了 Codex 编程能力与 Deep Research。

### 11. OpenClaw & Hermes Agent 等开源 agent 平台

- **OpenClaw**（Peter Steinberger, 11/2025）：开源自托管，将 WhatsApp/Slack/Teams 等聊天应用转化为 autonomous AI 助手，支持社区 Skills 市场。
- **Hermes Agent**（Nous Research, 2/2026）：开源自托管 agent 平台，具备**跨会话持久记忆**与**自演化技能系统**——可从经验中创建程序性技能（procedural skills）并跨会话复用，内置 40+ 工具（image gen、TTS、vision、Telegram/Discord/Slack/WhatsApp 等）。

两者代表了"开源 + 自托管 + 跨平台聊天集成"的 agent 平台趋势。

### 12. Anthropic 合作伙伴生态扩张（6 月）

- **TCS**（塔塔咨询服务）：将 Claude 引入受监管行业
- **DXC**：集成 Claude 到银行、航空、受监管行业的核心系统
- **首尔办公室 + 韩国生态合作**（6/17）
- **Claude Partner Network 新增 Services Track + Partner Hub**（6/3）
- **Project Glasswing 扩展到 150+ 新组织、15+ 国家**（6/2）

agent 不再只是开发者玩具，正在金融、医疗、政府等**受监管行业**快速落地。

---

## 四、当前顶级模型全景（Artificial Analysis 2026-06）

| 排名 | 模型 | 类型 | 备注 |
|---|---|---|---|
| 1 | Claude Fable 5 (with fallback) | Reasoning | 综合智能第一 |
| 2 | GPT-5.5 (xhigh) | Reasoning | OpenAI 当前旗舰 |
| 3 | GLM-5.2 (max) | Reasoning | 智谱 |
| 4 | Gemini 3.1 Pro Preview | Reasoning | Google DeepMind |
| 5 | MiniMax-M3 | Reasoning | MiniMax |
| 6 | DeepSeek V4 Pro (Max) | Reasoning | 深度求索 |
| 7 | Muse Spark | Reasoning | — |
| 8 | Kimi K2.6 | Reasoning | 月之暗面 |
| 9 | Nemotron 3 Ultra | Reasoning | NVIDIA |
| 10 | Grok 4.3 (high) | Reasoning | xAI |
| 11 | gpt-oss-120b (high) | Reasoning | OpenAI 开源 |

**输出速度 Top**：gpt-oss-120b（341 tok/s）、Nemotron 3 Ultra（171）、Grok 4.3（142）—— 开源 + 蒸馏小模型正在输出端逼近甚至超过闭源旗舰。

---

## 五、值得关注的趋势

1. **长horizon Agent 成为评估新前沿**：从单轮 tool-use 转向 multi-hour 自主任务。AA-Briefcase、GDPval-AA V2、𝜏³-Banking 等新基准涌现。

2. **行业垂直基准崛起**：Vals AI 在法律、医疗、金融、化学等垂直领域建立私有 agent 评估——通用 benchmark 已无法区分 agent 的真实工业价值。

3. **多 agent 经济学/市场实验**：Anthropic 的 Project Deal、Project Fetch、Vend 代表 agent 研究的新方向——让 agent 在真实经济环境中交互、博弈、协作。

4. **Agent 安全与对齐进入工程化阶段**：Anthropic 的 "Teaching Claude why"（减少 agentic misalignment）、LLM ATT&CK Navigator（AI 网络威胁建模）显示 agent safety 已从论文走向工具化、流程化。

5. **科学发现 Agent 成为新赛道**：化学、生物学方向的 agent 研究密集发布，LLM-for-Science 从问答走向"自主实验设计 + 工具调用"。

6. **开源 + 自托管 + 跨平台聊天集成**：Hermes Agent、OpenClaw 等项目反映了一种**去中心化 agent 部署**趋势——用户对自己的 agent、数据、技能市场拥有完整控制。

7. **受监管行业大规模采用**：TCS、DXC、Project Glasswing 等合作显示 agent 正在跨越"demo 阶段"，进入银行、医疗、政府等高门槛行业。

8. **速度 vs 智能的权衡**：开源小模型（gpt-oss-120b、Nemotron 3 Ultra）在输出速度上明显领先闭源旗舰，但智能分仍有差距——这是 agent 落地的核心成本/性能权衡。

---

## 六、推荐阅读

- [Anthropic Research Hub](https://www.anthropic.com/research) — 持续更新的 agent 安全/对齐/经济学研究
- [Artificial Analysis](https://artificialanalysis.ai/) — 模型与 agent 评测
- [Vals AI Benchmarks](https://www.vals.ai/benchmarks) — 行业垂直 agent 评估
- arXiv cs.MA（多 agent 系统）— 最新论文每日更新

---

*报告生成于 2026-06-20，所有数据均来自上述权威源在当日可访问的公开内容。*
