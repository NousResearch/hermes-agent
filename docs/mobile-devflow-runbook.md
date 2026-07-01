# Hermes 手机端开发闭环 Runbook

## 一句话

以后遇到开发任务,优先记住一个入口:

```text
/devflow repo=/path/to/project task=你的任务
```

小白模式也支持:

```text
/devflow 我想做一个登录功能
```

规则很简单:

- 如果 Hermes 记得最近项目,就自动用最近项目,并在确认页显示 `Repo source: recent project`.
- 如果 Hermes 不知道项目在哪里,它会先追问项目文件夹,不会瞎跑。
- 登录、注册、账号、密码、权限、鉴权这类任务会自动升级为中风险,先走预算确认,再进 OpenSquilla/Kanban。

Hermes 会自动判断是直接小改,还是先让 OpenSquilla 审查,再交给 Kanban 智能体军团拆解推进。

## 三个入口

### /devflow: 默认入口

适合你不想判断复杂度时使用。

```text
/devflow repo=/Users/yuxiansheng/dev/devrun-smoke task=复杂项目:把 README 扩展成正式项目说明页,并设计后续开发计划
```

自动路线:

- 小任务: DevRun 直接执行。
- 复杂任务: OpenSquilla 快速预审 -> Kanban triage 卡 -> Kanban 自动拆解和分派。
- OpenSquilla 不可用: 仍建 Kanban triage 卡,但卡片里标注审查待补。

### /devrun: 小而明确的执行

适合低风险、范围清楚、可以直接改和测的任务。

```text
/devrun repo=/Users/yuxiansheng/dev/devrun-smoke task=修复 README 里的一个错别字
```

特点:

- 低风险会后台执行。
- 中高风险会走审查或手机确认。
- 不适合一上来就丢“大项目”“系统重构”“设计全链路”。

### /devreview: 只审查不修改

适合你只想听独立意见,不希望它动文件。

```text
/devreview repo=/Users/yuxiansheng/dev/devrun-smoke task=评估这个方案的需求、架构、测试、安全和验收风险,不要修改文件
```

输出角度:

- 需求审查
- 架构审查
- 测试审查
- 安全审查
- 最终验收

## 和 Kanban 军团的关系

DevFlow 不是替代 Kanban,而是 Kanban 前面的手机端总闸门。

```text
你在手机发 /devflow
  -> Hermes 判断路线
  -> OpenSquilla 做独立预审
  -> Hermes 创建 Kanban triage 总卡
  -> Kanban 军团自动拆子任务/分派/推进
  -> Telegram 自动收到后续进展
```

也就是说:

- DevFlow 负责入口、风险判断、审查、建总卡。
- OpenSquilla 负责独立审查和质疑。
- Kanban 军团负责拆解、派工、长期推进。
- DevRun 负责小任务直接执行。

## 安全边界

自动允许:

- 读文件
- 搜索代码
- 生成计划
- 小范围普通代码修改
- 运行测试
- 创建 Kanban triage 卡

必须手机确认:

- 服务重启
- 跨仓库写入
- 删除文件
- 数据库迁移
- 改密钥、token、`.env`、核心配置
- 批量大改
- `git commit` / `git push` / reset / rebase

禁止自动:

- 输出 secrets
- 绕过 allowlist
- 远程强杀进程
- 无备份修改 Hermes/OpenClaw/Kanban 核心配置

## 常用验证

查看 job:

```text
/devstatus dev_xxx
```

查看最近 job:

```text
/devstatus
```

查看 Kanban 卡:

```text
/kanban show t_xxx
```

取消 job:

```text
/devcancel dev_xxx
```

## 什么时候用哪条

用 `/devrun`:

- 改一个小 bug
- 改一段文案
- 跑一次只读检查
- 范围小、风险低、你希望快

用 `/devreview`:

- 只想审方案
- 想要第三方独立意见
- 不想让它改任何文件

用 `/devflow`:

- 你懒得判断
- 任务有点复杂
- 涉及多个模块
- 需要先审查再拆解
- 需要 Kanban 军团继续推进

## 失败时怎么看

OpenSquilla 失败:

- DevFlow 仍会建 Kanban 卡。
- 卡里会写明审查失败或待补。
- 后续可以再发 `/devreview` 单独补审。

Kanban 没动:

- 先 `/kanban show t_xxx` 看卡片状态。
- 如果还在 triage/todo,说明需要等自动分解或人工触发。

Hermes 没回复:

- 不要在 Windows 远程强杀或重启。
- 在 Mac-A 本机终端重启 Hermes gateway。

## Mac-A 重启命令

```bash
cd /Users/yuxiansheng/hermes-agent
./venv/bin/hermes gateway restart
```

## 手机端 smoke test

```text
/devflow repo=/Users/yuxiansheng/dev/devrun-smoke task=复杂项目:验证沉淀链路自动建卡与通知,不要修改文件
```

预期:

- 先返回 DevFlow job id。
- 后台完成后返回 DevFlow status update。
- Tests/结果里出现 `Created t_xxx`。
- 同一条结果里出现 `Subscribed to Kanban updates for t_xxx.`
- 之后可以 `/kanban show t_xxx` 查看军团拆解进展。

## DevFlow Budget Gate

Complex `/devflow` tasks must pass a mobile budget confirmation before
OpenSquilla or Kanban starts.

The confirmation message shows:

- estimated agents
- estimated MiniMax-M3 calls
- route and risk
- whether OpenSquilla is available

Cancel means no OpenSquilla/Kanban budget is spent.

Typical estimates:

- Small `/devflow` routed to DevRun: 1 agent, about 1-4 MiniMax-M3 calls.
- Complex `/devflow` routed to Kanban: about 2-6 agents, about 3-18 MiniMax-M3 calls.
- Full/end-to-end/production wording can increase the estimate.
