# A2A 多 Agent + 右栏浏览器隔离 + 抖音操控 逻辑地图

> 状态：2026-07-09 实测 + Phase1 落地版  
> 范围：Hermes Desktop Dev 产品（`Hermes Dev/hermes-home`）、右栏 `<webview>` 浏览器、Agent `browser_*` 工具、抖音精选页  
> 约束：Video Studio / MoneyPrinter 仍见 `docs/capabilities/video-studio-logic-map.md`；本文件补齐 **A2A 编排、浏览器会话隔离、抖音控制能力**。

---

## 0. 编号体系

| 前缀 | 类型 | 说明 |
| --- | --- | --- |
| `CAP-*` | 能力 | 用户/产品可感知能力 |
| `PAGE-*` | 页面 | Desktop UI 或站外页面区域 |
| `BTN-*` | 按钮 | 可点击控件 |
| `ACT-*` | 动作 | UI 或 Agent 工具触发的动作 |
| `FLOW-*` | 流程 | 多动作业务链路 |
| `DATA-*` | 数据 | 会话、Cookie、Brief、任务包等 |
| `API-*` | 接口 | 内部 API / IPC / 工具 schema |
| `RULE-*` | 规则 | 隔离、权限、安全、验收 |
| `TEST-*` | 测试 | 自动/手动验证项 |
| `AG-*` | Agent | Profile 级智能体 |
| `TOOL-*` | 工具 | Hermes tool / toolset |

---

## 1. 能力地图

| ID | 能力 | 描述 | 相关 |
| --- | --- | --- | --- |
| `CAP-001` | A2A 主编排 | `default` 总指挥：Brief → 派工 → 验收 → 汇报 | `AG-001`, `FLOW-001` |
| `CAP-002` | 策划/抄作业 Agent | 文案、分镜、prompt 包、不强制出片 | `AG-002`, `FLOW-002` |
| `CAP-003` | 视频执行 Agent | 多模型 `video_generate` / 图生视频 / 可选 MoneyPrinter | `AG-003`, `TOOL-003`~`TOOL-006` |
| `CAP-004` | 聊天-浏览器绑定 | 一个聊天 session 对应一套右栏浏览器状态 | `PAGE-002`, `DATA-001`, `RULE-001` |
| `CAP-005` | 登录态持久化（右栏） | Electron `persist:` partition 按 sessionId 存 Cookie | `RULE-002`, `TEST-003` |
| `CAP-006` | 独立关闭浏览器 | 关闭当前 session 浏览器不影响其他聊天 | `ACT-014`, `RULE-003` |
| `CAP-007` | Agent 浏览器控制 | `browser_navigate/click/type/snapshot/vision` | `TOOL-010` |
| `CAP-008` | 抖音精选浏览 | 打开 feed、读卡片、关登录墙、汇总指标 | `PAGE-010`, `FLOW-010` |
| `CAP-009` | 抖音互动（登录后） | 点赞/评论/打开评论区（需持久登录） | `FLOW-011`, `RULE-010` |
| `CAP-010` | AI 视频生成多模型 | xAI Grok Imagine + FAL 族 | `TOOL-003`, skill `hermes-video-generate` |
| `CAP-011` | 右栏 ↔ Agent 同步 | gateway `browser.drive` → Desktop 右栏跟随 | `API-003`, `FLOW-020` |

---

## 2. Agent 地图

| ID | Profile | 人格/职责 | 工具建议 | Skills |
| --- | --- | --- | --- | --- |
| `AG-001` | `default` | 主 Agent：编排、验收、对用户说话 | 广（含 delegation） | `hermes-video-generate`, `a2a-video-pipeline`, profiles skill |
| `AG-002` | `viral-video-agents` | 抄作业/文案/策划/改编/分镜 | browser/web/vision/file；**不出片为主** | 策划类 + `hermes-video-generate`（仅作 prompt 参考） |
| `AG-003` | `video-studio` | 出片执行：静帧+视频+可选剪辑侧车 | **必须** `image_gen`+`video_gen` | `hermes-video-generate`, `moneyprinter-video`, `a2a-video-pipeline` |
| `AG-004` | `browser-controller` | 浏览器/CDP 专精（隔离浏览器态） | browser/computer_use | browser 相关 |
| `AG-005` | `default-agent` | Hermes 自研副驾（非内容流水线） | 开发向 | 工程 skills |

**A2A 派工信封（`DATA-010`）**

```text
TO: <profile>
FROM: default
GOAL: ...
INPUT: ...
CONSTRAINTS: ...
OUTPUT: 可验证 URL/路径/JSON
ACCEPTANCE: ...
```

---

## 3. 页面地图

| ID | 页面/区域 | 文件或 URL | 说明 |
| --- | --- | --- | --- |
| `PAGE-001` | 聊天主栏 | Desktop chat surface | 每 session 一条会话 |
| `PAGE-002` | 右栏 Browser tab | `apps/desktop/src/app/chat/right-rail/browser-pane.tsx` | `<webview partition=persist:hermes-browser-{sessionId}>` |
| `PAGE-003` | 右栏 Preview/Files | layout / preview stores | 与 browser tab 互斥展示由 layout 管理 |
| `PAGE-004` | Session 列表 | sidebar | 切换 session → 切换浏览器记录 |
| `PAGE-010` | 抖音精选 | `https://www.douyin.com/jingxuan` | 未登录可见 feed 卡片与点赞数 |
| `PAGE-011` | 抖音登录墙 | 弹窗「登录后免费畅享高清视频」 | 拦截高清播放与写互动 |
| `PAGE-012` | 抖音视频详情/播放 | 登录后 / 部分点击后 | 点赞、评论、评论区 |

---

## 4. 按钮 / 控件地图（抖音 + Desktop）

| ID | 控件 | 页面 | 动作 |
| --- | --- | --- | --- |
| `BTN-001` | 右栏浏览器打开/地址栏 | `PAGE-002` | `ACT-010` |
| `BTN-002` | 关闭当前 session 浏览器 | `PAGE-002` | `ACT-014` |
| `BTN-010` | 抖音「登录」 | `PAGE-010` | 打开 `PAGE-011` |
| `BTN-011` | 登录墙关闭 X / Esc | `PAGE-011` | `ACT-021` |
| `BTN-012` | Feed 视频卡片 | `PAGE-010` | `ACT-022` 点击进入/触发登录墙 |
| `BTN-013` | 点赞（心形） | `PAGE-012` | `ACT-023` **需登录** |
| `BTN-014` | 评论入口 | `PAGE-012` | `ACT-024` **需登录** |
| `BTN-015` | 评论发送 | `PAGE-012` | `ACT-025` **需登录** |
| `BTN-016` | 分类 Tab（全部/美食…） | `PAGE-010` | `ACT-026` 切换 feed |

---

## 5. 动作地图

| ID | 动作 | 实现 | 输出 |
| --- | --- | --- | --- |
| `ACT-001` | 写 Execution Brief | `AG-001` | `DATA-010` |
| `ACT-002` | 派工 viral-video-agents | profile 调用 / kanban | 策划包 |
| `ACT-003` | 派工 video-studio | profile 调用 | 视频 URL |
| `ACT-004` | 验收 public_url | 打开/HEAD URL | pass/fail |
| `ACT-010` | 打开右栏浏览器 | `openBrowserRail` / preload browser API | `DATA-001` 记录 |
| `ACT-011` | 导航 URL | webview `loadURL` / `driveBrowser(navigate)` | url 更新 |
| `ACT-012` | Agent browser_navigate | `TOOL-010` | 工具结果 + 可选 `browser.drive` |
| `ACT-013` | Agent browser_click/type | `TOOL-010` | DOM 交互 |
| `ACT-014` | 关闭当前浏览器 session | `closeBrowserRailForSession` | 删除 `DATA-001[sessionId]` |
| `ACT-020` | 打开抖音精选 | navigate | `PAGE-010` |
| `ACT-021` | 关闭登录墙 | Esc / 点 X | 恢复 feed 浏览 |
| `ACT-022` | 点击视频卡片 | click | 常再次弹出 `PAGE-011`（未登录） |
| `ACT-023` | 点赞 | click like | **登录后** |
| `ACT-024` | 打开评论区 | click comment | **登录后** |
| `ACT-025` | 发表评论 | type + send | **登录后；谨慎自动化** |
| `ACT-026` | 抓取 feed 指标 | snapshot / console 文本 | `DATA-020` |
| `ACT-027` | 归纳数据 | LLM 汇总 | 结构化报告 |

---

## 6. 流程地图

### `FLOW-001` A2A 视频流水线（产品主路径）

```text
用户链接/主题
  → AG-001 建 Brief (ACT-001)
  → AG-002 分析+策划 (ACT-002) → 交付分镜/prompt
  → AG-003 出片 (ACT-003) → image_generate? → video_generate
  → AG-001 验收 URL (ACT-004) → 汇报用户
```

相关：`CAP-001`~`CAP-003`, skill `a2a-video-pipeline` / `hermes-video-generate`

### `FLOW-010` 抖音只读分析（未登录可测）

```text
ACT-020 打开 jingxuan
  → ACT-021 关登录墙（若挡视野）
  → ACT-026 抓卡片：标题、作者、时长、点赞量
  → ACT-027 归纳 top hooks / 话题 / 互动量级
```

**2026-07-09 实测样本（Agent 工具浏览器，未登录）**

| 标题摘要 | 作者线索 | 时长 | 互动量 |
| --- | --- | --- | --- |
| 回复 @冷忆冰… 天空一望无际 #无人之岛 | @饺子WTF | 04:37 | 10.7 万 |
| 当你穿进老钱班29 #遗忘之海 | @侯绿萝 | 02:59 | 136.9 万 |
| 我最爱吃猪脚饭啦 #老吃家 | @邢三狗 | 02:46 | 103.7 万 |
| 你们15个被我包围了 #三角洲行动 | @成成大王 | — | feed 可见 |

结论：`ACT-026`/`ACT-027` **可用**；点击卡片会再次触发 `PAGE-011`。

### `FLOW-011` 抖音写互动（登录后）

```text
用户在「目标 session 的右栏浏览器」扫码登录抖音
  → Cookie 落在 persist:hermes-browser-{sessionId}
  → ACT-022 打开视频
  → ACT-023 点赞 / ACT-024 看评论 / ACT-025 评论（可选）
  → ACT-027 汇总评论区主题
```

**未登录时**：`ACT-023`~`ACT-025` **阻塞**（`RULE-010`）。

### `FLOW-020` Agent 工具驱动右栏同步

```text
Agent TOOL-010 browser_navigate 成功
  → tui_gateway 发 browser.drive (API-003)
  → Desktop use-preview-routing 仅处理 active session
  → driveBrowser → BrowserPane webview
```

规则：非 active session 的 drive 事件忽略（`RULE-004`）。

---

## 7. 数据地图

| ID | 数据 | 存储 | 隔离粒度 |
| --- | --- | --- | --- |
| `DATA-001` | 右栏浏览器 URL/title 注册表 | `localStorage` key `hermes.desktop.browserRail.v1` | **per chat sessionId** |
| `DATA-002` | Electron session partition | `persist:hermes-browser-{encodedSessionId}` | **per chat sessionId**（Cookie/LocalStorage/IndexedDB） |
| `DATA-003` | Camofox 持久身份 | `$HERMES_HOME/browser_auth/camofox` + userId hash | **per Hermes profile**（非 per chat） |
| `DATA-010` | A2A Execution Brief | 会话消息 / 作业目录 JSON | per job |
| `DATA-020` | 抖音 feed 抽取记录 | Agent 输出 / job 文件 | per job |
| `DATA-030` | video_gen 配置 | `config.yaml` → `video_gen.provider/model` | per Hermes profile |
| `DATA-031` | image_gen 配置 | `config.yaml` → `image_gen.*` | per Hermes profile |

---

## 8. 接口 / 工具地图

| ID | 接口 | 说明 |
| --- | --- | --- |
| `API-001` | `window.hermesDesktop.browser.*` | Desktop preload：open/navigate/reload/back/forward/onDrive |
| `API-002` | `hermes:browser:drive` IPC | main → renderer 驱动右栏 |
| `API-003` | gateway event `browser.drive` | Python 工具完成 → Desktop |
| `TOOL-001` | toolset `image_gen` / `image_generate` | 静帧 |
| `TOOL-002` | toolset `video_gen` | 视频工具集 |
| `TOOL-003` | `video_generate` | 统一 T2V/I2V |
| `TOOL-004` | `xai_video_edit` | xAI 改片 |
| `TOOL-005` | `xai_video_extend` | xAI 续片 |
| `TOOL-010` | `browser_*` | navigate/click/type/snapshot/vision/console… |
| `TOOL-011` | `computer_use` | 系统级桌面控制（另一通道） |

### 视频模型（`TOOL-003` 后端）

| Provider | Models |
| --- | --- |
| `xai` | `grok-imagine-video` (T2V), `grok-imagine-video-1.5` (I2V) |
| `fal` | `ltx-2.3`, `pixverse-v6`, `veo3.1`, `seedance-2.0`, `kling-v3-4k`, `happy-horse` |

---

## 9. 规则地图

| ID | 规则 |
| --- | --- |
| `RULE-001` | **一聊天一浏览器状态**：`$browserSessionId` = active/selected session 或 `draft`。 |
| `RULE-002` | **登录持久化（右栏）**：`partition = persist:hermes-browser-{sessionId}` → 同 session 重启 Desktop 后 Cookie 可保留（Electron persist 分区）。 |
| `RULE-003` | **独立关闭**：`closeBrowserRailForSession(sessionId)` 只删该 key；其它 session 的 `DATA-001`/`DATA-002` 不受影响。 |
| `RULE-004` | **drive 仅 active session**：非当前聊天的 `browser.drive` 不更新可见右栏。 |
| `RULE-005` | **双浏览器栈**：右栏 webview（per-session）≠ Agent `browser_*`（Camofox/Browserbase 等，常 **per Hermes profile**）。产品文案需区分「用户看得见的窗」与「Agent 无头/远程浏览器」。 |
| `RULE-006` | **Chat 模型 ≠ 媒体模型**：出片走 `video_generate`，不把 session 模型切成 Imagine Video。 |
| `RULE-007` | **Toolset 变更需新会话**：enable `video_gen` 后 `/reset` 或新 chat。 |
| `RULE-008` | **子 Agent 交付必须可验证**：`public_url` / path / task_id。 |
| `RULE-009` | **Dev home 边界**：只动 `Hermes Dev/hermes-home`，不动生产 `~/.hermes`（除非用户明确要求）。 |
| `RULE-010` | **抖音写操作需登录**：未登录可只读 feed；点赞/评论/完整评论区需用户在**对应 session 右栏**完成登录。 |
| `RULE-011` | **谨慎自动化互动**：自动点赞/刷评论有平台风控与伦理风险；默认仅分析，写操作需用户明确授权。 |

---

## 10. 浏览器隔离结论（直接回答产品问题）

| 问题 | 结论 | 证据 |
| --- | --- | --- |
| 登录信息能否持久化？ | **能（右栏 webview）** | `browserPartitionForSession` → `persist:hermes-browser-…` |
| 一个聊天框一个浏览器？ | **能（状态 + partition 均按 sessionId）** | `browser.ts` registry + partition 函数 |
| 关掉一个窗口影响其他聊天？ | **不影响** | `closeBrowserRailForSession` 只删当前 key；`browser.test.ts` 有用例 |
| Agent 能否控制浏览器？ | **能** | `browser_*` + `browser.drive` 同步右栏（active session） |
| Agent 栈与右栏是否同一 Cookie 罐？ | **默认否** | Camofox identity 是 **profile 级**；右栏是 **chat session 级** → 抖音登录若在右栏完成，Agent 无头浏览器**不一定**带上同一登录，除非做成同一 partition/adoption（见产品缺口） |

### 产品缺口（建议后续）

| Gap | 说明 | 建议 |
| --- | --- | --- |
| `GAP-001` | Agent 浏览器与右栏 Cookie 未默认共享 | 方案：Desktop-owned browser 作为唯一可视 shell，Agent 工具驱动同一 webview；或 Camofox adoption 绑定 session |
| `GAP-002` | 无视频模型 pill | 镜像 image-generation-pill |
| `GAP-003` | 专家 profile 工具仍偏宽 | 按 AG 表裁剪 terminal/cron 等 |
| `GAP-004` | 抖音未登录无法验证点赞/评论路径 | 在目标 session 右栏人工登录后跑 `FLOW-011` |

---

## 11. 测试地图

| ID | 测试 | 方式 | 结果（2026-07-09） |
| --- | --- | --- | --- |
| `TEST-001` | session 浏览器 registry 隔离 | `browser.test.ts` | 代码层存在；单元测覆盖 |
| `TEST-002` | 关闭当前 session 浏览器不影响其他 | `browser.test.ts` | 代码层存在 |
| `TEST-003` | persist partition 命名稳定 | 读 `browser-pane.tsx` | `persist:hermes-browser-{sessionId}` |
| `TEST-004` | video-studio 启用 video_gen/image_gen | `hermes --profile video-studio tools list` | **pass**（均 enabled） |
| `TEST-005` | video-studio config 写入 xai | 读 config.yaml | **pass** |
| `TEST-006` | 打开抖音精选 | `browser_navigate` | **pass** |
| `TEST-007` | 未登录抓 feed 指标 | console/snapshot | **pass**（见 `FLOW-010` 表） |
| `TEST-008` | 关登录墙 | Esc | **pass**（可再弹出） |
| `TEST-009` | 点击视频后互动 | click card | **blocked**：强制 `PAGE-011`（Agent 栈） |
| `TEST-010` | 点赞/评论 | — | Agent 栈 **blocked**；右栏已登录待在右栏内点击验证 |
| `TEST-011` | video_generate 烟测 | 既有会话已成功出 8s 片 | **pass**（历史） |
| `TEST-012` | 右栏登录态（用户截图） | Hermes 右栏 `douyin.com/jingxuan` | **pass**：头像可见、无「登录」按钮；`Agent: Navigate` 徽章可见 |
| `TEST-013` | Agent `browser_*` 与右栏登录共享 | 同时间点 Agent 再开 jingxuan | **fail**：仍扫码墙 → 坐实 `GAP-001` / `RULE-005` |
| `TEST-014` | 登录态 feed 只读归纳 | 用户右栏截图 | **pass**：无人之岛 88.0万赞/04:37；西昊挑战 36:22 @我是庄小周；消息角标 10 |

---

## 12. 关键代码锚点

| 主题 | 路径 |
| --- | --- |
| per-session 浏览器状态 | `apps/desktop/src/store/browser.ts` |
| partition / webview | `apps/desktop/src/app/chat/right-rail/browser-pane.tsx` → `browserPartitionForSession` |
| 关闭当前浏览器 | `closeBrowserRailForSession` / `closeCurrentBrowserSession` |
| drive 同步配方 | skill `hermes-desktop-app-development` → `references/right-rail-browser-agent-sync.md` |
| Camofox 持久身份 | `tools/browser_camofox_state.py` |
| 视频工具 | `tools/video_generation_tool.py`, `plugins/video_gen/xai`, `plugins/video_gen/fal` |
| SOUL（Dev） | `…/hermes-home/SOUL.md`, `profiles/viral-video-agents/SOUL.md`, `profiles/video-studio/SOUL.md` |
| Skills | `skills/media/hermes-video-generate`, profile 下 `a2a-video-pipeline` |

---

## 13. Phase1 落地检查（本次）

| 项 | 状态 |
| --- | --- |
| `default` SOUL 改为主编排 | done |
| `viral-video-agents` SOUL 策划不出片 | done |
| `video-studio` SOUL 执行出片契约 | done |
| `video-studio` `image_gen`+`video_gen` config | done |
| `hermes-video-generate` 链到 video-studio skills | done（symlink） |
| 抖音只读分析 | done |
| 抖音点赞/评论 | blocked（需用户在目标聊天右栏登录） |

---

## 14. 下一步建议（按优先级）

1. **用户在「要用的那个聊天」右栏打开抖音并扫码登录** → 复跑 `FLOW-011` 验证点赞/评论/评论区抓取。  
2. 推进 `GAP-001`：让 Agent 控制与右栏 **同一 Cookie 罐**，否则「你看到已登录、Agent 仍未登录」。  
3. Phase2：裁剪 AG-002/AG-003 工具面 + Desktop 视频 pill。  
4. 把本逻辑地图挂进 Desktop 产品文档索引（`docs/product/hermes-dev-history-index.md`）。
