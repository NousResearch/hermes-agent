# 交接:画布右侧 = 爱马仕 copilot(聊天气泡 UI)

> 目标:langflow 画布右侧放一个**爱马仕(EasyHermes)copilot**,要**聊天气泡 UI**(不要 xterm 终端),
> 能与本机 agent 对话、并经 MCP 驱动画布。**当前只做桌面端**(web 暂时不管)。

## 已完成(均在磁盘上,未提交)

### langflow `/Volumes/D/kari-all/langflow`(已重建前端 + 重启,运行在 :7860)
- **根因修复**:旧画布右侧面板 `KariCopilotPanel` 连的是**已废弃的独立 copilot `ws://127.0.0.1:3500`** → "连接中断"。
- `src/frontend/src/kari/mode.ts`:`copilotWsUrl()`/`copilotHttpBase()` 改为**同源**(`/kari/bridge`、`/kari/copilot`,即本机 langflow 7860),永不再连 3500。
- 新增 `src/frontend/src/kari/CanvasBridgeMount.tsx`:只起 `startCanvasBridge`(接收爱马仕经 MCP `/kari/canvas/op` 发来的画布操作并执行;op 执行逻辑本就在 `canvasBridgeClient.ts` 内,与聊天无关)。无聊天 UI。
- `src/frontend/src/pages/FlowPage/index.tsx`:`KariCopilotPanel` → `CanvasBridgeMount`。
- **`KariCopilotPanel.tsx` 已删除**(备份在 `/tmp/kari-backup/KariCopilotPanel.tsx`)。它本身就是一个**自包含的聊天气泡 UI**(用户/助手气泡、thinking 气泡、流式),原先用 `canvasBridgeClient` 的 WS 收发 `user_message`/`assistant_token`。**可作为气泡 UI 的现成蓝本**,把它的收发改接爱马仕即可。
- 重建命令:`cd langflow && rm -rf src/frontend/node_modules/.vite src/frontend/build && CI='' make build_frontend`(vite 会缓存旧构建,务必清 `.vite`)。验证服务的 bundle 里无 `连接中断`/`3500`,只有 `/kari/bridge`。
- **缓存坑**:画布在 Electron `<webview>`(partition `persist:hermes-workflow`)。重建后必须清
  `~/Library/Application Support/Hermes/Partitions/hermes-workflow/{Cache,Code Cache,Service Worker}` 再重启桌面端,否则一直显示旧的坏面板。

### 桌面端 `apps/desktop/src/app/workflow/index.tsx`(**未编译,这就是要改的文件**)
当前 `WorkflowView` 已加:
- `getConnection()` 拿本机 dashboard `baseUrl` → 右栏 webview 指向 `${baseUrl}/chat`。
- 折叠逻辑:`chatOpen`(默认 false=收起)、`chatMounted`(首次展开才挂载)、右上角 `Codicon` 开关按钮、`w-0↔w-[420px]` 折叠容器(内层固定 420px、`overflow-hidden`)。
- **问题(用户否决了这个方案)**:`${baseUrl}/chat` 是 **web 仪表盘的 xterm 终端 TUI**,在窄 webview 里渲染**空白**(只剩顶部图标 + 底部状态栏,中间对话区空)。用户要的是**聊天气泡**,不是终端。

### web(用户说先不管)
- `web/src/pages/WorkflowPage.tsx` "ready" 已改成 `画布 iframe + /chat iframe` 分栏,已构建。同样是 xterm,先放着。

## 待做(新会话的任务):桌面端右栏换成聊天气泡 UI

**核心问题:气泡 UI 怎么跟爱马仕 agent 对话?**(必须是 agent,不是裸 LLM —— agent 才有 `kari_canvas`/`kari_org` 等 MCP,能驱动画布)

三条候选路线,**建议按此优先级评估**:

1. **复用桌面原生 `ChatView`(`apps/desktop/src/app/chat/index.tsx`)** —— 它就是 assistant-ui 的**气泡聊天**(`@/components/assistant-ui/thread`、`@assistant-ui/react`、`ThreadMessage`),接 gateway,是"真·爱马仕"。
   - 难点:`ChatViewProps` 要 ~20 个回调(`gateway`、`onSubmit`、`onThreadMessagesChange`、`onSteer`、`onReload`…),全由 `app/desktop-controller.tsx` 用 `useGatewayBoot`/`useGatewayRequest` 等装配。要在 workflow 路由里再挂一个会与主 ChatView 的会话/runtime 冲突。
   - 评估点:能否让"工作流路由"在 shell 层做成 `[画布 | 现有 ChatView]` 分栏(复用同一个 ChatView 实例/runtime),而不是在 WorkflowView 内部再挂一个。看 `desktop-controller.tsx:970` 附近 `<ChatView .../>` 的装配 + `app/routes.ts` 路由模型。

2. **以 `/tmp/kari-backup/KariCopilotPanel.tsx` 为蓝本**做一个自包含气泡面板,但把收发从"canvasBridgeClient WS"改接**爱马仕 agent**(走 gateway `/api/ws` JSON-RPC,或一个新的流式聊天端点)。需要搞清 gateway 的发消息/收 token 协议(看 `apps/desktop/src/app/gateway/*` 与 `web/src/lib/gatewayClient.ts`、`web/src/components/ChatSidebar.tsx`)。

3. **新写一个极简气泡组件**,直接用 gateway client 发消息 + 订阅 `/api/events` 流式渲染气泡。最可控但要自己接协议。

**driving 画布**:气泡 agent 用的是本机爱马仕,已配 `~/.hermes/config.yaml` 的 `mcp_servers.kari_canvas`(+ `kari_org`)。只要对话走的是 agent(gateway),它就能调 `kari_canvas` 操作画布(画布 webview 已通过 `CanvasBridgeMount` 同源连上 `/kari/bridge`,等着收 op)。

## 运行中的服务(本机)
- kari-cloud `:8900`(管理端/计费/能力登记/org 扇出)
- langflow `:7860`(画布,已是 clean bundle)
- dashboard `:9119`(`hermes dashboard`,org responder 在跑)
- 桌面端 Electron 在跑(但加载的是上一版构建)

## 相关记忆
`[[kari-canvas-mcp]]`、`[[easyhermes-local-first-architecture]]`、`[[kari-org-capability-fanout]]`。
未做完:org 子账号扇出**单机演示**(用户之前选了要做,被这个 UI 问题打断)。
