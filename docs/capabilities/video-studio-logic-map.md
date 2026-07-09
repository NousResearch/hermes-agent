# Video Studio 逻辑地图

> 状态：MVP 实施版  
> 范围：Hermes Desktop 原生 Video Studio 页面 + MoneyPrinterTurbo adapter API  
> 约束：Video Studio 是 Desktop capability，不是 Hermes Agent Core 的一部分。

## 0. 编号体系

| 前缀 | 类型 | 说明 |
| --- | --- | --- |
| `CAP-*` | 能力 | 用户可感知的产品能力 |
| `PAGE-*` | 页面 | Desktop 页面/区域 |
| `BTN-*` | 按钮 | 用户可点击控件 |
| `ACT-*` | 动作 | UI 或 API 触发的动作 |
| `FLOW-*` | 流程 | 多动作组成的业务链路 |
| `DATA-*` | 数据 | 表单、任务、输出、配置等数据对象 |
| `API-*` | 接口 | Hermes adapter API 或上游 MoneyPrinter API |
| `RULE-*` | 规则 | 架构、安全、交互和边界规则 |
| `TEST-*` | 测试 | 自动/手动回归测试项 |

---

## 1. 能力地图

| ID | 能力 | 描述 | 页面 | 流程 | API | 测试 |
| --- | --- | --- | --- | --- | --- | --- |
| `CAP-001` | Video Studio 工作台 | 在 Hermes Desktop 内提供原生 React 视频生成页面 | `PAGE-001` | `FLOW-001` | `API-001`~`API-006` | `TEST-001`~`TEST-006` |
| `CAP-002` | MoneyPrinterTurbo sidecar | 将 `external/MoneyPrinterTurbo` 作为独立服务运行 | `PAGE-001` | `FLOW-002` | `API-001`, `API-002` | `TEST-003`, `TEST-004` |
| `CAP-003` | 高频视频生成链路 | 复刻主题、文案、语音、素材、字幕、合成、预览主链路 | `PAGE-002`~`PAGE-004` | `FLOW-003` | `API-003`~`API-006` | `TEST-001`, `TEST-002` |
| `CAP-004` | 任务状态管理 | 展示任务 id、状态、进度、错误和输出 | `PAGE-003` | `FLOW-004` | `API-004`, `API-005` | `TEST-005` |
| `CAP-005` | 输出预览 | 展示脚本、视频预览、下载入口 | `PAGE-004` | `FLOW-005` | `API-005` | `TEST-006` |

---

## 2. 页面地图

| ID | 页面/区域 | 文件 | 包含按钮 | 读写数据 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `PAGE-001` | Video Studio 页面壳 | `apps/desktop/src/app/video-studio/index.tsx` | `BTN-001`, `BTN-002` | `DATA-001`, `DATA-005` | 页面标题、流程说明、服务状态入口 |
| `PAGE-002` | 生成配置表单 | `apps/desktop/src/app/video-studio/index.tsx` | `BTN-003` | `DATA-001`, `DATA-002` | 输入主题、文案、比例、数量、素材、语音、字幕配置 |
| `PAGE-003` | 任务列表 | `apps/desktop/src/app/video-studio/index.tsx` | `BTN-004`, `BTN-005` | `DATA-003` | 展示和选择 MoneyPrinter 任务 |
| `PAGE-004` | 预览面板 | `apps/desktop/src/app/video-studio/index.tsx` | `BTN-006` | `DATA-003`, `DATA-004` | 展示脚本、状态、错误、视频输出 |
| `PAGE-005` | 左侧导航入口 | `apps/desktop/src/app/chat/sidebar/index.tsx` | `BTN-007` | `DATA-006` | 新增 Video Studio nav item |

---

## 3. 按钮地图

| ID | 按钮 | 所在页面 | 触发动作 | 接口 | 规则 |
| --- | --- | --- | --- | --- | --- |
| `BTN-001` | Health Check | `PAGE-001` | `ACT-001` | `API-001` | `RULE-001`, `RULE-004` |
| `BTN-002` | Start Service | `PAGE-001` | `ACT-002` | `API-002` | `RULE-002`, `RULE-004` |
| `BTN-003` | Generate Video | `PAGE-002` | `ACT-003` | `API-003` | `RULE-003`, `RULE-004` |
| `BTN-004` | Refresh | `PAGE-003` | `ACT-004` | `API-004` | `RULE-004` |
| `BTN-005` | Task Row | `PAGE-003` | `ACT-005` | 无 | `RULE-005` |
| `BTN-006` | Download | `PAGE-004` | `ACT-006` | adapter download URL（后续补齐） | `RULE-006` |
| `BTN-007` | Sidebar Video Studio | `PAGE-005` | `ACT-007` | 无 | `RULE-001` |

---

## 4. 动作地图

| ID | 动作 | 输入 | 输出 | 触发方 | 后续流程 |
| --- | --- | --- | --- | --- | --- |
| `ACT-001` | 查询 MoneyPrinter 健康状态 | 无 | `DATA-005` | `BTN-001` | `FLOW-002` |
| `ACT-002` | 请求启动 MoneyPrinter 服务 | 无 | `DATA-005` | `BTN-002` | `FLOW-002` |
| `ACT-003` | 创建视频任务 | `DATA-001` | `DATA-003` | `BTN-003` | `FLOW-003`, `FLOW-004` |
| `ACT-004` | 刷新任务列表 | 无 | `DATA-003[]` | `BTN-004` | `FLOW-004` |
| `ACT-005` | 选择任务 | task id | selected task | `BTN-005` | `FLOW-005` |
| `ACT-006` | 下载输出 | output url | media file | `BTN-006` | `FLOW-005` |
| `ACT-007` | 打开 Video Studio 页面 | route | `/video-studio` | `BTN-007` | `FLOW-001` |

---

## 5. 流程地图

### `FLOW-001` 页面进入流程

```text
BTN-007 → ACT-007 → routes.ts 解析 /video-studio → DesktopController lazy load VideoStudioView → PAGE-001
```

相关对象：`DATA-006`  
规则：`RULE-001`

### `FLOW-002` 服务健康/启动流程

```text
BTN-001/BTN-002 → ACT-001/ACT-002 → API-001/API-002 → capabilities.moneyprinter.adapter → external/MoneyPrinterTurbo
```

相关对象：`DATA-005`  
规则：`RULE-002`, `RULE-004`

### `FLOW-003` 视频生成流程

```text
PAGE-002 表单 → BTN-003 → ACT-003 → DATA-001 → DATA-002 → API-003 → MoneyPrinter /api/v1/videos → task id → DATA-003
```

MoneyPrinter 内部链路：

```text
主题 → 文案 → terms → TTS 语音 → 素材 → 字幕 → moviepy/ffmpeg 合成 → outputs
```

相关对象：`DATA-001`, `DATA-002`, `DATA-003`, `DATA-004`  
规则：`RULE-003`, `RULE-006`

### `FLOW-004` 任务刷新流程

```text
BTN-004 → ACT-004 → API-004 → MoneyPrinter /api/v1/tasks → DATA-003[] → PAGE-003
```

相关对象：`DATA-003`  
规则：`RULE-004`, `RULE-005`

### `FLOW-005` 预览流程

```text
BTN-005 → ACT-005 → selectedTask → PAGE-004 → video.streamUrl/downloadUrl → BTN-006
```

相关对象：`DATA-003`, `DATA-004`  
规则：`RULE-006`

---

## 6. 数据地图

| ID | 数据 | TypeScript/Python 位置 | 生产方 | 消费方 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `DATA-001` | `VideoGenerationForm` | `apps/desktop/src/app/video-studio/moneyprinter-client.ts` | `PAGE-002` | `ACT-003` | Hermes UI 表单模型，camelCase |
| `DATA-002` | `CreateVideoPayload` | `apps/desktop/src/app/video-studio/moneyprinter-client.ts` | `toCreateVideoPayload` | `API-003` | MoneyPrinter 请求模型，snake_case |
| `DATA-003` | `MoneyPrinterTask` | `apps/desktop/src/app/video-studio/moneyprinter-client.ts` / `capabilities/moneyprinter/adapter.py` | adapter | `PAGE-003`, `PAGE-004` | UI 稳定任务模型 |
| `DATA-004` | `MoneyPrinterVideoOutput` | `moneyprinter-client.ts` / `adapter.py` | adapter | `PAGE-004` | 视频文件、stream/download URL |
| `DATA-005` | `MoneyPrinterHealth` | `moneyprinter-client.ts` / `adapter.py` | `API-001`, `API-002` | `PAGE-001` | 服务安装、运行、storage、upstream commit |
| `DATA-006` | route metadata | `apps/desktop/src/app/routes.ts` | Desktop route config | Sidebar / Controller | `/video-studio` 路由、`video-studio` view id |

---

## 7. 接口地图

| ID | Hermes Adapter API | 方法 | 上游 API | 使用方 | 状态 |
| --- | --- | --- | --- | --- | --- |
| `API-001` | `/api/capabilities/moneyprinter/health` | `GET` | MoneyPrinter `/docs` reachability | `BTN-001` | 已实现 |
| `API-002` | `/api/capabilities/moneyprinter/service/start` | `POST` | local process start | `BTN-002` | 已实现基础版 |
| `API-002A` | `/api/capabilities/moneyprinter/config` | `GET/POST` | local ignored `config.toml` | config panel | 已实现；密钥不回传 |
| `API-003` | `/api/capabilities/moneyprinter/videos` | `POST` | `/api/v1/videos` | `BTN-003` | 已实现基础代理 |
| `API-004` | `/api/capabilities/moneyprinter/tasks` | `GET` | `/api/v1/tasks` | `BTN-004` | 已实现基础代理 |
| `API-005` | `/api/capabilities/moneyprinter/tasks/{task_id}` | `GET` | `/api/v1/tasks/{task_id}` | 后续轮询/详情 | 已实现基础代理 |
| `API-006` | `/api/capabilities/moneyprinter/tasks/{task_id}` | `DELETE` | `/api/v1/tasks/{task_id}` | 后续删除按钮 | 已实现基础代理 |
| `API-007` | `/api/capabilities/moneyprinter/{stream|download}/{file_path}` | `GET` | `/api/v1/stream`, `/api/v1/download` | `BTN-006` | 已实现代理 |

---

## 8. 规则地图

| ID | 规则 | 说明 | 覆盖对象 |
| --- | --- | --- | --- |
| `RULE-001` | 不触碰 Hermes Agent Core | Video Studio 只新增 Desktop route/page/API adapter，不改 Agent 主循环和核心 tool-call 架构 | `CAP-001`, `PAGE-005` |
| `RULE-002` | MoneyPrinterTurbo 独立 sidecar | 上游项目运行在独立 Python 进程，Hermes 只代理 API | `CAP-002`, `API-002` |
| `RULE-003` | UI 模型与上游 payload 分离 | `VideoGenerationForm` 使用 UI 友好字段，提交前转换为 MoneyPrinter snake_case | `DATA-001`, `DATA-002`, `TEST-001` |
| `RULE-004` | adapter 统一错误 envelope | Desktop 页面只消费 `{ok,data,error}`，不直接暴露上游错误结构 | `API-001`~`API-006` |
| `RULE-005` | 任务状态前端只展示 adapter 稳定模型 | 上游 task shape 变化由 adapter 吸收 | `DATA-003` |
| `RULE-006` | 输出路径必须经 adapter 白名单/stream/download | 不让 renderer 直接读取任意 MoneyPrinter 本地路径 | `DATA-004`, `API-007` |
| `RULE-007` | 密钥不进入 repo | MoneyPrinter config、API keys、storage 均由 `.gitignore` 排除或运行时配置 | `.gitignore`, `external/MoneyPrinterTurbo` |

---

## 9. 测试地图

| ID | 测试 | 命令/方式 | 覆盖能力 | 当前结果 |
| --- | --- | --- | --- | --- |
| `TEST-001` | UI 表单到 MoneyPrinter payload 映射 | `npm --prefix apps/desktop run test:ui -- src/app/video-studio/moneyprinter-client.test.ts` | `DATA-001`, `DATA-002`, `RULE-003` | 通过 |
| `TEST-002` | capability API path namespace | 同上 | `API-001`~`API-006`, `RULE-004` | 通过 |
| `TEST-003` | Python adapter 语法检查 | `python3 -m py_compile capabilities/moneyprinter/adapter.py gateway/platforms/api_server.py` | `API-001`~`API-006` | 通过 |
| `TEST-004` | Desktop typecheck | `npm --prefix apps/desktop run typecheck` | `PAGE-001`~`PAGE-005` | 通过 |
| `TEST-005` | Targeted Desktop lint | `npx eslint src/app/video-studio src/app/routes.ts src/app/chat/sidebar/index.tsx src/app/desktop-controller.tsx src/app/types.ts` | 新增 TS/TSX 代码风格 | 通过；完整 lint 当前仍有既有非本次改动错误 |
| `TEST-006` | Desktop build | `npm --prefix apps/desktop run build` | 路由、bundle、类型集成 | 通过；有既有 dirty-tree/CSS/chunk-size warnings |

---

## 10. 当前实现文件清单

| 文件 | 编号覆盖 | 说明 |
| --- | --- | --- |
| `.gitignore` | `RULE-007` | 忽略 MoneyPrinter runtime state |
| `external/MoneyPrinterTurbo/` | `CAP-002` | vendored upstream source，固定 commit `63113a3` |
| `capabilities/moneyprinter/upstream.json` | `CAP-002`, `RULE-002` | upstream 元数据 |
| `capabilities/moneyprinter/adapter.py` | `API-001`~`API-006`, `RULE-004` | Hermes adapter API 实现 |
| `hermes_cli/web_server.py` | `API-001`~`API-007` | Desktop `hermes serve` / dashboard backend route 注册 |
| `gateway/platforms/api_server.py` | `API-001`~`API-006` | 注册 Desktop capability API routes |
| `apps/desktop/src/app/routes.ts` | `DATA-006`, `PAGE-005` | 新增 `/video-studio` route/view |
| `apps/desktop/src/app/chat/sidebar/index.tsx` | `BTN-007`, `PAGE-005` | 新增 sidebar 入口 |
| `apps/desktop/src/app/desktop-controller.tsx` | `FLOW-001` | lazy load Video Studio page |
| `apps/desktop/src/app/video-studio/moneyprinter-client.ts` | `DATA-001`~`DATA-005` | 前端 API client 和模型转换 |
| `apps/desktop/src/app/video-studio/index.tsx` | `PAGE-001`~`PAGE-004` | React 页面首版 |
| `apps/desktop/src/app/video-studio/moneyprinter-client.test.ts` | `TEST-001`, `TEST-002` | Vitest 回归测试 |
