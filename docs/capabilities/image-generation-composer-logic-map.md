# Image Generation Composer Picker 逻辑地图

> 状态：MVP 实施版  
> 范围：Hermes Desktop 对话框内的图片生成后端/模型切换入口 + dashboard API  
> 约束：复用既有 `image_gen` tool/provider registry，不把图片生成业务逻辑塞进 Desktop renderer。

## 0. 编号体系

| 前缀 | 类型 | 说明 |
| --- | --- | --- |
| `CAP-*` | 能力 | 用户可感知的产品能力 |
| `PAGE-*` | 页面 | Desktop 页面/区域 |
| `BTN-*` | 按钮 | 用户可点击控件 |
| `ACT-*` | 动作 | UI 或 API 触发的动作 |
| `FLOW-*` | 流程 | 多动作组成的业务链路 |
| `DATA-*` | 数据 | 配置、provider、模型和状态对象 |
| `API-*` | 接口 | Hermes Desktop/backend API |
| `RULE-*` | 规则 | 架构、安全、交互和边界规则 |
| `TEST-*` | 测试 | 自动/手动回归测试项 |

---

## 1. 能力地图

| ID | 能力 | 描述 | 页面 | 流程 | API | 测试 |
| --- | --- | --- | --- | --- | --- | --- |
| `CAP-001` | 对话框图片生成切换 | 在 composer 控件区提供图片生成后端/模型选择入口 | `PAGE-001` | `FLOW-001` | `API-001`, `API-002` | `TEST-001`~`TEST-004` |
| `CAP-002` | Provider registry 可视化 | 将 CLI `hermes tools` 的 image_gen provider/model catalog 暴露给 Desktop | `PAGE-001` | `FLOW-002` | `API-001` | `TEST-001`, `TEST-002` |
| `CAP-003` | 配置写回 | 从 UI 选择 provider/model 后写入 `image_gen` 配置并启用 `image_gen` toolset | `PAGE-001` | `FLOW-003` | `API-002` | `TEST-002`, `TEST-003` |

---

## 2. 页面地图

| ID | 页面/区域 | 文件 | 包含按钮 | 读写数据 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `PAGE-001` | Chat composer 控件区 | `apps/desktop/src/app/chat/composer/controls.tsx` | `BTN-001` | `DATA-001`~`DATA-004` | 图片切换入口放在模型选择 pill 旁边，交互对齐模型选择器 |
| `PAGE-002` | 图片生成下拉菜单 | `apps/desktop/src/app/chat/composer/image-generation-pill.tsx` | `BTN-002`, `BTN-003` | `DATA-001`~`DATA-004` | 展示 provider、模型、能力和当前状态 |

---

## 3. 按钮地图

| ID | 按钮 | 所在页面 | 触发动作 | 接口 | 规则 |
| --- | --- | --- | --- | --- | --- |
| `BTN-001` | Image Generation Pill | `PAGE-001` | `ACT-001` | `API-001` | `RULE-001`, `RULE-004` |
| `BTN-002` | Provider Row | `PAGE-002` | `ACT-002` | `API-002` | `RULE-002`, `RULE-003` |
| `BTN-003` | Model Row | `PAGE-002` | `ACT-003` | `API-002` | `RULE-002`, `RULE-003` |

---

## 4. 动作地图

| ID | 动作 | 输入 | 输出 | 触发方 | 后续流程 |
| --- | --- | --- | --- | --- | --- |
| `ACT-001` | 打开图片生成菜单 | 无 | `DATA-001` | `BTN-001` | `FLOW-001`, `FLOW-002` |
| `ACT-002` | 选择图片 provider | provider id/name | refreshed `DATA-001` | `BTN-002` | `FLOW-003` |
| `ACT-003` | 选择图片模型 | provider id + model id | refreshed `DATA-001` | `BTN-003` | `FLOW-003` |
| `ACT-004` | 后端构建选项 | `config.yaml` + image_gen registry | `DATA-001` | `API-001` | `FLOW-002` |
| `ACT-005` | 后端保存选择 | provider/model selection | updated `config.yaml` | `API-002` | `FLOW-003` |

---

## 5. 流程地图

### `FLOW-001` Composer 入口流程

```text
PAGE-001 → BTN-001 → PAGE-002 DropdownMenu → 展示当前 provider/model
```

相关对象：`DATA-001`, `DATA-002`  
规则：`RULE-001`, `RULE-004`

### `FLOW-002` 选项加载流程

```text
BTN-001 mount/open → ACT-001 → API-001 → ACT-004 → image_gen registry + config → DATA-001 → PAGE-002
```

相关对象：`DATA-001`~`DATA-004`  
规则：`RULE-002`, `RULE-003`

### `FLOW-003` Provider/模型切换流程

```text
BTN-002/BTN-003 → ACT-002/ACT-003 → API-002 → ACT-005 → config.yaml + platform_toolsets.cli → DATA-001 refresh
```

如果 `image_gen` toolset 是本次才启用，UI 显示需要新对话或 `/reset` 的提示。  
相关对象：`DATA-001`, `DATA-003`, `DATA-004`  
规则：`RULE-001`, `RULE-004`

---

## 6. 数据地图

| ID | 数据 | TypeScript/Python 位置 | 生产方 | 消费方 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `DATA-001` | `ImageGenerationOptionsResponse` | `apps/desktop/src/types/hermes.ts` / `hermes_cli/web_server.py` | `API-001`, `API-002` | `PAGE-002` | 当前 enabled/provider/model 与 provider 列表 |
| `DATA-002` | `ImageGenerationProviderOption` | `apps/desktop/src/types/hermes.ts` | `API-001` | `PAGE-002`, `ACT-002` | provider id、名称、badge、可用性、模型列表 |
| `DATA-003` | `ImageGenerationModelOption` | `apps/desktop/src/types/hermes.ts` | `API-001` | `PAGE-002`, `ACT-003` | 模型 id、展示名、速度、价格、能力 |
| `DATA-004` | `image_gen` config | `~/.hermes/config.yaml` | `API-002` | `tools/image_generation_tool.py` | provider/model/use_gateway 的持久配置 |
| `DATA-005` | `platform_toolsets.cli` | `~/.hermes/config.yaml` | `API-002` | Agent session tool resolution | 确保切换后 `image_gen` toolset 已启用 |

---

## 7. 接口地图

| ID | Hermes API | 方法 | 使用方 | 状态 |
| --- | --- | --- | --- | --- |
| `API-001` | `/api/tools/image-generation/options` | `GET` | `ImageGenerationPill` | 已实现 |
| `API-002` | `/api/tools/image-generation/selection` | `PUT` | `ImageGenerationPill` | 已实现 |

---

## 8. 规则地图

| ID | 规则 | 说明 | 覆盖对象 |
| --- | --- | --- | --- |
| `RULE-001` | 不改变 Agent Core | UI 只选择配置；真实生成仍由既有 `image_generate` tool 执行 | `CAP-001`, `API-002` |
| `RULE-002` | Provider registry 是来源 | provider/model 列表来自 image_gen provider registry 与 CLI tools config，不在前端硬编码 | `CAP-002`, `DATA-002`, `DATA-003` |
| `RULE-003` | 凭证不进前端 | API 只返回 provider/model/capability；不返回 API key、OAuth token 或 secret | `API-001`, `API-002` |
| `RULE-004` | 新 toolset 需要新会话 | 如果本次选择才启用 `image_gen`，当前 Agent session 可能没有工具 schema，UI 要提示新对话或 `/reset` | `FLOW-003`, `DATA-005` |

---

## 9. 测试地图

| ID | 测试 | 命令/方式 | 覆盖能力 | 当前结果 |
| --- | --- | --- | --- | --- |
| `TEST-001` | image generation options API shape | `pytest tests/hermes_cli/test_dashboard_admin_endpoints.py::TestToolsConfigEndpoints::test_image_generation_options_shape` | `API-001`, `DATA-001`~`DATA-003` | 待验证 |
| `TEST-002` | provider/model selection writes config | `pytest tests/hermes_cli/test_dashboard_admin_endpoints.py::TestToolsConfigEndpoints::test_select_image_generation_provider_and_model` | `API-002`, `DATA-004`, `DATA-005` | 待验证 |
| `TEST-003` | invalid provider rejected | `pytest tests/hermes_cli/test_dashboard_admin_endpoints.py::TestToolsConfigEndpoints::test_select_image_generation_unknown_provider_400` | `API-002`, `RULE-002` | 待验证 |
| `TEST-004` | Desktop typecheck | `npm --prefix apps/desktop run typecheck` | `PAGE-001`, `PAGE-002`, `DATA-001`~`DATA-003` | 待验证 |

---

## 10. 当前实现文件清单

| 文件 | 编号覆盖 | 说明 |
| --- | --- | --- |
| `hermes_cli/web_server.py` | `API-001`, `API-002`, `DATA-001`~`DATA-005` | 新增 image generation options/selection endpoints |
| `apps/desktop/src/hermes.ts` | `API-001`, `API-002` | 新增 Desktop API client 方法 |
| `apps/desktop/src/types/hermes.ts` | `DATA-001`~`DATA-003` | 新增 image generation DTO types |
| `apps/desktop/src/app/chat/composer/image-generation-pill.tsx` | `PAGE-002`, `BTN-001`~`BTN-003`, `FLOW-001`~`FLOW-003` | 新增 composer 图片生成下拉菜单 |
| `apps/desktop/src/app/chat/composer/controls.tsx` | `PAGE-001`, `BTN-001` | 将图片生成 pill 放到模型选择 pill 旁边 |
| `apps/desktop/src/i18n/types.ts` / `en.ts` / `zh.ts` | `PAGE-002` | 新增菜单文案 |
| `tests/hermes_cli/test_dashboard_admin_endpoints.py` | `TEST-001`~`TEST-003` | 新增后端回归测试 |
