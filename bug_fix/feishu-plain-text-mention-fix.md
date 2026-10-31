# 修复：飞书 bot 间纯文本 @mention 不触发目标 bot 响应

## 问题现象

投研群中，bot A（如量化研究员）在回复里用纯文本 `@数据管家▶Mac` 交接给 bot B（如数据管家），但 bot B 不响应。同一条消息中如果用飞书原生 `<at>` 标签则正常。

## 根因

飞书只为消息 `mentions` 数组中包含目标 bot `open_id` 的原生 mention 创建 mention 事件。纯文本 `@岗位名` 不创建 mention 事件。

所有投研 profile 的群消息准入策略是 `allow_bots=mentions` + `require_mention=true`：
- `_admit()` → `_mentions_self(message)` 检查 `mentions` 数组是否包含目标 bot 的 open_id
- 纯文本 `@岗位名` 不在 `mentions` 数组里 → 返回 `group_policy_rejected`
- 消息被丢弃，目标 bot 永远不会收到处理请求

SOUL.md 中已写了 roster.json 路径和 `<at>` 标签格式说明，但 LLM 不总是遵守——glm-5.2 会退化为纯文本 `@岗位名`。

## 修复方案

在适配器 `format_message()` 中加入出站消息自动转换：扫描纯文本 `@岗位名`，用 roster.json 中的 open_id 替换为原生 `<at user_id="ou_...">岗位名</at>` 标签。

这是适配器层面的兜底，不依赖 LLM 遵守提示词指令。

### 为什么不做 SOUL.md 注入

所有 12 个投研 profile 的 SOUL.md 已经包含 roster.json 路径和 mention 格式说明。但 LLM 不总是遵守——问题已经发生了。适配器层转换是确定性兜底，保证无论 LLM 输出什么格式，出站消息的 mention 都能被飞书识别。

## 修改的文件

### 1. 适配器代码：`plugins/platforms/feishu/adapter.py`

4 处修改：

| # | 位置 | 修改内容 |
|---|------|----------|
| 1 | `FeishuAdapterSettings` (line ~325) | 新增字段 `mention_roster_path: str = ""` |
| 2 | `__init__` (line ~1348) | 初始化 roster 缓存：`_mention_roster_cache`, `_mention_roster_mtime`, `_mention_pattern` |
| 3 | `_load_settings` (line ~1424) | 从 `extra` 读取 `mention_roster_path` 配置 |
| 4 | `format_message` (line ~2021) | 新增 `_get_mention_roster()` 和 `_convert_plain_mentions()` 方法，`format_message()` 调用后者 |

### 2. 新增方法：`_get_mention_roster()`

- 从 `mention_roster_path` 指向的 JSON 文件加载 name→open_id 映射
- roster 格式：`{"agents": {"data-steward": {"name": "数据管家▶Mac", "bot_id": "ou_ac508ddf..."}}}`
- 对带 `▶Mac` 后缀的名字，同时注册去掉后缀的短名（如 `数据管家`），使 `@数据管家` 和 `@数据管家▶Mac` 都能匹配
- 按 mtime 缓存，roster 文件修改后自动重载，无需重启
- 预编译 regex：`@name1|@name2|...` 按名字长度降序排列（最长匹配优先）

### 3. 新增方法：`_convert_plain_mentions(content)`

- 扫描出站消息中的 `@岗位名` 模式
- 用 roster 中的 open_id 替换为 `<at user_id="ou_...">岗位名</at>`
- 已有原生 `<at>` 标签不会被重复转换（标签内的名字前是 `>` 不是 `@`，不匹配 `@name` 模式）
- `@_all`（飞书 @everyone）不会被转换（不在 roster 中）
- 无 `@` 字符的消息直接返回原内容（快速路径）

### 4. 配置：每个 profile 的 `config.yaml`

在 `platforms.feishu.extra` 下新增：

```yaml
platforms:
  feishu:
    extra:
      # ... existing config ...
      mention_roster_path: /path/to/deployment/feishu/roster.json
```

## roster.json 格式

```json
{
  "chat_id": "oc_xxx",
  "agents": {
    "data-steward": {
      "name": "数据管家▶Mac",
      "bot_id": "ou_ac508ddfca9b8712a88be59e805a09c9"
    }
  }
}
```

每个 agent 条目需要 `name`（显示名，含 ▶Mac 后缀）和 `bot_id`（open_id）。

## 测试验证

9 个测试用例全部通过：

| 测试 | 输入 | 预期 | 结果 |
|------|------|------|------|
| 全名纯文本 | `交接给@数据管家▶Mac处理` | 转换为原生标签 | ✅ |
| 短名纯文本 | `交接给@数据管家处理` | 转换为原生标签 | ✅ |
| 已有原生标签 | `<at user_id="ou_...">数据管家▶Mac</at>` | 不重复转换 | ✅ |
| 混合（一纯文本一原生） | `@量化研究员▶Mac...<at>数据管家▶Mac</at>` | 只转第一个 | ✅ |
| 无 mention | `今天天气不错` | 不变 | ✅ |
| @_all | `@_all 请注意` | 不转换 | ✅ |
| 多个纯文本 | `@数据管家▶Mac 和 @量化研究员▶Mac` | 都转换 | ✅ |
| 消息开头 | `@风险官▶Mac 请评估` | 转换 | ✅ |
| 冒号后 | `@投委主席:请裁决` | 转换 | ✅ |

## 部署记录

### Mac (macOS 14.8.7)

- **adapter.py 路径**：`/Users/season/workspace/github/hermes-agent/plugins/platforms/feishu/adapter.py`（github checkout，网关实际加载）
- **installs 环境同步**：2 个 installs 环境的 adapter.py 同步修改
- **config.yaml**：12 个投研 profile 全部添加 `mention_roster_path: /Users/season/investment-lab/deployment/feishu/roster.json`
- **roster.json**：`/Users/season/investment-lab/deployment/feishu/roster.json`
- **网关重启**：用户手动执行 `hermes gateway restart --yes`
- **验证**：日志确认 `Loaded mention roster (23 names)`

### Linux 服务器 (100.83.58.96, Ubuntu 24.04)

- **adapter.py 路径**：`/home/season/github/hermes-agent/plugins/platforms/feishu/adapter.py`（github checkout）
- **installs 环境同步**：3 个 installs 环境的 adapter.py 逐个打补丁
- **config.yaml**：12 个投研 profile 全部添加 `mention_roster_path: /home/season/investment-lab/deployment/feishu/roster.json`
- **roster.json**：`/home/season/investment-lab/deployment/feishu/roster.json`
- **网关重启**：通过 SSH 发送 SIGTERM，systemd `Restart=always` 自动重启（旧 PID 67517 → 新 PID 71166）
- **验证**：12 个 profile 全部重连

## 注意事项

- `mention_roster_path` 未配置时，转换功能静默关闭（`_get_mention_roster()` 返回空 dict），不影响原有行为
- roster 文件按 mtime 缓存，修改 roster.json 后无需重启网关，下一条出站消息自动加载新版本
- 此修改是 Hermes 核心代码修改，`hermes update` 可能覆盖。如被覆盖，需重新应用补丁
- 补丁脚本保存在 `/tmp/fix_feishu_mentions.py`（Linux 服务器），可重复执行（已修补的文件会跳过）
