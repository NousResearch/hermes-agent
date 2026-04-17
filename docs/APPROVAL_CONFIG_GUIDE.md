# Hermes Agent 授权配置指南

## 📋 概述

Hermes Agent 提供灵活的命令授权机制，确保危险命令在执行前得到用户确认。本指南详细介绍如何配置授权行为。

## 🎯 授权模式

在 `~/.hermes/config.yaml` 中配置 `approvals` 部分：

```yaml
approvals:
  mode: manual          # 授权模式：manual | blocking | smart | off
  timeout: 60          # CLI 授权超时（秒）
  gateway_timeout: 300 # Gateway 授权超时（秒）
  auto_allow:          # 自动允许的命令模式（正则表达式）
    - "^git\\s+status"
    - "^npm\\s+install\\s+--save-dev"
  auto_deny:           # 自动拒绝的命令模式（正则表达式）
    - "^rm\\s+-rf\\s+/"
    - "^chmod\\s+777"
```

### 授权模式说明

#### `manual` (默认)
- **行为**: 检测到危险命令时提示用户授权
- **等待**: Gateway 模式下等待，CLI 模式下交互式提示
- **适用**: 大多数使用场景

```yaml
approvals:
  mode: manual
```

#### `blocking` (新增 ⭐)
- **行为**: **必须**等待用户授权，否则命令不执行
- **等待**: 所有模式下都阻塞等待用户响应
- **超时**: 默认 300 秒（5 分钟），可通过 `timeout` 配置
- **适用**: 生产环境、关键任务、不允许跳过授权的场景

```yaml
approvals:
  mode: blocking
  timeout: 600  # 10 分钟超时
```

#### `smart`
- **行为**: 先使用 AI 评估风险，低风险自动通过，高风险提示用户
- **等待**: AI 判断为高风险时等待用户
- **适用**: 希望减少授权提示但保持安全的场景

```yaml
approvals:
  mode: smart
```

#### `off`
- **行为**: 完全禁用授权检查
- **等待**: 不等待
- **警告**: ⚠️ 仅在完全信任的环境中使用
- **适用**: 开发环境、沙箱、测试

```yaml
approvals:
  mode: off
```

---

## 🔐 自动允许/拒绝列表

### 自动允许 (`auto_allow`)

配置后，匹配的命令会**自动执行**，无需授权提示。

```yaml
approvals:
  auto_allow:
    # Git 操作
    - "^git\\s+(status|log|branch|remote)"
    
    # 包安装（非全局）
    - "^npm\\s+install\\s+(?!-g)"
    - "^pip\\s+install\\s+(?!-\\-user)"
    - "^yarn\\s+add"
    
    # 文件操作（安全路径）
    - "^mkdir\\s+-p\\s+\\./"
    - "^cp\\s+\\./"
    - "^mv\\s+\\./"
    
    # 构建命令
    - "^npm\\s+run\\s+(build|test|dev)"
    - "^make\\s+"
    - "^cargo\\s+build"
```

### 自动拒绝 (`auto_deny`)

配置后，匹配的命令会**自动拒绝**，甚至不会提示用户。

```yaml
approvals:
  auto_deny:
    # 危险删除
    - "^rm\\s+-rf\\s+/"
    - "^rm\\s+-rf\\s+\\$"
    
    # 危险权限
    - "^chmod\\s+(-[rwxRWX]*\\s+)*777"
    - "^chmod\\s+(-[rwxRWX]*\\s+)*666"
    
    # 系统文件
    - "^echo.*>\\s+/etc/"
    - "^tee.*>\\s+/etc/"
    
    # 网络危险操作
    - "^curl.*\\|.*sh"
    - "^wget.*\\|.*sh"
```

### 正则表达式语法

使用 Python 正则表达式语法：

| 模式 | 说明 | 示例 |
|------|------|------|
| `^` | 匹配命令开头 | `^rm` 匹配以 rm 开头的命令 |
| `$` | 匹配命令结尾 | `test$` 匹配以 test 结尾 |
| `\s+` | 匹配一个或多个空白 | `rm\s+-rf` 匹配 `rm -rf` |
| `?` | 非贪婪匹配 | `install\s+--save-dev` |
| `()` | 分组 | `git\s+(status|log)` |
| `!` | 否定预查 | `install\s+(?!-g)` 匹配非全局安装 |

---

## ⏱️ 超时配置

### CLI 超时

```yaml
approvals:
  timeout: 60  # CLI 模式下等待用户输入的超时（秒）
```

### Gateway 超时

```yaml
approvals:
  gateway_timeout: 300  # Gateway 模式下等待用户响应的超时（秒）
```

### Blocking 模式超时

```yaml
approvals:
  mode: blocking
  timeout: 600  # blocking 模式下的总超时（秒）
```

---

## 📝 完整配置示例

### 示例 1: 生产环境（严格）

```yaml
approvals:
  mode: blocking        # 必须等待授权
  timeout: 600         # 10 分钟超时
  gateway_timeout: 600
  
  # 自动拒绝极度危险的命令
  auto_deny:
    - "^rm\\s+-rf\\s+/"
    - "^mkfs"
    - "^dd\\s+"
    - "^chmod\\s+.*777"
  
  # 不允许自动通过任何危险命令
  auto_allow: []
```

### 示例 2: 开发环境（宽松）

```yaml
approvals:
  mode: smart          # AI 智能判断
  timeout: 120
  
  # 自动允许常见开发命令
  auto_allow:
    - "^git\\s+"
    - "^npm\\s+(install|run|test|build)"
    - "^yarn\\s+"
    - "^pnpm\\s+"
    - "^cargo\\s+(build|test|run)"
    - "^make\\s+"
    - "^docker\\s+(build|run|compose)"
  
  # 仍然拒绝危险命令
  auto_deny:
    - "^rm\\s+-rf\\s+/"
    - "^sudo\\s+rm"
```

### 示例 3: CI/CD 环境（自动化）

```yaml
approvals:
  mode: off  # 完全禁用（仅在可信环境）
  
  # 或者使用白名单模式
  # mode: manual
  # auto_allow:
  #   - ".*"  # 允许所有（不推荐）
```

### 示例 4: 混合模式（推荐）

```yaml
approvals:
  mode: manual
  timeout: 120
  gateway_timeout: 300
  
  # 自动允许安全的开发命令
  auto_allow:
    # 版本控制
    - "^git\\s+(status|log|diff|branch|tag|remote|fetch)"
    
    # 包管理（非全局）
    - "^npm\\s+install\\s+(?!-g)"
    - "^npm\\s+ci\\b"
    - "^yarn\\s+(install|add)"
    - "^pnpm\\s+(install|add)"
    
    # 构建和测试
    - "^npm\\s+run\\s+(build|test|lint|dev)"
    - "^yarn\\s+(build|test|lint)"
    - "^cargo\\s+(build|test|check)"
    - "^make\\s+(build|test|all)"
    
    # 文件操作（当前目录）
    - "^mkdir\\s+-p\\s+\\./"
    - "^cp\\s+\\./"
    - "^mv\\s+\\./"
    - "^rm\\s+(?!-rf)"  # 允许 rm 但不允许 -rf
  
  # 自动拒绝危险命令
  auto_deny:
    # 系统级删除
    - "^rm\\s+-rf\\s+/"
    - "^rm\\s+-rf\\s+\\$"
    - "^rm\\s+-rf\\s+~"
    
    # 危险权限
    - "^chmod\\s+.*777"
    - "^chmod\\s+.*666"
    
    # 系统文件修改
    - "^echo.*>\\s+/etc/"
    - "^cat.*>\\s+/etc/"
    
    # 远程代码执行
    - "^curl.*\\|.*sh"
    - "^wget.*\\|.*sh"
    - "^curl.*\\|.*bash"
```

---

## 🛠️ 命令行工具

### 查看待处理的授权

```bash
hermes approvals list
```

### 批准待处理的命令

```bash
# 批准特定会话
hermes approvals approve <session_key> --choice once

# 批准所有待处理
hermes approvals approve-all
```

### 拒绝待处理的命令

```bash
hermes approvals deny <session_key>
```

### 清除过期授权

```bash
hermes approvals cleanup
```

---

## 📊 授权流程图

```
命令执行请求
    ↓
检测是否危险？
    ├─ 否 → 直接执行 ✅
    └─ 是 → 继续
         ↓
    匹配 auto_deny？
         ├─ 是 → 自动拒绝 ❌
         └─ 否 → 继续
              ↓
         匹配 auto_allow？
              ├─ 是 → 自动通过 ✅
              └─ 否 → 继续
                   ↓
              检查 mode 配置
                   ├─ off → 直接执行 ✅
                   ├─ smart → AI 评估
                   │         ├─ 低风险 → 自动通过 ✅
                   │         ├─ 高风险 → 自动拒绝 ❌
                   │         └─ 不确定 → 等待用户
                   ├─ blocking → 阻塞等待用户 ⏳
                   └─ manual → 等待用户 ⏳
                        ↓
                   用户响应
                        ├─ once → 执行一次 ✅
                        ├─ session → 会话内通过 ✅
                        ├─ always → 永久通过 ✅
                        └─ deny/timeout → 拒绝 ❌
```

---

## 🔍 故障排除

### 问题 1: 授权提示不出现

**可能原因:**
- `approvals.mode` 设置为 `off`
- 命令匹配了 `auto_allow` 列表
- 在容器环境中（docker/modal 等）

**解决方案:**
```yaml
approvals:
  mode: blocking  # 强制等待
  auto_allow: []  # 清空自动允许列表
```

### 问题 2: 授权超时太快

**解决方案:**
```yaml
approvals:
  timeout: 300        # CLI 超时增加到 5 分钟
  gateway_timeout: 600 # Gateway 超时增加到 10 分钟
```

### 问题 3: 某些命令总是被拒绝

**可能原因:** 命令匹配了 `auto_deny` 列表

**解决方案:**
```yaml
approvals:
  auto_deny:
    # 注释掉或删除相关模式
    # - "^rm\\s+-rf"  # 注释后 rm -rf 会提示而不是直接拒绝
```

### 问题 4: Blocking 模式下命令卡住

**可能原因:** 用户没有收到授权通知

**解决方案:**
1. 检查通知渠道（Feishu/Telegram 等）是否正常
2. 手动批准：`hermes approvals list` 查看待处理，然后 `hermes approvals approve`
3. 增加超时：`timeout: 600`

---

## 📚 相关文件

- **授权检测:** `tools/approval.py`
- **IPC 等待:** `hermes_cli/approval_ipc.py`
- **配置文件:** `~/.hermes/config.yaml`
- **授权文件目录:** `~/.hermes/approvals/`

---

## 🎯 最佳实践

### 1. 根据环境选择模式

```yaml
# 生产环境
approvals:
  mode: blocking

# 开发环境
approvals:
  mode: smart

# 测试环境
approvals:
  mode: off
```

### 2. 精细化控制白名单

```yaml
# ✅ 好的做法：精确匹配
auto_allow:
  - "^git\\s+status"
  - "^git\\s+log"

# ❌ 不好的做法：过于宽泛
auto_allow:
  - "^git"  # 会允许 git filter-branch --force 等危险命令
```

### 3. 定期审查配置

```bash
# 查看哪些命令被自动允许/拒绝
hermes config show | grep -A 20 "approvals:"

# 查看授权历史
hermes logs --grep "approval"
```

### 4. 使用会话级授权

对于临时需要多次执行的命令，选择 `[s]ession` 选项而不是 `[a]lways`。

---

## 🆕 新增功能 (v0.9.0)

- ✅ **Blocking 模式**: 强制等待用户授权，不跳过
- ✅ **Auto Allow/Deny**: 预配置命令白名单/黑名单
- ✅ **文件 IPC**: 基于文件的授权等待机制
- ✅ **改进的 Gateway 等待**: 所有模式下都能正确等待

---

**最后更新:** 2026-04-17  
**版本:** v0.9.0  
**维护者:** Hermes Agent Team
