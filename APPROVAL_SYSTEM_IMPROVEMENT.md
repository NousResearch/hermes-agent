# Hermes Agent 授权系统改进 - 实现总结

## 📋 问题描述

用户反馈了一个关键的授权机制问题：

> "每次遇到需要授权的命令后，我发现 hermes 并不会停下来等待用户的授权反馈。当这条命令必须被执行而用户忙于其他事情而没有及时授权时，就导致这条需要授权的命令实际上就被跳过了。这样最终的结果就不正确了。希望出现授权命令时，一定要等待用户的授权反馈。或者增加一个配置项，让用户自己可以提前确定哪些命令是必须要授权的，哪些命令可以直接执行。"

### 核心问题

1. **授权不等待** - 在某些模式下（cron/batch），需要授权的命令直接被跳过
2. **缺少强制等待机制** - 没有配置项确保必须等待用户授权
3. **缺少预配置机制** - 用户无法提前配置哪些命令必须授权/可直接执行

---

## 🎯 解决方案

实现了三大核心功能：

### 1. 强制等待授权模式 (Blocking Mode) ✅

**新增配置项:**
```yaml
approvals:
  mode: blocking  # 强制等待用户授权
  timeout: 600    # 超时时间（秒）
```

**特性:**
- ✅ 所有模式下都阻塞等待用户响应
- ✅ 基于文件的 IPC 机制实现等待
- ✅ 可配置的超时时间
- ✅ 超时后自动拒绝命令

**实现文件:**
- `tools/approval.py` - 核心逻辑
- `hermes_cli/approval_ipc.py` - 文件 IPC 等待机制

---

### 2. 命令白名单/黑名单 (Auto Allow/Deny) ✅

**新增配置项:**
```yaml
approvals:
  auto_allow:  # 白名单 - 自动执行的命令
    - "^git\\s+status"
    - "^npm\\s+install"
  
  auto_deny:   # 黑名单 - 自动拒绝的命令
    - "^rm\\s+-rf\\s+/"
    - "^chmod\\s+.*777"
```

**特性:**
- ✅ 使用 Python 正则表达式匹配
- ✅ 支持复杂的匹配模式（否定预查等）
- ✅ 优先级：auto_deny > auto_allow > 默认检测
- ✅ 审计日志记录

**执行流程:**
```
命令 → 检查 auto_deny → 检查 auto_allow → 默认授权流程
         ↓                  ↓
      自动拒绝          自动通过
```

---

### 3. 改进的 Gateway 等待机制 ✅

**原有问题:**
- Gateway 模式下，当 `notify_cb` 为 None 时，命令被跳过

**改进方案:**
- 在 blocking 模式下，使用文件 IPC 强制等待
- 提供多种超时配置选项
- 确保所有场景都能正确等待

**实现逻辑:**
```python
if is_gateway and notify_cb:
    # 使用原有的 Gateway 回调机制
    wait_for_gateway_approval()
elif approval_mode == "blocking":
    # 新增：blocking 模式使用文件 IPC 等待
    wait_for_approval_blocking()
else:
    # 非 blocking 模式：返回 approval_required
    return {"status": "approval_required"}
```

---

## 📁 新增/修改的文件

### 新增文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `hermes_cli/approval_ipc.py` | 260 | 文件 IPC 等待机制 |
| `docs/APPROVAL_CONFIG_GUIDE.md` | 280 | 完整配置指南 |
| `config.approvals.example.yaml` | 100 | 配置示例 |
| `tests/test_approval_features.py` | 180 | 功能测试 |

### 修改文件

| 文件 | 修改行数 | 说明 |
|------|---------|------|
| `tools/approval.py` | +120 | 核心授权逻辑增强 |

---

## 🔧 技术实现细节

### 1. Blocking 模式实现

```python
# tools/approval.py
def _should_block_and_wait() -> bool:
    """Check if approval mode requires blocking wait."""
    mode = _get_approval_mode()
    return mode in ("blocking", "manual", "smart")

# In check_all_command_guards():
if approval_mode == "blocking":
    choice = wait_for_approval_blocking(
        session_key=session_key,
        command=command,
        description=combined_desc,
        timeout=timeout
    )
    if choice is None:  # Timeout
        return {"approved": False, "message": "BLOCKED: timed out"}
```

### 2. 文件 IPC 机制

```python
# hermes_cli/approval_ipc.py
def wait_for_approval_blocking(session_key, command, description, timeout):
    # 1. 写入请求文件
    write_approval_request(session_key, command, description)
    
    # 2. 轮询响应文件
    start_time = time.time()
    while time.time() - start_time < timeout:
        response = read_approval_response(session_key)
        if response:
            cleanup_approval_files(session_key)
            return response["choice"]
        time.sleep(poll_interval)
    
    # 3. 超时清理
    cleanup_approval_files(session_key)
    return None
```

### 3. 白名单/黑名单检查

```python
def _check_auto_allowlist(command: str, description: str) -> bool:
    config = _get_approval_config()
    auto_allow = config.get("auto_allow", [])
    
    for pattern in auto_allow:
        if re.search(pattern, command):
            return True
    return False

def _check_auto_denylist(command: str, description: str) -> bool:
    config = _get_approval_config()
    auto_deny = config.get("auto_deny", [])
    
    for pattern in auto_deny:
        if re.search(pattern, command):
            return True
    return False
```

---

## 🧪 测试结果

### 测试覆盖

```bash
$ python tests/test_approval_features.py

============================================================
Hermes Approval System - Feature Tests
============================================================

Test 1: Auto-Allow List ✓
Test 2: Auto-Deny List ✓
Test 3: Blocking Mode Detection ✓
Test 4: Dangerous Command Detection ✓
Test 5: Approval IPC (File-based) ✓

✅ All tests completed successfully!
```

### 功能验证

| 功能 | 测试状态 | 说明 |
|------|---------|------|
| Blocking 模式 | ✅ 通过 | 正确检测并等待 |
| Auto Allow | ✅ 通过 | 需要配置后生效 |
| Auto Deny | ✅ 通过 | 需要配置后生效 |
| 危险命令检测 | ✅ 通过 | 所有模式正确识别 |
| 文件 IPC | ✅ 通过 | 读写清理正常 |

---

## 📖 使用指南

### 场景 1: 生产环境（严格）

```yaml
approvals:
  mode: blocking        # 必须等待授权
  timeout: 600         # 10 分钟超时
  
  auto_deny:
    - "^rm\\s+-rf\\s+/"
    - "^mkfs"
    - "^dd\\s+"
  
  auto_allow: []       # 不允许自动通过
```

### 场景 2: 开发环境（平衡）

```yaml
approvals:
  mode: manual
  timeout: 120
  
  auto_allow:
    - "^git\\s+(status|log|branch)"
    - "^npm\\s+(install|run|test)"
    - "^cargo\\s+(build|test)"
  
  auto_deny:
    - "^rm\\s+-rf\\s+/"
    - "^chmod\\s+.*777"
```

### 场景 3: 开发环境（高效）

```yaml
approvals:
  mode: smart          # AI 智能判断
  
  auto_allow:
    - "^git\\s+"
    - "^npm\\s+(install|run)"
    - "^docker\\s+(build|run|compose)"
```

---

## 🎯 配置选项总览

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `mode` | string | "manual" | 授权模式：manual/blocking/smart/off |
| `timeout` | int | 60 | CLI 授权超时（秒） |
| `gateway_timeout` | int | 300 | Gateway 授权超时（秒） |
| `auto_allow` | list | [] | 自动允许的命令模式（正则） |
| `auto_deny` | list | [] | 自动拒绝的命令模式（正则） |

---

## 📊 改进效果

### Before（改进前）

```
用户：执行危险命令
  ↓
Hermes: 发送授权请求
  ↓
用户：（忙于其他事情，未及时响应）
  ↓
Hermes: 跳过命令，继续执行 ❌
  ↓
结果：不正确，用户期望等待
```

### After（改进后）

```
用户：执行危险命令
  ↓
Hermes: 检测配置 mode=blocking
  ↓
Hermes: 发送授权请求，阻塞等待 ⏳
  ↓
用户：（忙于其他事情）
  ↓
Hermes: 继续等待...（最多 timeout 秒）
  ↓
用户：收到通知，进行授权
  ↓
Hermes: 根据授权执行或拒绝 ✅
  ↓
结果：正确，符合用户期望
```

---

## 🔐 安全增强

### 多层防护

1. **Auto Deny** - 第一层：自动拒绝极度危险命令
2. **Pattern Detection** - 第二层：检测危险模式
3. **Smart Approval** - 第三层：AI 风险评估（smart 模式）
4. **User Approval** - 第四层：用户最终确认
5. **Blocking Wait** - 第五层：确保不跳过授权

### 审计日志

所有授权决策都会记录：
- Auto allow/deny 决策
- 用户授权决策（once/session/always）
- 超时事件
- 会话级授权状态

---

## 🚀 部署指南

### 1. 更新代码

```bash
cd ~/hermes-agent
git pull origin main
```

### 2. 配置授权模式

编辑 `~/.hermes/config.yaml`:

```yaml
approvals:
  mode: blocking
  timeout: 300
  auto_allow:
    - "^git\\s+status"
  auto_deny:
    - "^rm\\s+-rf\\s+/"
```

### 3. 测试配置

```bash
# 测试危险命令检测
hermes "rm -rf /tmp/test"

# 查看待处理授权
hermes approvals list
```

---

## 📝 待办事项

### 已完成 ✅

- [x] Blocking 模式实现
- [x] Auto allow/deny 列表
- [x] 文件 IPC 等待机制
- [x] Gateway 等待改进
- [x] 文档编写
- [x] 测试用例
- [x] 配置示例

### 可选增强 🔮

- [ ] CLI 命令：`hermes approvals approve/deny`
- [ ] 授权历史查询
- [ ] 统计面板
- [ ] 移动端通知集成
- [ ] 授权模板库

---

## 🎊 总结

### 核心价值

1. **可靠性** - 确保授权不被跳过
2. **灵活性** - 多种模式适应不同场景
3. **安全性** - 多层防护机制
4. **易用性** - 简单配置即可生效

### 用户收益

- ✅ 生产环境更安全（blocking 模式）
- ✅ 开发环境更高效（auto allow）
- ✅ 关键命令更可靠（强制等待）
- ✅ 配置管理更灵活（白名单/黑名单）

---

**实现日期:** 2026-04-17  
**实现者:** 澎湃时光 (JohnHarper)  
**版本:** v0.9.0  
**状态:** ✅ 完成并测试通过
