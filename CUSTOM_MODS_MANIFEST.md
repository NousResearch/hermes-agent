# Hermes Agent 自有修改清单

> **目的**: 记录所有自有修改点，确保升级后能精确恢复而非整体覆盖
> **创建时间**: 2026-05-17
> **当前版本**: fe8569571 (2026-05-29 升级后)
> **上游版本**: 2159d2a72 (v0.15.1)

---

## 修改文件清单

### 1. agent/auxiliary_client.py

**修改内容**: 添加 Alibaba/DashScope provider aliases

**修改位置**: `_PROVIDER_ALIASES` 字典，第 161 行后

**修改类型**: 添加（新增 6 行）

**Patch 内容**:
```diff
+"tencentmaas": "tencent-tokenhub",
+    # Alibaba / DashScope
+    "dashscope": "alibaba",
+    "bailian": "alibaba",
+    "aliyun": "alibaba",
+    "qwen": "alibaba",
+    "alibaba-cloud": "alibaba",
 }
```

**功能说明**: 支持百炼/DashScope API 的多种别名，简化用户配置

**验证方法**: `_PROVIDER_ALIASES["bailian"] == "alibaba"`

---

### 2. hermes_cli/kanban_db.py

**修改内容**: context.md convention 扩展

**修改类型**: 新增函数 + 常量（约 80 行）

**新增内容**:
- `_AGENT_SHARED` 常量
- `_KANBAN_BOARDS_CUSTOM` 常量
- `_AT_CONTEXT_RE` / `_FILE_CONTEXT_RE` 正则
- `resolve_context_path()` 函数
- `get_custom_boards_root()` 函数

**功能说明**: 支持 @ 路径简写引用 context.md，支持自定义看板目录

**验证方法**: `resolve_context_path` 函数存在且可导入

---

### 3. gateway/run.py

**修改内容**: 添加 Pending Task Injection Marker

**修改类型**: 添加注释和逻辑标记

**Patch 内容**:
```diff
+                                    # [PENDING_TASK_INJECT_MARKER] - Do not remove this comment
+                                    # Auto-injected by install-compression-hooks.sh
+                                    # Scans ~/agent-shared/tasks/_pending/ and injects waiting
+                                    # tasks as user messages into the compressed session history.
```

**功能说明**: 压缩会话时注入 pending task 逻辑的标记点

**验证方法**: 搜索文件中是否包含 `PENDING_TASK_INJECT_MARKER`

---

### 4. hermes_cli/config.py

**修改内容**: 删除配置警告跟踪代码（简化）

**修改类型**: 删除（上游已有相关删除）

**功能说明**: 简化配置加载逻辑，移除冗余警告缓存

**验证方法**: 检查文件中是否不存在 `_CONFIG_WARNED_CACHE`

---

### 5. plugins/memory/holographic/__init__.py

**修改内容**: temporal_decay_half_life 默认值改为 90

**修改类型**: 修改默认值

**Patch 内容**:
```diff
-temporal_decay_half_life: 0
+temporal_decay_half_life: 90

-temporal_decay = int(self._config.get("temporal_decay_half_life", 0))
+temporal_decay = int(self._config.get("temporal_decay_half_life", 90))
```

**功能说明**: 记忆时效衰减半衰期设为 90 天

**验证方法**: 检查 `temporal_decay_half_life` 默认值为 90

---

### 6. plugins/memory/__init__.py

**修改内容**: register_hook 转发到全局 PluginManager

**修改类型**: 修改 `_ProviderCollector.register_hook` 方法

**Patch 内容**:
```diff
-    def register_hook(self, *args, **kwargs):
-        pass
+    def register_hook(self, hook_name: str, callback, **kwargs):
+        """转发到全局 PluginManager._hooks，实现真实的 hook 注册。"""
+        try:
+            from hermes_cli.plugins import get_plugin_manager
+            pm = get_plugin_manager()
+            pm._hooks.setdefault(hook_name, []).append(callback)
+        except Exception:
+            pass
```

**功能说明**: 让 mem0-session-memory 的 on_session_end hook 能真正生效

**验证方法**: 检查文件中是否包含 `转发到全局 PluginManager`

---

### 7. gateway/platforms/feishu.py

**修改内容**: ~~Markdown 元素文本长度上限注释~~ **已失效**

**说明**: 上游 v0.15 重构了消息分段逻辑，使用 `_DEFAULT_TEXT_BATCH_MAX_CHARS = 4000` 替代原 1500 常量。原有注释已不适用。

**处理**: 注释性修改不影响功能，**无需重建**。

**验证方法**: 检查 `_DEFAULT_TEXT_BATCH_MAX_CHARS` 是否存在（验证新架构）

---

## 配置文件自有修改 (config.yaml)

> 注：config.yaml 在 ~/.hermes/ 目录，不在 git 仓库内，升级不会覆盖

**自有配置项**:
1. `model.provider: alibaba-coding-plan` - 主模型使用百炼
2. `model.fallback_providers` - 失败回退到 dashscope
3. `auxiliary.*.provider` - 各辅助任务路由到 minimax-cn / alibaba

---

## 升级保护策略

### Pre-Upgrade 备份

```bash
# 1. 备份当前自有修改 diff
cd ~/.hermes/hermes-agent
git diff HEAD > ~/.hermes/upgrade-protection/backups/custom-mods-$(date +%Y%m%d-%H%M%S).diff

# 2. 创建 stash 作为保险
git stash push -m "pre-upgrade-backup-$(date +%Y%m%d-%H%M%S)"
```

### Post-Upgrade 合并

```bash
# 1. 检查新版本是否有冲突函数
# 2. 使用精确 patch 恢复自有修改（而非整体覆盖）
# 3. 运行验证脚本确认功能正常
```

---

## 验证检查清单

| 文件 | 检查项 | 预期结果 |
|------|--------|----------|
| auxiliary_client.py | `_PROVIDER_ALIASES["bailian"]` | `"alibaba"` |
| feishu.py | 包含 "1500" 注释 | 存在 |
| run.py | 包含 `PENDING_TASK_INJECT_MARKER` | 存在 |
| config.py | `_CONFIG_WARNED_CACHE` | 不存在 |
| holographic/__init__.py | `temporal_decay_half_life` 默认值 | 90 |

---

## 风险等级

| 修改 | 风险等级 | 说明 |
|------|----------|------|
| provider aliases | **低** | 纯添加，不影响现有逻辑 |
| feishu 注释 | **低** | 注释性质，不影响执行 |
| run.py marker | **中** | 可能与压缩逻辑冲突 |
| config.py 删除 | **中** | 删除代码，需检查上游是否有新依赖 |
| holographic 默认值 | **低** | 参数调整，不影响核心逻辑 |

---

**维护者**: Hermes Agent 自动保护系统
**更新频率**: 每次 git status 检测到修改时自动更新