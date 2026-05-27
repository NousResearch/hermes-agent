# 任务包：修复 npm prefix 配置导致 GitHub MCP 连接失败

**任务 ID**: TASK-20260521-001
**状态**: 待执行
**优先级**: P1
**环境**: macOS 26.5 (Hermes Agent)

---

## 1. 根因分析

### 1.1 配置现状

```
# ~/.npmrc
prefix=~/.npm-global
registry=https://registry.npmmirror.com
```

该配置将 npm 全局安装目录指向 `~/.npm-global`，用于 TokScale、Repomix、openclaw、playwright-mcp 等工具的全局安装。

### 1.2 冲突机制

npm v9+ 引入了 config 冲突检测。当同时存在以下情况时触发错误：

1. `~/.npmrc` 设置了 `prefix=~/.npm-global`
2. Hermes（或 npx 自身）尝试在 subprocess 中使用 `--prefix` flag 或 `npm prefix` 命令

**具体错误**:
```
npm error config prefix cannot be changed from project config: /Users/gu/.npmrc.
```

### 1.3 受影响场景

| 场景 | 影响 | 严重程度 |
|------|------|----------|
| `npx -y @modelcontextprotocol/server-github` — 直接运行 | ✅ **当前可工作**（测试验证通过） | 低 |
| `npm prefix` — 任何位置 | ❌ **报错**（退出了但返回错误信息到 stderr） | 中 |
| `npm install --prefix <path>` — 有自定义 prefix | ⚠️ **模糊行为**（--prefix 被全局 prefix 覆盖风险） | 中 |
| `npm install -g <pkg>` — 全局安装 | ✅ **可工作**（写入 ~/.npm-global） | 低 |
| Hermes LSP install（agent/lsp/install.py） | ⚠️ **潜在风险**（详见下文） | 中 |

### 1.4 LSP Install 风险详情

`agent/lsp/install.py:235` 中的 `_install_npm()` 函数执行：
```python
subprocess.run([npm, "install", "--prefix", str(staging), ...])
```
当 `~/.npmrc` 已有 `prefix=~/.npm-global` 时，`--prefix` flag 可能被 npm 忽略或冲突，导致 LSP 工具安装到错误位置。

### 1.5 日志证据

从 `agent.log` 中可以确认：
- 2026-05-20 22:58:29: `npm install --prefix /Users/gu/.hermes/lsp yaml-language-server` 在 `prefix` 设置下执行
- 2026-05-20 22:59:40: GitHub MCP (stdio) 成功注册 26 个工具（说明 npx 在当前版本可工作）

### 1.6 核心结论

**`prefix=~/.npm-global` 不是目前 GitHub MCP 连接失败的直接原因**（直接原因是 Copilot HTTP endpoint 不稳定 + 切换到 npx/stdio 已修复），**但它是 npm 生态的雷区，会在以下情况下引爆**：

1. npm 版本升级（v11+ 对 prefix 冲突的校验越来越严格）
2. Hermes LSP 安装失败（`--prefix` 被全局 prefix 干扰）
3. 其他使用 `npm install --prefix` 的子系统

---

## 2. 修复方案（按优先级排序）

### 方案 A（推荐）：移除 `~/.npmrc` 的 `prefix`，改用 shell alias/软链

**原理**：将全局安装行为从 `.npmrc` 配置转移到 PATH 层级解决。

**步骤**:

1. **创建 `~/.npm-global/bin` 到 PATH 的持久化链接**（如果尚未存在）
   ```bash
   # ~/.zshrc 中确保以下路径（实际已存在）
   export PATH="$HOME/.npm-global/bin:$PATH"
   ```

2. **移除 `~/.npmrc` 中的 `prefix` 行**
   ```bash
   # 编辑 ~/.npmrc，删除 prefix=~/.npm-global
   # 保留 registry=https://registry.npmmirror.com
   ```

3. **将现有的全局包迁移到系统默认 prefix**
   ```bash
   # 重新安装所有已安装的全局包
   npm ls -g --depth=0 --parseable | tail -n +2 | xargs -I{} basename {} | xargs npm install -g
   ```
   或使用以下方法逐个迁移：
   ```bash
   npm install -g $(npm ls -g --depth=0 --parseable | tail -n +2 | xargs -I{} basename {})
   ```

4. **验证**
   ```bash
   npm prefix  # 应该返回 /usr/local 或类似路径，无错误
   npx -y @modelcontextprotocol/server-github --help  # 应正常输出
   npm install -g cowsay  # 安装到系统 prefix，应成功
   cowsay "Hello"  # 应可执行
   npm uninstall -g cowsay
   ```

**验收标准**:
- [ ] `npm prefix` 返回无错误
- [ ] `npx -y @modelcontextprotocol/server-github --help` 输出 "GitHub MCP Server running on stdio"
- [ ] `npm install -g <pkg>` 可正常安装到系统 prefix
- [ ] Hermes 重启后 GitHub MCP 工具正常加载（`/tools` 可见 `mcp_github_*` 系列工具）
- [ ] 已有的全局工具（openclaw, playwright-mcp, context7-mcp, deepseek-tui 等）仍然可用

**约束检查**:
- [x] 不破坏现有 npm global install 行为
- [x] MCP server npx 启动正常工作
- [x] GitHub MCP 正常连接

**风险**:
- 系统 prefix 可能需要 `sudo`（macOS 上 `/usr/local` 通常不需要，但如果 node 安装在 `/usr/local/lib/node_modules` 可能需要权限调整）
- 如果遇到权限问题，回退到方案 B

---

### 方案 B：保留 `prefix`，为 npx 创建独立无 `prefix` 的 subprocess 环境

**原理**：不修改 `.npmrc`，而是让 Hermes 在启动 npx subprocess 时覆盖 `npm_config_prefix` 环境变量。

**步骤**:

1. **修改 `tools/mcp_tool.py`** 中的 `_build_safe_env()` 或子进程启动逻辑

   在 `_resolve_stdio_command()` 之后，添加环境变量覆盖：
   ```python
   # 在 server_params 构建之前
   safe_env["npm_config_prefix"] = ""  # 或 safe_env.pop("npm_config_prefix", None)
   ```

   或者在 `StdioServerParameters` 构建时确保 `npm_config_prefix` 被重置。

2. **定位代码**: `tools/mcp_tool.py` 约 1266-1281 行

   在 `_run_stdio` 方法中：
   ```python
   safe_env = _build_safe_env(user_env)
   command, safe_env = _resolve_stdio_command(command, safe_env)
   # === 新增 ===
   # 覆盖 npm_config_prefix 避免 ~/.npmrc 的 prefix 干扰 npx
   if "npm_config_prefix" in safe_env:
       del safe_env["npm_config_prefix"]
   # === 新增结束 ===
   ```

3. **验证**
   ```bash
   # 确认 npx 在无 npm_config_prefix 环境下工作
   env -u npm_config_prefix npx -y @modelcontextprotocol/server-github --help
   ```

4. **重启 Hermes 并验证 GitHub MCP**

**验收标准**:
- [ ] `~/.npmrc` 保留 `prefix=~/.npm-global`
- [ ] `npm install -g <pkg>` 仍正常安装到 `~/.npm-global`
- [ ] npx 启动 MCP server 无 prefix 冲突错误
- [ ] Hermes 重启后 GitHub MCP 工具正常加载
- [ ] 其他 npx 启动的 MCP server（playwright, context7）也正常工作

**约束检查**:
- [x] 不破坏现有 npm global install 行为（保留 prefix）
- [x] MCP server npx 启动正常工作
- [x] GitHub MCP 正常连接

**额外测试**:
- [ ] `env -u npm_config_prefix npx -y @modelcontextprotocol/server-github --help` 成功
- [ ] `npm --location=global prefix` 仍返回正确路径

---

### 方案 C：使用 `npm config set prefix` 替代 `~/.npmrc` 硬编码

**原理**：使用 npm 用户级配置命令来设置 prefix，这样对 subprocess 环境的影响更可控。

**步骤**:

1. **清理 `~/.npmrc` 中的 `prefix` 行**
   ```bash
   # 编辑 ~/.npmrc，删除 prefix=~/.npm-global
   ```

2. **通过 npm config 命令设置**
   ```bash
   npm config set prefix ~/.npm-global
   ```

3. **验证**
   ```bash
   npm prefix  # 应返回 ~/.npm-global，但无错误
   npx -y @modelcontextprotocol/server-github --help
   npm install -g cowsay && cowsay "Hello" && npm uninstall -g cowsay
   ```

**验收标准**:
- [ ] `npm prefix` 返回 `/Users/gu/.npm-global` 且无错误
- [ ] 全局安装仍正常写入 `~/.npm-global`
- [ ] npx 启动 MCP server 正常
- [ ] Hermes 重启后 GitHub MCP 正常

**风险**:
- `npm config set prefix` 实际上也会写入 `~/.npmrc`，格式类似但可能使用不同的写权限
- 如果 npm 版本对 `npmrc` 中的 `prefix` 校验变得更严格，这个方案可能无效

---

### 方案 D（激进）：将 MCP server 从 npx 改为本地安装

**原理**：完全绕过 npx，将 `@modelcontextprotocol/server-github` 本地安装到项目目录，直接通过 node 启动。

**步骤**:

1. **在 Hermes 项目目录下安装**
   ```bash
   cd /Users/gu/.hermes/hermes-agent
   npm install @modelcontextprotocol/server-github
   ```

2. **修改 `config.yaml`** 使用本地路径
   ```yaml
   mcp_servers:
     github:
       enabled: true
       command: node
       args:
       - node_modules/@modelcontextprotocol/server-github/dist/index.js
       env:
         GITHUB_PERSONAL_ACCESS_TOKEN: "gho_..."
       timeout: 120
   ```

3. **验证**
   ```bash
   node node_modules/@modelcontextprotocol/server-github/dist/index.js --help
   ```

**验收标准**:
- [ ] 本地 node_modules 中包含 `@modelcontextprotocol/server-github`
- [ ] 直接 node 启动可工作
- [ ] Hermes 重启后 GitHub MCP 正常加载
- [ ] `~/.npmrc` 的 prefix 设置不干扰本地安装

**风险**:
- 每次升级需要手动 `npm update`
- `node_modules` 膨胀
- 不是标准 MCP 的推荐部署方式

---

## 3. 推荐方案及理由

### 🏆 推荐：方案 A（移除 prefix，改用系统默认 prefix + PATH）

**理由**:

1. **根治问题**：完全消除 `.npmrc` 中 `prefix` 配置带来的冲突风险，而不是打补丁
2. **长期安全**：npm v11+ 对 prefix 的校验越来越严格，未来版本可能直接拒绝执行而不是只给警告
3. **行业标准做法**：大部分 npm 用户不使用自定义 prefix，macOS 的 `/usr/local/lib/node_modules` 是标准位置
4. **修复面广**：一次修复解决所有受影响场景（npx、npm install --prefix、npm prefix 等）
5. **Hermes 兼容性**：Hermes 的 MCP 子进程使用过滤后的环境（`_SAFE_ENV_KEYS`），移除 prefix 后子进程继承干净的 PATH，不会再有配置冲突

**前提检查**:
- macOS 的 `/usr/local/lib/node_modules` 通常不需要 sudo 访问（Homebrew 安装的 node 有此权限）
- 当前 `~/.npm-global/bin` 已有 15+ 个全局工具的软链，需要逐个迁移

**如果权限成为问题**：
执行方案 A 过程中如果遇到 `/usr/local` 的写权限问题，立即切换到 **方案 B**（更安全、侵入性更小）。

---

## 4. 执行计划

```
[Phase 1] 诊断确认（已完成） → [Phase 2] 执行修复（方案 A） → [Phase 3] 验收测试
```

### Phase 1 ✅ 已完成
- [x] 确认 `prefix=~/.npm-global` 导致 `npm prefix` 报错
- [x] 确认 npx 在当前 npm 版本（11.11.0）仍可工作
- [x] 确认 `~/.npm-global/bin` 有 15+ 全局工具
- [x] 确认 Hermes MCP 子进程环境过滤逻辑
- [x] 确认 GitHub MCP 当前使用 npx/stdio 模式

### Phase 2 — 执行修复
分配给: Claude Code

### Phase 3 — 验收测试
分配给: Claude Code + 人工确认

---

## 5. 回滚方案

如果修复后出现全局安装失效或 npx 异常：

1. **恢复 `~/.npmrc`**:
   ```bash
   echo 'prefix=~/.npm-global' >> ~/.npmrc
   ```
2. **恢复所有全局工具**:
   ```bash
   # 从 ~/.npm-global/bin 重新建立软链（如果被清理）
   ```
3. **切换到方案 B** 并重试

---

*任务包创建时间: 2026-05-21*
*版本: v1.0*
