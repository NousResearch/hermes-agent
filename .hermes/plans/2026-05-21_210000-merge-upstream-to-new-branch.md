# 从 feature/shenshan 创建新分支并合并上游改动

## Goal

基于当前 `feature/shenshan` 分支创建一个新的合并分支，安全地将 GitHub 上游 (`upstream/main`) 的最新改动合入，不影响正在开发的 `feature/shenshan`。

## 当前上下文

| 项目 | 值 |
|------|-----|
| 当前分支 | `feature/shenshan` (已推送到 Gitee, 工作区干净) |
| 最新提交 | `440dbc340` fix(title_generator): 强制会话标题使用简体中文 |
| 本地 remote | `origin` → Gitee, `upstream` → GitHub |
| 上游最新 SHA | `48be2e0` (2026-05-21) |
| 上游落后天数 | ~24 天 (origin/main 停在 4/27) |

## 冲突风险评估

| 文件 | feature/shenshan 改动 | 上游改动 | 冲突风险 |
|------|----------------------|---------|---------|
| `run_agent.py` | +128 行 (x_user_token, Langfuse 等) | vision 路由微调 | 低 — 改动区域不同 |
| `pyproject.toml` | +langfuse optional | 移除 pytest-xdist | **中** — 可能冲突 |
| `uv.lock` | +339 行差异 | 锁文件更新 | **中** — 可重新生成 |
| `agent/image_routing.py` | 无改动 | vision 路由 | 零 |
| `agent/file_safety.py` | 无改动 | .env 写保护 | 零 |
| `hermes_cli/runtime_provider.py` | 无改动 | API Key 安全修复 | 零 |
| `gateway/platforms/telegram.py` | 无改动 | observe 扩展 | 零 |
| 其余 ~25 个测试文件 | 无改动 | 测试架构重构 | 低 |

## 步骤

### 1. 确保 upstream remote 数据最新

```bash
cd /home/renyh/.hermes/hermes-agent
git fetch upstream
```

**注意**: 国内直连 GitHub 可能超时。如果超时，改用代理或 mirror:
```bash
# 备选方案：通过 ghcr proxy 或 HTTP 代理
# 或从 origin/main 已有的基础上 cherry-pick
```

### 2. 从 feature/shenshan 创建新分支

```bash
git checkout -b feature/shenshan-upstream-merge feature/shenshan
```

### 3. 合并上游 main

```bash
git merge upstream/main --no-edit
```

**预期**: 
- 大部分文件 fast-forward 或自动合并
- 可能冲突: `pyproject.toml`, `uv.lock`

### 4. 处理冲突 (如有)

**pyproject.toml 冲突解决策略:**
- 保留 `feature/shenshan` 新增的 `langfuse` optional dependency
- 接受上游移除 `pytest-xdist` 的改动
- 手动合并后验证:
```bash
python3 -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"
```

**uv.lock 冲突解决策略:**
- 直接接受上游版本，然后重新 lock:
```bash
uv lock --no-update  # 或
uv lock
```

### 5. 验证合并结果

```bash
# 确认合并提交存在
git log --oneline -5

# 确认关键文件语法正确
python3 -c "import ast; ast.parse(open('run_agent.py').read())"
python3 -c "import ast; ast.parse(open('hermes_cli/runtime_provider.py').read())"

# 确认上游安全修复已合入
git log --oneline --all | grep -c "security\|file-safety\|API key leakage"
```

### 6. 推送到 Gitee

```bash
git push origin feature/shenshan-upstream-merge
```

## 备选方案: cherry-pick 精选提交

如果 merge 冲突太多或不想引入测试架构大改，可以 cherry-pick 高价值提交：

```bash
git checkout -b feature/shenshan-security-updates feature/shenshan

# 安全修复 (必选)
git cherry-pick 5edb346 c6a992e 9514ddb 5908822 eead464

# 功能改进 (可选)
git cherry-pick 32aea11 24c7ce0 b4afc65 a9db0e2

git push origin feature/shenshan-security-updates
```

## 验证清单

- [ ] `git status` 工作区干净
- [ ] `pyproject.toml` 语法正确
- [ ] `run_agent.py` 语法正确
- [ ] 安全修复文件已包含上游改动
- [ ] 新功能分支已推送到 Gitee
- [ ] 本地 `feature/shenshan` 不受影响

## 风险

1. **GitHub 网络超时** — `git fetch upstream` 可能失败。备选：从 origin/main 已有 SHA 基础上 cherry-pick
2. **uv.lock 冲突** — 锁文件变更量大。解决：重新 `uv lock`
3. **测试架构改动** — 上游测试从 xdist 改为 subprocess，影响 CI 但**不影响生产运行**
4. **pyproject.toml pytest-xdist 移除** — 如果你本地用 `pytest -n auto` 跑测试需要改回串行

## 文件可能变化

- `agent/file_safety.py` — .env 写保护
- `hermes_cli/runtime_provider.py` — API Key 安全修复
- `agent/image_routing.py` — vision 路由配置化
- `gateway/platforms/telegram.py` — Telegram observe 扩展
- `pyproject.toml` — 依赖变更
- `uv.lock` — 锁文件更新
- `scripts/run_tests.sh` — 测试脚本重构
- `tests/conftest.py` — 测试配置重构
- ~30 个测试文件
