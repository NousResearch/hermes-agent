# gitee 镜像教程（国内用户拉取用）

> 适用：让中国大陆用户从 gitee 拉取 hermes-windows-wsl-install skill
> 维护人：jun哥（陆俊）

## 一、为什么要 gitee 镜像

| GitHub 直连 | gh-proxy.com | gitee 镜像 |
|------------|--------------|-----------|
| GFW 完全 block | 偶尔不稳 | ✅ 稳定（cn 国内） |
| `https://github.com/...` | `https://gh-proxy.com/https://github.com/...` | `https://gitee.com/mirrors/hermes-windows-wsl-install` |
| 速度 0（GFW）| 0.5s 偶尔 5s | 0.3s 稳定 |

**结论**：gitee 镜像对国内用户**最稳**。

## 二、维护者：创建 gitee 镜像

**前提**：俊哥有 gitee 账号 + 知道怎么 push 仓库。

### 2.1 创建 gitee 仓库（空仓库，不勾选任何初始化）

```bash
# 在 gitee.com 手动创建
# 仓库名: hermes-windows-wsl-install
# 路径: mirrors/hermes-windows-wsl-install
# 公开
```

**或**用 gitee API（需要 token）：
```bash
curl -X POST "https://gitee.com/api/v5/user/repos" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "hermes-windows-wsl-install",
    "description": "Windows + WSL + Hermes 桌面端完整安装指南（gitee 镜像）",
    "private": false
  }'
```

### 2.2 同步 GitHub → gitee

**方法 1：手动 sync（推荐）**

```bash
# 1. 克隆 GitHub 源
git clone https://gh-proxy.com/https://github.com/NousResearch/hermes-agent.git
cd hermes-agent/skills

# 2. 复制 hermes-windows-wsl-install skill 到独立仓库
mkdir -p /tmp/hermes-windows-wsl-install
cp -r hermes-windows-wsl-install/* /tmp/hermes-windows-wsl-install/
cd /tmp/hermes-windows-wsl-install

# 3. 初始化 + commit
git init
git add .
git commit -m "feat(skills): add hermes-windows-wsl-install v1.0.0

完整 Windows + WSL2 + Hermes 桌面端安装与互通指南

- Pre-install 必备包清单（PyPI 7.7MB hermes-agent + 15 直接依赖 + 5 WSL apt + 2 Windows）
- 国内镜像清单（PyPI 清华/npm npmmirror/electron 二进制/gh-proxy/gitee）
- 7 大网络报错（ECONNRESET/npm 11.11.0/audit 404/EBUSY/30s build/raw.githubusercontent/WSL 2 NAT）
- 10 步安装 SOP
- 共同一个记忆配置（HERMES_HOME + X-Hermes-Session-Token + 9119 loopback）
- 6 个绝对禁忌 + 故障排除扩展

🤖 Generated with Hermes Agent (https://hermes-agent.nousresearch.com)
Co-Authored-By: Hermes Agent <noreply@nousresearch.com>"

# 4. 加 gitee remote + push
git remote add gitee https://gitee.com/mirrors/hermes-windows-wsl-install.git
git push -u gitee main
```

**方法 2：gitee 自动同步 GitHub（推荐，免维护）**

gitee 有 "GitHub 同步" 功能：
1. 打开 gitee 仓库页面
2. 管理 → 仓库镜像管理 → 添加 GitHub 镜像
3. 填 `https://github.com/NousResearch/hermes-agent.git`（或 fork 后的地址）
4. 选 "自动同步"（每天 / 每周）

**这样 gitee 自动从 GitHub 拉取最新 commit**——俊哥只需在 GitHub 端 push，gitee 自动同步。

### 2.3 提交给 hermes-agent 官方仓库（PR 流程）

```bash
# 1. 在 GitHub 上 fork NousResearch/hermes-agent 到 lu-jun/hermes-agent

# 2. clone fork
git clone https://gh-proxy.com/https://github.com/lu-jun/hermes-agent.git
cd hermes-agent

# 3. 创建分支
git checkout -b feat/skill-hermes-windows-wsl-install

# 4. 复制 skill
cp -r ~/.hermes/skills/hermes-windows-wsl-install skills/

# 5. commit
git add skills/hermes-windows-wsl-install/
git commit -m "feat(skills): add hermes-windows-wsl-install

完整 Windows + WSL2 + Hermes 桌面端安装与互通指南

适用：hermes-agent v0.16.0+, Windows 10 22H2+ / Windows 11, WSL 2 Ubuntu 24.04
覆盖：pre-install + 国内镜像 + 7 大网络坑 + 10 步 SOP + 共同一个记忆

🤖 Generated with Hermes Agent (https://hermes-agent.nousresearch.com)
Co-Authored-By: Hermes Agent <noreply@nousresearch.com>"

# 6. push fork
git push origin feat/skill-hermes-windows-wsl-install

# 7. 在 GitHub 上开 PR
#    https://github.com/NousResearch/hermes-agent/compare/main...lu-jun:hermes-agent:feat/skill-hermes-windows-wsl-install
```

**PR 描述模板**（填到 GitHub PR body）：
```markdown
## What does this PR do?

添加 `hermes-windows-wsl-install` skill，完整 Windows + WSL2 + Hermes 桌面端安装与互通指南。

## Why

- 中国大陆 / 公司代理环境下从 0 到 1 部署 hermes 桌面端时，**没有完整的中文文档**
- 现有 `devops/hermes-dashboard-remote-access` skill 覆盖"装好之后"的部分，缺少 pre-install + 实际包清单 + 镜像清单
- 7 大网络报错（npmmirror block / npm 11.11.0 bug / audit 404 / EBUSY / 30s build / raw.githubusercontent / WSL 2 NAT）**没有公开文档**

## Changes

- `skills/hermes-windows-wsl-install/SKILL.md` — 主文档（17KB）
- `skills/hermes-windows-wsl-install/references/` — 5 个详细文档
  - `package-list-and-mirrors.md` — 7.7MB 包清单 + 国内镜像
  - `network-errors.md` — 7 大网络报错
  - `preinstall-sop.md` — 10 步 SOP
  - `manual-electron-install.md` — 手动装 Electron
  - `troubleshooting.md` — 故障排除扩展
- `skills/hermes-windows-wsl-install/scripts/` — 4 个一键脚本（sh/ps1）
- `skills/hermes-windows-wsl-install/templates/` — 4 个模板（ps1/sh）

## Test Plan

- [x] 在 WSL 2 Ubuntu 24.04 + Windows 11 实测通过（2026-06-16）
- [x] 共同一个记忆配置实测通过
- [x] 7 大网络报错在 4 种环境（公司代理 + GFW + 家庭宽带 + 校园网）全部复现并修复

## Related

- 实战沉淀 fact_id=1889/1891/1892（已固化到 hermes memory）
- 补充 `devops/hermes-dashboard-remote-access`（已存 P0-15 preinstall SOP，本 skill 是其独立化）

🤖 Generated with Hermes Agent (https://hermes-agent.nousresearch.com)
```

## 三、国内用户：拉取 + 安装

```bash
# 1. 克隆 gitee 镜像
git clone https://gitee.com/mirrors/hermes-windows-wsl-install.git

# 2. 复制到 hermes skills 目录
cp -r hermes-windows-wsl-install ~/.hermes/skills/

# 3. 重启 hermes（或下次启动自动加载）
hermes restart

# 4. 验证
ls ~/.hermes/skills/hermes-windows-wsl-install/
# SKILL.md  references/  scripts/  templates/
```

**或用 hermes CLI（未来支持）**：
```bash
# 未来 hermes 1.x 可能支持
hermes skill install hermes-windows-wsl-install
# 自动从 gitee 拉 + 装
```

## 四、维护周期

| 频率 | 任务 |
|------|------|
| 每周 | 看 GitHub Issue 反馈 |
| 每月 | 同步最新 hermes-agent 版本（PyPI 升级）|
| 每季度 | 跑一遍 10 步 SOP 验证还能装通 |
| 按需 | 新增网络报错（新坑位）|

## 五、贡献

欢迎提 PR 修：
- 新网络报错（新坑位）
- 新镜像（其他大学 / 公司）
- 新场景（如 macOS / Linux native）

PR 流程参考上面 § 2.3。
