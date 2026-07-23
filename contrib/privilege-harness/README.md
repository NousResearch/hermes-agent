# Hermes Privilege Harness — `vip_sudo`

> 🔐 **LLM 只有一个提权通道，永远经过你批准。**
> 🔐 **One path to root. Always user-approved.**

[English](#english) | [中文](#chinese)

---

## English

This is my first exploration into **constraining LLM agents at the infrastructure level** — starting with the most dangerous capability: `sudo`.

Hermes' built-in sudo works by storing your password in plaintext (`SUDO_PASSWORD` in `.env`). That password is visible in files, environment variables, and process memory — reachable by any injected code.

**Privilege Harness eliminates passwords entirely.** Instead, a dedicated system user (`_hermesvip`) holds NOPASSWD sudo, but the LLM can never touch it directly. A daemon speaks for it. Every privileged command flows through a single gate: `vip_sudo`. Every execution requires your explicit approval via Hermes' native interactive card.

### Two Branches, Two Philosophies

| | `main` (Active Guard) | `passive-vip` (Community) |
|---|---|---|
| **Philosophy** | VIP polices ALL sudo | Hermes governs, VIP executes |
| **Terminal sudo** | ❌ Blocked, redirected to vip_sudo | Hermes manages it (dangerous patterns) |
| **vip_sudo approval** | Native card + anti-loop + session state | Native card + stamp verification |
| **Target** | Personal use, high-security environments | Community PR, minimalist, easy to review |
| **Code size** | guard.py ~200 lines + gateway handler | guard.py ~140 lines |

**`main`** is the full-featured version I use daily. It actively intercepts any `terminal("sudo ...")` call and redirects to vip_sudo. Includes anti-loop protection and session-scoped approval memory.

**`passive-vip`** is designed for community adoption. It removes the active guard — Hermes' native dangerous command detection handles blocking. VIP only does one thing: accept pre-approved commands and execute them via daemon. A stamp verification system ensures unstamped commands are rejected, even if someone bypasses the approval card.

### Install

```bash
git clone https://github.com/CortonKwok-GGD/hermes-privilege-harness.git
cd hermes-privilege-harness
git checkout main          # or passive-vip
sudo bash install.sh
```

Requires Hermes Agent >= v0.18.0. macOS & Linux.

### License

MIT

---

## 中文

这是我对 **LLM Agent 在基础设施层面进行约束** 的初步探索——第一步就是管住最危险的能力：`sudo`。

Hermes 原生的 sudo 方案是把密码明文存在 `.env` 里，LLM 调 `sudo` 时自动注入。密码在文件、环境变量、进程内存中到处飘——任何注入代码都能读到。

**Privilege Harness 完全不需要密码。** 创建一个专用系统用户（`_hermesvip`），它有 NOPASSWD sudo 但 LLM 永远不能直接访问。一个 daemon 替它发声。所有提权命令走唯一入口 `vip_sudo`，每次执行都要你在 Hermes 原生交互卡片上批准。

### 两个分支，两种哲学

| | `main`（主动守卫） | `passive-vip`（社区版） |
|---|---|---|
| **哲学** | VIP 管住所有 sudo | Hermes 守门，VIP 执行 |
| **终端 sudo** | ❌ 拦截，重定向到 vip_sudo | Hermes 原生危险检测管理 |
| **vip_sudo 审批** | 原生卡片 + 防循环 + 会话状态 | 原生卡片 + stamp 防绕过验证 |
| **适用场景** | 自用、高安全环境 | 社区 PR、最小化、易审阅 |
| **代码量** | guard.py ~200行 + gateway handler | guard.py ~140行 |

**`main`** 是我日常使用的完整版。它主动拦截任何 `terminal("sudo ...")` 调用并重定向到 vip_sudo，包含防循环保护和会话级审批记忆。

**`passive-vip`** 为社区采纳而设计。去掉了主动守卫——Hermes 原生的危险命令检测负责拦截。VIP 只做一件事：接收已被批准的指令，通过 daemon 执行。内置 stamp 验证机制，即使绕过审批卡片也会被拒绝执行。

### 安装

```bash
git clone https://gitee.com/cortonkwok/hermes-privilege-harness.git
cd hermes-privilege-harness
git checkout main          # 或 passive-vip
sudo bash install.sh
```

需要 Hermes Agent >= v0.18.0。支持 macOS 和 Linux。

### 许可

MIT
