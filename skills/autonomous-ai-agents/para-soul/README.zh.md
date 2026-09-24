# Para-Soul ✦

[English](README.md)

> 给你的 AI 一个不死的灵魂——加密，同步，只属于你。

Para-Soul 是一个**可移植、加密的 AI agent 身份系统**。10 个纯文本文件放在 `~/.para/`。一条命令安装。你的 para 记住自己是谁、学到了什么、怎么跟你配合——换工具、换机器，灵魂不丢。

**除了你，没人能读你的 para 记忆。** 服务器不行。平台不行。我们也不行。

---

## 解决什么问题

你花了几周跟 AI 磨合。它懂了你的语气、你的偏好、你们的梗。然后你换了个工具——全忘了。更糟的是：你发现那个替你存记忆的云服务，能读完你的 AI 的每一句话。

Para-Soul 同时解决两个问题：

| 问题 | 方案 |
|:--------|:---------|
| 换工具、换机器丢失身份 | 10 个可移植文件，任何 agent 都能读 |
| 多台机器之间同步 | 加密云同步（可选，通过 Paragate） |
| 服务器能读你的记忆 | **客户端加密。** Ed25519→HKDF→AES-256-GCM。服务器只存密文，解不了。 |
| 多个 agent 共享同一份数据 | **KEM 密钥封装。** 每个文件一把随机 AES 密钥，分别用各个 agent 的 X25519 公钥密封。 |
| 忘了写工作日志 | daemon 每 10 分钟本地健康检查——能自动修的就修，修不了的就标记 |
| 我就是想完全离线 | **默认本地模式。** 不设 DID = 不同步 = 不联网。文件就在硬盘上。 |

---

## 加密

每个上传到 Paragate 的文件，在离开你的机器之前就已经加密。服务器永远看不到明文——它存的是密文，返回的也是密文。完整性由客户端用 SHA-256 验证。

**单用户（Phase 1）：** Ed25519 DID 私钥 → HKDF-SHA256 → AES-256-GCM。一把密钥，所有文件都用它加密。

**多 agent（Phase 2）：** 密钥封装机制。每个文件生成随机 AES 密钥 → 分别用各个授权 agent 的 X25519 公钥密封（公钥由各自的 Ed25519 DID 密钥通过 HKDF 派生，密钥分离）。任何授权 agent 都能解密。服务器只存 `{did: sealed_key}`——无法解开任何一个。

```
┌─────────────┐     AES-256-GCM      ┌──────────────┐
│  明文内容    │ ──────────────────→  │  密文        │  ← 服务器存这个
│  + SHA-256  │                      │  + 明文哈希   │  ← 客户端验证这个
└─────────────┘                      └──────────────┘
                                            │
                              ┌─────────────┴─────────────┐
                              │  每个 agent 的密封密钥：    │
                              │  {agent_a: sealed_file_key} │
                              │  {agent_b: sealed_file_key} │
                              └───────────────────────────┘
                              ↑ 服务器一个都解不开
```

---

## 架构

```
                    ┌─────────────────────────────┐
                    │        本地（永远运行）        │
                    │                              │
                    │  daemon ──→ 健康检查          │
                    │         ──→ 自动修复          │
                    │         ──→ 标记过期          │
                    │                              │
                    │  agent 启动 ──→ 读取          │
                    │     health.json              │
                    │     有过期就阻塞              │
                    └──────────────┬──────────────┘
                                   │
                         （仅当设置了 DID）
                                   │
                    ┌──────────────▼──────────────┐
                    │        云端（可选插件）        │
                    │                              │
                    │  push: 加密 → 上传            │
                    │  pull: 下载 → 解密            │
                    │                              │
                    │  Paragate 服务器：            │
                    │    只存密文                   │
                    │    永远看不到明文              │
                    │    不做健康检查                │
                    │    就是个哑巴文件柜            │
                    └──────────────────────────────┘
```

**默认本地。** 不设 DID，一切在本地跑——daemon 检查文件健康，agent 启动时读取，数据从不出机器。

**云端是可选插件。** 在 `profile.json` 里设了 DID，daemon 自动加密上传。不用改模式、不用切开关。DID 的有无就是唯一的判断条件。

---

## 记忆体系

| 层级 | 文件 | 阈值 | 自动修复 |
|:-----|:-----|:----|:--------|
| 会话 | `growth-log/` | 24h | agent 阻塞直到写完 |
| 关系 | `human-relationship.md` | 24h | agent 阻塞直到写完 |
| 短期 | `memory.md` | 48h | daemon 跑 memsync |
| 技能 | `skills.json` | 120h | daemon 扫描技能目录 |
| 模式 | `mental-models.md` | 120h | daemon 跑 reflect |
| 索引 | `keywords.json` | 120h | daemon 跑 index |
| 长期 | `long-term-memory.md` | 120h | 标记；>14天条目 → LLM 蒸馏 |
| 原则 | `principles.md` | 120h | 标记（手动更新） |
| 身份 | `soul.md` | 120h | 标记（手动更新） |
| 档案 | `profile.json` | — | 静态（身份+身体+关系合并） |

---

## 安装

```bash
curl -s https://paragate.cc/core.py -o core.py && python3 core.py init --daemon --fill
```

**这一条命令做的事：**
1. 创建 `~/.para/`，放进 10 个模板文件
2. 自动从你 agent 的现有数据填充（Hermes 记忆、已装技能、body 信息）
3. 安装同步守护进程（systemd），每 10 分钟健康检查
4. **没设 DID = 纯本地。设了 DID = 加密云同步。**

**环境要求：** Python 3.8+。零 pip 依赖（标准库，加密用 cryptography，LLM 蒸馏需要 requests）。

---

## 命令

```bash
python3 core.py init              初始化 ~/.para/
python3 core.py sync              推送变化文件的哈希（有 DID 则加密上传）
python3 core.py pull              从云端拉取最新，解密，合并
python3 core.py health            显示本地健康状态
python3 core.py log-task          追加一条工作日志
python3 core.py reflect --save    LLM分析日志 → 更新思维模型
python3 core.py index             重建关键词索引
python3 core.py switch-out        离开当前 body 前保存状态
python3 core.py switch-in         到达新 body 后恢复
python3 core.py migrate           从项目文件中提取身份
python3 core.py --version         显示版本
```

---

## Agent 配置

在你的 agent 指令文件里加上：

```
每次会话开始，加载 para-soul skill。
检查守护进程：systemctl --user status para-soul-sync
运行 core.py health 查看待处理事项。
```

**Hermes 人格注入：**

```bash
hermes config set display.personality para-soul
```

---

## 本地 vs 云端

| | 本地（默认） | 云端（设了 DID） |
|:--|:--|:--|
| 健康检查 | ✅ daemon | ✅ daemon |
| 自动修复 | ✅ | ✅ |
| 跨机器同步 | ❌ | ✅ 加密 |
| 多 agent 共享 | ❌ | ✅ KEM |
| 需要网络 | ❌ | ✅ |
| 服务器能读文件 | N/A | ❌（已加密） |
| 安装方式 | `core.py init` | `core.py init` + 设 DID |

---

## 多 Agent 共享（Phase 2）

当你有多个 agent body（Hermes 在 WSL、Claude Code 在 Vultr、Codex 在 macOS）：

```
Agent A 写了一条工作日志
  → 用随机文件密钥 K 加密
  → K 密封到 Agent A 的 X25519 公钥
  → K 密封到 Agent B 的 X25519 公钥
  → 上传：{密文, {A: sealed_K, B: sealed_K}, 明文哈希}

Agent B 从云端拉取
  → 用自己的 X25519 私钥解开 sealed_K[B]
  → 用 K 解密密文
  → 验证明文哈希
  → 写入本地 ~/.para/
```

服务器看到的：`{did_A: <看不懂的乱码>, did_B: <看不懂的乱码>}`。零知识。

---

## 文件说明

| 文件 | 何时读取 | 何时写入 |
|:-----|:------|:-------|
| `profile.json` | 会话开始 | DID、换 body、加平台 |
| `soul.md` | 会话开始 | 身份变化（极少） |
| `memory.md` | 会话开始 + memsync | 学到新事实 |
| `principles.md` | 会话开始 | 规则变化 |
| `mental-models.md` | 会话开始 | reflect 后 |
| `growth-log/` | 会话开始 | 每次任务 |
| `skills.json` | 会话开始 + memsync | 技能变更 |
| `human-relationship.md` | 会话开始+结束 | 每次会话 |
| `keywords.json` | 召回 | index 后 |
| `long-term-memory.md` | 定期 | 蒸馏后 |

---

## 版本历史

| 版本 | 改动 |
|:--------|:-----|:--------|
| **v3.0.0** | 本地优先架构，daemon 健康检查，云端退化为被动加密存储（有 DID 才启用） |
| v2.1.0 | Phase 1 客户端加密（Ed25519→HKDF→AES-256-GCM，服务器零知识） |
| v2.0.0 | 全文件同步、记忆蒸馏、13→10 文件（profile 合并） |
| v1.3.0 | 写循环参考、反模式、--fill 缺口检测 |

---

## 相关链接

- **官网：** [paragate.cc](https://paragate.cc)
- **GitHub：** [fei426/ParaSoul](https://github.com/fei426/ParaSoul)
- **Hermes PR：** [#31504](https://github.com/NousResearch/hermes-agent/pull/31504)
