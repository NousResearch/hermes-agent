# Hermes Agent 源码学习指南

> 深入理解 Hermes Agent 的内部构造与核心机制

**版本**: v0.6.0 | **更新日期**: 2026-04-02

---

## 📖 文档目录

### 第一部分：整体架构
| 章节 | 标题 | 描述 |
|------|------|------|
| [01-overview/00-architecture](01-overview/00-architecture.md) | 系统架构总览 | Hermes 的整体架构设计、模块关系图 |
| [01-overview/01-core-concepts](01-overview/01-core-concepts.md) | 核心概念 | Agent 对话循环、Tool 系统、Memory、Skills |

### 第二部分：核心模块详解
| 章节 | 标题 | 描述 |
|------|------|------|
| [02-core/01-aiagent](02-core/01-aiagent.md) | AIAgent 核心 | run_agent.py 对话循环、迭代控制 |
| [02-core/02-message-flow](02-core/02-message-flow.md) | 消息流转机制 | 消息处理、上下文管理、压缩触发 |

### 第三部分：Tool 系统
| 章节 | 标题 | 描述 |
|------|------|------|
| [03-tool-system/01-registry](03-tool-system/01-registry.md) | 工具注册与调度 | 中心化 registry、装饰器注册机制 |
| [03-tool-system/02-discovery](03-tool-system/02-discovery.md) | 工具发现机制 | _discover_tools() 自动扫描 |
| [03-tool-system/03-core-tools](03-tool-system/03-core-tools.md) | 核心工具集 | File、Terminal、Web、Browser、MCP、Delegate |

### 第四部分：CLI 系统
| 章节 | 标题 | 描述 |
|------|------|------|
| [04-cli-system/01-main](04-cli-system/01-main.md) | CLI 入口与调度 | hermes_cli/main.py 命令体系 |
| [04-cli-system/02-setup](04-cli-system/02-setup.md) | 交互式设置向导 | setup.py 配置向导实现 |
| [04-cli-system/03-config](04-cli-system/03-config.md) | 配置管理系统 | config.py 默认配置、环境变量 |
| [04-cli-system/04-auth](04-cli-system/04-auth.md) | 认证与凭证 | auth.py 多 provider 凭证管理 |
| [04-cli-system/05-skin](04-cli-system/05-skin.md) | 主题皮肤系统 | skin_engine.py 数据驱动皮肤 |

### 第五部分：Gateway 系统
| 章节 | 标题 | 描述 |
|------|------|------|
| [05-gateway/01-overview](05-gateway/01-overview.md) | 网关架构总览 | gateway/run.py 消息网关架构 |
| [05-gateway/02-session](05-gateway/02-session.md) | 会话管理 | session.py 持久化与状态管理 |
| [05-gateway/03-platforms](05-gateway/03-platforms.md) | 平台适配器 | base.py 基类设计、Telegram/Discord 实现 |

### 第六部分：记忆与状态管理
| 章节 | 标题 | 描述 |
|------|------|------|
| [06-memory-state/01-sessiondb](06-memory-state/01-sessiondb.md) | SessionDB | SQLite FTS5 搜索、hermes_state.py |
| [06-memory-state/02-memory-tool](06-memory-state/02-memory-tool.md) | Memory Tool | 长期记忆、语义存储 |
| [06-memory-state/03-honcho](06-memory-state/03-honcho.md) | Honcho 集成 | dialectic 用户建模 |

### 第七部分：Skills 系统
| 章节 | 标题 | 描述 |
|------|------|------|
| [07-skills/01-overview](07-skills/01-overview.md) | Skills 架构 | Skill 定义、生命周期 |
| [07-skills/02-hub](07-skills/02-hub.md) | Skills Hub | agentskills.io 集成 |

### 第八部分：高级功能
| 章节 | 标题 | 描述 |
|------|------|------|
| [08-advanced/01-acp](08-advanced/01-acp.md) | ACP 适配器 | VS Code/Zed/JetBrains 集成 |
| [08-advanced/02-cron](08-advanced/02-cron.md) | Cron 调度 | 定时任务、scheduler.py |
| [08-advanced/03-rl](08-advanced/03-rl.md) | RL 训练环境 | Atropos、checkpoint 管理 |

### 附录
| 章节 | 标题 | 描述 |
|------|------|------|
| [appendix/01-file-index](appendix/01-file-index.md) | 文件索引 | 核心文件与功能映射表 |
| [appendix/02-config-ref](appendix/02-config-ref.md) | 配置项参考 | 所有配置项说明 |
| [appendix/03-glossary](appendix/03-glossary.md) | 术语表 | 核心概念解释 |

---

## 🚀 快速导航

### 按入口点查找
- **CLI 入口** → [04-cli-system/01-main](04-cli-system/01-main.md)
- **Gateway 入口** → [05-gateway/01-overview](05-gateway/01-overview.md)
- **Agent 入口** → [02-core/01-aiagent](02-core/01-aiagent.md)

### 按功能查找
- **工具系统** → [03-tool-system](03-tool-system/)
- **消息平台** → [05-gateway](05-gateway/)
- **记忆存储** → [06-memory-state](06-memory-state/)
- **扩展技能** → [07-skills](07-skills/)

---

## 📊 模块依赖图

```
hermes_cli/main.py
       ↓
    cli.py
       ↓
run_agent.py (AIAgent)
       ↓
model_tools.py (Tool orchestration)
       ↓
tools/registry.py (Central registry)
       ↓
tools/*.py (Individual tools)
```

---

## 🔗 相关资源

- [官方 README](../README.md)
- [项目贡献指南](../CONTRIBUTING.md)
- [ACP 安装指南](../docs/acp-setup.md)
- [Honcho 集成说明](../docs/honcho-integration-spec.md)

---

*本文档由 OMC Agent 自动生成，如有问题请提交 Issue。*
