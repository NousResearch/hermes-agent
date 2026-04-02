# Skills 系统架构

## 7.1 概述

**功能**: Hermes 的自改进技能系统，允许 Agent 从经验中创建和优化技能。

---

## 7.2 Skill 结构

```
┌─────────────────────────────────────────────────────────────────┐
│                      Skill 结构                                 │
└─────────────────────────────────────────────────────────────────┘

optional-skills/
└── <skill-name>/
    ├── DESCRIPTION.md       # 元数据 (名称、版本、作者、标签)
    ├── SKILL.md            # 技能指令定义
    └── references/          # 参考文档
        ├── api.md
        └── examples.md
```

---

## 7.3 Skill 生命周期

```
┌─────────────────────────────────────────────────────────────────┐
│                    Skill 生命周期                                 │
└─────────────────────────────────────────────────────────────────┘

1. 定义 (Definition)
   └─ 在 optional-skills/ 创建 skill 目录

2. 注册 (Registration)
   └─ skills_tool.py 扫描并注册

3. 激活 (Activation)
   └─ Agent 根据上下文自动激活或用户手动触发
   └─ SKILL.md 内容注入 system prompt

4. 执行 (Execution)
   └─ Agent 使用 Skill 关联的工具

5. 学习 (Learning)
   └─ Agent 从经验中改进 Skill
```

---

## 7.4 内置 Skills

| Skill | 目录 | 功能 |
|-------|------|------|
| `index-cache` | `skills/index-cache/` | 索引缓存优化 |
| `github` | `skills/github/` | GitHub 集成 |

---

## 7.5 Skills Hub

Skills Hub (agentskills.io) 允许从社区共享和安装技能。

```python
# tools/skills_hub.py
class SkillsHub:
    """Skills Hub 集成"""

    def install(self, skill_name: str) -> None:
        """从 Hub 安装技能"""
        # 1. 获取技能元数据
        metadata = self.api.get_skill(skill_name)

        # 2. 下载技能文件
        self.download_skill(metadata)

        # 3. 安装到 ~/.hermes/skills/
        self.install_to_user_dir(metadata)

    def list_available(self) -> List[dict]:
        """列出 Hub 上的可用技能"""
        return self.api.list_skills()
```

---

## 7.6 关键源码索引

| 功能 | 文件 | 关键类/函数 |
|------|------|------------|
| Skill 注册 | `tools/skills_tool.py` | `SkillRegistry` |
| Hub 集成 | `tools/skills_hub.py` | `SkillsHub` |
| Skill 执行 | `agent/skill_commands.py` | `SkillExecutor` |

---

## 7.7 下一步

- [Skills Hub](02-hub.md) - Hub 集成详解
- [ACP 适配器](../../08-advanced/01-acp.md) - 编辑器集成

---

[← 返回总目录](../SUMMARY.md) | [← 记忆系统](../../06-memory-state/03-honcho.md) | [Hub →](02-hub.md)
