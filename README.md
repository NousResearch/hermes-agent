# 🧠 ThinkCheck × Hermes — 水晶之心

> 为开源智能体 [Hermes Agent](https://github.com/NousResearch/hermes-agent) 装上自我审视的"逻辑之眼"。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![ThinkCheck 3.0](https://img.shields.io/badge/ThinkCheck-3.0-orange.svg)](https://github.com/luoxuejian000/hermes-agent)
[![Forked from NousResearch/hermes-agent](https://img.shields.io/badge/forked%20from-NousResearch%2Fhermes--agent-green.svg)](https://github.com/NousResearch/hermes-agent)

本项目是 NousResearch 出品的 [Hermes Agent](https://github.com/NousResearch/hermes-agent) 的增强分支。它在保留原项目所有强大能力（自我进化、三层记忆、广泛模型支持）的基础上，深度集成了我自研的 **ThinkCheck 3.0 推理评估引擎**。

**它的独特之处在于**：Hermes Agent 在生成文本后，会自动调用 ThinkCheck 进行"逻辑体检"，评估其推理质量（U统一性/D发展性/A对抗性/H和谐度），并给出通俗的改进建议。这使得"水晶之心"不止是一个能干的 Agent，更是一个能自我审视、持续进化的可靠伙伴。

---

## ✨ 与官方版本的核心区别

| 能力维度 | 官方 Hermes Agent | 🧠 水晶之心 (本仓库) |
| :--- | :--- | :--- |
| **推理质量评估** | ❌ 不支持 | ✅ 支持，提供U/D/A/H四维诊断报告 |
| **内容逻辑自检** | ❌ 不支持 | ✅ 支持，可自动发现并标注逻辑矛盾 |
| **概念漂移检测** | ❌ 不支持 | ✅ 支持，精准定位术语含义偏移 |
| **自我审视工具** | 无 | `thinkcheck_evaluate`，可被 Agent 自主调用 |

---

## 🧪 ThinkCheck 3.0 评估引擎

本仓库集成的 **ThinkCheck 3.0**，是一款基于晶脉哲学与谐振理论开发的AI推理质量诊断系统。

**四维评估指标**：

- **U (统一性)**：概念在文本中使用的语义一致性。
- **D (发展性)**：论证层次递进与新信息引入的节奏。
- **A (对抗性)**：文本内部逻辑矛盾的密度。
- **H (和谐度)**：综合前三项后得出的整体推理健康度。

**核心文件一览**：

- `thinkcheck_harmony/`：ThinkCheck 3.0 核心引擎的完整代码。
- `tools/thinkcheck_tool.py`：将引擎注册为 Hermes Agent 可调用工具的入口。
- `demo_thinkcheck.py`：一个可直接运行的测试脚本，直观展示诊断效果。

---

## 🚀 快速体验

1.  **克隆仓库**
    ```bash
    git clone https://github.com/luoxuejian000/hermes-agent.git
    cd hermes-agent
    ```