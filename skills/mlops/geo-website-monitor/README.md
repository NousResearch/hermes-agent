# GEO官网监测工具 - 快速入门指南

## 概述

本工具用于监测客户官网在大模型回答中的引用情况，并提供GEO优化建议。

## 论文依据

- **标题**: GEO: Generative Engine Optimization
- **会议**: KDD 2024
- **作者**: Aggarwal, P., Murahari, V. S., Rajpurohit, S., Kalyan, A., Narasimhan, K., & Deshpande, A.
- **arXiv**: 2311.09735
- **代码和数据**: https://generative-engines.com/GEO/

## 安装依赖

```bash
pip install requests beautifulsoup4
```

## 快速使用

```bash
# 仅技术审计
python templates/geo-website-monitor.py audit https://example.com

# 仅内容评估
python templates/geo-website-monitor.py content https://example.com

# 完整测试（技术审计+内容评估+Agent引用测试结果回填；最多5个场景）
# 触发skill一开始必须先询问用户：场景提示词由Agent自主生成，还是由用户提供？以及本次测试几个（1-5）
python templates/geo-website-monitor-v2.py full https://example.com --scenarios scenarios.csv --scenario-count 5 --citation-results agent-citation-results.json
```

## 场景词CSV格式示例

场景词最多支持5个；完整监测执行一开始，必须先问用户：场景提示词由 Agent 自主生成，还是由用户提供？以及本次需要测试几个场景提示词（1–5）。

```csv
场景词,优先级
快速换模系统厂家推荐,高
苏州快速换模系统厂家,高
液压快速换模系统厂家,中
```

## 设计原则

- **必须为每一个结论提供明确来源**: 不要只给出结果，要说明是如何得出的
- **不要提供主观的预期提升数据**: 只陈述事实和方法依据
- **必须引用论文出处**: 当提到GEO优化方法时，明确标注来源

## 文件说明

- `SKILL.md`: 完整的技能文档
- `templates/geo-website-monitor.py`: 主程序脚本
- `templates/llms.txt.template`: llms.txt模板
- `references/arxiv-2311-09735-summary.md`: GEO论文摘要解读
- `references/gelefu-evaluation-example.json`: 实际评估报告示例

## 如何安装为Hermes Skill

将本目录复制到 `~/.hermes/skills/mlops/geo-website-monitor/`（或其他合适的分类目录）。

## 常见陷阱

- **网站404页面返回200 OK**: 有些网站的404页面会返回200状态码并显示HTML内容，必须同时检查响应内容是否真的是robots.txt/llms.txt
