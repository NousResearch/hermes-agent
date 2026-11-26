---
name: creative-drawing-board-iteration-workflow
description: creative-drawing-board 项目从 V1 到 V37 的高速迭代工作流 — PRD起草→dev委托→验证→交付→下一方向
---

# creative-drawing-board 高速迭代工作流

## 背景
creative-drawing-board 项目从 V1（745行）迭代到 V37（23567行），形成了一套成熟的高速迭代模式。

## 触发条件
- 项目：PRJ-20260418-002
- boss 选择迭代方向（P0/P1/P2）
- 当前版本 V33+，单 HTML 文件项目

## 工作流（5步）

### Step 1: 选方向 → PRD 起草
boss 选定方向后，立即起草 PRD：
```
PRD Path: ~/.hermes/proposals/workspace-pm/proposals/P-YYYYMMDD-NNN-prd.md
```
PRD 包含：背景/功能概览/数据结构/UI HTML+CSS/控制函数/验收目标/非功能需求/版本估算

### Step 2: 索引登记 → 委托 dev
在 `proposal-index.md` 添加新条目（status: approved_for_dev）
委托 subagent：
```
delegate_task goal=实现 VXX 功能...
max_iterations=50
```

### Step 3: subagent 实现 → 推送
subagent 负责：
- 读取 PRD
- 实现所有功能到 index.html
- git add + commit + push origin master
- git push origin master:gh-pages --force

### Step 4: 验证
```bash
cd /home/hermes/creative-drawing-board
git log --oneline -2 && wc -l index.html
```
检查：
- commit 是否完成
- 行数是否合理增长（+200~500行）
- 是否已 push

### Step 5: 索引更新 → 交付报告
更新 proposal-index.md：
- 修正实际行数
- 填入 Dev Commit SHA
- 移除 Acceptance: pending

向 boss 报告：
- 部署 URL + cache-bust 版本号
- 新增功能列表
- 版本演进表
- 下一步迭代方向（5选项 A/B/C/D/E）

## 关键约束

| 约束 | 说明 |
|------|------|
| 单文件 | 所有代码在 index.html |
| 零外部依赖 | 禁止 CDN |
| 向后兼容 | 每版继承所有之前功能 |
| 缓存 bust | `?v=N` URL 参数递增 |

## 版本估算规律

| 功能类型 | 预估增量 |
|----------|----------|
| 简单面板 | +200~300行 |
| 中等功能（商店/成就） | +300~400行 |
| 复杂功能（3D/动画） | +400~500行 |

实际行数通常比预估少 5~10%。

## Subagent 交付质量验证（重要）

**V43 教训**：dev agent 可能把功能加到错误的地方（如把动物模板加到点画游戏而非动画书）。

### 验证清单（每次必须）
```bash
# 1. 检查 commit 是否完成
git log --oneline -2

# 2. 检查实际代码变更（关键！）
git diff index.html | head -50
git diff index.html --stat

# 3. 搜索关键函数/字段确认在正确位置
grep -n "ANIMABOOK_PRESET_TEMPLATES" index.html  # 动画书模板
grep -n "dotTemplates" index.html                  # 点画模板
```

### subagent 未提交（偶发）
症状：`git status --short` 显示 `M index.html` 但无 commit。

处理：
```bash
cd /home/hermes/creative-drawing-board
git add index.html
git commit -m "VXX: 功能名"
git push origin master
```

### 需要手动 patch（偶发）
症状：subagent 达到 max_iterations 但功能不完整或在错误位置。

处理：
1. 检查代码缺失部分
2. 使用 patch 工具修复
3. 手动 commit + push

### Git push 网络问题（高频）
症状：`git push origin master:gh-pages -f` 超时或被拒绝。

解决方案（按优先级）：
1. **subprocess 方式**（最稳）：
```python
import subprocess, os
os.environ['GIT_TERMINAL_PROMPT'] = '0'
os.environ['GIT_ASKPASS'] = 'echo'
result = subprocess.run(
    ['git', 'push', 'origin', 'master:gh-pages', '-f', '--quiet'],
    cwd='/home/hermes/creative-drawing-board',
    capture_output=True, text=True, timeout=180
)
```

2. **重试+延迟**：
```python
for attempt in range(3):
    result = subprocess.run(...)
    if result.returncode == 0: break
    time.sleep(3)
```

## 版本历史（关键节点）

| 版本 | 行数 | 增量 | 核心功能 |
|------|------|------|----------|
| V38 | 23910 | +343 | 每日挑战 |
| V39 | 24453 | +543 | 翻页动画书+配音录音 |
| V40 | 24552 | +99 | GIF导出 |
| V41 | 24717 | +165 | 故事板时间轴 |
| V42 | 24924 | +207 | 预设动画模板(8种) |
| V43 | 25154 | +230 | 动物动画模板(8种) |
| V44 | 25402 | +248 | 配音变声特效(8种) |
| V45 | 25457 | +55 | 模板分类浏览(Tab) |

**总计**：V38→V45 共 8 个版本，+1547 行

## 工具使用
- `delegate_task`: 委托 subagent 实现
- `terminal`: git 验证、wc 统计
- `patch`: 手动修复 subagent 遗漏
- `write_file`: PRD 起草
- `read_file`: PRD 读取（供 subagent）

## 注意事项
- PRD 只需起草，无需 boss 确认（boss 已选方向）
- 索引更新在委托后立即做，交付后再修正实际行数
- subagent 完成后立即验证 git log，不要等
- 报告永远提供 5 个方向选项让 boss 选
