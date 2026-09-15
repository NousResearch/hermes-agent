---
name: dbg-card-game-workflow
description: PRJ-20260421-001 DBG卡牌游戏标准化开发流程 - PRD起草→dev agent委托→验收→部署
---

# DBG 卡牌游戏开发流程

## 概述
PRJ-20260421-001 DBG卡牌游戏的标准化开发流程。

## 标准迭代流程

1. **起草 PRD** → `workspace-pm/proposals/P-YYYYMMDD-NNN-prd.md`
2. **更新 proposal-index.md** → `approved_for_dev` + `confirmed`
3. **委托 dev agent** → `delegate_task`，传 PRD 路径 + 游戏文件路径
4. **Dev 完成检查** → `git log --oneline origin/gh-pages -3` 确认 commit 存在
5. **验收** → browser 验证功能，console 检查关键函数
6. **更新 proposal-index.md** → `accepted` + dev commit hash
7. **询问 boss** → 是否继续下一迭代

## Dev Agent 委托模板

```
项目路径: /mnt/c/Users/YeZhimin/Desktop/card-game-prototype/index.html
PRD: /home/hermes/.hermes/proposals/workspace-pm/proposals/P-XXXXXXXX-XXX-prd.md
GitHub Token: (见记忆文件中的 gh_token)
```

## 关键检查点

### 验证部署成功
```bash
git log --oneline origin/gh-pages -1
```
输出包含版本号和内容说明即成功。

### 验证功能存在
```javascript
// browser console
typeof functionName !== 'undefined'
Object.keys(RELICS || {}).length
FLOORS.length
```

### GitHub Pages 缓存刷新
```
https://yeluo45.github.io/card-game-prototype/?t=1746288000
```
时间戳每次不同即可。

## 常见问题

### Dev agent 未完成 git push
当 dev agent 用 max_iterations 限制时，可能在 git push 前达到上限。
**处理**：检查 `git status --short`，如有修改则手动 commit + push。

### 并行 Subagent 委托后的 Git 流程（V38 教训）
两个 subagent 并行处理后，各自可能只完成了部分修改。
**完整处理流程**：
```bash
# 1. 检查状态
git status --short

# 2. 合并冲突时（subagent 互相冲突）取新版本
git checkout --theirs index.html
git add index.html

# 3. 提交
git add index.html
git commit -m "V38: 功能描述"

# 4. pull --rebase 处理远程新commit
git pull origin main --rebase

# 5. push
git push origin main

# 6. 验证 gh-pages（因为该项目的Actions不是push-to-main触发）
git log --oneline origin/gh-pages -1
```

### GitHub Actions 部署触发机制（card-game-prototype 特例）
该项目的 `pages-build-deployment` workflow 使用 `dynamic` 事件触发，**不是**标准的 push-to-main 自动触发。
**部署到 GitHub Pages 的正确方式**：
```bash
# 推送 main 到 gh-pages 触发 Actions
git push origin main:gh-pages -f
```
验证：`gh run list --workflow pages-build-deployment --limit 3`

### 浏览器显示旧版本
- 用 `?v=N` 或 `?t=timestamp` 强制刷新
- 检查 HTML `<title>` 确认实际版本
- 用 browser console 验证 `gameState` 字段是否存在

## 主 Agent 手写 + Subagent 并行的模式（PvZ 教训）

PvZ (PRJ-20260421-001) 项目中，当需要同时实现多个独立功能模块时（如成就系统+3个小游戏模式），采用以下策略：

**核心系统 → 主 Agent 手写**：
成就系统 (achievements.py) 是整个系统的基础，如果出问题会影响所有模式，由主 Agent 直接编写，确保正确性。

**独立功能 → 并行 Subagent 委托**：
当多个功能完全独立、无相互依赖时（如 lawnbowling.py/endless.py/zen.py），用 `delegate_task` 的 `tasks` 参数并行委托给多个 subagent（最多3个），各自独立完成一个文件。

**整合验证 → 主 Agent 接管**：
Subagent 完成后，主 Agent 负责：
1. 验证所有文件语法正确 (`py_compile`)
2. 更新 main.py 整合所有新状态
3. 验证集成测试通过
4. GitHub push

**委托模板（多并行）**：
```
上下文：游戏路径 /home/hermes/prj-plants-vs-zombies
目标：写 source/state/<feature>.py
游戏技术栈：Python + Pygame
关键常量：SCREEN_WIDTH=800, SCREEN_HEIGHT=600, GRID_OFFSET_X=50, GRID_OFFSET_Y=100, CELL_WIDTH=83, CELL_HEIGHT=90, GRID_ROWS=5, GRID_COLS=9
```

**Subagent 常见问题**：
- Terminal 调用可能报错（权限问题），但文件写入通常成功
- 完成后用 `py_compile` 验证语法，不用 terminal lint

## 版本历史

| 版本 | 提案 | 主要内容 |
|------|------|----------|
| V1 | P-20250421-001 | 核心战斗循环 |
| V2 | P-20260502-003 | 卡牌扩充 + 敌人扩充 |
| V3 | P-20260502-006 | 地图/进度系统 |
| V4 | P-20260502-011 | 诅咒牌与特殊牌系统 |
| V5 | P-20260502-012 | 战斗奖励卡牌选择系统 |
| V6 | P-20260502-013 | 遗物/神器系统 |
| V7 | P-20260502-015 | 敌人与Boss扩充 |
| V8 | P-20260502-016 | 卡牌升级系统 |
| V9 | P-20260503-001 | 成就系统 |
| V10 | P-20260503-002 | 章节扩展（第4/5层） |
| V11 | P-20260503-003 | 更多遗物效果 |
| V37 | P-20260503-019 | 敌人意图图标化 |
| V38 | P-20260503-020 | 新手教程+卡牌使用率追踪 |

## 游戏访问
https://yeluo45.github.io/card-game-prototype/
