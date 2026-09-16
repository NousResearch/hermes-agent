---
name: dont-step-white-workflow
description: 别踩白块（dont-step-white）Web游戏标准化迭代流程 - React/Vite/PWA项目，从PRD到交付的完整流程
---

# 别踩白块 迭代工作流

## 项目概览

- **GitHub**: https://github.com/YeLuo45/dont-step-white
- **部署**: https://yeluo45.github.io/dont-step-white/ (gh-pages)
- **技术栈**: React 18 + Vite 5 + vite-plugin-pwa
- **开发目录**: `/home/hermes/.hermes/proposals/workspace-dev/proposals/dont-step-white/`
- **源码**: master 分支，gh-pages 分支部署

## 版本历史

| Version | Key Features |
|---------|--------------|
| V1 | 经典钢琴块玩法，4列8行网格，分数/速度递增，历史最高分，PWA |
| V2 | 无尽模式（命机制+combo计分），道具系统（护盾/冰冻/双倍） |
| V3 | 金币经济，皮肤商店（4套主题），关卡挑战（6关） |
| V4 | 分享式排行榜（URL base64编码，无需后端） |
| V5 | 移动端适配（安全区域/防缩放/横屏锁定/PWA安装优化） |

## 标准迭代流程

```
1. boss 选择迭代方向（P0-P4）
2. 小墨起草 PRD → `proposals/workspace-pm/proposals/P-YYYYMMDD-NNN-prd.md`
3. 更新 proposal-index.md → 新条目插在同级旧版本**之后**，Status: approved_for_dev
4. 委托 dev agent（opencode）→ 传 PRD 路径 + 项目路径 + max_iterations
5. dev 完成后：合并到 master → npm run build → gh-pages -d dist → 部署验证
6. 更新 proposal-index.md → Status: delivered，Stage: V{N}（已交付）
```

## Git 分支策略

- `master`: 主分支，所有功能合并到这里
- `feature/v{N}-{name}`: 每个版本的功能分支
- `gh-pages`: 部署分支，通过 `gh-pages -d dist` 更新

```bash
# 开发流程
git checkout master && git pull
git merge feature/v{N}-{name} --no-edit
git push origin master

# 构建并部署
npm run build
gh-pages -d dist  # 自动推送到 gh-pages
```

## 部署注意事项

**GitHub Pages CDN 传播延迟**：推送后资源不是立即可访问的，需要等待约 **20-30 秒** 才能通过 `yeluo45.github.io` 访问到新资源。

验证方法：
```bash
gh-pages -d dist
sleep 20
curl -sI https://yeluo45.github.io/dont-step-white/assets/index-XXXX.js | head -3
# HTTP/2 200 = 成功，HTTP/2 404 = 还在传播中，再等几秒
```

**gh-pages 更新原理**：`gh-pages -d dist` 会用本地 dist 目录的内容强制更新远程 gh-pages 分支。每次构建的 JS/CSS 文件名都不同（Vite hash），但 index.html 会同步更新。

## Subagent 委托模板

```
delegate_task(
  acp_command="opencode",
  goal="根据 PRD 实现别踩白块 V{N} 功能",
  context="""
  项目路径: /home/hermes/.hermes/proposals/workspace-dev/proposals/dont-step-white/
  PRD: ~/.hermes/proposals/workspace-pm/proposals/P-YYYYMMDD-NNN-prd.md
  部署地址: https://yeluo45.github.io/dont-step-white/
  
  要求:
  1. git pull 确保最新
  2. 创建 feature/v{N}-{name} 分支
  3. 实现所有功能
  4. npm run build 验证
  5. git commit + push
  6. 提供交付报告
  """,
  max_iterations=80
)
```

## Subagent 常见问题

**dev agent hit max_iterations 在 git push 之前**：大多数情况下 subagent 完成了代码实现但未能 push。需要手动检查：
```bash
git status
git log --oneline feature/v{N}-{name} -3
```
如果分支有提交但未推送，手动 push：
```bash
git push origin feature/v{N}-{name}
```

**npm install 失败**：检查是否引入了新依赖，如果 package.json 有变化：
```bash
npm install
npm run build
```

## V4 重要教训：后端服务宕机切换方案

**问题**：V4 原计划使用 LeanCloud 实现全球排行榜，但 LeanCloud 服务已下线。

**解决方案**：切换到纯前端 URL 分享方案
- 分享数据通过 base64 编码在 URL 参数中传递
- 无需任何后端服务
- 使用 `btoa(JSON.stringify(shareData))` 编码
- 接收时 `JSON.parse(atob(rankData))` 解码

**结论**：涉及第三方服务的功能， PRD 中应包含**备选纯前端方案**，避免服务不可用时需要临时重构。

## 关键 localStorage Keys

| Key | 用途 |
|-----|------|
| `dsw_v2_best` | V2 最高分 + 昵称 |
| `dsw_v3_coins` | V3 金币数据 |
| `dsw_v3_equipped` | V3 当前装备皮肤 |
| `dsw_v3_owned` | V3 已拥有皮肤列表 |
| `dsw_v3_progress` | V3 关卡进度 |
| `dsw_v4_shared` | V4 分享历史记录 |
| `dsw_v4_friends` | V4 好友列表 |

## 架构要点

- **皮肤系统**：CSS 变量方案，通过 `document.documentElement.style.setProperty('--skin-bg', skin.bg)` 切换
- **路由**：App.jsx 检查 `window.location.search` 中的 `rank` 参数决定显示主菜单还是战绩页面
- **PWA**：vite-plugin-pwa 生成 Service Worker + manifest，gh-pages 分支部署
- **音效**：Web Audio API，通过 localStorage 控制开关

## 验证命令

```bash
cd /home/hermes/.hermes/proposals/workspace-dev/proposals/dont-step-white

# 构建验证
npm run build

# 检查 dist 内容
ls dist/assets/

# 确认部署资源
gh-pages -d dist && sleep 20 && curl -sI https://yeluo45.github.io/dont-step-white/ | head -3

# 检查分支状态
git branch -a
git log --oneline master -3
```
