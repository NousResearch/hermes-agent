---
name: production-timeline-manager
description: 白梦客排期管理技能 - 项目时间线,里程碑设定,进度跟踪
version: 1.0.0
author: 白梦客AI创作团队
tags: [timeline, scheduling, milestones, deadline, 白梦客]
---

# Production Timeline Manager - 白梦客排期管理

项目时间线、里程碑设定、进度跟踪。

## 核心职责

- 创建项目时间线
- 设定里程碑
- 跟踪进度
- 管理 Deadline

## 阶段结构

| 阶段 | 周期 | 交付物 |
|------|------|--------|
| 预产 | 1-3天 | Brief解析、创意方向、分镜脚本 |
| 资产制作 | 2-7天 | 视觉素材、音频、动效 |
| 后期 | 1-3天 | 剪辑、调色、混音、输出 |

## 里程碑模板

```
Milestone: [名称]
类型: [阶段门/检查点/硬截止/软截止]
日期: YYYY-MM-DD
负责人: [角色]
状态: [pending/in_progress/at_risk/complete/overdue]
```

## 进度跟踪

### 日报
```
日期: [今天]
项目: [项目名]
阶段: [当前阶段]
进度: [X/Y里程碑]
状态: [正常/风险/延期]
阻塞: [问题列表]
```

## 预警阈值

- **黄**：截止前1天
- **橙**：截止日
- **红**：已过期

## 协作接口

- **production-manager**：资源配置、风险评估
- **creative-director**：范围变更、创意决策
- **director**：拍摄/视觉时间线对齐
- **editor**：后期制作排期