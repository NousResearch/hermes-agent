---
name: canvas-touch-click-coordinate-debug
description: 移动端 canvas 元素 touch/click 坐标转换调试方法
---

# Canvas Touch/Click Coordinate Debug Skill

## Problem
移动端 canvas 元素，当 CSS 将 canvas 显示尺寸缩小（如 width:100%, max-width:400px 配合 canvas.width=800）时，touch 事件和 click 事件传入的 `clientX/clientY` 是 viewport 相对坐标，但 canvas 内部绘制用的是 canvas 坐标系（0-800）。

## Symptom
- 地图节点点击无反应（鼠标正常，触摸异常）
- 调试：click 和 touchstart 事件同时触发，坐标不一致

## Root Cause
`handleMapClick(e)` 用 `clientX - rect.left` 计算坐标，但 canvas CSS 显示宽度小于内部宽度时，这个差值不等于 canvas 内部坐标。

正确公式：
```javascript
const rect = canvas.getBoundingClientRect();
const scaleX = canvas.width / rect.width;   // canvas内部 / CSS显示
const scaleY = canvas.height / rect.height;
const x = (e.clientX - rect.left) * scaleX;
const y = (e.clientY - rect.top) * scaleY;
```

## Prevention
所有 canvas 上的 click/touch 事件处理函数都要做坐标转换。

## Context
card-game-prototype index.html ~line 10494-10508
