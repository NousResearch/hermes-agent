---
name: pixelpal-v21-emotion-alert-integration
description: PixelPal V21 EmotionAlert 组件集成教训 - subagent 交付后未集成的常见问题处理
tags: [pixelpal, react, integration, subagent]
---

# PixelPal V21 EmotionAlert 集成教训

## 背景
Dev agent 创建了 EmotionAlert 组件但未集成到任何 UI 页面。这是 subagent 常见失败模式之一。

## 问题
- EmotionAlert 放在 `src/services/emotion/` 而不是 `src/components/`（非标准）
- 组件内部有 `setInterval` + `useEffect`，在 JSX 内联回调中使用会违反 React Hook Rules
- 内联回调 `() => useStore.getState()` 会在渲染时执行 store 操作，产生副作用

## 解决方案
**不直接使用 EmotionAlert 组件**，而是：
1. 用纯函数 `checkEmotionAlertState()` 获取预警状态
2. 在 Sidebar 的 `useEffect` 中订阅 `emotion:logAdded` 事件
3. 自建 badge UI（pulsing heart icon）

```typescript
// Sidebar.tsx
const [showAlertBadge, setShowAlertBadge] = useState(false);
const [alertMessage, setAlertMessage] = useState('');

useEffect(() => {
  const checkAlert = () => {
    const alert = checkEmotionAlertState();
    setShowAlertBadge(alert.type !== 'none');
    setAlertMessage(alert.suggestion || alert.message);
  };
  checkAlert();
  window.addEventListener('emotion:logAdded', checkAlert);
  return () => window.removeEventListener('emotion:logAdded', checkAlert);
}, []);
```

## 教训
- Subagent 交付后必须检查组件是否真的被页面使用
- 组件位置非标准时，用纯函数包装比强行集成更稳定
- 涉及副作用的回调不要内联在 JSX 里
