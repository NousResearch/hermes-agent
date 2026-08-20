---
name: react-print-isolation
description: React 应用中实现打印内容隔离的动态样式注入法
---
# React Print Isolation — 动态样式注入法

## 何时使用

当 React 应用需要实现"只打印特定区域，其他全部隐藏"的功能时使用。

**常见场景：** 报告打印、发票打印、导出 PDF、打印预览等。

## 核心问题

React 应用挂载在 `#root` div 内，元素嵌套层级深。传统的 CSS `@media print` 选择器（如 `body > *:not(.print\:block)` 或 `body *`）无法正确工作：

- `body > *:not(...)` 只匹配 body 的直接子元素，而 `#root` 才是直接子元素
- `body * { display: none }` 配合 `.print\:block { display: block }` 也不够，因为 `.print\:block *` 的 display 被 `.print\:block` 的 `display: block !important` 覆盖后又被子元素重新设为 none

## 解决方案：动态 style 注入

在进入报告模式时，通过 JavaScript 动态注入 `<style>` 标签到 document head：

```tsx
useEffect(() => {
  if (!reportMode) return
  const styleId = 'report-print-only'
  if (document.getElementById(styleId)) return

  const style = document.createElement('style')
  style.id = styleId
  style.innerHTML = `
    @media print {
      body > #root > * { display: none !important; }
      body > #root > .print\\:block { display: block !important; }
      .print\\:block * { display: block !important; }
      @page { margin: 0.5cm; size: A4; }
    }
  `
  document.head.appendChild(style)
  setTimeout(() => window.print(), 100)

  return () => {
    const el = document.getElementById(styleId)
    if (el) el.remove()
  }
}, [reportMode])
```

## 原理

| 选择器 | 作用 |
|--------|------|
| `body > #root > *` | 隐藏 #root 下所有子元素（即整个 App） |
| `body > #root > .print\:block` | 只显示报告容器本身 |
| `.print\:block *` | 报告内部所有元素都显示 |

## 关键要点

1. **时机**：在 reportMode 状态变为 true 时注入样式，打印对话框由 `window.print()` 触发
2. **清理**：在 useEffect return 中移除 style 标签，避免污染其他页面
3. **逃逸**：使用 `.print\:block` 类名标记要打印的区域，确保该元素及其子树在打印时可见
4. **@page**：通过 `@page { margin: 0.5cm; size: A4; }` 控制打印页面布局

## Tailwind CSS 的 print 类失效问题

Tailwind 的 `@tailwind utilities` 生成的类名带有反斜线（如 `print\:block`），在 CSS 中需要双反斜线转义 `print\\:block`。但在 React 的 `style.innerHTML` 中，只需单反斜线即可。

## 验证方法

1. 进入报告模式
2. 打开浏览器 DevTools → Elements，检查 document.head 是否注入 style 标签
3. 触发 Ctrl+P 打印预览，确认只有报告区域显示
