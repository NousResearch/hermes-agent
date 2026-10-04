---
name: uni-app-h5-build-debug
description: uni-app H5 build debugging — diagnosing empty bundle output when uni build reports "DONE Build complete" but produces no Vue code
tags: [uni-app, vite, vue, debug]
version: 1.0.0
---

# uni-app H5 Build Debugging

## Context
uni-app + Vite + Vue3 项目，`npm run build:h5` 始终只产出 882 字节的 pinia bootstrap JS，无 Vue 应用代码。

## Key Discoveries

### 1. CLI 静默吞掉错误

`@dcloudio/vite-plugin-uni/dist/cli/action.js` 的 `runBuild()` 函数：

```javascript
catch (e) {
    console.error(e.message || e);
    // BUG: H5 platform does NOT exit(1), just continues and prints "DONE"
    if (options.platform !== 'h5') {
        process.exit(1);
    }
}
```

即使 build 失败，H5 平台也继续执行到 `console.log(M['build.done'])`，导致 "DONE Build complete" 是误导。

**Fix**: 在 action.js 的 catch 块加 `process.exit(1)`，或加 debug 日志：

```javascript
catch (e) {
    console.error('[DEBUG runBuild] caught error:', e.message);
    if (process.env.UNI_EXIT_ON_BUILD_ERROR || options.platform !== 'h5') {
        process.exit(1);
    }
}
```

### 2. Vite build() 返回值

Vite `build()` 返回 `{ output: Rollup.OutputChunk[] }`，可用以下方式检查：

```javascript
const result = await build({...})
console.log('output length:', result.output.length)
result.output.forEach((chunk, i) => {
    console.log('chunk', i, ':', chunk.type, chunk.fileName)
})
```

### 3. "At least one <template>" 错误的真正含义

直接 Vite API 调用时报：
```
[vite:vue] src/App.vue: At least one <template> or <script> is required
```

这是 `@vitejs/plugin-vue` 的报错，说明 Vue SFC compiler 的 `parse()` 返回了空的 `descriptor`。App.vue 本身有 `<template>` 和 `<script>`，但 parse 失败。

可能原因：
- polyfill `rewriteCompilerSfcParse()` 在 `dist/utils/polyfill.js` 中用 `resolveBuiltIn('@vue/compiler-sfc')` 获取到的路径指向 `@dcloudio/uni-cli-shared` 捆绑的版本
- 但 CLI 和 API 调用时 module resolution 上下文不同，导致获取到的 compiler-sfc 实例不同

### 4. CLI vs API 结果不同

- CLI (`uni build --platform h5`)：返回 3 个 chunk 但写入位置不对
- 直接 Vite API (`vite.build()`)：报错但错误更明确

### 5. index.html 引用必须是 /src/main.js

uni-app 项目中 `index.html` 必须有：
```html
<script type="module" src="/src/main.js"></script>
```
NOT `/src/main.ts`（官方模板用 ts，但项目用 js）

### 6. uni CLI 不识别 "type": "module"

vite.config.js 在 CLI 模式下会报 CJS 错误。解决方案：重命名为 `vite.config.cjs`。

### 7. 静态资源导入错误

```
Unable to resolve `@import "./static/iconfont.css"` from /path/src/App.vue
```

创建空文件占位：
```bash
mkdir -p src/static
touch src/static/iconfont.css
```

### 8. package.json 丢失 "type": "module"

在修改 package.json 后（如 npm install），`"type": "module"` 可能被移除。需手动恢复。

## 文件路径

- CLI 入口：`node_modules/.bin/uni`
- action.js：`node_modules/@dcloudio/vite-plugin-uni/dist/cli/action.js`
- polyfill：`node_modules/@dcloudio/vite-plugin-uni/dist/utils/polyfill.js`
- copy 插件：`node_modules/@dcloudio/vite-plugin-uni/dist/plugins/copy.js`

## Debug 方法

### 添加 action.js debug 补丁
```javascript
// In runBuild() try block after build():
const result = await build(...)
console.log('[DEBUG] build() returned, result:', JSON.stringify(result ? { outputLength: result.output?.length } : 'null'))
if (result?.output) {
    result.output.forEach((chunk, i) => {
        console.log('[DEBUG] chunk', i, ':', chunk.type, chunk.fileName)
    })
}
console.log('[DEBUG] options.outDir:', options.outDir)
```

### 绕过 CLI 直接调 Vite API
```javascript
// 写一个临时脚本 /tmp/direct-build.js
const { createRequire } = require('module')
const req = createRequire(projectRoot + '/')
process.env.UNI_INPUT_DIR = projectRoot + '/src'
process.env.UNI_OUTPUT_DIR = projectRoot + '/dist'
process.env.UNI_CLI_CONTEXT = projectRoot
process.env.UNI_PLATFORM = 'h5'
process.env.NODE_ENV = 'production'
const polyfillModule = req('@dcloudio/vite-plugin-uni/dist/utils/polyfill.js')
polyfillModule.rewriteCompilerSfcParse()
const uniPlugin = req('@dcloudio/vite-plugin-uni').default
const plugins = uniPlugin()
const { build } = req('vite')
await build({ root: projectRoot, plugins, build: { outDir: '...' } })
```

## 根本性解决方案

如果调试无法解决，考虑放弃 uni-app CLI 构建，改用标准 Vite + Vue3 + hash 路由。成本最高但最可控。
