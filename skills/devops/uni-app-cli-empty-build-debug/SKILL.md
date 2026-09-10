---
name: uni-app-cli-empty-build-debug
category: devops
description: uni-app CLI 构建产物为空的调试方法 — index.html 路径问题导致 vite-plugin-uni 编译 0 个页面
tags: [uni-app, vue, vite, debug, build]
---

# uni-app CLI 构建产物为空的调试方法

## 触发条件
uni-app 3.0.0-406xx 版本使用 `npm run build:h5` 或 `npx vite build` 时，构建显示 "Build complete" 但产物只有 2 个文件（index.html + 1 个极小的 js/css），实际编译时间极短（4 秒 vs 正常的 60-90 秒）。

## 根因
uni-app vite-plugin 要求 `index.html` 必须位于 `src/` 目录下（与 `pages.json`、`manifest.json`、`App.vue`、`main.js` 同级），而不是项目根目录。index.html 在根目录时，vite-plugin-uni 解析不到任何页面，生成空 bundle。

## 修复步骤

1. 把 `index.html` 从项目根目录移到 `src/` 目录：
   ```bash
   mv /path/to/project/index.html /path/to/project/src/index.html
   ```

2. 检查 `vite.config.js` 中的 `inputDir` 配置：
   ```js
   inputDir: path.resolve(__dirname, 'src')
   ```
   确保 uni 插件能正确定位到 src 目录。

3. 清理缓存后重新构建：
   ```bash
   rm -rf node_modules/.vite node_modules/.cache node_modules/@dcloudio/.cache dist/
   npm run build:h5
   ```

4. 验证构建了正确的页面数：
   - 正常构建：产物包含所有页面的 chunk js 文件，体积较大（通常 > 1MB）
   - 空构建：只有 index.html + 1 个极小的 js/css（< 1KB）

## 预防措施
- 创建 uni-app 项目时，确保 `index.html` 在 `src/` 下而非根目录
- 使用 CLI 模式（而非 HBuilderX IDE 模式）时，页面文件必须全在 `src/pages/` 下
- 所有 `pages.json` 中引用的页面路径必须有对应的 `.vue` 文件存在

## 相关版本
- uni-app: 3.0.0-4060620250520001
- @dcloudio/vite-plugin-uni: ~3.0.0-406xx
- 问题表现：vite-plugin-uni/build.js 使用 `options.inputDir` 解析页面，inputDir 错误导致 0 页面被编译
