---
name: wslinux-uni-app-h5-build
description: WSL Linux 环境下 uni-app H5 构建与 GitHub Pages 部署
---

# WSL Linux 环境下 uni-app H5 构建与 GitHub Pages 部署

## 问题背景

WSL 环境下 uni-app 项目（Vue 3 + Vite）从零创建 CLI 构建配置并部署到 GitHub Pages 子目录 (`/future-little-leaders/`)，项目原为 HBuilderX IDE 模式，无 package.json 等构建文件。

## 已知坑点

### 坑点 1：旧 index.html 破坏 Vite 构建
**现象**：构建时报错 `Rollup failed to resolve import "/future-little-leaders/assets/index-xxx.js" from "index.html"`

**根因**：旧项目根目录的 `index.html` 引用了旧的 bundle 文件名（如 `index-COw3GvFH.js`），Vite 把根目录 index.html 当作入口模块来解析，而不是 uni-app 的内部入口。

**解决方案**：构建前用干净的 minimal index.html 替代旧的，内容只需：
```html
<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, user-scalable=no, initial-scale=1.0, maximum-scale=1.0, minimum-scale=1.0" />
    <title>应用名</title>
    <script type="module" crossorigin src="/src/main.js"></script>
  </head>
  <body>
    <div id="app"></div>
  </body>
</html>
```
uni-app 的 vite 插件会自动在构建时替换为正确的引用路径。

### 坑点 2：git reset --hard 丢失所有本地修改
**现象**：`git checkout origin/gh-pages -- .` 或 `git reset --hard` 后本地修改全部丢失，包括 pages.json 补丁、dist 构建产物、node_modules

**根因**：这些文件在工作目录但未被 git 追踪，或者 reset 直接回退到上一个 commit 状态

**教训**：WSL 开发时，**不要在 gh-pages 分支上直接做开发**。构建产物和源码应该分离：
- 源码修改在独立分支或 worktree
- 构建产物只通过 `git add dist/` 一次性提交，不要 reset

### 坑点 3：npm install 超时（网络问题）
**现象**：`npm install` 长时间超时甚至 300s+ 超时

**解决方案**：
```bash
npm cache clean --force
npm install --registry https://registry.npmmirror.com
```
中国镜像 `npmmirror.com` 比 npmmirror 官方源更快。

### 坑点 4：GitHub Token 401 但 gh CLI 正常
**现象**：Python 脚本用 Token 调用 GitHub API 返回 401 Unauthorized，但 `gh auth status` 显示 YeLuo45 已登录

**原因**：Token 已失效/被撤销，但 gh CLI 用自己的凭证缓存

**解决方案**：用 `gh api` 代替 Python urllib 直接调 API：
```bash
# GET
gh api repos/{owner}/{repo}/{endpoint}
# POST
gh api repos/{owner}/{repo}/{endpoint} -X POST -F 'content=...' -F 'encoding=base64'
```

### 坑点 5：git push via API 422 when network unstable
**现象**：WSL 网络不稳定时，git push 和 GitHub API 都间歇性失败

**解决方案**：分步重试 + 确认 `ghp_` Token 仍然有效

### 坑点 6：ESM 模式下 uni 插件导入失败
**现象**：`vite.config.js` 加了 `"type": "module"` 后，构建报错 `uni is not a function` 或 `default is undefined`

**根因**：`@dcloudio/vite-plugin-uni` 是 CJS 模块，`import uni from '@dcloudio/vite-plugin-uni'` 在 ESM 下拿不到 `default` 导出

**解决方案**：用 `createRequire` 处理 CJS/ESM 桥接：
```js
import { createRequire } from 'module'
import { defineConfig } from 'vite'
import path from 'path'
const require = createRequire(import.meta.url)
const uni = require('@dcloudio/vite-plugin-uni').default

export default defineConfig({
  plugins: [uni({ inputDir: path.resolve(__dirname, 'src') })],
  build: { outDir: 'dist/build/h5', emptyOutDir: true },
  resolve: { alias: { '@': path.resolve(__dirname, 'src') } }
})
```

### 坑点 7：uni CLI 静默吞掉错误 — "Build complete" 不等于构建成功
**现象**：`DONE  Build complete` 显示成功，但只有空的 `index.html`（303 字节），无任何 JS bundle，耗时仅 4-8 秒（正常 60-90 秒）

**根因**：`uni` CLI 包装了 `vite build`，当 Vite 的 transform 阶段失败（如 `compilerSfc.parse()` 报错）时，uni-cli 自己的 `onwarn` 处理器会把错误吞掉：
```js
// @dcloudio/vite-plugin-uni/dist/utils/logger.js
export function createHookLogger(name, warnFn) {
  return {
    onwarn(warning, warn) {
      if (warning.code === 'EMPTY_BUNDLE') return  // 静默忽略
      warnFn(warning.message)  // 其他警告也只打一行 message
    }
  }
}
```
这导致所有 Vite plugin 的 transform 错误被降级为普通 warning，Rollup 继续走完流程但产出了空 bundle。

**如何暴露真实错误**：用 `node scripts/build.js` 直接调用 Vite API，绕过 uni CLI 的错误处理链：

```js
// scripts/build.js
import { createRequire } from 'module'
import { build } from 'vite'
import path from 'path'
const require = createRequire(import.meta.url)

const uni = require('@dcloudio/vite-plugin-uni').default
const uniPlugin = (typeof uni === 'function') ? uni() : uni

await build({
  plugins: uniPlugin,
  logLevel: 'info',
  build: { outDir: 'dist/build/h5', emptyOutDir: true }
})
```

运行 `node scripts/build.js` 会直接抛出被 uni CLI 吞掉的核心错误，例如：
```
[vite:vue] src/App.vue: At least one <template> or <script> is required
```

**已知根因组合**：uni-cli-shared 捆绑的 `@vue/compiler-sfc@3.4.21` 在被 Vite/Rollup 封装后，调用 `parse()` 时行为异常 — 直接 Node.js 调用同一个 `.cjs.js` 文件可以成功，但通过 Vite plugin 链调用就报 `At least one <template> or <script> is required`（实际上文件有 template）。这不是版本冲突，而是 uni CLI 包装层和 Vite plugin 链的模块作用域问题。

**关键调试命令**：
```bash
# 检查 uni() 返回类型（不同版本返回不同东西）
node -e "const u = require('@dcloudio/vite-plugin-uni'); console.log(typeof u.default, typeof u)"

# 402xx: default 是数组
# 406xx/408xx: default 是函数，必须调用 uni() 才得到数组

# 直接测试 compiler-sfc parse 是否正常
node -e "
const { parse } = require('./node_modules/@dcloudio/uni-cli-shared/lib/@vue/compiler-sfc/dist/compiler-sfc.cjs.js');
const fs = require('fs');
try {
  const result = parse(fs.readFileSync('src/App.vue', 'utf-8'), { filename: 'src/App.vue' });
  console.log('Parse OK, template:', !!result.template);
} catch(e) {
  console.log('Parse FAILED:', e.message);
}
"
# 如果这里报 OK 但 build 报 fail，说明是 Vite/Rollup 封装层问题

# 确认 Vite plugin 是否实际被调用
# 在 vite.config.js 的 plugins 数组里加 console.log 调试
```

**解决方案（按优先级）**：
1. **换更老的 uni-app 版本**：`3.0.0-401` 系列可能没有这个问题（那时还没有 `rewriteCompilerSfcParse` polyfill）
2. **用官方 degit 模板验证**：`npx degit dcloudio/uni-preset-vue#vite-ts my-test` 创建干净模板，确认官方能正常构建
3. **降级 Vue 版本**：uni-app 406xx 自带 vue@3.4.21，不要用 overrides 改它
4. **放弃 uni-app CLI，换标准 Vite + Vue3 + hash 路由**：成本最高但最可控

### 坑点 7b：构建显示成功但产物为空（最常见！）
**现象**：`DONE Build complete` 但只有空的 `index.html`，无任何 JS bundle，耗时仅 4-8 秒（正常 60-90 秒）

**排查步骤**：
```bash
# 1. 确认所有页面文件存在
node -e "const ps = require('@dcloudio/uni-cli-shared').parsePagesJsonOnce('/path/src', 'h5'); ps.pages.forEach(p => { const exists = require('fs').existsSync('/path/src/' + p.path + '.vue'); console.log((exists?'OK':'MISSING')+' '+p.path) })"

# 2. 用 debug 模式看构建详情
npm run build:h5 -- --debug 2>&1 | grep -v "uni:require\|^[0-9]" | head -60

# 3. 确认 manifest.json 的 router.base 配置正确
cat src/manifest.json

# 4. 检查 build 中是否有 EMPTY_BUNDLE
npm run build:h5 -- --debug 2>&1 | grep EMPTY
```

**已知根因**：Vue 版本 overrides 与 uni-app 内部编译器版本链不兼容时，Vite 的 transform 阶段会静默跳过所有 .vue 文件，导致产出空 bundle。uni-app 的 `config/build.js` 里有 `EMPTY_BUNDLE` 警告被静默忽略的逻辑：
```js
if (warning.code === 'EMPTY_BUNDLE') {
  return; // 被吞掉了
}
```

**解决方案（按优先级）**：
1. 移除 package.json 中的 Vue `overrides`，让 uni-app 使用自带版本
2. 降级 uni-app 到更稳定的版本：`3.0.0-4020920240930001`
3. 如果必须用特定 Vue 版本，换用标准 Vite+Vue3 手写 uni-app API 兼容垫片，不用 `@dcloudio/vite-plugin-uni`

### 坑点 8：构建产物路径结构异常
**现象**：加了 `rollupOptions.input` 后，产物出现在 `dist/build/h5/src/index.html`

**根因**：uni-app 插件自己管理入口，不需要也不应该手动指定 `rollupOptions.input`

**解决方案**：删除 `rollupOptions.input`，让 uni 插件自己处理入口解析。

## 项目结构（CLI 模式）

```
project/
├── package.json          # 必须：uni-app CLI 依赖 + "type": "module"
├── vite.config.js        # 必须：createRequire CJS 桥接 + uni 插件
├── index.html            # 必须（minimal 版，放项目根目录）
├── src/
│   ├── main.js           # 必须：createSSRApp 入口
│   ├── App.vue           # 必须：iconfont 导入
│   ├── manifest.json     # 必须：uni-app 配置（H5 路由 base 必须设对）
│   ├── pages.json        # 必须：页面路由注册
│   ├── uni.scss          # 可选：全局样式变量
│   ├── pages/            # 页面组件
│   └── stores/           # Pinia stores
├── static/               # 静态资源（iconfont.css 等）
└── dist/build/h5/        # 构建产物（不提交到源码分支）
```

## GitHub Pages 子目录部署配置

`src/manifest.json` 中必须指定：
```json
"h5": {
  "router": {
    "mode": "hash",
    "base": "/future-little-leaders/"
  }
}
```

构建后 `dist/build/h5/index.html` 中引用路径为 `/future-little-leaders/assets/...`，对应 GitHub Pages URL `https://username.github.io/future-little-leaders/`。

## GitHub Pages 部署工作流

### 方式 A：源码分支分离（推荐）
1. `main` 分支：只放源码
2. `gh-pages` 分支：只放构建产物（index.html + assets/）
3. 在源码目录构建后，切换到 gh-pages 目录拉取新构建产物，提交推送

### 方式 B：在同一 repo 用 CLI 部署脚本
```python
import subprocess, json, os, base64

# 用 gh api 验证 token 有效
# 1. GET /git/refs/heads/gh-pages → SHA
# 2. GET /git/trees/{sha}?recursive=1 → existing files
# 3. 对比 SHA，过滤 unchanged 文件
# 4. POST /git/blobs 上传 changed 文件
# 5. POST /git/trees 创建新 tree
# 6. POST /git/commits 创建 commit
# 7. PATCH /git/refs/heads/gh-pages 更新 ref
```

## 验证清单

构建完成后检查：
- [ ] `dist/build/h5/index.html` 中 CSS/JS 路径包含 `/future-little-leaders/assets/`
- [ ] `dist/build/h5/assets/` 包含 `pages-achievement-achievement.*.js`（V2 产物）
- [ ] `dist/build/h5/assets/` 包含 `pages-report-report.*.js`（V2 产物）
- [ ] `dist/build/h5/assets/` 包含 `pages-task-template-picker.*.js`（V2 产物）
- [ ] GitHub Pages URL 可访问 `index.html`

## 关键路径
- 项目目录：`/home/hermes/future-little-leaders/`
- GitHub repo：`YeLuo45/future-little-leaders`
- GitHub Pages：`https://yeluo45.github.io/future-little-leaders/`
