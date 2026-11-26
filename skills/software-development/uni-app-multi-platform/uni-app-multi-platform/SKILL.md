---
name: uni-app-multi-platform
description: uni-app 多平台项目开发指南 — 项目创建、构建配置、依赖坑点、H5 部署到 GitHub Pages 子目录（含 Git Data API 推送与 .nojekyll 处理）
version: 1.0.0
triggers:
  - uni-app
  - uni-app H5
  - uni-app 多平台
  - uni-app build:h5
  - uni-app 微信小程序
metadata:
  hermes:
    tags: [uni-app, Vue 3, Vite, H5, 微信小程序, GitHub Pages]
    related_skills: [github-pages-uni-app-deploy, taro, app-development-guide]
    source_project: future-little-leaders
---

# uni-app 多平台项目开发指南

## 来源经验
本技能沉淀自 `future-little-leaders` 项目（P-20250418-004）的完整开发流程：
- 技术栈：uni-app + Vite + Vue 3 + Pinia（H5 + 微信小程序多端）
- 部署：GitHub Pages 子目录 (`/future-little-leaders/`)
- 部署方式：本地构建 → Git Data API 推送 dist 到 gh-pages 分支

---

## 环境要求

| 组件 | 版本要求 |
|------|----------|
| Node.js | >= 18（推荐 20，uni-app Vite 模式要求） |
| npm / yarn / pnpm | 最新稳定版 |
| @vue/cli / Vite | 内置（uni-app 项目自带） |
| 微信开发者工具 | Windows/Mac 桌面应用（开发小程序用，非必选）|

---

## 项目创建

### 方式一：HBuilderX 可视化（推荐新手）

1. 下载 [HBuilderX](https://www.dcloud.io/hbuilderx.html)
2. 新建项目 → 选择 `uni-app` → 填写项目名
3. 选择模板（默认 Vue 3）

### 方式二：CLI 命令行

```bash
# 使用 npx 创建（需已安装 @dcloudio/uni-app-cli）
npx degit dcloudio/uni-preset-vue#vite-ts my-project

cd my-project
npm install
```

> 注意：future-little-leaders 使用的是 Vue 3 + Vite 模式，非 TS 版本。CLI 创建时确认选择 `Vue 3` 而非 `Vue 2`。

---

## 目录结构（典型 Vue 3 Vite 项目）

```
src/
├── pages/                  # 页面
│   └── index/
│       └── index.vue
├── static/                 # 静态资源（图片、字体等）
├── App.vue                 # 应用入口
├── main.js                 # Vue 入口
├── manifest.json           # App 配置（H5、小程序、App 的元信息）
├── pages.json              # 页面路由配置
├── uni.scss                # 全局样式变量
└── vite.config.js          # Vite 配置
```

---

## 关键配置文件

### manifest.json（路由 base 配置）

部署到 GitHub Pages 子目录时，**必须在构建前**配置 `router.base`：

```json
{
  "name": "My App",
  "appid": "__UNI__XXXXXX",
  "description": "",
  "h5": {
    "router": {
      "base": "/{REPO_NAME}/"
    }
  }
}
```

**关键规则：**
- `router.base` 必须在构建前配置，构建后修改无效
- 修改后必须重新 `npm run build:h5`
- `/{REPO_NAME}/` 中的仓库名不能有前后空格

### pages.json（路由与 tabBar）

```json
{
  "pages": [
    {
      "path": "pages/index/index",
      "style": {
        "navigationBarTitleText": "首页"
      }
    }
  ],
  "tabBar": {
    "color": "#7A7E83",
    "selectedColor": "#3cc51f",
    "borderStyle": "black",
    "backgroundColor": "#ffffff",
    "list": [
      { "pagePath": "pages/index/index", "text": "首页" },
      { "pagePath": "pages/profile/profile", "text": "我的" }
    ]
  }
}
```

### vite.config.js（代理与构建配置）

future-little-leaders 的 vite.config.js 包含 API 代理（用于小程序开发阶段）：

```js
import { defineConfig } from 'vite'
import uni from '@dcloudio/vite-plugin-uni'

export default defineConfig({
  plugins: [uni()],
  server: {
    proxy: {
      '/api': {
        target: 'https://your-backend.sealoshzh.site',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  }
})
```

---

## 常见依赖坑点

### Vue 版本链 mismatch（Build complete 但无产物）

**问题表现**：`uni build` 或 `npm run build:h5` 显示 "DONE Build complete" 或 "Build complete"，但产物目录中只有 `index.html`，零 JS 文件或 JS 文件极小（仅几百字节 bootstrap 代码）。

**真实错误被 uni CLI 静默处理**，Rollup 报错时 CLI 只显示欢迎文本。检查产物目录确认：

```bash
ls dist/build/h5/
find dist/build/h5/ -name '*.js' | wc -l
wc -c dist/build/h5/assets/*.js
```

**典型错误信息**（从 Rollup 来，被 CLI 吞掉）：
```
"normalizeCssVarValue" is not exported by "@vue/shared"
imported by "@vue/runtime-core"
```
或
```
[vite:vue] src/App.vue: At least one <template> or <script> is required
```

**根因**：npm 安装了多个版本的 Vue 包（vue@3.5 + @vue/shared@3.4），导致 `@vue/runtime-core` 需要的 API 在实际加载的 `@vue/shared` 中不存在。

**诊断**：
```bash
npm ls vue @vue/runtime-core @vue/shared
```

**修复**：在 `package.json` 中加 `overrides` 强制统一版本：

```json
{
  "overrides": {
    "vue": "3.4.21",
    "@vue/runtime-core": "3.4.21",
    "@vue/runtime-dom": "3.4.21",
    "@vue/shared": "3.4.21",
    "@vue/compiler-core": "3.4.21",
    "@vue/compiler-dom": "3.4.21",
    "@vue/compiler-sfc": "3.4.21",
    "@vue/server-renderer": "3.4.21",
    "@vue/reactivity": "3.4.21"
  }
}
```

然后 `rm -rf node_modules package-lock.json && npm install`。

### uni CLI 行为差异

uni-app 不同版本的 `uni()` 插件返回类型不同：

| 版本 | uni() 返回类型 | 调用方式 |
|------|---------------|----------|
| 401xx | 数组（9个插件） | 直接 `plugins: [uni()]` 或 `plugins: [...uni()]` |
| 402xx | 数组 | 同上 |
| 406xx+ | 函数（返回包装对象） | `plugins: [uni()]` — 函数调用返回 `{ name: 'uni', apply, config }` |

当 CLI 行为异常时，直接 Vite API 调用可以暴露被静默的错误。

### Pinia 版本兼容性

**问题**：`npm run build:h5` 报错 `pinia@3.0.2 缺少 ./dist/pinia.mjs 导出`

**根因**：uni-app 某些版本自带的 pinia 3.x 与 Vite 构建不兼容

**解决**：降级到兼容版本

```bash
npm install pinia@2.2.6
```

**验证**：构建不再报错即可。构建成功后 lock 文件会固定版本。

### Node 版本问题

- Node 18：大多数情况兼容
- Node 20：完全兼容，推荐使用
- Node < 18：Vite 模式可能有问题

如遇奇怪报错，先确认 Node 版本：

```bash
node -v
```

---

## H5 构建与本地预览

### 构建 H5

```bash
npm install
npm run build:h5
```

产物输出到 `dist/build/h5/`，包含：
- `index.html` — 入口页面
- `assets/` — JS/CSS/图片资源（含 `_plugin-*.js` 等带下划线的文件）

### 本地预览

```bash
# 或开发模式预览
npm run dev:h5
```

---

## GitHub Pages 子目录部署

> 详见 `github-pages-uni-app-deploy` 技能，此处总结核心要点。

### 完整部署流程

**第一步：构建前配置**（必须）

在 `src/manifest.json` 的 `h5.router.base` 配置仓库名，然后：

```bash
npm run build:h5
```

**第二步：添加 .nojekyll**

uni-app 打包产物含大量 `_plugin-*.js`、`chunks/_plugin-*` 等带下划线的文件。GitHub Pages 默认启用 Jekyll，会静默忽略这些文件，导致白屏。

```bash
touch dist/build/h5/.nojekyll
```

**第三步：推送 dist 到 gh-pages（Git Data API）**

WSL 环境 Git push 不稳定时，用 Git Data API 直接推送：

```python
import urllib.request, json, base64, os, time

token = "ghp_..."          # GitHub PAT
owner, repo = "user", "repo-name"
dist_dir = "dist/build/h5"

# 1. 遍历所有文件并上传 blob
files = []
for root, dirs, fnames in os.walk(dist_dir):
    for fname in fnames:
        full_path = os.path.join(root, fname)
        rel_path = os.path.relpath(full_path, dist_dir)
        with open(full_path, "rb") as f:
            content = base64.b64encode(f.read()).decode()
        files.append((rel_path, content))

def upload_blob(content_b64, retries=4):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                f"https://api.github.com/repos/{owner}/{repo}/git/blobs",
                data=json.dumps({"content": content_b64, "encoding": "base64"}).encode(),
                headers={"Authorization": f"token {token}", "Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read())["sha"]
        except Exception:
            if attempt < retries - 1:
                time.sleep(3)
    return None

blob_shas = {}
for i, (path, content) in enumerate(files):
    sha = upload_blob(content)
    if sha:
        blob_shas[path] = sha
    if (i+1) % 20 == 0:
        print(f"  {i+1}/{len(files)}")

# .nojekyll 必须上传，禁用 Jekyll
with open(os.path.join(dist_dir, ".nojekyll"), "rb") as f:
    nojekyll_sha = upload_blob(base64.b64encode(f.read()).decode())
blob_shas[".nojekyll"] = nojekyll_sha

# 2. 获取当前 gh-pages SHA
req = urllib.request.Request(f"https://api.github.com/repos/{owner}/{repo}/git/ref/heads/gh-pages")
req.add_header("Authorization", f"token {token}")
with urllib.request.urlopen(req, timeout=15) as r:
    ghp_sha = json.loads(r.read())["object"]["sha"]

# 3. 创建 tree
tree_entries = [{"path": p, "mode": "100644", "type": "blob", "sha": s} for p, s in blob_shas.items()]
req = urllib.request.Request(
    f"https://api.github.com/repos/{owner}/{repo}/git/trees",
    data=json.dumps({"tree": tree_entries}).encode(),
    headers={"Authorization": f"token {token}", "Content-Type": "application/json"}
)
with urllib.request.urlopen(req, timeout=30) as r:
    new_tree_sha = json.loads(r.read())["sha"]

# 4. 创建 commit
req = urllib.request.Request(
    f"https://api.github.com/repos/{owner}/{repo}/git/commits",
    data=json.dumps({"message": "Deploy H5 dist", "tree": new_tree_sha, "parents": [ghp_sha]}).encode(),
    headers={"Authorization": f"token {token}", "Content-Type": "application/json"}
)
with urllib.request.urlopen(req, timeout=15) as r:
    new_commit_sha = json.loads(r.read())["sha"]

# 5. 更新 gh-pages ref
req = urllib.request.Request(
    f"https://api.github.com/repos/{owner}/{repo}/git/refs/heads/gh-pages",
    data=json.dumps({"sha": new_commit_sha}).encode(),
    headers={"Authorization": f"token {token}", "Content-Type": "application/json"}
)
with urllib.request.urlopen(req, timeout=15) as r:
    json.loads(r.read())

print("Deploy complete!")
```

**第四步：配置 GitHub Pages**

```bash
# 仓库必须先设为 public，或 token 需有 repo scope
curl -X POST "https://api.github.com/repos/{owner}/{repo}/pages" \
  -H "Authorization: token {token}" \
  -d '{"build_type":"legacy","source":{"branch":"gh-pages","path":"/"}}'
```

**第五步：验证**

```bash
# 检查资源路径是否包含仓库名前缀
curl -s "https://user.github.io/repo-name/" | grep -o 'src="/repo-name/assets/[^"]*"'
```

---

## 常见问题排查

### 白屏 + `<!--app-html-->`

**原因**：资源路径未匹配子目录

**排查**：
1. 检查 `manifest.json` 的 `router.base` 是否配置了正确的仓库名
2. 确认是否在**构建前**配置，构建后修改必须重新构建
3. 验证部署后 HTML 中的资源路径是否包含仓库名

### 资源 404（特别是 `_plugin-*.js`）

**原因**：GitHub Pages 启用 Jekyll，忽略了 `_` 开头的文件

**解决**：在 gh-pages 根目录保留 `.nojekyll` 文件

### GitHub Actions workflow 持续失败

**原因**：Actions 环境网络问题、依赖安装超时等

**替代方案**：放弃 Actions，本地构建后用 Git Data API 推送（如上流程）

### API 上传 blob 超时

**解决**：分小批量（每批 5 个文件），每批之间 `time.sleep(1)`。约 100 个文件需 5-10 分钟。

### 私有仓库部署

**解决**：GitHub Pages 源分支选 `gh-pages` 需仓库为 public，或 token 需 `repo` scope

---

## Pinia 状态管理（如项目用到）

uni-app Vue 3 项目推荐使用 Pinia 管理全局状态：

```js
// stores/user.js
import { defineStore } from 'pinia'

export const useUserStore = defineStore('user', {
  state: () => ({
    token: '',
    userInfo: null
  }),
  actions: {
    setToken(token) {
      this.token = token
      uni.setStorageSync('token', token)
    },
    loadToken() {
      this.token = uni.getStorageSync('token') || ''
    }
  }
})
```

> 注意：future-little-leaders 项目曾遇到 pinia@3.0.2 兼容问题，降级到 2.2.6 解决。

---

## 多端构建命令

| 平台 | 命令 | 输出目录 |
|------|------|----------|
| H5 | `npm run build:h5` | `dist/build/h5/` |
| 微信小程序 | `npm run build:mp-weixin` | `dist/build/mp-weixin/` |
| H5 开发模式 | `npm run dev:h5` | 内存实时构建 |

---

## 经验总结

1. **构建前配置 router.base** — 这是最容易忘记的步骤，忘了就要重来
2. **.nojekyll 不可忘** — 每次推送都要保留，否则带下划线的 JS 文件全部 404
3. **pinia 版本兼容性** — 如遇构建报错，先尝试降级 pinia 到 2.2.6
4. **Git Data API 部署** — WSL 环境 Git push 不稳定时，用 API 直接推送更可靠
5. **私有仓库** — 部署 GitHub Pages 前需先设为 public，或确认 token 有 repo scope
