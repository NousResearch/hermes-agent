---
name: subagent-delegation-checklist
description: subAgent 委托后的验收清单与常见问题预防
version: 1.0.0
author: Hermes Agent
license: MIT
---

# subAgent 委托验收清单

## 何时使用

每次使用 delegate_task 委托子Agent后，应用此清单进行验收。

## 验收检查清单

### 1. 文件位置检查
- [ ] 文件是否写入正确目录（检查完整绝对路径）
- [ ] 不是 home/ 或 session 默认目录

### 2. PRD 需求核对
- [ ] 每一条 PRD 需求是否有对应实现
- [ ] 常见遗漏项：
  - UI 按钮（开始/重新开始）
  - 游戏结束判定逻辑
  - 边界条件处理
  - 错误处理

### 3. 功能完整性
- [ ] 核心逻辑可正常运行
- [ ] 主要功能流程无崩溃
- [ ] 控制台无 Error

### 4. 关键 Bug 模式检查（web UI 项目）
- [ ] `grep -n "WIN_VALUE\|hardcode\|写死的值" src/` — 检查配置是否被硬编码覆盖
- [ ] `grep -n "import.*组件名" src/` — 检查组件是否被 import（文件存在 ≠ 被使用）
- [ ] `grep -n "组件名\|<组件 " src/` — 检查组件是否在 JSX 中实际渲染
- [ ] 检查 hooks 返回值是否被调用方正确解构（如 `mode` vs `playMode` vs `gameMode`）

### 5. 存储 key 验证
- [ ] 切换模式后 localStorage key 是否真的变化
- [ ] 不同概念（如 gameMode + playMode）的存储 key 是否各自独立

### 6. 文件行数验证（针对已有项目增强）
- [ ] 交付后立即检查文件行数：`wc -l <file>`
- [ ] 如果行数大幅减少（如减少50%+），说明发生了重写，必须回滚
- [ ] 回滚命令：`git checkout <previous_commit> -- <file>`
- [ ] 回滚后重新委托，明确强调「禁止重写，只能修改」

## 常见问题

| 问题 | 原因 | 预防 |
|------|------|------|
| 文件写入 ~/animal-forest/ 而不是 proposals/workspace-dev 目录 | subAgent 自作主张在 home 目录创建项目 | **delegate_task 的 context 里写明：`项目根目录 = ~/.hermes/proposals/workspace-dev/proposals/<slug>/`，明确告知"不要在其他位置创建文件，创建后用 ls 验证"** |
| 找到错误的代码库路径 | subAgent 根据模糊描述自行推断项目位置 | **委托前先用 terminal + find 确认实际路径**，特别对于 web 项目，实际可能是一个单文件 HTML 而非 React 目录 |
| 组件文件存在但从未渲染 | subAgent 创建了组件但忘记在父组件中 import 和使用 | **验收时必须验证：grep -n "import.*组件名" 检查是否有 import，grep "组件名" 检查是否有实际渲染** |
| 本地 build 通过但 CI 失败 | subAgent 创建了文件但从未 `git add`，本地 dist/ 有旧缓存掩盖问题，CI 全新构建时找不到文件 | **触发 CI 前必须：① git status 检查 untracked 文件 ② git ls-files 确认关键文件已 commit ③ 清理 dist/ 后重新 build 验证：`rm -rf dist && npm run build`** |
| 配置对象定义但从未读取 | subAgent 定义了 MODES/Achievements 等配置，但实际逻辑用硬编码绕过 | **验收时必须验证：grep 配置文件中的 key 名，确认在 hook/逻辑代码中有实际调用** |
| 两套概念混用（如 gameMode vs playMode） | subAgent 定义了 A/B 两种概念但存储 key 只用了一种 | **验收时必须验证：每种概念都有对应的存储 key，切换概念时 key 确实变化** |
| PRD 要求遗漏 | subAgent 未完整理解需求 | 交付后逐一核对 |
| 直接跳过草稿确认 | subAgent 自主执行 | 先要求输出结构，确认后再写文件 |
| GitHub push 失败（WSL 网络） | WSL DNS/网络问题 | 委托前确认网络，备好手动 push 方案（API创建仓库+本地commit待网络恢复后push） |
| dev agent 完全重写游戏而非增强 | subAgent 倾向于从头创建而非修改现有代码 | **对于已有项目（特别是单文件项目如HTML游戏），必须在 context 中明确强调：「禁止重写游戏，只能修改现有代码」。交付后立即检查文件行数，如果行数大幅减少（如5000行→1000行）说明发生了重写，必须回滚并重新委托** |
| subagent 达到 max_iterations 但代码未 push | subagent 完成代码修改但卡在 git push 阶段，超时后停在"未提交"状态 | **subagent 交付后立即执行 `git status` 检查，如果有 modified/untracked 文件说明代码已改但未 commit。正确流程：1) `git add` 所有变更 2) `git commit` 3) `git push`。多次实践发现：subagent 通常能在 max_iterations 内完成代码修改，但几乎无法完成 git add/commit。建议：代码修改完成后主 agent 直接接手 commit/push，不用等待 subagent** |
| subagent completed 但从未 git commit | subagent session 显示 completed（30+ api_calls），但 `git log` 显示本地无新 commit，`git status` 有 modified/untracked 文件 | **表现特征：subagent 报告完成，summary 列出所有文件修改，但 `git status` 显示文件未 commit。根因：subagent 在 max_iterations 前完成了代码修改并可能 `git add` 了部分文件，但最后 iteration 用于报告而非执行 git 操作。**这不是 max_iterations 问题，而是 subagent 把"完成"理解为"写完代码"而非"push 完成"。每次都需要主 agent 手动 commit/push。**教训：不要假设 subagent 报告完成就等于已 commit。必须立即在目标目录执行 `git status` 确认。** |
| subagent 完成但 git push 未执行 | subagent 在 max_iterations 内正常完成，report 显示 completed，但从未执行 git push。git log 显示本地有 commit 但 remote 没有更新 | **表现特征：subagent session 显示 completed 且有合理的 API 调用量（如50+次），但 git fetch origin 后发现 remote 分支未更新。根因：subagent 通常不会主动执行 git push。建议：即使 subagent 报告完成，也要立即在目标目录执行 `git push` 验证。如果 push 失败（网络/认证），手动处理。典型模式：subagent 完成了 `git add` + `git commit` 本地提交，但 push 操作不在其执行计划中** |
| subagent 只做调查不实现 | 委托后 `git status` 显示 no changes 或只有 SPEC.md 等文档更新，实质代码未修改 | **立即检查 `git diff --stat` 确认实质变更。如果无代码变更：1) 不再重新委托（codex 多次会重复同样行为）2) 直接自己实现。creative-drawing-board V50 案例：codex 输出了详细调查（代码位置、实现步骤）但未写任何代码，后续重委托同样只调查不实现。教训：对于简单功能迭代，直接自己 patch 比等待 subagent 更高效** |
| Android: UI 布局类型不符 | PRD 要求 ConstraintLayout + View Binding，subagent 却用 LinearLayout + findViewById | **Android 项目验收必须检查：1) layout XML 根元素是否匹配（ConstraintLayout vs LinearLayout）2) build.gradle 是否配置了 `viewBinding true` 3) Activity 是否用 View Binding（`ActivityMainBinding.inflate`）而非 `findViewById`。如果不符，主 agent 直接自己 patch，不用重新委托** |
| Android: subagent 创建 Material Demo UI 替代原有功能 | v1.3.0 案例：subagent 看到 "Material Design" 就在 layout 里添加 FAB、多个 button variants、email input 等 demo 组件，把原有 greeting 功能全部覆盖。PRD 要求的是「MaterialButton 替代普通 Button」「TextInputLayout 替代 EditText」，subagent 理解为「做一个 Material 组件展示页」 | **Android 迭代项目验收时：1) 必须 grep 验证原有 ID（如 `btnGreet`、`etName`、`tvGreeting`）是否仍在 layout XML 中 2) MainActivity 的 onClick 逻辑是否仍处理这些 ID 3) 如果 subagent 全面改写了 UI 但丢失原有功能，直接自己重写 layout + Activity，不用重新委托。教训：subagent 会「发挥创意」添加 PRD 没要求的组件，对于已有功能的迭代项目，必须明确要求「保留 XXX 功能不变，只替换组件样式」** |
| Android: 测试针对旧架构而非新架构 | V2.1.0 单元测试案例：项目刚完成 V2.0.0 架构升级（MVVM+Compose+Hilt），委托单元测试时 subagent 测试了 V1.x 的旧 Activity（View Binding + XML layouts）而非 V2.0.0 的 Compose ViewModel。根因：subagent 看到 "android-hello" 项目名，直接从 commit 历史中找了最早的 Activity 文件测试，没有意识到项目已经过架构升级 | **对于刚完成架构升级的 Android 项目，委托测试前必须：1) 在 context 中明确写当前版本号（如 V2.0.0）和架构（如 MVVM+Compose+Hilt）2) 列出要测试的具体类名和它们的完整包路径（如 `com.hello.android.ui.viewmodel.MainViewModel`）3) 明确禁止测试旧架构代码（如 SplashActivity、View Binding layouts）。教训：迭代项目中不同版本架构差异巨大，subagent 倾向于从最新代码往前找"入口"而不是从版本历史理解当前状态** |
| Android: 缺少 View Binding 配置 | subagent 创建了 binding layout 文件但 build.gradle 未开启 viewBinding，导致编译失败 | **Android 项目验收时：1) `grep -n "viewBinding" app/build.gradle` 确认已配置 2) 如果缺失，直接 patch 添加。教训：不能假设 subagent 会配置，必须主 agent 验证** |
| 代码加到错误的功能区域 | 单文件 HTML 项目有多个功能区（如点画游戏 + 动画书），subagent 把动物模板加到了点画游戏而非动画书模板 | **对于大单文件项目，必须在 context 中明确：1) 具体要修改哪个数组/对象（如 `ANIMABOOK_PRESET_TEMPLATES` 而非 `dotTemplates`）2) 具体在哪个函数/方法附近增加代码。交付后用 grep 验证代码确实在正确位置** |
| `git checkout -- .` 撤销了 subagent 所有修改 | 交互式 git 操作时误以为只是撤销 staging area，实际撤销了 working tree 所有未提交修改 | **在 subagent 交付后，不要用 `git checkout -- .` 或 `git reset --hard` 来清理工作区**。正确做法：`git status` 查看哪些文件有修改，用 `git diff <file>` 或 `git diff --stat` 确认变更内容。需要撤销特定文件时用 `git checkout HEAD -- <file>` 指定文件撤销；需要清理 untracked 文件时用 `git clean -fd` 单独处理。**总之：绝对不要在 subagent 交付后执行全文件撤销命令** |
| subagent 完成后留下 TypeScript 类型错误 | subagent 实现功能但留下类型错误（如 undefined 检查缺失、接口属性不匹配、组件 Props 缺失） | **subagent 交付后立即运行 `npm run build 2>&1 \| grep -E "error TS" \| head -20` 检查类型错误。常见错误模式：1) `currentProject.id` 可能为 undefined → 用 `?? 0` 或 `as number` 修复 2) 组件 Props 缺失（如 `isOpen`）→ 直接 patch 添加 3) 服务类方法签名与调用处不匹配 → 添加适配方法（如 `ReminderService.setCallback()`）。修复后重新 build 验证** |
| subagent 完成但代码在 phantom `frontend/` 子目录 | subagent 在 `/home/hermes/<project>/frontend/` 下创建文件，但实际项目在 `/home/hermes/.hermes/proposals/workspace-dev/proposals/<project>/`。项目结构中不存在 `frontend/` 这一层 | **特征：`git status` 显示 "nothing to commit" 且 `git log` 未推进，但 `find /home/hermes -name "<filename>"` 能找到文件。根因：subagent 根据 "React + Vite 项目" 描述自行创建了 `frontend/` 子目录，但实际项目根目录就是 src/。发现后直接自己实现，不用重新委托。教训：对于已在 workspace-dev 中的项目，context 中必须写「项目根目录 = /home/hermes/.hermes/proposals/workspace-dev/proposals/<slug>/，禁止创建额外子目录」 |
| JSX 组件放在 `return()` 之外导致语法错误 | subagent 添加了组件 import 和 state，但在 JSX 中把组件放在了 return 的闭合括号之外（如 `</div>` 和 `)` 之间），导致 `error TS1005: ')' expected` | **构建失败且错误指向文件末尾几行时，检查是否有组件被意外放在 return 块之外。修复方法：`git checkout master -- <file>` 还原文件，重新应用修改，或手动把组件移回 return JSX 树内正确位置** |
| `dist/` 目录在 `.gitignore` 中导致无法 `git add` | gh-pages 部署需要提交 dist/ 构建产物，但 `git add dist` 报错 "The following paths are ignored by one of the .gitignore files" | **使用 `git add -f dist` 强制添加被忽略的目录** |

### 委托前必做：项目结构预检

对于 web UI 类项目，在委托前先用以下命令确认实际代码位置：

```bash
# 在预期项目目录下执行
find . -name "*.html" -not -path "./node_modules/*" 2>/dev/null | head -5
find . -name "package.json" -not -path "./node_modules/*" 2>/dev/null
ls -la
```

这样可以避免 subAgent 跑到错误路径（如 `hermes-collab-web/` 而非 `collaboration/web/index.html`）。

## 关键教训

1. **必须指定完整绝对路径**，不能只给目录名
2. **subAgent 跳过草稿确认是常见问题**，主Agent必须主动验收
3. **PRD 每一条都要核对**，不能假设交付即合格
4. **web UI 项目在委托前预检实际代码位置**，实际实现可能与描述不符（单文件 vs 框架项目）
