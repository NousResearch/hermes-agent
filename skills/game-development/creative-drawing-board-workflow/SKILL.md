---
name: creative-drawing-board-workflow
description: PRJ-20260418-002 creative-drawing-board 标准化开发流程 - 迭代模式+部分交付验证+内容扩展技巧
---

# creative-drawing-board 迭代工作流

## 概述

儿童 HTML5 绘画板（3-6岁），单 HTML 文件，零外部依赖。GitHub: YeLuo45/creative-drawing-board，部署于 https://yeluo45.github.io/creative-drawing-board/。

## 版本历史

| Version | Lines | Key Features |
|---------|-------|--------------|
| V1 | 745 | Drawing/brush/eraser/bubble game |
| V2 | 1981 | +贴纸(24)/描红(8)/填色/背景(6)/涂色(4)/undo-redo |
| V3 | 2401 | +音效引擎(Web Audio API, 11种)+内容扩展(贴纸56/描红20/涂色10/背景10) |
| V4 | 3547 | +学习记录(localStorage)+成就系统(12徽章)+画廊+贴纸编辑(拖拽/缩放/旋转) |
| V5 | 3966 | +气泡积分系统(4种气泡/连击/每50分奖励贴纸)+本地排行榜+新纪录动画 |
| V6 | 5305 | +引导教学模式(8跟画模板/年龄分级/逐笔画演示)+PWA离线(sw.js+manifest.json)+打印功能 |
| V7 | 6324 | +自定义贴纸创作(200x200绘制/换色/翻转/重绘)+最多20个+localStorage |
| V8 | 6878 | +4套节日主题包(万圣节10月/圣诞节12月/春节1-2月/复活节4月，每套8贴纸+2背景+动画) |
| V9 | 7896 | +每日挑战系统(7种任务/进度条/奖励/连续打卡)+成就进化(⭐⭐⭐三级)+限定成就(challenge/early_bird/night_owl/streak) |
| V10 | 9444 | +游戏模式(拼图3×3~4×4/连线6模板/迷宫递归回溯7×7~15×11)+游戏状态管理(GAME_STATES) |
| V11 | 10418 | +画作分享(📥PNG导出/水印/画廊管理20幅/批量操作)+🔗分享面板(Web Share API/navigator.share) |
| V12 | 11091 | +30涂色模板/分类浏览/难度标记/荧光笔刷(5色+彩虹)/粒子爆发动画/星星飘落/C-E-G-C合成旋律 |
| V13 | 11548 | +5首BGM/场景切换(switchBGM调用)/音量控制/500ms渐变/页面可见性降速 |
| V14 | 12412 | +时长限制(15-120分钟)/周报柱状图/内容解锁5规则/数学题家长入口 |
| V15 | 12688 | +30描红模板(数字0-9/字母8个/汉字5个/物品5个/交通2个)/9连线模板(水果3/动物3/交通2/自然1)/分类筛选/学习路径/描红进度 |
| V16 | 13114 | +对称绘画(左右/上下/四角)/智能形状(直线/圆/矩形/三角+预览)/多步撤销(20步+Ctrl+Z/Y)/画布缩放(双指+按钮50%-300%)/橡皮擦大小滑块 |
| V17 | 13379 | +中英文双语界面(t(key)函数/data-i18n属性)/语言切换按钮🌐/全UI双语化/语言记忆localStorage |
| V19 | 15143 | +5节日主题(端午/中秋/重阳/元宵/春节,共38贴纸+14背景)/节日自动检测/节日选择面板/主题激活/节日特效(纸屑/灯笼/月光/涟漪/落叶) |
| V20 | 15995 | +海报打印(A3/A2/A1分割+拼接线+页码)/涂色卡打印(纯净黑白线稿)/描红卡打印(虚线/实线/点线)/每日练习纸/打印菜单5选项 |
## 标准迭代流程

1. **小墨提议方向** → boss 选定
2. **起草 PRD** → `proposals/workspace-pm/proposals/P-YYYYMMDD-NNN-prd.md`
3. **更新 proposal-index.md** → 在旧版本条目**之后**插入新条目，Status: `approved_for_dev`
4. **委托 dev agent** → `delegate_task`，传 PRD 路径 + 项目路径 + max_iterations=35-40
5. **关键教训：dev agent 几乎总是 hit max_iterations 在 git push 之前** → 委托后总是手动检查 `git status`，如有修改则 commit + push
6. **验证数量** → 检查各数组实际数量是否与 PRD 一致
7. **Commit → push to master → force-push to gh-pages**
8. **更新 proposal-index.md** → Dev Commit SHA

## Subagent 部分交付验证模式

Dev agent 完成后，**总是**需要手动验证数量：
```bash
cd /home/hermes/creative-drawing-board
sed -n '/const STAMPS = \[/,/\];/p' index.html | grep -c "id: '"
sed -n '/const TRACING_TEMPLATES = \[/,/\];/p' index.html | grep "id: '" | grep -v "paths" | wc -l
sed -n '/const COLORING_PAGES = \[/,/\];/p' index.html | grep "^\s*id:" | wc -l
sed -n '/const BACKGROUNDS = \[/,/\];/p' index.html | grep "id:" | wc -l
```

**数量不对时**：用 `patch` 逐个补充缺失条目。**不要用 execute_code 字符串替换**——它会贪心匹配，跨数组边界覆盖，导致文件损坏。

**文件损坏后恢复**：
```bash
git checkout -- index.html  # 恢复到上一个 commit
# 然后用 patch 逐个添加缺失内容
```

## 内容数组扩展技巧（重要教训）

V3 开发教训：
- 扩展 STAMPS 数组时，`patch` 的 old_string 如果包含与 TRACING 数组相同的代码片段，会**意外覆盖 TRACING 数组**
- 每次 `patch` 只操作一个数组，完成后立即验证所有数组数量
- 新增内容放在数组末尾 `];` 前，避免破坏数组结构

**execute_code 绝对不能用**：尝试用 Python 脚本替换 STAMPS 数组时，写入内容出现重复，导致 745 行文件变成 7MB。原因是字符串替换逻辑在处理大型 HTML 时出现错误。**只能用 patch 逐个添加数组元素。**

**数组边界唯一性**：TRACING 数组的 `];` 在文件中重复出现多次，定位时要包含数组内最后一个元素的 id 来精确定位。例如：
```bash
# 准确定位到 TRACING 数组的末尾
sed -n '876,879p' index.html  # 先找到 ] 和下一个 ]; 之间的行
```

## PWA 多文件交付注意事项

V6 引入了 `manifest.json` 和 `sw.js` 两个辅助文件。**总是验证新文件是否被 git 追踪**：
```bash
git status --short
ls -la manifest.json sw.js  # 检查文件存在
```
如果新文件未被追踪：
```bash
git add manifest.json sw.js && git commit -m "add: PWA support files"
```

## 架构要点

- **7层画布**: backgroundCanvas / drawingCanvas / stampLayer / tracingLayer / bubbleCanvas / uiCanvas / textCanvas
- **音效引擎**: Web Audio API (AudioContext + OscillatorNode + GainNode)，soundEnabled 全局变量，localStorage key `drawing_board_sound`
- **统计数据**: localStorage key `drawing_board_stats` (JSON)，`drawing_board_gallery` 缩略图
- **自定义贴纸**: localStorage key `custom_stickers` (JSON)
- **主题系统**: V8 新增，`THEME_STAMPS` (Object，按节日key分组) + `THEME_BACKGROUNDS`，`getActiveSeason()` 自动检测，`preferred_theme` localStorage 记住偏好，`initThemeSystem()` 必须在 `init()` 中调用
- **游戏状态管理**: V10 新增，`GAME_STATES = { NONE, JIGSAW, DOT_CONNECT, MAZE }`，`currentGame` 变量追踪，`initJigsawGame()` / `initDotConnectGame()` / `initMazeGame()` 分别初始化，`exitGame()` 返回正常模式
- **迷宫算法**: V10 递归回溯生成，`generateMaze(cols, rows)` 返回 `{maze, start, end}`，`drawMazePath()` 检测碰墙
- **画廊**: V4 初版，V11 扩展至20幅，结构 `{id, thumbnail, fullImage, title, date}`，localStorage key `gallery_items`

- **画廊**: V4 初版，V11 扩展至20幅，结构 `{id, thumbnail, fullImage, title, date}`，localStorage key `gallery_items`

- **背景音乐**: V13 新增，`bgmGainNode` + `sfxGainNode` 独立增益，5首程序化生成(happy/peaceful/playful/dreamy/festive)，`switchBGM(type)` 切换，`bgmEnabled`/`bgmVolume`/`sfxVolume` localStorage，页面隐藏时音量降至30%

- **导出/分享**: V11 新增，`exportAsPNG(options)` 合并所有图层生成PNG，`navigator.share()` / `navigator.canShare()` Web Share API，`copyToClipboard()` 剪贴板复制
- **导出/分享**: V11 新增，`exportAsPNG(options)` 合并所有图层生成PNG，`navigator.share()` / `navigator.canShare()` Web Share API，`copyToClipboard()` 剪贴板复制

## Git 工作流

```bash
cd /home/hermes/creative-drawing-board
git add index.html && git commit -m "V<N>: <features>"
git push origin master && git push origin master:gh-pages --force
```

# 验证命令

```bash
# 检查行数
wc -l index.html

# 验证音效引擎函数
grep -c "playBtnClick\|playStampPlace" index.html

# 验证贴纸编辑函数
grep "placedStamps\|stampScaleUp\|stampRotate\|stampBringToFront" index.html

# 验证统计/画廊函数
grep -E "showStatsPanel|openGallery|drawingBoardStats" index.html

# 验证主题系统
grep "initThemeSystem\|THEME_STAMPS\|getActiveSeason" index.html
```

## V4-V18 运营教训

**Dev agent 经常 hit max_iterations 在 git push 之前**：V4-V18 十五次迭代，大多数 dev agent 完成了代码实现但未能执行 `git push`。V15 和 V18 例外（subagent 自己 push 完了）。**解决方案：委托后总是执行 `git status`，如有修改则 commit + push**。

**Subagent 重复函数问题**：V9 中 subagent 复制粘贴了旧版 `renderAchievements()` 函数，导致文件中存在两个同名函数。需要手动删除旧版函数体。

**Subagent 漏写初始化调用**：V8 中 `initThemeSystem()` 函数已实现但未在 `init()` 中被调用。需要检查 `init()` 中是否包含新增的 init 函数调用。

**Subagent 部分交付是常态**：STAMPS/TRACING/BG 等数组扩展时，经常部分完成。每次委托后都要验证各数组数量。

**Subagent 漏接事件监听器（V12 新教训）**：subagent 可能完整实现了 DOM 结构和函数逻辑，但忘记给新增按钮绑定 `addEventListener`。V12 荧光笔按钮、V16 shape/symmetry dropdown 都发生过。需要找到同类型按钮的事件注册模式，手动补充监听器。

**Subagent 漏写辅助函数调用（V12 新教训）**：subagent 定义了 `animateRainbowHighlighter()` 但调用方式错误（如放在了脚本末尾立即执行而非在颜色选择时触发）。需要补充 `let rainbowAnimRunning = false` 状态变量，并在颜色选择处理器中启动/停止动画。

**Subagent 漏写完整功能（V12 新教训）**：subagent 可能只实现了部分功能。例如 V12 的粒子爆发动画和 C-E-G-C 合成旋律函数需要自己补充实现。

**Subagent 漏写 switchBGM 调用（V13 新教训）**：V13 实现 BGM 场景切换时，subagent 定义了 `switchBGM()` 函数但在部分游戏入口/出口函数中忘记调用。**总是检查所有游戏和面板函数是否都有 switchBGM 调用**：
- `startJigsawGame()` → `switchBGM('playful')`
- `startDotConnectGame()` → `switchBGM('playful')`
- `startMazeGame()` → `switchBGM('playful')`
- `exitGame()` → `switchBGM('happy')`

**Subagent 漏写游戏时间限制检查（V14 新教训）**：V14 实现家长时长限制后，subagent 漏掉了 `startJigsawGame()` / `startDotConnectGame()` / `startMazeGame()` 三个游戏函数的 `checkTimeLimitBlock()` 调用。补充到这三个函数开头。

**Subagent dropdown 事件处理器遗漏（V16 新教训）**：V16 实现 shape/symmetry dropdown 时，HTML 和函数都写好了但 `.dropdown-item` 的 click 事件监听器没有绑定到按钮上。需要手动补充：
```javascript
shapeOptions.querySelectorAll('.dropdown-item').forEach(item => {
    item.addEventListener('click', () => {
        shapeType = item.dataset.shape;
        // ...
    });
});
```
同理 symmetryOptions 也需要。还要加上 document click 关闭 dropdown。

**V17 rate limit 问题**：subagent 在 40 次迭代后触发 API rate limit（429），但代码已完整写入文件，只是没有 push。检查 `git status` 发现有修改，手动 push 即可。

**V21-V22 教训**：subagent 完成代码实现并 push 成功，但 commit message 不是预期格式，或未执行 force-push 到 gh-pages。总是运行 `git log --oneline -2` 确认。

**V23 教训**：subagent 实现了新工具系统（渐变/文字/形状库/图层）但在集成到现有绘画管道时未完成：
**Subagent 完成模式总结**：
- **V22-V23**：需要手动 commit + 集成修复
- **V24-V25**：subagent 完整交付，无需手动修复
- **V26**：subagent 完整交付（hit max_iterations 但 push 在限制前已完成）
- **V27**：subagent 完成基础设施，但 hook 只完成 1/4，需要手动补充其余 3 个 hooks
- **V28**：subagent 完整交付并 push 成功（音频录制功能）
- **V29**：subagent hit max_iterations，缺失 twoDBtn 事件监听 + 3D 鼠标交互事件，需要手动补充后 push

**V28 教训（音频录制）**：subagent 完整实现 MediaRecorder 音频录制，包括 UI 面板、存储管理、播放/导出功能。V24-V26-V28 证明当功能范围明确、实现路径标准时 subagent 可完整交付。

**V27 教训（视频录制 - Hook 模式）**：subagent 完成了录制基础设施（状态变量/UI面板/回放/导出）但只 hook 了 `STROKE_START`。原因是 subagent 在 `drawingCanvas.addEventListener('pointermove')` 内部的事件处理器中添加 hook，而不是在该事件处理器所调用的 `draw` 函数中添加。**当需要 hook 到现有事件流时，必须在事件处理器直接调用的函数内部添加 hook**。

V27 需要手动添加的 hooks：
- `strokeMove` → 在 `draw` 函数内部（行 ~9741），而非创建新函数
- `stopDrawing`（strokeEnd）→ 在 `stopDrawing()` 函数开头（行 ~9746）
- `selectColor` → 在 `selectColor()` 函数开头（行 ~9294）
- `selectTool` → 在 `selectTool()` 函数开头（行 ~11981）

验证命令：
```bash
grep -n "recordAction" index.html | grep -E "STROKE_MOVE|STROKE_END|COLOR_CHANGE|TOOL_CHANGE"
```
应该有 4 条（加上 STROKE_START 共 5 条）。

**Hook 插入位置判断**：
1. 找到现有函数定义（`function name(` 或 `name = function(`）
2. 在函数体**第一行有效代码**之后添加 hook
3. 不要创建新 wrapper 函数——事件流不会调用它们

**V26 教训（AR）**：subagent 完整实现并 push 成功，但在报告总结中说自己"无法验证 AR 按钮可见性"（需要浏览器）。这是预期内的限制——subagent 无法做视觉验证。关键验证还是靠 `git log --oneline -2` 和 `wc -l`。**只要 push 成功、代码量合理增长，就是合格交付**。

**V29 教训（3D 绘画 + 外部依赖冲突）**：
1. subagent hit max_iterations，遗漏两个关键部分：工具栏按钮事件监听 + canvas 鼠标交互事件
2. **外部依赖冲突**：PRD 阶段发现 Three.js（~500KB CDN）违反项目零依赖原则
   - 主动向 boss 澄清选择：接受外部依赖 / 坚持零依赖改方案 / 换方向
   - boss 选择坚持零依赖，改用 Canvas2D 伪 3D（透视变换模拟）
   - 手动补充了 mouse wheel 缩放和 mousedown/mousemove/mouseup 旋转交互

**修复后的验证顺序**：发现问题 → 先用 `git status` + `wc -l` 确认文件完好 → 再用 `grep` 定位具体缺失 → 用 `patch` 精确修复 → 再次验证。**不要在未确认文件状态时直接 patch**。

**Subagent 收尾检查清单**：
```bash
cd /home/hermes/creative-drawing-board
git status --short                    # 确认有 M index.html
wc -l index.html                      # 确认行数符合预期
# V29 3D 验证
grep "threeDBtn.*addEventListener" index.html   # 工具栏按钮事件
grep "is3DRotating\|canvas3D.rotationY" index.html  # 鼠标交互
grep "canvas.addEventListener.*wheel" index.html   # 滚轮缩放
```
# V8+ 主题
grep "initThemeSystem" index.html
# V10+ 游戏状态
grep "GAME_STATES\|initJigsawGame" index.html
# V11+ 导出
grep "exportAsPNG\|openExportPanel" index.html
# V12+ 涂色增强
grep "triggerColoringCelebration\|playColoringCompleteMelody" index.html
# V13+ BGM
grep "switchBGM.*playful" index.html
grep "switchBGM.*happy" index.html
# V14+ 时长限制（游戏函数也要检查）
grep "checkTimeLimitBlock" index.html
# V16+ 绘画工具
grep "shapeBtn\|symmetryBtn" index.html | grep "addEventListener"
grep "langBtn\|switchLanguage" index.html | grep "addEventListener"
# V17+ 多语言
grep -c "data-i18n" index.html
git add . && git commit -m "V{N}: ..." && git push && git push --force
```
