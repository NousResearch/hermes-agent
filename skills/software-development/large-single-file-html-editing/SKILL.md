---
name: large-single-file-html-editing
description: 大型单文件 HTML 项目（如 creative-drawing-board ~28000+ 行）的安全迭代编辑规范
---

# Large Single-File HTML Project 迭代编辑规范

## 适用场景
- 单文件 HTML 项目（如 creative-drawing-board ~28000+ 行）
- 连续多个版本迭代，每个版本添加新功能
- 需要同时修改 CSS/HTML/JS 的场景

## 核心问题
多次连续 patch 在大文件上容易产生：
1. 重复函数（同名但不同签名）
2. 函数顺序错乱
3. 编辑器解析断裂
4. **CSS 层叠顺序错误（CSS 依赖上下文，后插入可能导致样式覆盖失效或样式冲突）**

## CSS Patching 安全规范

### CSS 的特殊性
CSS 不同于 JS，**顺序敏感**且**级联覆盖**。同一选择器重复定义时，后面的生效。插入位置错误会导致：
- 新样式被意外覆盖
- 样式结构断裂（缺少 `}` 导致整段样式错乱）
- 与现有选择器冲突

### CSS Patch 最佳实践
1. **精确 old_string**：必须包含完整的 CSS 规则块（从选择器到 `}`）
2. **避免模糊匹配**：不要用 `.class` 这种泛泛的选择器，因为同名类可能多处存在
3. **用唯一特征定位**：
   ```javascript
   // 危险：.volume-row 在文件中出现 20+ 次
   patch(old_string: "        .volume-row {", ...)

   // 安全：包含足够上下文的完整规则
   patch(old_string: "        .eraser-preset.active {\n            background: #ff6b6b;\n            color: white;\n            border-color: #ff6b6b;\n        }\n\n        .volume-row {", ...})
   ```
4. **插入点选择原则**：插入到**两个完整 CSS 规则块之间**，不要插入到规则块中间

### CSS Patch 错误处理
如果 patch 后发现 CSS 结构损坏（文件无法正确解析）：
```bash
cd /home/hermes/creative-drawing-board
git checkout -- .
# 重新实现
```

### 验证 CSS Patch 成功
```bash
# 检查变更行数是否合理（CSS patch 通常 50-150 行）
git diff --stat
```

## 安全工作流

### 1. 小改动（单点修改）
直接用 patch() 工具，old_string 精确到能唯一识别

### 2. 中等改动（2-3个相关修改）
按顺序执行，每个 patch 完成后再进行下一个

### 3. 大改动（涉及函数替换/重排）
**必须先 git checkout -- . 还原，再用新补丁逐个添加**

```bash
# 步骤
cd /home/hermes/creative-drawing-board
git checkout -- .   # 还原所有未提交更改
# 然后重新应用所有需要的 patch
```

### 4. 函数签名变更的处理
当旧函数 `foo(id)` 变成新函数 `foo(id, event)` 时：
- 删除旧函数和添加新函数**分开两个 patch**
- 第一个 patch 只改函数调用签名
- 第二个 patch 才改函数定义签名
- 避免一次性替换导致重复函数

### 5. 提交前验证
```bash
git diff --stat   # 检查变更大小
git log --oneline -3  # 确认提交历史
```

## 常用文件定位模式
```javascript
// 搜索 CSS 位置
search_files(pattern: "old CSS class", output_mode: "content")

// 搜索 JS 函数
search_files(pattern: "function functionName", output_mode: "content")

// 搜索 HTML 按钮
search_files(pattern: 'id="buttonId"', output_mode: "content")
```

## creative-drawing-board 项目特定位置
- CSS 样式区：文件开头 ~500-700 行
- HTML 工具栏按钮：~line 8300-8400
- 图层系统：~line 24300
- 背景/BGM 系统：~line 12000
- 分享功能：~line 13400

## Git 推送模式（subprocess 绕过 terminal 阻塞）
```python
import subprocess, os
os.environ['GIT_TERMINAL_PROMPT'] = '0'
os.environ['GIT_ASKPASS'] = 'echo'

subprocess.run(
    ['git', 'push', 'origin', 'master', '--quiet'],
    cwd='/home/hermes/creative-drawing-board',
    capture_output=True, text=True, timeout=180
)
subprocess.run(
    ['git', 'push', 'origin', 'master:gh-pages', '-f', '--quiet'],
    cwd='/home/hermes/creative-drawing-board',
    capture_output=True, text=True, timeout=180
)
```

## V55 图层管理优化的教训
- 第一轮 patch 添加了新版本函数但没有删除旧版本
- 第二轮 patch 尝试删除旧版本时用错了 old_string
- 结果：getActiveContext 前面多了一大段重复代码
- 解决：git checkout 还原，从头重新实现

### 正确的函数替换流程
```javascript
// 旧函数 (line X)
function oldFunc(id) { ... }

// 新函数 (应直接替换旧函数)
function oldFunc(id, event) { ... }

// 如果旧函数不在新位置，不要替换，直接在新位置添加新函数
function newFuncWithDifferentName(id, event) { ... }
// 然后删除旧函数
```
