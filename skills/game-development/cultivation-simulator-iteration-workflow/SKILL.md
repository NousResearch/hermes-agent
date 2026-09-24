---
name: cultivation-simulator-iteration-workflow
description: cultivation-simulator 单文件HTML游戏从PRD到交付的高速迭代流程 — PRD起草→subagent实现→语法验证→手动push→交付报告
---

# cultivation-simulator 迭代工作流

## 项目信息
- **仓库**: https://github.com/YeLuo45/cultivation-simulator
- **代码路径**: `/home/hermes/cultivation-simulator/index.html`
- **游戏URL**: https://yeluo45.github.io/cultivation-simulator/
- **类型**: 单文件HTML5游戏（约330KB JS + HTML混合）
- **版本**: V10为基础（bug修复后），持续迭代V11/V12/A1-A6等

## 触发条件
Boss 从 A/B/C/D/E 方向列表中选择迭代方向（A=战斗系统扩展, C=经济平衡, D=界面优化, E=AI功能）

## 工作流程

### 1. PRD起草（脑内）
- 功能范围和实现方案脑内确定
- 确认：数据结构变更、UI修改、是否需要新全局变量
- 直接口述给boss确认

### 2. Boss确认后委托subagent
- 路径: `/home/hermes/cultivation-simulator/index.html`
- 当前git commit通过 `git log --oneline -1` 获取
- 提供详细的代码实现方案（因为subagent没有游戏上下文）

### 3. subagent实现
- 48 tool calls限制，约5分钟
- **关键**：subagent的git push经常失败（需要手动push）

### 4. 验收步骤
```bash
# 1. 检查commit
git log --oneline -2

# 2. 语法验证（必须）
python3 -c "
import re
with open('/home/hermes/cultivation-simulator/index.html','r',encoding='utf-8') as f:
    content = f.read()
m = re.search(r'<script>(.*?)</script>', content, re.DOTALL)
if m:
    with open('/tmp/verify.js','w',encoding='utf-8') as f: f.write(m.group(1))
    print('Extracted', len(m.group(1)), 'chars')
" && node --check /tmp/verify.js && echo "SYNTAX OK"

# 3. 手动push（如subagent未推送）
git push origin main && git push origin main:gh-pages -f
```

### 5. 向Boss报告
- 简述交付内容
- 列出下一步 A/B/C/D/E 选项

## subagent委托模板
```
context:
/home/hermes/cultivation-simulator/index.html
当前git commit: <hash> (<desc>)
文件约<行数>行，需要从现有代码中观察实际的字段名和函数结构。

goal:
在 /home/hermes/cultivation-simulator/index.html 中实现 <功能名>。

## 任务
<详细实现步骤>

## 约束
- 所有已有功能不变
- 语法必须通过 node --check
- 提交信息：< meaningful commit msg>
```

## 常见坑点

### 1. subagent不push
每次都需要检查 `git status`，如果 ahead of 'origin/main' 就手动 `git push origin main && git push origin main:gh-pages -f`

### 2. 语法错误
单文件HTML中 `<script>` 内容超过300KB，node --check 提取后验证。常见问题：
- 模板字符串 `` ` `` 配对错误（最危险的bug，会导致后续函数全部消失）
- 字符串中的单引号/双引号不匹配

### 3. gameState vs combatState
- gameState: 游戏全局状态（灵气、灵石、境界、背包等）
- combatState: 战斗状态（HP、攻击、buff等）
- 新功能需确认修改哪个state

### 4. 变量作用域
全局变量直接声明（如 `let combatEnergy = 0`），在函数内访问时确保已初始化

## 迭代历史（2025-05-08）

| 版本 | 内容 | Commit |
|------|------|--------|
| V10 | Bug修复 + MiniMax配置面板 | 多次 |
| V11 | 装备强化系统 1-9星 | d69b72c |
| V12 | 必杀技系统 能量积蓄 | b14f88b |
| A1 | 秘境敌人指数成长+渡劫失败倒退 | b9e9f9a |
| A2 | 奇遇种类扩充+体质剧情分支 | ef3c8d1 |
| A3 | 装备强化系统完成 | d69b72c |
| A4 | 套装共鸣系统 | e268c17 |
| A5 | 防御反击系统 | 075d0d2 |
| A6 | 绝技分支/升级 每武器3系 | b81c460 |

## 经验记录
- 每次迭代平均耗时：约5分钟（委托+验收+推送）
- subagent push失败率：约50%，需手动补救
- 语法验证通过率：约80%，需手动patch修复
- 单文件HTML最大风险：模板字符串配对错误（会静默吞噬后续函数）
- **subagent hit max_iterations 后果**：可能丢失部分代码（如选择按钮逻辑），必须人工检查

## 已知风险

### showSerendipityModal 选择按钮丢失
subagent 修改 `showSerendipityModal` 时，E4 AI描述分支容易丢失 `showChoice` 的「接受/拒绝」按钮。
**修复方法**：在两个分支（AI/非AI）的 `showRealmBattle` 判断后，手动patch添加：
```javascript
if (result.showChoice && result.choices && result.choices.length > 0) {
    const choiceLabels = { 0: '接受', 1: '拒绝' };
    html += `<div style="text-align:center;margin-top:15px;">`;
    result.choices.forEach((label, idx) => {
        const btnLabel = choiceLabels[idx] || label;
        html += `<button class="btn btn-cultivate" onclick="handleSerendipityChoice('${name}', ${idx})" style="margin-left:${idx > 0 ? '8px' : '0'}">${btnLabel}</button>`;
    });
    html += `</div>`;
}
```

### NPC对话入口可能不存在
E1 NPC对话记忆要求 `generateAiDialogue` 函数被实际调用。如果游戏UI中没有NPC对话入口，该功能不会生效。subagent实现时可能未验证实际UI入口是否存在。

## 迭代历史（2025-05-08）

| 版本 | 内容 | Commit |
|------|------|--------|
| V10 | Bug修复 + MiniMax配置面板 | 多次 |
| V11 | 装备强化系统 1-9星 | d69b72c |
| V12 | 必杀技系统 能量积蓄 | b14f88b |
| A1 | 秘境敌人指数成长+渡劫失败倒退 | b9e9f9a |
| A2 | 奇遇种类扩充+体质剧情分支 | ef3c8d1 |
| A3 | 装备强化系统完成 | d69b72c |
| A4 | 套装共鸣系统 | e268c17 |
| A5 | 防御反击系统 | 075d0d2 |
| A6 | 绝技分支/升级 每武器3系 | b81c460 |
| C | 经济与资源平衡 | 5fec097 |
| D | 界面体验优化（移动端/日志/存档/布局） | 467dcaa |
| E | AI功能增强（NPC记忆/渡劫场景/秘境名称/奇遇描述） | 8bf0be3 |
