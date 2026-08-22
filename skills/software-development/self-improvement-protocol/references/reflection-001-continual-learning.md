# 反思 #1 — 2026-05-30 — 持续学习论文学习与系统改进

## 任务摘要
用户要求学习 Chen 2026 的持续学习综述论文，并结合现有记忆和 skill 系统让自己更聪明。

## Verifier 信号
- [x] 用户纠正了？ → 无 (首次执行)
- [x] 工具失败了？ → pymupdf gettext 无输出 (第一次尝试)，改用 pymupdf Python API 成功
- [x] 需要回退？ → 从 pymupdf CLI 回退到 Python API
- [x] 空响应？ → 无

## 提取的知识

### 知识 1: pymupdf CLI vs Python API
- 类型: 工具使用陷阱
- 稳定性: 永不过期
- 内容: `pymupdf gettext <file>` 可能无输出，用 `pymupdf.open()` + `page.get_text()` 更可靠
- 归属: Skill patch (ocr-and-documents)

### 知识 2: 持续学习框架可映射到 Agent 系统
- 类型: 架构洞察
- 稳定性: 永不过期
- 内容: 论文的三维分类法 (What×How×When) 可映射到 Memory/Skills/USER.md
- 归属: Archive (self-improvement-protocol skill)

### 知识 3: 自我改进必须有 verifier
- 类型: 理论洞察
- 稳定性: 永不过期
- 内容: 无 verifier 的自我改进会退化 (model collapse)。对 Agent 而言，verifier = 用户纠正 + 工具失败
- 归属: Skill (self-improvement-protocol)

## 行动执行

### ✅ 已完成
1. 创建 skill `hermes-agent/self-improvement-protocol`
   - OODA-Reflect 循环
   - 纠错结构化捕获
   - Skill 自动修补机制
   - 跨 skill 知识传播

2. 添加参考文件
   - `references/correction-patterns.md` — 纠错模板
   - `references/continual-learning-insights.md` — 论文洞察

3. 更新 skill `archived-memory-recall`
   - 添加"主动重放 (Replay)"机制
   - 重放触发时机表
   - 重放 vs 普通检索的区别

4. 归档论文学习
   - 论文全文: `/tmp/cl_survey_full.txt`
   - 关键洞察: 已写入 self-improvement-protocol skill

### ⏳ 待后续执行
1. ocr-and-documents skill patch (pymupdf 陷阱) — 下次使用时修补
2. 跨 skill 传播检查 — 下次修补其他 skill 时检查相关 skills

## 反思质量自检

- [x] 是否提取了可操作的知识？ → 是 (3 条)
- [x] 是否选择了正确的存储位置？ → 是 (Skill + Archive)
- [x] 是否有 verifier 信号？ → 是 (工具失败 + 回退)
- [x] 7 天后这些知识还有用吗？ → 是 (架构洞察永不过期)

## 下次改进

- 反思流程可以更自动化 (检测 tool failure → 自动触发)
- 跨 skill 传播可以更系统化 (patch 时自动检查 related_skills)
