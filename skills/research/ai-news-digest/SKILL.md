---
name: ai-news-digest
description: 多源 AI 资讯汇总，汇聚 The Decoder、TechCrunch、MarkTechPost、DeepMind、BAIR、KDnuggets、The Batch、Hacker News、Artificial Analysis 等源。自动抓取并生成带中文摘要和源链接的结构化日报，同时包含 AI 模型排行榜。执行后：1）自动写入飞书文档（已配置 lark-cli 时）；2）同时通过消息把完整内容返回给用户。
version: "8.1"
author: Judy (朱迪) / Hermes adaptation
license: MIT
---

# AI News Digest Skill (v8.1)

多源 AI 资讯汇总，覆盖 10+ 个权威来源。每次输出**必须包含中文摘要和源链接**。

---

## 触发关键词

```
AI资讯
AI新闻
AI日报
AI动态
最新AI
多源AI
AI digest
AI汇总
今日AI
```

---

## 数据来源（11个）

| # | 来源 | 类型 | Fetch 方式 |
|---|---|---|---|
| 1 | **The Decoder** | AI 深度分析 | curl RSS |
| 2 | **TechCrunch AI** | 创业/融资新闻 | curl |
| 3 | **MarkTechPost** | AI 论文/工具 | curl 主页 |
| 4 | **DeepMind Blog** | 官方研究 | curl |
| 5 | **BAIR Blog** | Berkeley 学术 | curl 主页 |
| 6 | **KDnuggets** | 数据科学/ML | curl 主页 |
| 7 | **The Batch** | 吴恩达周报 | curl 主页 |
| 8 | **Hacker News** | AI 技术讨论 | JSON API |
| 9 | **AI News** | 商业 AI 新闻 | curl RSS |
| 10 | **Artificial Analysis** | AI 模型排行榜 | Playwright |
| 11 | **smol.ai** | AI 社区聚合 | curl RSS |

---

## Workflow

### Step 1: Fetch News Sources

```bash
# The Decoder RSS
curl -s "https://the-decoder.com/feed/" | python3 -c "
import sys, xml.etree.ElementTree as ET
tree = ET.parse(sys.stdin)
root = tree.getroot()
for item in root.findall('.//item')[:10]:
    title = item.find('title').text
    link = item.find('link').text
    pub = item.find('pubDate').text if item.find('pubDate') is not None else ''
    print(f'TITLE: {title}')
    print(f'LINK: {link}')
    print(f'DATE: {pub}')
    print('---')
"

# Hacker News AI stories (filter for AI-related keywords)
curl -s "https://hacker-news.firebaseio.com/v0/topstories.json" | python3 -c "
import sys, json, urllib.request
ids = json.load(sys.stdin)[:80]
count = 0
for id in ids:
    try:
        data = json.loads(urllib.request.urlopen(f'https://hacker-news.firebaseio.com/v0/item/{id}.json', timeout=5).read())
        title = data.get('title','')
        if any(k in title.lower() for k in ['ai','llm','gpt','claude','gemini','model','neural','openai','anthropic','deepmind','mistral','nvidia','gpu']):
            url = data.get('url', f'https://news.ycombinator.com/item?id={id}')
            score = data.get('score', 0)
            print(f'TITLE: {title}')
            print(f'LINK: {url}')
            print(f'SCORE: {score}')
            print('---')
            count += 1
            if count >= 15:
                break
    except:
        pass
"

# MarkTechPost (use Python — grep -P variable-length lookbehind is BROKEN on many systems)
curl -s -L "https://www.marktechpost.com/" | python3 -c "
import sys, re
content = sys.stdin.read()
pattern = re.compile(r'<h2[^>]*><a[^>]*href=\"([^\"]+)\"[^>]*>([^<]+)</a>', re.IGNORECASE)
matches = pattern.findall(content)[:15]
for url, title in matches:
    if title.strip():
        print(f'TITLE: {title.strip()}')
        print(f'LINK: {url}')
        print('---')
"

# KDnuggets (use Python — grep -oP variable-length lookbehind is unreliable)
curl -s -L "https://www.kdnuggets.com/" | python3 -c "
import sys, re
content = sys.stdin.read()
pattern = re.compile(r'<h[23][^>]*><a[^>]*href=\"([^\"]+)\"[^>]*>([^<]+)</a>', re.IGNORECASE)
matches = pattern.findall(content)[:15]
for url, title in matches:
    if title.strip() and len(title.strip()) > 10:
        print(f'TITLE: {title.strip()}')
        print(f'LINK: {url}')
        print('---')
"
```

### Step 2: Fetch AI Leaderboard (Playwright)

```bash
NODE_PATH=/root/.hermes/node/lib/node_modules node -e "
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  await page.goto('https://artificialanalysis.ai/', { waitUntil: 'networkidle', timeout: 60000 });
  await page.waitForTimeout(3000);
  const data = await page.evaluate(() => document.body.innerText);
  console.log(data);
  await browser.close();
})().catch(e => console.error(e.message));
"
```

### Step 3: Format Output

```markdown
# 🤖 AI 资讯日报 · {YYYY年MM月DD日}

> 汇聚 The Decoder、TechCrunch、MarkTechPost、DeepMind、BAIR、KDnuggets 等源

---

## 🏎️ The Decoder 热点

### [1] {新闻标题}
**摘要**：{核心信息 + 为什么重要，2-3 句话}
📅 {日期} | 📎 来源：[The Decoder]({链接})

---

## 📰 Hacker News AI 热议

### [1] {新闻标题}
**摘要**：{核心信息}
🔺 {分数} | 📎 来源：[HN Thread]({链接})

---

## 🏆 AI 模型排行榜（Artificial Analysis）

### 综合智能指数 TOP 10
| # | 模型 | 得分 |
|---|---|---|
| 1 | GPT-5.5 (xhigh) | **60** |
| ... | ... | ... |

### 速度排名 TOP 5（Output Tokens/s）
| # | 模型 | 速度 |
|---|---|---|
| 1 | gpt-oss-120B (high) | **231** |
| ... | ... | ... |

### 性价比排名 TOP 5（$/1M Tokens）
| # | 模型 | 价格 |
|---|---|---|
| 1 | gpt-oss-120B (high) | **$0.3** |
| ... | ... | ... |

📎 完整榜单：https://artificialanalysis.ai

---

**共抓取 X 条资讯** ✅
```

---

## 必填字段

| 字段 | 要求 |
|---|---|
| **标题** | 原文保留 |
| **摘要** | 必须中文，2-3 句话 |
| **日期** | YYYY-MM-DD |
| **链接** | 必须可点击 |

---

## Step 4: 双输出 — 写入飞书文档 + 消息返回

> ⚠️ **CRITICAL**: 执行本 skill 时，必须同时做两件事：
> 1. **写入飞书文档**（已配置 lark-cli 时）
> 2. **通过消息把完整格式化内容直接返回给用户**（无论是否写入文档）
>
> 文档只是备份和分享链接，**消息返回才是主要输出**。

### 4a: 检查 lark-cli 权限

```bash
# 检查是否已登录（注意：--json flag 避免 lark-cli stdout 混入 [lark-cli] 前缀）
lark-cli auth status --json 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('HAS_LARK_USER=true' if d.get('identity') == 'user' else 'HAS_LARK_USER=false')
" 2>/dev/null || echo "HAS_LARK_USER=false"
```

### 4b: 写入临时 Markdown 文件

```bash
# 生成日期
DATE=$(date +%Y%m%d)
DATE_DISPLAY=$(date +%Y年%m月%d日)

# 写入临时文件（必须在 /tmp 或当前工作目录，因为 lark-cli @file 只支持相对路径）
# ⚠️ CRITICAL: 使用双引号 "DOCEOF" 而非单引号 'DOCEOF'，否则 ${DATE_DISPLAY}
# 等变量会被写成字面量而非展开值（heredoc 单引号禁止变量展开）
cat << DOCEOF > /tmp/ai-digest-${DATE}.md
# 🤖 AI 资讯日报 · ${DATE_DISPLAY}

...（完整格式化内容）...

**共抓取 N 条资讯** ✅
DOCEOF
```

### 4c: 创建飞书文档（使用 v1 API）

> ⚠️ **CRITICAL PITFALL**: `--api-version v2` with `--markdown @file` does NOT work (returns `--content is required` error). You MUST use **v1 API** (default) for this operation.

```bash
cd /tmp
lark-cli docs +create --title "AI 资讯日报 · ${DATE_DISPLAY}" --markdown @ai-digest-${DATE}.md
```

**响应示例**：
```json
{
  "ok": true,
  "data": {
    "doc_id": "KQekd7bHmoqIAGxdlyLcic92nne",
    "doc_url": "https://www.feishu.cn/docx/KQekd7bHmoqIAGxdlyLcic92nne",
    "message": "文档创建成功"
  }
}
```

### 4d: 验证写入

```bash
# ⚠️ lark-cli 输出会在 JSON 前混入 [lark-cli] 前缀行，必须先过滤掉
lark-cli docs +fetch --doc {doc_id} 2>&1 | grep -v '^\[lark-cli\]' | python3 -c "
import sys, json
data = json.load(sys.stdin)
doc = data.get('data', {}).get('document', {})
print('Title:', doc.get('title'))
print('Length:', doc.get('length', 0))
"
```

### 4e: 返回文档链接（仅辅助）

将 doc_url 返回给用户，格式：
```
📄 **AI 资讯日报 · {日期}**
🔗 {url}
```

**但这不是唯一的输出** —— 还必须同时把完整日报内容通过消息直接发给用户（见 4f）。

### 4f: 通过消息返回完整内容（必须）

> ⚠️ **这是最重要的输出步骤**。在返回给用户的消息中，**必须包含完整的格式化日报内容**（所有新闻条目、所有排行榜），而不仅仅是文档链接。

在 Skill 执行完毕后，Agent 必须用 `send_message` 或在最终回复中直接输出：

```markdown
# 🤖 AI 资讯日报 · {YYYY年MM月DD日}

> 汇聚 The Decoder、TechCrunch、MarkTechPost、Hacker News、Artificial Analysis 等源

---

## 🏎️ The Decoder 热点

### [1] {新闻标题}
**摘要**：{中文摘要，2-3句话}
📅 {日期} | 📎 来源：[The Decoder]({链接})

...（所有条目完整列出）...

## 📰 Hacker News AI 热议

### [1] {新闻标题}
**摘要**：{中文摘要}
🔺 {分数} | 📎 来源：[HN Thread]({链接})

...（所有条目完整列出）...

## 🏆 AI 模型排行榜（Artificial Analysis）

### 综合智能指数 TOP 10
| # | 模型 | 得分 |
|---|---|---|
| 1 | ... | ... |

...（所有排行榜）...

📎 完整榜单：https://artificialanalysis.ai

---
**共抓取 N 条资讯** ✅
```

然后再附上飞书文档链接（如果有）。

---

## Step 5: 提交到 Git（自动）

```bash
cd /root/.hermes/skills/research/ai-news-digest

# 配置 git（如果未配置）
git config user.email "agent@hermes" 2>/dev/null
git config user.name "Hermes Agent" 2>/dev/null

# 初始化 git（如果是首次）
git init 2>/dev/null || true
git add SKILL.md
git diff --cached --stat

# 提交
git commit -m "Update ai-news-digest skill: add Feishu doc auto-write (v8.0)

- Auto-create Feishu doc after generating daily AI digest
- Check lark-cli auth before creating doc
- Use v1 API for --markdown @file (v2 broken)
- Verify doc content after creation
- Auto-commit to git"

# 显示提交结果
git log --oneline -3 2>/dev/null || echo "No commits yet"
```

---

## Pitfalls

1. **MarkTechPost/KDnuggets grep 失败**：使用 `grep -oP` 的 variable-length lookbehind 在很多系统上不工作。**必须用 Python re** 代替。

2. **lark-cli docs +create --api-version v2 失败**：v2 API 不支持 `--markdown @file`，会报 `--content is required`。**用 v1 API**（不传 `--api-version` 参数）。

3. **Hacker News 过滤**：先用更宽泛的关键词列表（加入 `openai`、`anthropic`、`nvidia` 等），数量上限从 50 提到 80，确保不漏热门 AI 新闻。

4. **lark-cli @file 路径限制**：只支持**相对路径**，不支持绝对路径 `/tmp/ai-digest.md`。**必须先 cd /tmp**，然后用 `@ai-digest.md`。

5. **lark-cli proxy 警告**：如果看到 `[WARN] proxy detected: HTTPS_PROXY=http://127.0.0.1:7890`，忽略即可，不影响功能。

6. **Git not configured**：如果 `git config` 失败（没有全局配置），不影响功能，只是不会自动 commit。手动配置：`git config --global user.email "you@email" && git config --global user.name "Your Name"`

7. **heredoc 单引号阻止变量展开**：写入 markdown 文件时，`cat << 'DOCEOF'` 会把 `${DATE_DISPLAY}` 等变量写成字面量。**必须用双引号 `cat << DOCEOF`**，让 shell 展开变量。双引号写法：`cat << DOCEOF > /tmp/ai-digest-${DATE}.md`

8. **auth status 检查的 grep 假阴性**：`lark-cli auth status` 的 JSON 输出包含其他带引号的字段，简单的 `grep -q '"identity":"user"'` 可能匹配到其他 JSON 字段内容导致结果错误。**用 `lark-cli auth status --json | python3 -c "..."` 解析**，而非 grep。

9. **`lark-cli docs +fetch` 输出混有 [lark-cli] 前缀**：`lark-cli docs +fetch` 的 stdout 会先输出若干 `[lark-cli]` 行再输出 JSON，导致 `python3 -c "json.load(sys.stdin)"` 报 `JSONDecodeError`。**必须先 `grep -v '^\\[lark-cli\\]'` 再 pipe 给 python3**。

10. **lark-cli v1 API 已标记 deprecated**：从 2026-05 起，`lark-cli docs +create` 的 v1 API 输出 `[deprecated] docs +create with v1 API is deprecated and will be removed in a future release.`。但目前 v2 仍然不支持 `--markdown @file`，所以暂时继续用 v1。**未来若 v1 被移除，需要探索 v2 的替代方案**（可能需要先 `lark-cli docs +create --title "..."` 再 `lark-cli docs +update --doc {id} --markdown @file`）。

11. **MarkTechPost / KDnuggets 持续无输出（已确认）**：自 2026-04 起这两个网站的 HTML 结构持续变化，Python regex 抓取方式不可靠。**2026-05-05 会话再次确认返回空结果**。建议：
    - **移除**这两个源，或
    - **替换为**可靠的 RSS 源：Hugging Face blog (https://huggingface.co/blog/feed.xml)、MIT News AI (https://news.mit.edu/topic/artificial-intelligence2-rss.xml)、或 Google AI blog。
    - 如果保留，必须添加 fallback：当输出为空时，跳过该源而不是让日报出现空白区块。

12. **lark-cli auth status 检查更可靠方法（2026-05-05 验证）**：`lark-cli auth status --json` 输出可能被 `[lark-cli]` 前缀污染，直接用 `json.load(sys.stdin)` 会失败。正确做法：
    ```bash
    lark-cli auth status --json 2>/dev/null | grep -v '^\[lark-cli\]' | python3 -c "
    import sys, json
    try:
        data = json.load(sys.stdin)
        print('HAS_LARK_USER=true' if data.get('identity') == 'user' else 'HAS_LARK_USER=false')
    except:
        print('HAS_LARK_USER=false')
    "
    ```
    避免使用 `grep '"identity":"user"'` 简单匹配，因为 JSON 中可能其他字段值也包含该字符串，导致假阳性。

13. **lark-cli docs +fetch 输出混有 [lark-cli] 前缀（2026-05-05 验证）**：`lark-cli docs +fetch --doc {id}` 的 stdout 会先输出若干 `[lark-cli]` 行再输出 JSON，直接 pipe 给 `python3 -c "json.load(sys.stdin)"` 报 `JSONDecodeError`。正确做法：
    ```bash
    lark-cli docs +fetch --doc {id} 2>&1 | grep -v '^\[lark-cli\]' | python3 -c "
    import sys, json
    data = json.load(sys.stdin)
    doc = data.get('data', {}).get('document', {})
    print('Title:', doc.get('title'))
    print('Length:', doc.get('length', 0))
    "
    ```
    验证文档时，应该检查 `length` 字段是否 > 5000（完整日报通常 8000+ 字符），而不是仅检查 `ok: true`。

---

## 完整执行示例

```
用户: 执行 AI news digest

Agent:
1. Fetch The Decoder RSS
2. Fetch Hacker News AI stories
3. Fetch MarkTechPost
4. Fetch KDnuggets
5. Fetch Artificial Analysis leaderboard via Playwright
6. Format all data into markdown
7. Check lark-cli auth → HAS_LARK_USER=true
8. Write to /tmp/ai-digest-20260503.md
9. cd /tmp && lark-cli docs +create --title "AI 资讯日报 · 2026年05月03日" --markdown @ai-digest-20260503.md
10. Verify doc length > 5000
11. Auto-commit SKILL.md to git
12. ★ 通过消息返回完整日报内容（所有条目 + 所有排行榜）
13. ★ 再附上飞书文档链接（格式：📄 + 🔗）
```
