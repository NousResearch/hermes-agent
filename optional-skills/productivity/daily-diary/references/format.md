# Diary Format

日记格式规范和排版规则。
（此文件通过 `skill_view('daily-diary', file_path='references/format.md')` 加载）

## 存储结构

```
{STORAGE_DIR}/
├── 2026-W21/
│   ├── 2026-05-18.md
│   └── images/
│       ├── 2026-05-18-001.jpg
│       └── 2026-05-18-002.jpg
├── 2026-W22/
│   └── …
```

- 根目录：`{STORAGE_DIR}`（可配置）
- 内部：ISO 周目录 `YYYY-Www`
- 每周目录内：每天一个 `.md` 文件 + `images/` 子目录

## 条目格式

```markdown
# 2026-05-18 Monday

> ☀️ 21°C · Beijing

## 08:34

正文内容在前。

![描述](images/2026-05-18-001.jpg)

## 20:15

追加内容。
```

### 规则

- **H1:** `# YYYY-MM-DD DayOfWeek` — 每天只出现一次
- **引用行:** `> <emoji> <天气> · <地点>` — 默认地点可配置（memory key: `diary_default_location`）
- **H2:** `## HH:MM` — 每次追加的时间戳
- **正文:** 文字在前，图片在后
- **图片引用:** `![描述](images/YYYY-MM-DD-NNN.jpg)` — 相对路径
- **同一天追加:** 追加到同一文件，不加重复 H1
- **新一天:** 创建新文件，写完整头

## 天气获取

尝试 `curl -s wttr.in/{location}?format=3`，失败则跳过天气行。

默认地点：`Beijing`（可从 memory `diary_default_location` 覆盖）。
