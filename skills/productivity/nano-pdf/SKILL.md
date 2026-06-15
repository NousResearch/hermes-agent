---
name: nano-pdf
description: "Edit PDF text/typos/titles via nano-pdf CLI (NL prompts)."
version: 1.0.0
author: community
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [PDF, Documents, Editing, NLP, Productivity]
    homepage: https://pypi.org/project/nano-pdf/
---

# nano-pdf

Edit PDFs using natural-language instructions. Point it at a page and describe what to change.

## Prerequisites

```bash
# Install with uv (recommended — already available in Hermes)
uv pip install nano-pdf

# Or with pip
pip install nano-pdf
```

## Usage

```bash
nano-pdf edit <file.pdf> <page_number> "<instruction>"
```

## Examples

```bash
# Change a title on page 1
nano-pdf edit deck.pdf 1 "Change the title to 'Q3 Results' and fix the typo in the subtitle"

# Update a date on a specific page
nano-pdf edit report.pdf 3 "Update the date from January to February 2026"

# Fix content
nano-pdf edit contract.pdf 2 "Change the client name from 'Acme Corp' to 'Acme Industries'"
```

**API key setup**: Set `NANOPDF_API_KEY` env var, or run `nano-pdf config set api_key <your-key>`. Get a key at https://pypi.org/project/nano-pdf/

## 頁碼偵測

nano-pdf 使用 1-based 頁碼（從 1 開始）。如果編輯到錯誤頁面，尝试 ±1 偏移。

**自動偵測腳本：**
```bash
# 列出 PDF 前 3 頁的內容，確認頁碼對應
python -c "import PyMuPDF; doc=PyMuPDF.open('file.pdf'); [print(f'Page {i+1}:', doc[i].get_text()[:100]) for i in range(min(3, len(doc)))]"
```

## 驗證

編輯後執行差異檢查：
```bash
nano-pdf diff <file.pdf>
```

或直接開啟 PDF 確認特定頁面。

## 錯誤處理

| 錯誤 | 原因 | 解決方案 |
|------|------|----------|
| `API key not configured` | NANOPDF_API_KEY 未設定 | 執行 `nano-pdf config set api_key <your-key>` 或設定環境變數 |
| `Edit failed` | 頁碼錯誤或內容不符 | 確認頁碼，參考上方自動偵測腳本 |
| `File not found` | 檔案路徑錯誤 | 使用絕對路徑，確認檔案存在 |
| `Permission denied` | 無法寫入輸出檔 | 檢查目錄權限或使用 `sudo` |
