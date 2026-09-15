---
name: github-api-push-when-network-blocks-git
description: 当 WSL 网络完全阻塞 Git HTTPS push 时，通过 GitHub REST API 直接创建 commit 推送源码
tags: [github, api, wsl, workaround]
---

# GitHub API 直接推送（当 Git Push 被网络阻塞时）

## 场景
WSL 环境中 `git push` 超时（HTTPS 和 SSH 都超时），但 `gh api` 和 `curl` 可以访问 GitHub API。

## 核心原理
绕过 git 协议，直接用 GitHub REST API 创建 blob → tree → commit → ref 四步完成推送。

## 完整流程（推荐：subprocess curl，比 http.client 更稳定）

### Step 1: 创建仓库
```bash
gh repo create <repo-name> --public --source=. --push
```

### Step 2: 准备文件
确保所有变更文件在本地已创建（不需要 git add/commit）。

### Step 3: 通过 API 推送（5步）

**Phase 1: 获取当前分支完整 SHA（必须分两步！）**
```bash
# Step A: 从 ref 获取当前 commit SHA
GET https://api.github.com/repos/{owner}/{repo}/git/refs/heads/{branch}
→ {object: {sha: "7b5d7062579c4566..."}}

# Step B: 用完整 SHA 获取 tree SHA
GET https://api.github.com/repos/{owner}/{repo}/git/commits/{full_sha}
→ {tree: {sha: "edd881b167f8d2eacb..."}}
```

⚠️ **关键坑**：直接用 `gh api ... --jq '.object.sha'` 返回的是**缩写 SHA**，用在 `/git/commits/{sha}` 会 404。必须分两步，先 ref → 再 commit。

**Phase 2: 创建所有变更文件的 blob（每个文件一次 POST）**
```bash
POST https://api.github.com/repos/{owner}/{repo}/git/blobs
body: {"content": "<file_content>", "encoding": "utf-8"}
→ 返回 {"sha": "39bfcdc36a073f8ea04d9a2c21e8bfc5535ea68a"}
```

⚠️ encoding 选择：
- 小文件（<1MB）：用 `"encoding": "utf-8"`（推荐，直接传字符串）
- 大文件（>1MB）：用 `"encoding": "base64"`

**Phase 3: 获取完整 tree**
```bash
GET https://api.github.com/repos/{owner}/{repo}/git/trees/{tree_sha}?recursive=1
→ 返回 200+ entries 的完整 tree
```

⚠️ tree SHA 必须是完整的 40 字符，缩写会 422。

**Phase 4: 创建 tree（只传变更文件，base_tree 自动合并其余）**
```bash
POST https://api.github.com/repos/{owner}/{repo}/git/trees
body: {
    "base_tree": "<parent_tree_sha>",
    "tree": [
        {"path": "src/file1.ts", "sha": "<blob_sha>", "mode": "100644", "type": "blob"},
        {"path": "src/file2.ts", "sha": "<blob_sha>", "mode": "100644", "type": "blob"}
    ]
}
→ 返回 {"sha": "3bee99d12bf212efeeb2e14e6f1b22b9996195b5"}
```

⚠️ 只传变更的文件，用 `base_tree` 让 GitHub 自动合并未包含的文件（避免 422 tree too large）。

**Phase 5: 创建 commit 并创建分支**
```bash
# 创建 commit
POST https://api.github.com/repos/{owner}/{repo}/git/commits
body: {
    "message": "feat: description",
    "tree": "<new_tree_sha>",
    "parents": ["<parent_commit_sha>"]
}
→ 返回 {"sha": "5f08c8b9f8688579f3e33f97acc598b0b9d63ae0"}

# 创建分支 ref（推荐，比 PATCH ref 更干净）
POST https://api.github.com/repos/{owner}/{repo}/git/refs
body: {"ref": "refs/heads/feature-branch", "sha": "<new_commit_sha>"}
→ 返回 {"ref": "refs/heads/feature-branch", ...}
```

## 推荐代码模板：Python + subprocess curl

```python
#!/usr/bin/env python3
import json, subprocess, os

GH_TOKEN = subprocess.run(['gh', 'auth', 'token'], capture_output=True, text=True).stdout.strip()
OWNER = 'YeLuo45'
REPO = 'pixel-pal-web'
PARENT = '3ba74812a18c64d4f47945bab404c1e76079333d'  # 完整 SHA
BASE_DIR = '/home/hermes/pixel-pal-web'

def api(method, path, data=None):
    """通过 curl 调用 GitHub API，比 http.client 更稳定"""
    cmd = ['curl', '-s', '-X', method,
           '-H', f'Authorization: Bearer {GH_TOKEN}',
           '-H', 'Accept: application/vnd.github.v3+json',
           '-H', 'Content-Type: application/json']
    url = f'https://api.github.com/repos/{OWNER}/{REPO}{path}'
    if data:
        cmd += ['-d', json.dumps(data)]
    result = subprocess.run(cmd + [url], capture_output=True, text=True, timeout=30)
    return json.loads(result.stdout)

# Step 1: 获取 parent commit 的 tree SHA
commit_data = api('GET', f'/git/commits/{PARENT}')
tree_sha = commit_data['tree']['sha']  # 用 commit 的 tree SHA，不是 commit SHA

# Step 2: 获取完整 tree
tree_data = api('GET', f'/git/trees/{tree_sha}?recursive=1')

# Step 3: 为每个变更文件创建 blob
files = [
    'src/services/rag/messageEmbedding.ts',
    'src/services/storage/embeddingStorage.ts',
    'src/components/ChatPanel/ChatHistorySearchDialog.tsx',
]
blob_map = {}
for filepath in files:
    full_path = f'{BASE_DIR}/{filepath}'
    with open(full_path) as f:
        content = f.read()
    resp = api('POST', '/git/blobs', {'content': content, 'encoding': 'utf-8'})
    blob_map[filepath] = resp['sha']

# Step 4: 构建新 tree entries（只包含变更文件）
new_entries = []
for item in tree_data['tree']:
    if item['path'] in blob_map:
        new_entries.append({'path': item['path'], 'sha': blob_map[item['path']], 'mode': item['mode'], 'type': item['type']})
    else:
        new_entries.append({'path': item['path'], 'sha': item['sha'], 'mode': item['mode'], 'type': item['type']})

# 添加新文件（blob_map 中有但 tree 中没有的）
existing_paths = {item['path'] for item in tree_data['tree']}
for filepath, sha in blob_map.items():
    if filepath not in existing_paths:
        new_entries.append({'path': filepath, 'sha': sha, 'mode': '100644', 'type': 'blob'})

# Step 5: 创建新 tree
new_tree = api('POST', '/git/trees', {'tree': new_entries, 'base_tree': tree_sha})

# Step 6: 创建 commit
new_commit = api('POST', '/git/commits', {
    'message': 'feat: description',
    'tree': new_tree['sha'],
    'parents': [PARENT]
})

# Step 7: 创建分支
api('POST', '/git/refs', {'ref': 'refs/heads/v58-voice-interaction', 'sha': new_commit['sha']})
print("Branch created!")
```

## 关键要点

| 要点 | 说明 |
|------|------|
| 必须用完整 40 字符 SHA | 缩写会 404/422，分两步：ref → commit → tree SHA |
| blob 用 `"encoding": "utf-8"` | 最简单，不需要 base64 |
| tree 只传变更文件 | `base_tree` 自动合并其余，避免 422 |
| 创建分支用 `POST /git/refs` | 比 PATCH ref 更干净，不需要 force |
| 用 `subprocess.run(['curl'])` | 比 Python http.client 在 WSL 网络下更稳定 |

## 已知限制

### 422 Unprocessable Entity — Tree 太复杂
当仓库已有大量文件时（>3000 entries），GitHub Trees API 会 422。

**解决**：只传变更的文件（约 15-20 个），用 `base_tree` 自动合并其余。

### 404 on /git/commits/{sha}
通常是 SHA 缩写导致的。用完整 SHA 重试。

## 触发 GitHub Actions
```bash
POST https://api.github.com/repos/{owner}/{repo}/actions/workflows/{workflow_filename}/dispatches
body: {"ref": "master"}
→ 204 No Content = 成功
```

## 使用场景
- 网络完全阻塞 Git 协议但 API 可用
- 需要快速推送代码触发 CI/CD
- git push 反复超时的临时解决方案
