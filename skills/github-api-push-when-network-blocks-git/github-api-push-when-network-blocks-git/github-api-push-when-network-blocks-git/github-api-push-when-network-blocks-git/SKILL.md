---
name: github-api-push-when-network-blocks-git
description: 当 WSL 网络完全阻塞 Git HTTPS push 时，通过 GitHub REST API 直接创建 commit 推送源码
tags: [github, api, wsl, workaround]
---

# GitHub API 直接推送（当 Git Push 被网络阻塞时）

## 场景
WSL 环境中 `git push` 超时（HTTPS 和 SSH 都超时），但 `gh api` 和 `curl` 可以访问 GitHub API。

## 完整流程

### Step 1: 创建仓库
```bash
gh repo create <repo-name> --public --source=. --push
# 或者通过 GitHub API 创建空仓库后用 API 推送
```

### Step 2: 准备文件
确保所有文件在本地准备好（已 commit）。

### Step 3: 通过 API 推送（4步）

**Phase 1: 获取当前分支信息**
```python
GET https://api.github.com/repos/{owner}/{repo}/git/ref/heads/main
→ 获取 current_sha

GET https://api.github.com/repos/{owner}/{repo}/git/commits/{current_sha}
→ 获取 parent_tree_sha
```

**Phase 2: 创建所有文件 blob**
```python
POST https://api.github.com/repos/{owner}/{repo}/git/blobs
body: {
    "content": <base64_content>,
    "encoding": "base64"
}
→ 返回每个文件的 full_sha（40字符）
```

**Phase 3: 创建 tree**
```python
POST https://api.github.com/repos/{owner}/{repo}/git/trees
body: {
    "base_tree": <parent_tree_sha>,
    "tree": [{"path": <path>, "mode": "100644", "type": "blob", "sha": <full_sha>}, ...]
}
→ 返回 new_tree_sha
```

**Phase 4: 创建 commit 并更新分支**
```python
POST https://api.github.com/repos/{owner}/{repo}/git/commits
body: {
    "message": <commit_message>,
    "tree": <new_tree_sha>,
    "parents": [<current_sha>]
}
→ 返回 new_commit_sha

PATCH https://api.github.com/repos/{owner}/{repo}/git/refs/heads/main
body: {"sha": <new_commit_sha>, "force": false}
```

## 关键要点
- blob 必须用 `encoding: "base64"`（不能只用 utf-8 字符串）
- 创建 blob 时，`sha` 字段返回值是完整的 40 字符 SHA-1
- tree 的 `base_tree` 参数必须指向 parent commit 的 tree SHA（不是 commit SHA）
- commit 的 `parents` 数组包含当前分支的最新 commit SHA
- 更新 ref 用 PATCH，force=false 防止覆盖远程新提交

## 已知限制

### 422 Unprocessable Entity — Tree 太复杂
当仓库已有大量文件时（>3000 entries），GitHub Trees API 会返回 422 错误：
```
POST /git/trees → 422 Unprocessable Entity
```
**原因**：GitHub 对单次 tree 创建有 entry 数量限制（约 3000 个）。

**症状**：
```python
{"error": "HTTP 422", "body": "Unprocessable Entity"}
# parent_tree_sha 可能有 3956+ entries
```

**解决方向**（已验证，2026-05-02）：
1. **只传变更文件**：不要把所有 3956 个文件都传进 tree，只传变更的 15-20 个文件
2. GitHub 的 Trees API 会自动用 `base_tree` 合并未包含的文件
3. 示例：
```python
# 错误做法：传整个树的 3956 entries
tree_entries = [{"path": path, ...} for path in all_3956_files]  # 422!

# 正确做法：只传变更的文件（约 15-20 个）
changed_files = {"src/App.jsx": "abc123...", "src/New.jsx": "def456..."}
tree_entries = [{"path": path, "mode": "100644", "type": "blob", "sha": sha}
                for path, sha in changed_files.items()]
# 使用 base_tree=parent_tree_sha，GitHub 自动合并未变更的文件
tree_result = api_request("POST", f"/repos/{owner}/{repo}/git/trees", {
    "base_tree": parent_tree_sha,
    "tree": tree_entries
})
```

## 最可靠方案：Fresh Temp Directory + Embedded PAT Push

当 git push 和 API tree creation 都遇到问题时（大量文件、大文件、超时），最简单有效的方法：

```bash
# 1. 创建新目录
mkdir -p /tmp/gh-pages-fresh && cd /tmp/gh-pages-fresh

# 2. 初始化 git（只用 dist 内容）
git init
git config user.email "hermes@agent.local"
git config user.name "Hermes Agent"
git remote add origin https://github.com/{owner}/{repo}.git

# 3. 放入要推送的内容（dist 或其他）
cp -r /path/to/dist/. .
git add .
git commit -m "deploy message"

# 4. 关键：把 PAT 直接嵌入 URL，然后 push
#    这样避免了 git credential prompt 在网络阻塞时挂起的问题
git push https://ghp_YOUR_TOKEN@github.com/{owner}/{repo}.git master:gh-pages --force
```

**为什么有效**：
- 避免了 API 创建 blob/tree/commit 的复杂流程
- 避免了 credential helper 在网络阻塞时挂起
- 新建的 git repo 只有要推送的文件，没有历史负担
- PAT 嵌入 URL 不经过 credential helper，直接 HTTP 认证

**适用场景**：
- 推送 gh-pages 部署产物（dist 目录）
- 推送少量文件到新分支
- `git push` 超时但 `curl https://github.com/` 可达的情况

**已知限制**：
- 不能 push 带 git history 的分支（用 `--force` 覆盖整个分支）
- 需要 `git config user.email/name`，否则 commit 会失败
- PAT 嵌入 URL 会留在 shell 历史中，注意清理

## 422 "Update is not a fast forward" 解决

当 PATCH ref 时报 422，是因为远程分支已向前推进，当前 refsha 不是远程的最新 commit。

**正确流程**：
1. 先 GET /git/refs/heads/{branch} 获取远程当前 SHA
2. GET /git/commits/{sha} 获取其 tree_sha
3. 创建 blob → 以远程 tree_sha 为 base_tree 创建新 tree → 创建 commit（parent 为远程 SHA）→ PATCH ref
4. 这样是 fast-forward，不会 422

```python
# 获取远程当前 SHA（必须先做！）
GET https://api.github.com/repos/{owner}/{repo}/git/refs/heads/gh-pages
→ {"object": {"sha": "2a42e97..."}}
remote_sha = response["object"]["sha"]

# 获取其 tree_sha（作为 base_tree）
GET https://api.github.com/repos/{owner}/{repo}/git/commits/{remote_sha}
→ {"tree": {"sha": "43007ba..."}}
parent_tree = response["tree"]["sha"]

# 上传变更文件 blobs
# ...

# 用 remote_sha 作为 parent 创建 tree（传入 base_tree=parent_tree）
# GitHub Trees API 会以 base_tree 为基础，自动合并未包含的文件

# 创建 commit，parents=[remote_sha]（fast-forward）
# PATCH ref → 成功！
```

## 大文件上传超时处理

Blob 创建 API 对 > 1MB 的文件可能超时（curl 60s 限制）。

**判断方法**：
- POST /git/blobs 返回空响应或超时
- `json.decoder.JSONDecodeError: Expecting value` 表示服务器返回了非 JSON（如超时页面）

**解决方案**：
1. **GitHub Actions workflow**（最可靠）：在仓库创建 `.github/workflows/deploy.yml`，用 contents API 触发 workflow dispatch
   ```python
   POST https://api.github.com/repos/{owner}/{repo}/actions/workflows/deploy.yml/dispatches
   body: {"ref": "master"}
   → 204 = 成功
   ```
2. **分块上传**：将大文件拆分为 <500KB 的块分别上传（复杂，不推荐）

## 触发 GitHub Actions
```python
POST https://api.github.com/repos/{owner}/{repo}/actions/workflows/{workflow_filename}/dispatches
body: {"ref": "main"}
→ 204 No Content = 成功
```

## 验证部署
```bash
curl -sI https://yeluo45.github.io/<repo-slug>/
→ HTTP/2 200 = 部署成功
```

## 使用场景
- 网络完全阻塞 Git 协议但 API 可用
- 需要快速推送代码触发 CI/CD
- git push 反复超时的临时解决方案
