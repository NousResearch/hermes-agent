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

### 已知坑点

### Token 被 GitHub Push Protection 拦截的修复

**症状**：
```bash
git push origin feature/xxx
# BLOCKED: GitHub Push Protection found a secret in commit 8d0c120d:
# "GitHub token as it appears in a commit"
remote: Blocked due to token detection
```
但 `git log` 显示的 token 是截断的（如 `ghp_REDACTED_EXAMPLE...7J6I`），看起来不像完整 token。

**根本原因**：终端显示 commit 时截断了输出（如 `ghp_REDACTED_EXAMPLE...7J6I`），但 commit 对象中的完整字节仍然包含真实 token。GitHub 的 Push Protection 扫描的是 commit 的原始字节内容，不是终端显示的截断文本。

**排查方法**（二进制搜索）：
```python
# 检查文件是否包含真实 token（不用字符串，用字节）
with open('skills/xxx/SKILL.md', 'rb') as f:
    content = f.read()
token_bytes = b'ghp_REDACTED_EXAMPLE'
if token_bytes in content:
    pos = content.find(token_bytes)
    print(f"Token at byte {pos}: {content[pos:pos+40]}")
```

**修复流程**：
1. 找到所有包含 token 的文件路径
2. 将 token 替换为占位符 `$GITHUB_TOKEN`（注意用字节替换：`content.replace(token_bytes, b'$GITHUB_TOKEN')`）
3. `git add <files>` → `git commit --amend --no-edit`（不要 new commit，否则 SHA 又变）
4. `GIT_TERMINAL_PROMPT=0 git push origin <branch>`

**重要**：`git commit --amend` 替换的是当前 commit，而不是创建新 commit。如果 amend 后的 commit 中仍包含 token（比如 amend 前没 add 最新文件），Push Protection 会再次拦截。

**预防**：所有技能文档和脚本中使用 `$GITHUB_TOKEN` 环境变量占位符，不要写死真实 token。

---

## Token 不一致问题
`gh auth status` 显示登录 ≠ `git push` 能用。**原因**：`git remote -v` 中的 token 和 `gh auth token` 返回的当前 token 可能不同（旧 token 被 `git remote set-url` 写入后未更新）。

**症状**：push 时 `Authentication failed`，但 `gh auth status` 显示正常。

**排查**：
```bash
# 检查 git remote 中的 token
git remote -v  # 看到完整 URL

# 检查 gh 当前 token
gh auth token  # 两者对比

# 如果不一致，更新 remote URL
git remote set-url origin https://YeLuo45:$(gh auth token)@github.com/YeLuo45/ash-echoes.git
```

### Temp Directory 方案排除 .git
用独立目录做 `git init` + push 时，`cp -r` 会复制 `.git` 目录，导致 blobs 里包含 git internal objects，后续 commit 引用这些对象时会失败（找不到）。

**解决**：只复制 dist 内容，不要整个目录：
```bash
mkdir -p /tmp/deploy-clean
cp -r /path/to/dist/. /tmp/deploy-clean/  # 用 /. 确保不复制隐藏文件（但 .git 仍会被复制！）
# 更安全：
rsync -r --exclude='.git' /path/to/dist/ /tmp/deploy-clean/
```

### Python urllib vs curl
创建大文件 blob 时（base64 编码），curl subprocess 命令行可能超长导致 `OSError: [Errno 7] Argument list too long`。Python urllib 直接在内存中处理，无此限制：
```python
import urllib.request
# 比 subprocess.run(['curl', ..., '-d', large_base64_string]) 更可靠
```

## 422 Unprocessable Entity — Tree 太复杂
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
git push https://ghp_REDACTED_EXAMPLE_TOKEN@github.com/{owner}/{repo}.git master:gh-pages --force
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

## hermes-agent 仓库推送要点

**仓库**：`git@github.com:YeLuo45/hermes-agent.git`（SSH remote）
- SSH push 在 WSL 中可能超时，但这次（2026-05-05）成功了
- 默认分支是 `main`（不是 `master`），创建 PR 时用 `--base main`
- `gh pr create` 需要 `main` 作为 base

**推送前检查**：
```bash
git remote -v  # 确认为 SSH URL (git@github.com:...)
git branch -a  # 确认分支存在
git log --oneline main..feature/xxx  # 确认有 commits 要推送
```

**创建 PR**：
```bash
gh pr create \
  --title "title" \
  --body "body" \
  --base main \
  --head feature/xxx
```

## 使用场景
- 网络完全阻塞 Git 协议但 API 可用
- 需要快速推送代码触发 CI/CD
- git push 反复超时的临时解决方案
