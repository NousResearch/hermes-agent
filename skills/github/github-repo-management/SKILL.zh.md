---
name: github-repo-management
description: 克隆、创建、分叉、配置和管理 GitHub 仓库。管理远程、密钥、发布和工作流。使用 gh CLI 或回退到 git + GitHub REST API 通过 curl。
version: 1.1.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [GitHub, Repositories, Git, Releases, Secrets, Configuration]
    related_skills: [github-auth, github-pr-workflow, github-issues]
---

# GitHub 仓库管理

创建、克隆、分叉、配置和管理 GitHub 仓库。每个部分首先显示 `gh` 命令，然后显示 `git` + `curl` 回退方案。

## 先决条件

- 已通过 GitHub 认证（参见 `github-auth` 技能）

### 设置

```bash
if command -v gh &>/dev/null && gh auth status &>/dev/null; then
  AUTH="gh"
else
  AUTH="git"
  if [ -z "$GITHUB_TOKEN" ]; then
    if [ -f ~/.hermes/.env ] && grep -q "^GITHUB_TOKEN=" ~/.hermes/.env; then
      GITHUB_TOKEN=$(grep "^GITHUB_TOKEN=" ~/.hermes/.env | head -1 | cut -d= -f2 | tr -d '\n\r')
    elif grep -q "github.com" ~/.git-credentials 2>/dev/null; then
      GITHUB_TOKEN=$(grep "github.com" ~/.git-credentials 2>/dev/null | head -1 | sed 's|https://[^:]*:\([^@]*\)@.*|\1|')
    fi
  fi
fi

# 获取你的 GitHub 用户名（多个操作需要）
if [ "$AUTH" = "gh" ]; then
  GH_USER=$(gh api user --jq '.login')
else
  GH_USER=$(curl -s -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/user | python3 -c "import sys,json; print(json.load(sys.stdin)['login'])")
fi
```

如果你已经在仓库内：

```bash
REMOTE_URL=$(git remote get-url origin)
OWNER_REPO=$(echo "$REMOTE_URL" | sed -E 's|.*github\.com[:/]||; s|\.git$||')
OWNER=$(echo "$OWNER_REPO" | cut -d/ -f1)
REPO=$(echo "$OWNER_REPO" | cut -d/ -f2)
```

---

## 1. 克隆仓库

克隆是纯 `git` 操作 — 两种方式都相同：

```bash
# 通过 HTTPS 克隆（适用于凭证助手或嵌入令牌的 URL）
git clone https://github.com/owner/repo-name.git

# 克隆到特定目录
git clone https://github.com/owner/repo-name.git ./my-local-dir

# 浅克隆（大型仓库更快）
git clone --depth 1 https://github.com/owner/repo-name.git

# 克隆特定分支
git clone --branch develop https://github.com/owner/repo-name.git

# 通过 SSH 克隆（如果配置了 SSH）
git clone git@github.com:owner/repo-name.git
```

**使用 gh（简写）：**

```bash
gh repo clone owner/repo-name
gh repo clone owner/repo-name -- --depth 1
```

## 2. 创建仓库

**使用 gh：**

```bash
# 创建公共仓库并克隆
gh repo create my-new-project --public --clone

# 私有，带描述和许可证
gh repo create my-new-project --private --description "一个有用的工具" --license MIT --clone

# 在组织下
gh repo create my-org/my-new-project --public --clone

# 从现有本地目录
cd /path/to/existing/project
gh repo create my-project --source . --public --push
```

**使用 git + curl：**

```bash
# 通过 API 创建远程仓库
curl -s -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/user/repos \
  -d '{
    "name": "my-new-project",
    "description": "一个有用的工具",
    "private": false,
    "auto_init": true,
    "license_template": "mit"
  }'

# 克隆它
git clone https://github.com/$GH_USER/my-new-project.git
cd my-new-project

# -- 或者 -- 将现有本地目录推送到新仓库
cd /path/to/existing/project
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/$GH_USER/my-new-project.git
git push -u origin main
```

在组织下创建：

```bash
curl -s -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/orgs/my-org/repos \
  -d '{"name": "my-new-project", "private": false}'
```

### 从模板创建

**使用 gh：**

```bash
gh repo create my-new-app --template owner/template-repo --public --clone
```

**使用 curl：**

```bash
curl -s -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/owner/template-repo/generate \
  -d '{"owner": "'"$GH_USER"'", "name": "my-new-app", "private": false}'
```

## 3. 分叉仓库

**使用 gh：**

```bash
gh repo fork owner/repo-name --clone
```

**使用 git + curl：**

```bash
# 通过 API 创建分叉
curl -s -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/owner/repo-name/forks

# 等待 GitHub 创建它，然后克隆
sleep 3
git clone https://github.com/$GH_USER/repo-name.git
cd repo-name

# 添加原始仓库作为 "upstream" 远程
git remote add upstream https://github.com/owner/repo-name.git
```

### 保持分叉同步

```bash
# 纯 git — 适用于任何地方
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

**使用 gh（快捷方式）：**

```bash
gh repo sync $GH_USER/repo-name
```

## 4. 仓库信息

**使用 gh：**

```bash
gh repo view owner/repo-name
gh repo list --limit 20
gh search repos "machine learning" --language python --sort stars
```

**使用 curl：**

```bash
# 查看仓库详情
curl -s \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO \
  | python3 -c "
import sys, json
r = json.load(sys.stdin)
print(f\"Name: {r['full_name']}\")
print(f\"Description: {r['description']}\")
print(f\"Stars: {r['stargazers_count']}  Forks: {r['forks_count']}\")
print(f\"Default branch: {r['default_branch']}\")
print(f\"Language: {r['language']}\")"

# 列出你的仓库
curl -s \
  -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/user/repos?per_page=20&sort=updated" \
  | python3 -c "
import sys, json
for r in json.load(sys.stdin):
    vis = 'private' if r['private'] else 'public'
    print(f\"  {r['full_name']:40}  {vis:8}  {r.get('language', ''):10}  ★{r['stargazers_count']}\")"

# 搜索仓库
curl -s \
  "https://api.github.com/search/repositories?q=machine+learning+language:python&sort=stars&per_page=10" \
  | python3 -c "
import sys, json
for r in json.load(sys.stdin)['items']:
    print(f\"  {r['full_name']:40}  ★{r['stargazers_count']:6}  {r['description'][:60] if r['description'] else ''}\")"
```

## 5. 仓库设置

**使用 gh：**

```bash
gh repo edit --description "更新的描述" --visibility public
gh repo edit --enable-wiki=false --enable-issues=true
gh repo edit --default-branch main
gh repo edit --add-topic "machine-learning,python"
gh repo edit --enable-auto-merge
```

**使用 curl：**

```bash
curl -s -X PATCH \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO \
  -d '{
    "description": "更新的描述",
    "has_wiki": false,
    "has_issues": true,
    "allow_auto_merge": true
  }'

# 更新主题
curl -s -X PUT \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.mercy-preview+json" \
  https://api.github.com/repos/$OWNER/$REPO/topics \
  -d '{"names": ["machine-learning", "python", "automation"]}'
```

## 6. 分支保护

```bash
# 查看当前保护
curl -s \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/branches/main/protection

# 设置分支保护
curl -s -X PUT \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/branches/main/protection \
  -d '{
    "required_status_checks": {
      "strict": true,
      "contexts": ["ci/test", "ci/lint"]
    },
    "enforce_admins": false,
    "required_pull_request_reviews": {
      "required_approving_review_count": 1
    },
    "restrictions": null
  }'
```

## 7. 密钥管理（GitHub Actions）

**使用 gh：**

```bash
gh secret set API_KEY --body "your-secret-value"
gh secret set SSH_KEY < ~/.ssh/id_rsa
gh secret list
gh secret delete API_KEY
```

**使用 curl：**

密钥需要使用仓库的公钥加密 — 通过 API 操作更复杂：

```bash
# 获取仓库的公钥用于加密密钥
curl -s \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/actions/secrets/public-key

# 加密并设置（需要带 PyNaCl 的 Python）
python3 -c "
from base64 import b64encode
from nacl import encoding, public
import json, sys

# 获取公钥
key_id = '<key_id_from_above>'
public_key = '<base64_key_from_above>'

# 加密
sealed = public.SealedBox(
    public.PublicKey(public_key.encode('utf-8'), encoding.Base64Encoder)
).encrypt('your-secret-value'.encode('utf-8'))
print(json.dumps({
    'encrypted_value': b64encode(sealed).decode('utf-8'),
    'key_id': key_id
}))"

# 然后 PUT 加密的密钥
curl -s -X PUT \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/actions/secrets/API_KEY \
  -d '<output from python script above>'

# 列出密钥（仅名称，值隐藏）
curl -s \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/actions/secrets \
  | python3 -c "
import sys, json
for s in json.load(sys.stdin)['secrets']:
    print(f\"  {s['name']:30}  updated: {s['updated_at']}\")"
```

注意：对于密钥，`gh secret set` 要简单得多。如果需要设置密钥且 `gh` 不可用，建议仅为该操作安装它。

## 8. 发布

**使用 gh：**

```bash
gh release create v1.0.0 --title "v1.0.0" --generate-notes
gh release create v2.0.0-rc1 --draft --prerelease --generate-notes
gh release create v1.0.0 ./dist/binary --title "v1.0.0" --notes "Release notes"
gh release list
gh release download v1.0.0 --dir ./downloads
```

**使用 curl：**

```bash
# 创建发布
curl -s -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/releases \
  -d '{
    "tag_name": "v1.0.0",
    "name": "v1.0.0",
    "body": "## Changelog\n- Feature A\n- Bug fix B",
    "draft": false,
    "prerelease": false,
    "generate_release_notes": true
  }'

# 列出发布
curl -s \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/releases \
  | python3 -c "
import sys, json
for r in json.load(sys.stdin):
    tag = r.get('tag_name', 'no tag')
    print(f\"  {tag:15}  {r['name']:30}  {'draft' if r['draft'] else 'published'}\")"

# 上传发布资产（二进制文件）
RELEASE_ID=<id_from_create_response>
curl -s -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Content-Type: application/octet-stream" \
  "https://uploads.github.com/repos/$OWNER/$REPO/releases/$RELEASE_ID/assets?name=binary-amd64" \
  --data-binary @./dist/binary-amd64
```

## 9. GitHub Actions 工作流

**使用 gh：**

```bash
gh workflow list
gh run list --limit 10
gh run view <RUN_ID>
gh run view <RUN_ID> --log-failed
gh run rerun <RUN_ID>
gh run rerun <RUN_ID> --failed
gh workflow run ci.yml --ref main
gh workflow run deploy.yml -f environment=staging
```

**使用 curl：**

```bash
# 列出工作流
curl -s \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/actions/workflows \
  | python3 -c "
import sys, json
for w in json.load(sys.stdin)['workflows']:
    print(f\"  {w['id']:10}  {w['name']:30}  {w['state']}\")"

# 列出最近的运行
curl -s \
  -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/repos/$OWNER/$REPO/actions/runs?per_page=10" \
  | python3 -c "
import sys, json
for r in json.load(sys.stdin)['workflow_runs']:
    print(f\"  Run {r['id']}  {r['name']:30}  {r['conclusion'] or r['status']}\")"

# 下载失败运行的日志
RUN_ID=<run_id>
curl -s -L \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/actions/runs/$RUN_ID/logs \
  -o /tmp/ci-logs.zip
cd /tmp && unzip -o ci-logs.zip -d ci-logs

# 重新运行失败的工作流
curl -s -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/actions/runs/$RUN_ID/rerun

# 仅重新运行失败的作业
curl -s -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/actions/runs/$RUN_ID/rerun-failed-jobs

# 手动触发工作流（workflow_dispatch）
WORKFLOW_ID=<workflow_id_or_filename>
curl -s -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/actions/workflows/$WORKFLOW_ID/dispatches \
  -d '{"ref": "main", "inputs": {"environment": "staging"}}'
```

## 10. Gists

**使用 gh：**

```bash
gh gist create script.py --public --desc "有用的脚本"
gh gist list
```

**使用 curl：**

```bash
# 创建 gist
curl -s -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/gists \
  -d '{
    "description": "有用的脚本",
    "public": true,
    "files": {
      "script.py": {"content": "print(\"hello\")"}
    }
  }'

# 列出你的 gists
curl -s \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/gists \
  | python3 -c "
import sys, json
for g in json.load(sys.stdin):
    files = ', '.join(g['files'].keys())
    print(f\"  {g['id']}  {g['description'] or '(no desc)':40}  {files}\")"
```

## 快速参考表

| 操作 | gh | git + curl |
|--------|-----|-----------|
| 克隆 | `gh repo clone o/r` | `git clone https://github.com/o/r.git` |
| 创建仓库 | `gh repo create name --public` | `curl POST /user/repos` |
| 分叉 | `gh repo fork o/r --clone` | `curl POST /repos/o/r/forks` + `git clone` |
| 仓库信息 | `gh repo view o/r` | `curl GET /repos/o/r` |
| 编辑设置 | `gh repo edit --...` | `curl PATCH /repos/o/r` |
| 创建发布 | `gh release create v1.0` | `curl POST /repos/o/r/releases` |
| 列出工作流 | `gh workflow list` | `curl GET /repos/o/r/actions/workflows` |
| 重新运行 CI | `gh run rerun ID` | `curl POST /repos/o/r/actions/runs/ID/rerun` |
| 设置密钥 | `gh secret set KEY` | `curl PUT /repos/o/r/actions/secrets/KEY` (+ 加密) |