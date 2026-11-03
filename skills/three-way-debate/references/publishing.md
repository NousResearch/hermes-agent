# 发布技能到 Hermes 技能商店 — 实战笔记

## 流程

```
hermes skills publish <path> --to github --repo NousResearch/hermes-agent
```

实际做的事情：
1. Fork `NousResearch/hermes-agent` 到 `jh195845886/hermes-agent`
2. 把技能文件复制到 fork 的 `community-skills/<skill-name>/`
3. 提交 + 推送
4. 创建 PR 到上游

## 踩过的坑

### 1. 401 Failed to fork

**原因**：hermes CLI 尝试 fork 时用的 token 没有写到 hermes config 里。

**解法**：
```bash
TOKEN=$(grep -oP 'ghp_[A-Za-z0-9]+' ~/.git-credentials | head -1)
hermes config set github.token "$TOKEN"
```

### 2. Fork 已存在但仍报 401

**原因**：hermes 的 fork 接口在 fork 已存在时返 401 而非 409/422。

**解法**：先删 fork 再重试，或者手动走 fork → clone → 加文件 → push → PR 流程。

### 3. PR 更新

源仓库更新后要同步到 fork 的 PR 分支：
```bash
cd /tmp/hermes-agent-fork
git checkout add-<skill-name>
git pull <source-repo-url> main  # 拉最新
# 重新复制技能文件
cp -r ~/.hermes/skills/<name>/SKILL.md community-skills/<name>/
git add community-skills/<name>/
git commit -m "update"
git push
# PR 自动更新，无需重新创建
```

### 4. token 权限

需要 `repo` scope 的 GitHub personal access token。classic token 默认有，fine-grained token 需要手动勾。

## 备用方案：手动 PR 流程

当 `hermes skills publish` 一直失败时：

```bash
TOKEN=$(grep -oP 'ghp_[A-Za-z0-9]+' ~/.git-credentials | head -1)

# 1. Fork（如果还没 fork）
curl -s -X POST -H "Authorization: token $TOKEN" \
  "https://api.github.com/repos/NousResearch/hermes-agent/forks"

# 2. Clone fork
git clone "https://<user>:${TOKEN}@github.com/<user>/hermes-agent.git" /tmp/hermes-fork

# 3. 复制技能
mkdir -p /tmp/hermes-fork/community-skills/<skill-name>
cp -r ~/.hermes/skills/<skill-name>/{SKILL.md,README.md,scripts/,references/} \
  /tmp/hermes-fork/community-skills/<skill-name>/

# 4. 提交推送
cd /tmp/hermes-fork
git checkout -b add-<skill-name>
git add community-skills/<skill-name>/
git commit -m "community-skills: add <skill-name>"
git push origin add-<skill-name>

# 5. 创建 PR
curl -s -X POST "https://api.github.com/repos/NousResearch/hermes-agent/pulls" \
  -H "Authorization: token $TOKEN" \
  -H "Accept: application/vnd.github+json" \
  -d '{"title":"community-skills: add <skill-name>","head":"<user>:add-<skill-name>","base":"main","body":"..."}'
```
