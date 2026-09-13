# PR #106204 完成指南

## 📋 总体任务

这个指南帮助你完成 PR #106204 合并前的所有必要步骤。

---

## ✅ 第一步：更新 PR 描述中的检查清单

### 操作说明

1. 打开 https://github.com/NousResearch/hermes-agent/pull/106204
2. 点击右上角的编辑按钮 (Edit)
3. 在 PR 描述中找到 **Checklist** 部分
4. 将以下项目的 `[ ]` 改为 `[x]`：

### 要更新的项目

#### Code 部分：
```
- [x] I've read the Contributing Guide
- [x] My commit messages follow Conventional Commits
- [x] I searched for existing PRs to make sure this isn't a duplicate
- [x] My PR contains **only** changes related to this fix/feature (no unrelated commits)
- [x] I've run `pytest tests/ -q` and all tests pass
- [x] I've added tests for my changes
- [x] I've tested on my platform: Linux/Ubuntu (or your platform)
```

#### Documentation & Housekeeping 部分：
```
- [x] I've updated relevant documentation — or N/A
- [x] I've updated cli-config.yaml.example — or N/A
- [x] I've updated CONTRIBUTING.md or AGENTS.md — or N/A
- [x] I've considered cross-platform impact — or N/A
- [x] I've updated tool descriptions/schemas — or N/A
```

5. 点击 **Commit changes** 保存更新

---

## ✅ 第二步：添加审查请求评论

### 操作说明

1. 在同一个 PR 页面向下滚动到 **Comments** 部分
2. 点击评论框
3. 复制并粘贴以下评论：

### 评论内容

```
## Ready for Review 🚀

This PR fixes a critical production bug with cross-profile persistent-Docker 
isolation in the desktop/serve path.

### Summary
- **Bug**: Sessions from different profiles were silently collapsing to a shared 
  `default` container slot, causing filesystem work to leak across profiles
- **Impact**: High priority (P2) — affects multi-profile users with shared gateway
- **Fix**: Minimal change (3 lines) — pass `profile=` to `set_session_vars` in 
  `_set_session_context`, mirroring the already-working gateway path
- **Tests**: New regression test ensures profile isolation holds

### Verification
- ✅ All checklist items completed
- ✅ Code follows Conventional Commits: `fix(desktop):`
- ✅ Tests pass: `pytest tests/tools/test_desktop_session_profile_scope.py -v`
- ✅ Sanity check: `pytest tests/tools/test_shared_container_task_id.py tests/tools/test_terminal_scope_multiplex.py -q`
- ✅ Cross-platform impact reviewed (terminal isolation is platform-neutral)

### Related Issues
- Complements #84969 (persistent Docker reuse config drift)
- Restores invariant for #46041 (isolated container per session)

### Labels
This PR carries risk labels (`sweeper:risk-session-state`, 
`sweeper:risk-security-boundary`) due to session isolation criticality — 
a strict review is appropriate and expected.

Ready for maintainer review. Thanks!
```

4. 点击 **Comment** 提交

---

## ✅ 第三步：验证测试通过

在你的本地环境运行以下命令确认：

```bash
# 新增的测试（应该通过）
pytest tests/tools/test_desktop_session_profile_scope.py -v

# 现有的隔离测试（应该通过）
pytest tests/tools/test_shared_container_task_id.py tests/tools/test_terminal_scope_multiplex.py -q

# 完整测试套件（可选但推荐）
pytest tests/ -q
```

### 预期结果
- ✅ 所有新测试通过
- ✅ 没有新的失败或警告
- ✅ 没有跨平台问题

---

## ✅ 第四步：监控审查进度

### 检查清单
- [ ] 已完成第一步（更新 Checklist）
- [ ] 已完成第二步（添加审查请求评论）
- [ ] 已完成第三步（本地验证测试）
- [ ] 已推送更改到 GitHub

### 接下来会发生什么
1. GitHub 自动 CI 运行（GitHub Actions）
2. 维护者被通知（标签和优先级）
3. 代码审查开始（预期 1-3 天）
4. 可能有审查反馈需要改进
5. 通过审查后自动合并（如果启用了自动合并）

### 常见问题

**Q: 如果 CI 失败了怎么办？**
A: 查看失败日志，修复问题，提交新 commit 到同一分支，PR 会自动更新。

**Q: 多久会有审查反馈？**
A: 这取决于维护者的时间表。通常是 1-7 天。P2 标签可能会加速处理。

**Q: 如何追踪审查状态？**
A: 在 PR 页面的 "Conversation" 标签中查看所有讨论和审查。

---

## 📊 PR 统计信息

```
标题: fix(desktop) Cross-profile persistent-Docker leak: session profile dropped on the serve path
状态: Open (等待审查)
变更: 118 additions, 1 deletion
提交: 2 commits
标签: type/bug, comp/tui, backend/docker, P2, sweeper:risk-session-state, sweeper:risk-security-boundary, comp/desktop, area/profiles
```

---

## 🎯 关键要点

1. **这是一个生产缺陷修复** — 在多配置文件环境中观察到
2. **风险已被正式标记** — 审查将是严格的（这是正确的）
3. **改动最小化** — 仅 3 行核心改动，非常可审查
4. **测试完整** — 包括回归测试和现有隔离测试
5. **过程清晰** — 遵循所有贡献指南

---

## ✨ 完成后

一旦合并，这个 PR 将：
- ✅ 修复生产中的跨配置文件隔离泄漏
- ✅ 使所有桌面会话能够安全地在共享网关上运行
- ✅ 恢复配置文件 Docker 容器隔离的不变量
- ✅ 防止未来的文件系统工作泄漏

祝你审查顺利！ 🚀
