# 78 个问题修复任务 — 当前验收报告

更新时间：2026-05-01

## 当前定位

未在历史会话或本地文件中找到明确题为“78 个问题”的单独清单；现场更符合 Hermes repo 当前批量修复工作区：

- `git status --short`：69 个已改/新增路径。
- `git diff --stat`：62 个 tracked 文件变更，约 2181 insertions / 861 deletions。
- 变更覆盖 agent runtime、gateway、tools、CLI、web、docs、tests。

因此本轮按“脏工作区批量修复闭环”处理：先分组、静态扫描、编译、重点测试，再遇红修红。

## 分组

- agent_runtime：15 个路径
- gateway：17 个路径
- cli：9 个路径
- tools：20 个路径
- web：4 个路径
- misc/docs：4 个路径

## 安全扫描

新增行静态风险扫描规则：

- 长 token/API key/password/secret/credential 赋值
- `shell=True`
- `eval()` / `exec()`
- `pickle.load(s)`

结果：0 hit。

## 已通过验证

### 当前任务继承验证

- Weixin/Hermes 上下文连续性相关组合：114 passed in 4.40s
- 新增/独立烟测：12 passed in 1.25s

### Python 编译

- 已改/新增 Python 文件：63 个
- `python -m py_compile`：exit 0

### 分组测试

- tools-focused：297 passed, 2 skipped in 11.43s
- gateway-focused：295 passed in 78.96s
- cli-focused：138 passed, 1 warning in 7.61s
  - warning 为第三方 `discord/player.py` 使用 Python 3.13 将废弃的 `audioop`，非本轮代码失败。
- agent-changed：506 passed in 7.30s
- run-agent-changed：370 passed in 74.05s

## 正在验证

- web tests/build：已完成。`web/package.json` 无 `test` script，因此不能以 `npm test` 作为验收项；改用实际存在的 `npm run build`。

## Web 验证结果

- 系统 PATH：Node `v18.19.1` / npm `9.2.0`。
- fallback PATH：`~/.local/lib/nodejs/current/bin` 下 Node `v22.22.0` / npm `10.9.4`。
- 直接用系统 Node18 构建会失败：Vite 7 要 Node >=20.19，报 `crypto.hash is not a function`。
- 按既有 fallback 方案执行：
  - `PATH=$HOME/.local/lib/nodejs/current/bin:$PATH npm --prefix web run build`
  - 结果：成功，`✓ built in 8.02s`。
- 仅余 Vite chunk size warning：产物 JS 约 932 KB，属于性能建议，不是构建失败。

## 当前结论

截至目前未发现需要立即修复的红灯。主要 Python / gateway / tools / CLI / agent/runtime / web build 改动均通过聚焦分组验收。下一步生成最终交接报告；如继续深挖，可再跑全量 test suite 或按剩余业务优先级继续分片。

## 边界

本轮未执行以下高风险动作：

- 未重启 live gateway
- 未修改 cron
- 未微信实发测试
- 未删除资产
- 未运行 `hermes update`
- 未输出或保存密钥
