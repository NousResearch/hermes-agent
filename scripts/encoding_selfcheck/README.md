# 编码契约自检套件（encoding_selfcheck）

**来源**：AgentSwarm `AUTO-20261109-009`（配套契约 `specs/payload-encoding-contract.md`，
提案 `AUTO-20261109-007`）。**本副本零依赖 AgentSwarm**，可独立运行。

## 这是什么

一套**可执行**的编码自检。回答三个问题，每个都给出**可判定的结论**（不是打印一堆行）：

| 模式 | 回答的问题 | 退出码 |
|---|---|---|
| `probe` | **本机环境**现在安全吗？（文件面 / 载荷面 / 输出面三个面分别判） | 安全 0 / 危险 1 |
| `static` | **我的代码里**哪些地方读改写没显式 encoding？哪些是**高危**？ | 有高危 1 / 无 0 |
| `mojibake` | **存量数据**里已经有乱码吗？ | 有命中 1 / 无 0 |

## 怎么跑

```bash
cd /aispace/hermes/hermes-agent/scripts/encoding_selfcheck

# 1) 本机环境实测
python3 encoding_selfcheck.py probe

# 2) 自身读改写点扫描（默认排除 tests/ 与 vendored 目录）
python3 encoding_selfcheck.py static /aispace/hermes/hermes-agent

# 3) 存量乱码审计
python3 encoding_selfcheck.py mojibake /home/hector/.hermes
python3 encoding_selfcheck.py mojibake /aispace/hermes/hermes-agent --json
```

加 `--json` 得到机读输出（便于接 CI）。

## CI 门禁（本仓已接入）

**接入形态**：`repo-guards.yml` 的**两个步骤**（上游 2026-11 已把 PR-only 守卫收敛成单一 `repo-guards`
job 共用一次 checkout ⇒ 独立 workflow 会多一次 runner 跳转，故按上游范式走步骤）：

| 步骤 | 条件 | 命令 |
|---|---|---|
| `Probe the payload encoding environment` | `!cancelled()`（始终） | `python3 scripts/encoding_selfcheck/encoding_selfcheck.py probe` |
| `Reject new unencoded read-modify-write points` | `!cancelled() && inputs.python` | `static . --baseline scripts/encoding_selfcheck/baseline.txt --fail-on high,payload` |

⇒ `all-checks-pass` 已 `needs: repo-guards` ⇒ 门禁红即**阻塞合并**。

> **版本号同步**：本仓的聚合检查名带版本（`All required checks pass (v3)`），
> protect-main ruleset **显式点名**该 context ⇒ 改 ci.yaml 的版本名时**必须同批**改 ruleset，
> 否则 ruleset 等一个永远不出现的 context（PR 会一直挂在"等待检查"）。
> 规则来源见 `.github/workflows/ci.yaml` 中该 job 上方的注释。

**为什么有 baseline**：本仓当前有 **17 处载荷入口 + 若干未知 + 82 处输出面**存量命中。
若"命中即红"按全量语义执行，CI **从上线第一天起就红**，且永远无法变绿 ⇒ 门禁变成噪音 ⇒ 必被加豁免 ⇒ 失效。
所以基线把"已知欠账"显式化：**存量不红，新增必红**（只减不增）。

**基线指纹是行号无关的**（`rel:kind:call`）：上游持续演进、代码行号会漂移 ⇒ 若指纹含行号，
存量会被误判成"新增" ⇒ 门禁恒红 ⇒ 基线迅速腐烂。

**刷新 baseline**（问题真修好之后）：

```bash
python3 scripts/encoding_selfcheck/encoding_selfcheck.py static . \
  --write-baseline scripts/encoding_selfcheck/baseline.txt
```

**失败时看到的输出**：

```
  ── 🔴 新增命中 1 处 (不在 baseline 内 ⇒ 门禁红) ──
   [payload] agent/lsp/xxx.py  sys.stdin.read()  (in _read_payload)
```

**套件来源与漂移**：本副本 vendored 自 AgentSwarm `AUTO-20261110-001`，
canonical 是该仓的 `scripts/encoding_selfcheck.py`。两份必须**逐字节一致** ——
AgentSwarm 侧的 CI 有**漂移守卫**（sha256 比对）盯着这件事。
改了 canonical ⇒ 同步本副本 ⇒ 刷新 AgentSwarm 侧 manifest。

## 判定怎么读

### probe

```
编码契约自检 · probe (本机环境实测)
  filesystem encoding : utf-8
  preferred encoding : UTF-8
  stdin encoding      : utf-8
  stdout encoding     : utf-8
  文件面 (形状1) 无 encoding 读回 : ✅ 逐字一致
  载荷面 (形状3) UTF-8 字节误解码 : ✅ 安全
  输出面 (形状3) stdout 写中文     : ✅ 可写
⇒ 判定: 当前环境安全
```

- **文件面（形状 1）**：`Path.read_text()` 不带 `encoding` 时按**环境默认编码**解析；
  往返不一致 = 已损坏；抛 `UnicodeDecodeError` = **响亮失败**（不是静默损坏，但链路同样断）。
- **载荷面（形状 3 入）**：`sys.stdin` 的文本模式解码。**这才是静默乱码的活险** ——
  UTF-8 字节被按 latin-1 解码后进内存，肉眼在管道层看不出来，落进 UTF-8 存储才显形。
- **输出面（形状 3 出）**：`sys.stdout.write(中文)` 会不会抛 `UnicodeEncodeError`。

> ⚠️ **"当前环境安全"≠"安全"**。本机 Python 3.14 + 只有 C/UTF-8 locale ⇒ 文件面恒为 UTF-8；
> 但 `PYTHONIOENCODING=latin-1` 之类的一行环境变量就能把载荷面打成危险。
> 契约要求**显式** encoding，不依赖环境恰好是 UTF-8。

### static

命中分三档 —— 这是与"撒网式告警"的关键差别：

| 档 | 含义 | 处置优先级 |
|---|---|---|
| **高危 high** | 字面量路径指向的文件**当前含非 ASCII 字节** ⇒ locale 非 UTF-8 即损坏 | 先治 |
| **潜伏 latent** | 路径可解析但目标当前**纯 ASCII** ⇒ 今天无害，写入中文即变高危 | 排队 |
| **未知 unknown** | 路径是变量/动态拼装 ⇒ 静态不可判 | 人工判读 |

`shape-3`（文本模式 `sys.stdin.read()` / `sys.stdout.write()`）**全部列出**，不参与三档分级 ——
它们是**载荷入口**，优先级由环境决定，不由文件内容决定。

### mojibake

输出命中文件路径 + C1 控制符计数 + 片段。C1 控制符（U+0080-U+009F）是 latin-1 误读的最强信号 ——
合法文本几乎不含 C1，而 UTF-8 续字节经 latin-1 误读恰好落进 C1 区。

判据与 AgentSwarm 仓内 canonical 实现**逐条一致**（有单元测试锁：vendored 版 vs
`swarm.db.proposal_yaml_parser.is_mojibake_text` 在同一批样本上判定相同）⇒ **零口径漂移**。

## hermes 侧的已知热点（2026-11-09 实测）

`static` 在 hermes 上的实跑结论见
`/home/hector/DevSpace/AgentSwarm/docs/audits/external-runtime-selfcheck-20261109.md`。

**优先关注 shape-3 载荷入口**（5 处生产代码）：

- `hermes_cli/main_query_file.py:27` — `args.query = sys.stdin.read()`
- `hermes_cli/gateway_chat.py:378` — `args.query = sys.stdin.read()`
- `hermes_cli/peer.py:186` — `message = sys.stdin.read().strip()`
- `hermes_cli/send_cmd.py:34` — `return sys.stdin.read()`
- `hermes_cli/kanban.py:344` — `body = sys.stdin.read()`

**改法**（契约规则 4：进程间文本载荷必须是 UTF-8 字节流，接收端显式 UTF-8 解码）：

```python
# 改前 —— 文本模式, 解码口径 = locale
body = sys.stdin.read()

# 改后 —— 显式 UTF-8, 不看 locale
raw = sys.stdin.buffer.read()          # 字节流
body = raw.decode("utf-8")             # 显式解码
# 无字节流时 (测试替身) 回退:
#   body = sys.stdin.read() if not hasattr(sys.stdin, "buffer") else ...
```

> 本目录**只提供检测能力，不改业务源码** —— 修改 hermes 业务代码属于该仓的决策范围，
> 不由本套件代劳。上面给的是改法建议。

## 注意

- 套件**自身**不会踩它检测的坑：启动即 `reconfigure(encoding="utf-8", errors="replace")`，
  因此在 `LC_ALL=C` 下也不会因 print 中文而崩。
- 探针载荷固定为"中文 + 重音 + emoji"三类（重音是**负样本**保护：真实 `café` 不该被判成乱码）。
- 本套件**只读**，不修改被扫的任何文件。
