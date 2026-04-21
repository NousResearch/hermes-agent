# Hermes Digital Office 🏢

A **game-like web UI for Hermes** that turns digital employees into little
characters wandering between four office zones — **work / talk / rest / learn** —
and lets you hire, configure, and watch them in real time. Designed so that an
8-year-old can hire their first employee in three clicks.

![Hermes Office cover](docs/img/cover.png)

> **Status:** v0.1.0 — all 66 backend tests green; frontend builds clean; live
> on `127.0.0.1:8765`.

---

## Quickstart (English)

```bash
# 1) install (from the repo root)
pip install -e .[office]

# 2) build the React frontend (one-time)
cd hermes_office/frontend
npm install
npm run build
cd ../..

# 3) launch the office (standalone) **or** use the Gateway user API server
hermes office
# → opens http://127.0.0.1:8765 in your browser

# Integrated mode (same process as Open WebUI’s backend — default port 8642):
# Enable the API server in ~/.hermes/.env (API_SERVER_ENABLED=true, API_SERVER_KEY=…)
# then:  hermes gateway
# → open http://127.0.0.1:8642/office/  (REST under /api/office/, WS /ws/office)
```

If you skip step (2), the API still runs and the page shows a friendly
"frontend not built" card with the exact build command.

### Common flags

```bash
hermes office --no-browser              # don't auto-open the browser
hermes office --port 9000               # bind a different port
hermes office --runtime hermes          # spawn real Hermes agents (LLM)
hermes office --log-level debug         # verbose server logs
```

### Using it in 30 seconds

1. Click the green **＋ Hire** button.
2. **Step 1**: pick a body 🤖🐱🦊 + color.
3. **Step 2**: pick a job (Researcher · Coder · Writer · Designer · …) **or**
   describe it in your own words ("write product copy and run A/B tests"). The
   skill resolver auto-picks toolsets + skills.
4. **Step 3**: confirm — the new employee walks into the office.
5. Type a task in the bottom composer (`@dept` or `@name` to direct it). Watch
   their speech bubbles, tool calls, and zone transitions stream in real time.

---

## 中文快速上手

Hermes 数字办公室 — 把每个数字员工画成一个会在 **工作区 / 交流区 / 休息区
/ 学习区** 之间走动的小人。**像玩游戏一样**地招人、调教、观察。

```bash
# 1) 安装
pip install -e .[office]

# 2) 构建前端（只需一次）
cd hermes_office/frontend
npm install
npm run build
cd ../..

# 3) 启动
hermes office
# 浏览器自动打开 http://127.0.0.1:8765

# 或与用户侧 Gateway API 合一（与 Open WebUI 同一后端，默认 8642）：
# ~/.hermes/.env 打开 API_SERVER_ENABLED，设置 API_SERVER_KEY，再运行 hermes gateway
# 浏览器访问 http://127.0.0.1:8642/office/（接口前缀 /api/office/，WebSocket /ws/office）
```

### 30 秒上手

1. 右上角按绿色 **＋ 招人**。
2. **第 1 步**：选个形象 🤖🐱🦊 和颜色。
3. **第 2 步**：选一个工种（研究员/程序员/作者/设计师/…）**或**直接用大白话描述
   （比如"帮我写产品文案并跑 A/B 测试"）。系统会自动推荐工具集和技能。
4. **第 3 步**：点 **入职!** —— 新员工立刻走进办公室。
5. 在底部输入框打任务，可用 `@部门名` 或 `@员工名` 指派。员工头顶的对话气泡、
   工具调用、状态切换会实时流出来。

---

## What's inside

```
hermes_office/
├── models.py             # Pydantic v2 schema (single source of truth)
├── store.py              # Atomic JSON persistence under ~/.hermes/office/
├── capacity.py           # Deterministic hardware/cost calculator
├── skill_resolver.py     # Bag-of-words classifier (no LLM, no network)
├── data/role_keywords.json
├── presets.py            # 8 built-in roles for the Hire wizard
├── eventbus.py           # asyncio fan-out for the WebSocket activity feed
├── runtime/
│   ├── simulated.py      # Synthetic activity for demos & tests
│   └── hermes.py         # Bridge to a real run_agent.AIAgent
├── server.py             # FastAPI app — REST + /ws/office
├── launcher.py           # `hermes office` subcommand
└── frontend/             # Vite + React + TypeScript + Tailwind + Canvas2D
    ├── src/
    │   ├── components/   # OfficeCanvas, HireWizard, EmployeeEditor, …
    │   ├── api.ts        # Typed REST client
    │   ├── ws.ts         # Reconnecting WebSocket subscriber
    │   └── i18n.ts       # EN + 中文 bundles
    └── dist/             # `npm run build` output (served by FastAPI)
```

The full design lives under `.kiro/specs/digital-office-ui/`:
[`requirements.md`](../.kiro/specs/digital-office-ui/requirements.md) →
[`design.md`](../.kiro/specs/digital-office-ui/design.md) →
[`tasks.md`](../.kiro/specs/digital-office-ui/tasks.md).

---

## Why these choices?

| Decision | Why |
| --- | --- |
| **FastAPI + uvicorn** | Hermes already uses it for the gateway; one process, async-native, schema-first. |
| **Local JSON store** under `$HERMES_HOME/office/` | Zero extra deps, atomic writes, easy export. Profile-aware (works with `HERMES_HOME` overrides). |
| **Canvas2D** (not DOM) for the office | 60 FPS for ≥200 sprites on a 2020 laptop; no animation library. |
| **Bag-of-words skill resolver** (not LLM) | Deterministic, instant, audit-able, and fast enough that the wizard feels live. |
| **Two runtimes** (`simulated`, `hermes`) | Demos & tests run free; live runs cost only when you flip the switch. |
| **WebSocket fan-out** | One backend event reaches every open tab without polling. |

---

## Tests

```bash
pytest hermes_office/tests -q   # 66 tests, < 5 s
```

Coverage of note:
- `test_capacity.py` locks the hardware-math outputs to hand-derived numbers.
- `test_skill_resolver.py` covers EN/中文 inputs and the optimisation loop.
- `test_server_api.py` is a full happy/error path REST + WebSocket sweep.

---

## Self-improving loop

Every accepted Hire creates a tiny telemetry record under
`~/.hermes/office/telemetry.jsonl`. The optimiser in
`skill_resolver.optimize()` re-fits the keyword weights from that file and
writes them to `~/.hermes/office/role_weights.user.json`, which the resolver
prefers over the bundled defaults at next boot. The wizard therefore gets
better at guessing *your* roles the more you hire.

To trigger an optimisation pass manually:

```bash
python -m hermes_office.skill_resolver \
  --telemetry ~/.hermes/office/telemetry.jsonl \
  --out      ~/.hermes/office/role_weights.user.json
```

---

## Security notes

- Server binds **127.0.0.1 only** by default. The WebSocket also rejects
  non-loopback `Origin` headers.
- Every persisted JSON document and every emitted ActivityEvent goes through
  `hermes_office.store.redact_secrets()` before write.
- Backups under `office/_quarantine/` are created automatically when a JSON
  file fails to load (e.g. after a power-loss mid-write).

---

## Reference

- Design / requirements / tasks: `.kiro/specs/digital-office-ui/`
- API contract: see `design.md` §5 or hit `http://127.0.0.1:8765/docs` while
  the server is running.
- Spec workflow: this module follows the [Kiro spec workflow]
  (requirements → design → tasks → implementation), all four documents are
  checked in.
