# Hermes 通用做T系统 Phase 2 Batch 2（shared CLI + 多 Profile Orchestrator）实施计划

> **For Hermes:** 严格遵守 `test-driven-development`。先写失败测试，再最小实现，再跑定向回归，最后跑全量 `tests/test_hermes_olin_runtime.py`。

**Goal:** 在已完成 `hermes_t` 通用入口 facade 的基础上，继续推进第二阶段第二批：抽出 shared CLI builder 消除 `hermes_olin` / `hermes_t` 的重复解析逻辑，并补一个最小可用的多 profile orchestrator 骨架，为后续多股票调度做正式起点。

**Architecture:** 不重写 runtime 主链，继续复用 `RuntimeProfile`、`TradingStateStore`、`run_runtime_cycle()`。新增一个共享 CLI 模块，负责统一 parser / args -> profile / args -> store 的装配；再新增 orchestrator 模块，负责把多个 profile 定义批量编排为一次多 runtime 调用。

**Tech Stack:** Python 3.11, pytest, argparse, json, dataclasses/pathlib（如需要）, 现有 `hermes_olin` / `hermes_t` 代码。

---

## 1. 本批目标

1. 抽 shared CLI builder，去掉两个 `__main__.py` 中的重复参数定义
2. 新增多 profile orchestrator 骨架
3. 保持现有 `hermes_olin` / `hermes_t` 行为不回退
4. README / 正式口径同步到“已支持 generic facade + multi-profile skeleton”

## 2. 文件范围

### 新增
- `hermes_t/cli_shared.py`
- `hermes_t/orchestrator.py`

### 修改
- `hermes_t/__main__.py`
- `hermes_olin/__main__.py`
- `tests/test_hermes_olin_runtime.py`
- `hermes_olin/README.md`
- `hermes_t/README.md`

## 3. TDD 任务拆解

### Task A: shared CLI builder
- 先写失败测试，锁定：
  - shared builder 能按传入参数生成 parser
  - shared builder 能构造 `RuntimeProfile`
  - shared builder 能按 generic / legacy 模式构造 `TradingStateStore` 或 `OlinStateStore`
- 再最小实现 `hermes_t/cli_shared.py`
- 最后让 `hermes_olin.__main__` 与 `hermes_t.__main__` 都改为调用 shared helper

### Task B: multi-profile orchestrator skeleton
- 先写失败测试，锁定：
  - 可从 JSON 文件读取多个 profile 定义
  - 能对每个 profile 依次创建 `TradingStateStore`
  - 能调用 runner（默认 `run_runtime_cycle`）并返回汇总结果
- 再最小实现 `hermes_t/orchestrator.py`

### Task C: docs + regression
- README 同步
- 全量回归

## 4. 本批完成判定

完成后应满足：
- `hermes_olin.__main__` / `hermes_t.__main__` 共用同一套 CLI 构建逻辑
- `hermes_t` 存在正式 `orchestrator` 模块
- 可通过 API 对多个 profile 做最小批量编排
- 现有单 profile 行为和默认欧林兼容行为不回退
