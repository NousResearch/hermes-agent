# Hermes 通用股票做T系统升级实施计划

> **For Hermes:** 执行本计划时，严格遵守 `test-driven-development`；每个代码行为先写失败测试，再最小实现，再回归。必要时再做独立代码审查。

**Goal:** 把当前 `hermes_olin` 的“欧林专用最小正式 runtime”升级为“可迁移到其他股票、可剥离欧林数据、具备后续自动迭代能力”的通用做T系统骨架，并在第一阶段先完成运行时配置与状态存储的去欧林耦合。

**Architecture:** 采用“通用运行时内核 + 股票策略/画像配置 + 每标的独立状态命名空间”的分层结构。第一阶段不直接重命名整个包，也不一次性引入复杂多股票调度，而是在保持当前 `hermes_olin` 兼容的前提下，先抽出通用 `RuntimeProfile/StateStore` 语义，让 runtime 的关键常量、状态目录、标的身份从欧林硬编码变成可注入配置。后续再分阶段推进策略插件化、观测闭环、自迭代闭环。

**Tech Stack:** Python 3.11, pytest, pathlib, dataclass/pydantic风格轻量配置对象, 现有 `hermes_olin` 测试体系。

---

## 0. 当前审计结论

### 已确认的现状

当前 `hermes_olin` 还只是“欧林专用最小 runtime”，离老王要求的目标还有明显差距：

1. **包名和公共语义仍然绑定欧林**
   - `hermes_olin/__init__.py`
   - `hermes_olin/__main__.py`
   - `hermes_olin/store.py`
   - `hermes_olin/runtime.py`
   - `tests/test_hermes_olin_runtime.py`

2. **状态存储类名直接写死为 `OlinStateStore`**
   - `hermes_olin/store.py`
   - 这意味着未来迁移到别的股票时，连基础状态层都带着欧林语义。

3. **执行参数仍是 runtime 常量，不是 profile/config**
   - `TRADE_UNIT`
   - `MAX_T_TRADES`
   - 文案里也直接写“第N次买入/卖出 10000 股”

4. **状态目录只有单命名空间，没有“按标的隔离”能力**
   - 目前是 `base_dir/state/realtime/`
   - 删除欧林数据后复用别的股票，缺少正式 profile/symbol 目录边界

5. **当前自愈主要是“派发/挂起状态自愈”，不是“策略自动迭代”**
   - `recover_signal_runtime()`
   - `deliver_pending_signal()`
   - `confirm_dispatch_sent()`
   - 它解决的是 runtime 状态一致性，不是参数日级进化、策略评估、标的画像更新

### 第一阶段设计目标

先做最小但方向正确的通用化收口：

- 把“欧林专属状态存储”升级为“通用状态存储”
- 把 `trade_unit/max_trades/symbol/profile_id` 变成可配置 profile
- 把状态路径升级成“按 profile/symbol 隔离”的结构
- 保持当前 CLI 和现有欧林测试大体兼容
- 为第二阶段“通用股票画像/策略插件化”留接口

---

## 1. 分阶段路线图

### Phase A — 运行时去欧林耦合（现在开始）

目标：
- 提炼通用运行时配置对象
- 提炼通用状态存储类
- 让 runtime 使用 profile 驱动，而不是硬编码欧林
- 保持兼容别名，避免一次性破坏当前调用方

### Phase B — 策略层参数化

目标：
- 把买卖阈值、trade unit、max trades、消息模板、交易窗口规则从 runtime 主流程中抽离
- 形成 `profile -> signal policy -> runtime decision` 的可替换结构

### Phase C — 多标的/多画像正式支持

目标：
- 同一套内核可运行多个 symbol/profile
- 每个 symbol 独立状态目录、独立 dispatch ledger、独立 execution state
- 删除欧林数据后不影响系统骨架

### Phase D — 自动迭代/演进闭环

目标：
- 加入复盘输入、策略效果统计、参数建议器
- 支持日级/周级评估与参数推荐
- 把“能修 runtime 状态”升级为“能演进策略参数”

---

## 2. 本轮实施范围

本轮只执行 **Phase A 的第一刀**，避免过度设计：

1. 新增通用运行时 profile 对象
2. 新增通用状态存储类 `TradingStateStore`
3. 让 `OlinStateStore` 成为兼容别名/子类，而非唯一正式语义
4. 让 runtime 核心函数支持从 store/profile 读取 `symbol / trade_unit / max_trades`
5. 将状态目录从固定 `state/realtime` 升级为 `profiles/<profile_id>/state/realtime`
6. 为兼容当前欧林调用方，默认 profile 仍指向 `olin-688319`

**本轮不做：**
- 不重命名包 `hermes_olin`
- 不一次性改所有外部脚本
- 不接入真实多股票调度器
- 不实现自动调参算法
- 不动盘中线上引擎

---

## 3. 目标文件

### 重点修改文件
- Modify: `hermes_olin/store.py`
- Modify: `hermes_olin/runtime.py`
- Modify: `hermes_olin/__main__.py`
- Modify: `hermes_olin/__init__.py`
- Modify: `hermes_olin/README.md`
- Test: `tests/test_hermes_olin_runtime.py`

### 可能新增文件
- Create: `hermes_olin/profile.py`

---

## 4. 实施任务拆解

## Task 1: 新增运行时 profile 对象

**Objective:** 把 symbol、profile_id、trade_unit、max_trades 从 runtime 常量中抽出来，形成最小通用配置对象。

**Files:**
- Create: `hermes_olin/profile.py`
- Modify: `hermes_olin/__init__.py`
- Test: `tests/test_hermes_olin_runtime.py`

**Step 1: Write failing test**

在 `tests/test_hermes_olin_runtime.py` 新增测试，验证默认 profile 具备稳定的欧林兼容值，例如：

```python
def test_default_runtime_profile_keeps_olin_compatible_values():
    from hermes_olin.profile import DEFAULT_RUNTIME_PROFILE

    assert DEFAULT_RUNTIME_PROFILE.profile_id == "olin-688319"
    assert DEFAULT_RUNTIME_PROFILE.symbol == "688319"
    assert DEFAULT_RUNTIME_PROFILE.trade_unit == 10000
    assert DEFAULT_RUNTIME_PROFILE.max_trades == 4
```

再补一个测试，验证可创建其他股票 profile：

```python
def test_runtime_profile_supports_non_olin_symbol():
    from hermes_olin.profile import RuntimeProfile

    profile = RuntimeProfile(
        profile_id="test-600519",
        symbol="600519",
        trade_unit=200,
        max_trades=6,
    )

    assert profile.profile_id == "test-600519"
    assert profile.symbol == "600519"
    assert profile.trade_unit == 200
    assert profile.max_trades == 6
```

**Step 2: Run test to verify failure**

Run:
```bash
pytest tests/test_hermes_olin_runtime.py::test_default_runtime_profile_keeps_olin_compatible_values tests/test_hermes_olin_runtime.py::test_runtime_profile_supports_non_olin_symbol -q
```

Expected:
- FAIL
- 原因应是 `hermes_olin.profile` 或 `RuntimeProfile` 尚不存在

**Step 3: Write minimal implementation**

在 `hermes_olin/profile.py` 中新增最小实现：

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeProfile:
    profile_id: str
    symbol: str
    trade_unit: int = 10000
    max_trades: int = 4


DEFAULT_RUNTIME_PROFILE = RuntimeProfile(
    profile_id="olin-688319",
    symbol="688319",
    trade_unit=10000,
    max_trades=4,
)
```

并在 `hermes_olin/__init__.py` 导出。

**Step 4: Run test to verify pass**

Run:
```bash
pytest tests/test_hermes_olin_runtime.py::test_default_runtime_profile_keeps_olin_compatible_values tests/test_hermes_olin_runtime.py::test_runtime_profile_supports_non_olin_symbol -q
```

Expected:
- PASS

**Step 5: Commit**

```bash
git add hermes_olin/profile.py hermes_olin/__init__.py tests/test_hermes_olin_runtime.py
git commit -m "feat: add runtime profile for generalized trading runtime"
```

---

## Task 2: 引入通用状态存储类并保留欧林兼容别名

**Objective:** 让正式状态存储语义从 `OlinStateStore` 升级为通用 `TradingStateStore`，同时不破坏已有调用。

**Files:**
- Modify: `hermes_olin/store.py`
- Modify: `hermes_olin/__init__.py`
- Test: `tests/test_hermes_olin_runtime.py`

**Step 1: Write failing test**

新增测试验证：

```python
def test_trading_state_store_uses_profile_scoped_state_directory(tmp_path):
    from hermes_olin.profile import RuntimeProfile
    from hermes_olin.store import TradingStateStore

    profile = RuntimeProfile(profile_id="demo-600519", symbol="600519")
    store = TradingStateStore(tmp_path, profile=profile)

    assert store.base_dir == tmp_path
    assert store.profile.profile_id == "demo-600519"
    assert store.state_dir == tmp_path / "profiles" / "demo-600519" / "state" / "realtime"
```

再补一个兼容测试：

```python
def test_olin_state_store_remains_backward_compatible(tmp_path):
    from hermes_olin.store import OlinStateStore

    store = OlinStateStore(tmp_path)

    assert store.profile.profile_id == "olin-688319"
    assert store.profile.symbol == "688319"
```

**Step 2: Run test to verify failure**

Run:
```bash
pytest tests/test_hermes_olin_runtime.py::test_trading_state_store_uses_profile_scoped_state_directory tests/test_hermes_olin_runtime.py::test_olin_state_store_remains_backward_compatible -q
```

Expected:
- FAIL
- 因为 `TradingStateStore` 不存在，或 `profile` / 新目录结构尚未实现

**Step 3: Write minimal implementation**

在 `hermes_olin/store.py`：

- 新增 `TradingStateStore`
- 支持构造参数 `profile: RuntimeProfile = DEFAULT_RUNTIME_PROFILE`
- `state_dir` 改为：
  - `base_dir / "profiles" / profile.profile_id / "state" / "realtime"`
- `OlinStateStore` 改为：
  - `class OlinStateStore(TradingStateStore): ...`
  - 默认 profile 固定为 `DEFAULT_RUNTIME_PROFILE`

示意：

```python
class TradingStateStore:
    def __init__(self, base_dir: str | Path, profile: RuntimeProfile = DEFAULT_RUNTIME_PROFILE):
        self.base_dir = Path(base_dir)
        self.profile = profile
        self.profile_dir = self.base_dir / "profiles" / profile.profile_id
        self.state_dir = self.profile_dir / "state" / "realtime"
        self.state_dir.mkdir(parents=True, exist_ok=True)


class OlinStateStore(TradingStateStore):
    def __init__(self, base_dir: str | Path):
        super().__init__(base_dir, profile=DEFAULT_RUNTIME_PROFILE)
```

**Step 4: Run test to verify pass**

Run:
```bash
pytest tests/test_hermes_olin_runtime.py::test_trading_state_store_uses_profile_scoped_state_directory tests/test_hermes_olin_runtime.py::test_olin_state_store_remains_backward_compatible -q
```

Expected:
- PASS

**Step 5: Commit**

```bash
git add hermes_olin/store.py hermes_olin/__init__.py tests/test_hermes_olin_runtime.py
git commit -m "refactor: introduce generic trading state store"
```

---

## Task 3: 让执行建议读取 profile 参数，而不是 runtime 常量

**Objective:** 让 runtime 的建议生成逻辑真正支持其他股票画像，而不是只支持欧林 10000 股 / 4 笔。

**Files:**
- Modify: `hermes_olin/runtime.py`
- Test: `tests/test_hermes_olin_runtime.py`

**Step 1: Write failing test**

新增测试：

```python
def test_build_execution_suggestion_uses_profile_trade_unit_and_max_trades(tmp_path):
    from hermes_olin.profile import RuntimeProfile
    from hermes_olin.store import TradingStateStore
    from hermes_olin.runtime import build_execution_suggestion

    profile = RuntimeProfile(
        profile_id="demo-600519",
        symbol="600519",
        trade_unit=200,
        max_trades=6,
    )
    store = TradingStateStore(tmp_path, profile=profile)

    suggestion = build_execution_suggestion(
        store,
        {"summary_signal": "sell", "score": {"total": 20}},
        "20260501",
    )

    assert suggestion["trade_unit"] == 200
    assert suggestion["max_trades"] == 6
    assert suggestion["text"] == "第1次卖出 200 股"
```

再加边界测试：

```python
def test_build_execution_suggestion_respects_profile_max_trades_limit(tmp_path):
    from hermes_olin.profile import RuntimeProfile
    from hermes_olin.store import TradingStateStore

    profile = RuntimeProfile(
        profile_id="demo-600519",
        symbol="600519",
        trade_unit=200,
        max_trades=1,
    )
    store = TradingStateStore(tmp_path, profile=profile)
    store.save_execution_state({"trade_date": "20260501", "sell_count": 1})

    suggestion = build_execution_suggestion(
        store,
        {"summary_signal": "sell", "score": {"total": 20}},
        "20260501",
    )

    assert suggestion["next_action"] == "hold"
```

**Step 2: Run test to verify failure**

Run:
```bash
pytest tests/test_hermes_olin_runtime.py::test_build_execution_suggestion_uses_profile_trade_unit_and_max_trades tests/test_hermes_olin_runtime.py::test_build_execution_suggestion_respects_profile_max_trades_limit -q
```

Expected:
- FAIL
- 因为当前代码仍读取 `TRADE_UNIT` / `MAX_T_TRADES`

**Step 3: Write minimal implementation**

在 `hermes_olin/runtime.py` 中：

- 增加 profile 读取辅助函数，例如：

```python
def _profile_trade_unit(store: OlinStateStore) -> int:
    return int(getattr(store, "profile", DEFAULT_RUNTIME_PROFILE).trade_unit)


def _profile_max_trades(store: OlinStateStore) -> int:
    return int(getattr(store, "profile", DEFAULT_RUNTIME_PROFILE).max_trades)
```

- 在 `build_execution_suggestion()` 中替换硬编码常量
- 在相关分支中用 profile 值生成文本和阈值判断
- 保持旧默认行为不变

**Step 4: Run test to verify pass**

Run:
```bash
pytest tests/test_hermes_olin_runtime.py::test_build_execution_suggestion_uses_profile_trade_unit_and_max_trades tests/test_hermes_olin_runtime.py::test_build_execution_suggestion_respects_profile_max_trades_limit -q
```

Expected:
- PASS

**Step 5: Run nearby regression tests**

Run:
```bash
pytest tests/test_hermes_olin_runtime.py::test_build_execution_suggestion_uses_execution_state_sequence tests/test_hermes_olin_runtime.py::test_build_execution_suggestion_holds_when_active_signal_exists tests/test_hermes_olin_runtime.py::test_stage_pending_signal_sets_active_signal_in_execution_state -q
```

Expected:
- PASS

**Step 6: Commit**

```bash
git add hermes_olin/runtime.py tests/test_hermes_olin_runtime.py
git commit -m "refactor: drive execution suggestions from runtime profile"
```

---

## Task 4: CLI 注入 profile，并维持欧林默认入口兼容

**Objective:** 让 CLI 默认仍能跑欧林，但技术上已经具备切换 profile 的能力。

**Files:**
- Modify: `hermes_olin/__main__.py`
- Test: `tests/test_hermes_olin_runtime.py`

**Step 1: Write failing test**

新增测试：

```python
def test_cli_main_supports_profile_override(monkeypatch, tmp_path):
    import hermes_olin.__main__ as main_mod

    captured = {}

    def fake_run_runtime_cycle(store, **kwargs):
        captured["profile_id"] = store.profile.profile_id
        captured["symbol"] = store.profile.symbol
        return {
            "trade_date": kwargs["effective_trade_date"],
            "now": "2026-05-01 09:30:00",
            "recovery": {},
            "suggestion": {},
            "pending": {},
            "result": {},
            "execution_state": {},
            "push_state": {},
        }

    monkeypatch.setattr(main_mod, "run_runtime_cycle", fake_run_runtime_cycle)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hermes-olin",
            "--base-dir", str(tmp_path),
            "--trade-date", "20260501",
            "--profile-id", "demo-600519",
            "--symbol", "600519",
        ],
    )

    main_mod.main()

    assert captured["profile_id"] == "demo-600519"
    assert captured["symbol"] == "600519"
```

**Step 2: Run test to verify failure**

Run:
```bash
pytest tests/test_hermes_olin_runtime.py::test_cli_main_supports_profile_override -q
```

Expected:
- FAIL
- 因为 CLI 还没有 `--profile-id` / `--symbol`

**Step 3: Write minimal implementation**

在 `hermes_olin/__main__.py`：

- 新增 CLI 参数：
  - `--profile-id`
  - `--symbol`
  - 可选：`--trade-unit`
  - 可选：`--max-trades`
- 组装 `RuntimeProfile`
- 用 `TradingStateStore(base_dir, profile=profile)` 替换直接 `OlinStateStore(base_dir)` 的正式路径
- 若不传参数，则退回 `DEFAULT_RUNTIME_PROFILE`

**Step 4: Run test to verify pass**

Run:
```bash
pytest tests/test_hermes_olin_runtime.py::test_cli_main_supports_profile_override -q
```

Expected:
- PASS

**Step 5: Run existing CLI regressions**

Run:
```bash
pytest tests/test_hermes_olin_runtime.py::test_cli_main_defaults_base_dir_under_home_even_after_chdir tests/test_hermes_olin_runtime.py::test_cli_main_passes_explicit_dispatch_target tests/test_hermes_olin_runtime.py::test_cli_main_uses_env_dispatch_target tests/test_hermes_olin_runtime.py::test_cli_main_rejects_invalid_trade_date -q
```

Expected:
- PASS

**Step 6: Commit**

```bash
git add hermes_olin/__main__.py tests/test_hermes_olin_runtime.py
git commit -m "feat: add profile-aware CLI entry for trading runtime"
```

---

## Task 5: README 与正式语义收口

**Objective:** 把 README 从“欧林专用最小 runtime”描述升级为“兼容欧林默认画像的通用化第一阶段内核”。

**Files:**
- Modify: `hermes_olin/README.md`

**Step 1: Update docs**

README 至少要同步以下几点：

1. 当前仍是第一阶段，不是完整多股票产品
2. 默认 profile 为 `olin-688319`
3. 状态路径是 `~/.hermes_olin_runtime/profiles/<profile_id>/state/realtime/`
4. CLI 支持 profile/symbol 覆盖
5. 后续 Phase B/C/D 仍待推进

**Step 2: Verification**

人工检查 README 不再把系统描述成“只能服务欧林”的最终形态。

**Step 3: Commit**

```bash
git add hermes_olin/README.md
git commit -m "docs: document generalized runtime phase-1 semantics"
```

---

## 5. 全量验证命令

完成本轮后至少执行：

```bash
pytest tests/test_hermes_olin_runtime.py -q
```

如果新增了 profile/store 独立测试文件，也要跑：

```bash
pytest tests/test_hermes_olin_runtime.py tests/test_hermes_olin_profile.py -q
```

如有必要，补一次更广回归：

```bash
pytest tests/ -q
```

---

## 6. 验收标准

本轮完成后，必须满足：

- [ ] `RuntimeProfile` 已存在，默认仍兼容欧林
- [ ] `TradingStateStore` 已存在，`OlinStateStore` 仍可兼容使用
- [ ] 状态目录已按 `profile_id` 隔离
- [ ] `build_execution_suggestion()` 不再依赖硬编码 `TRADE_UNIT/MAX_T_TRADES`
- [ ] CLI 可注入 profile/symbol
- [ ] 现有欧林回归仍通过
- [ ] README 已同步第一阶段通用化语义

---

## 7. 设计边界与注意事项

1. **先做兼容抽象，不做激进重命名**
   - 当前包名 `hermes_olin` 暂不改
   - 先把内部语义做通用，再考虑第二阶段拆包

2. **坚持 TDD**
   - 每个行为改动先写失败测试
   - 必须显式验证 RED，再写实现

3. **避免过度设计**
   - 本轮不引入策略注册中心
   - 本轮不引入数据库
   - 本轮不做多进程调度器

4. **兼容优先**
   - 旧的欧林默认入口不能被破坏
   - 老测试如果表达的是正确业务语义，应继续通过

5. **正式目标不是“换个类名”，而是建立可迁移边界**
   - 真正关键是 profile/symbol/状态命名空间解耦
   - 这是未来删除欧林数据后继续复用系统的前提

---

## 8. 本计划后的下一步

完成本轮后，下一轮优先做：

1. 把信号阈值和风控参数从 runtime 主函数抽到 profile/policy
2. 建立 `symbol profile -> strategy policy -> runtime outcome` 三层结构
3. 为“自动迭代”预留评估记录和参数建议输入面
4. 再决定是否把 `hermes_olin` 演进为更中性的顶层包，例如 `hermes_t` 或 `hermes_trading`
