# Hermes 通用做T系统 Phase 2（去欧林化入口层）实施计划

> **For Hermes:** 执行本计划时，严格遵守 `test-driven-development`；每个代码行为先写失败测试，再最小实现，再跑定向回归，最后跑全量 `tests/test_hermes_olin_runtime.py`。必要时做独立代码审查。

**Goal:** 在不破坏现有 `hermes_olin` 兼容入口的前提下，引入一个正式的通用包入口层，把“对外入口语义仍绑定欧林”的问题先拆掉，为后续策略层/多标的调度层继续通用化铺路。

**Architecture:** 保持当前 runtime/store/profile 核心实现继续驻留在现有模块中，第二阶段第一批改造先新增一个通用 facade 包 `hermes_t/`。`hermes_t` 负责提供去欧林化的导出 API、generic CLI 入口和更中性的默认 runtime 目录；`hermes_olin` 保留为兼容入口，不立即删除。这样能先把“对外品牌/入口”与“内部兼容层”分开，降低后续重命名和多标的扩展成本。

**Tech Stack:** Python 3.11, pytest, argparse, pathlib, 现有 `hermes_olin` runtime/store/profile 实现。

---

## 0. 第二阶段边界审计结论

### 已确认的事实

1. `hermes_olin` 内部核心逻辑已经完成第一阶段 profile-aware 改造：
   - `RuntimeProfile`
   - `TradingStateStore`
   - `run_runtime_cycle()` 主链按 profile 读取 `trade_unit/max_trades`

2. 当前真正还“绑欧林”的主要是**对外入口层**：
   - 包名：`hermes_olin`
   - CLI 描述：`Run one Hermes Olin minimal runtime cycle`
   - 默认状态根目录：`~/.hermes_olin_runtime`
   - `__init__` 顶层文案仍写 Olin runtime

3. 当前兼容层仍合理存在：
   - `OlinStateStore` 仍有保留价值，因为默认欧林 profile 还要继续兼容旧调用方
   - 但它不应该继续作为未来通用做T系统的唯一对外语义

4. 现阶段最小高 ROI 改造不是重写 runtime，而是**补一个 generic facade 包**：
   - 低风险
   - 不扰动现有测试主链
   - 立即把后续系统叙事从“欧林专用包”推进到“有正式通用入口、欧林入口仅作兼容”

### 本批改造不做

- 不在这一批里把所有内部模块重命名成 `hermes_t.*`
- 不立即删除 `hermes_olin`
- 不做多 profile 调度器
- 不做策略插件系统
- 不做自动调参/自进化引擎

---

## 1. 本批目标

这批只做三件事：

1. 新增 `hermes_t` 通用 facade 包
2. 新增 `python -m hermes_t` generic CLI 入口
3. 更新 README/正式口径，明确：
   - `hermes_t` 是第二阶段开始后的正式通用入口
   - `hermes_olin` 仍保留为兼容入口

---

## 2. 文件范围

### 新增文件
- Create: `hermes_t/__init__.py`
- Create: `hermes_t/__main__.py`
- Create: `hermes_t/README.md`

### 修改文件
- Modify: `pyproject.toml`
- Modify: `tests/test_hermes_olin_runtime.py`
- Modify: `hermes_olin/README.md`

---

## 3. TDD 任务拆解

### Task 1: 为通用 facade 包写失败测试

**Objective:** 先用测试定义 `hermes_t` 作为正式通用入口必须存在，并且导出 generic API。

**Files:**
- Modify: `tests/test_hermes_olin_runtime.py`

**Step 1: Write failing tests**

新增测试覆盖：

```python
def test_hermes_t_exports_generic_runtime_api():
    import hermes_t

    assert hermes_t.RuntimeProfile is RuntimeProfile
    assert hermes_t.TradingStateStore is TradingStateStore
    assert callable(hermes_t.run_runtime_cycle)
```

再补一个测试确认兼容层仍保留：

```python
def test_hermes_t_does_not_expose_olin_named_store_in_generic_all():
    import hermes_t

    assert "TradingStateStore" in hermes_t.__all__
    assert "OlinStateStore" not in hermes_t.__all__
```

**Step 2: Run test to verify failure**

Run:
```bash
cd /Users/wj/hermes-agent && uv run python3 -m pytest tests/test_hermes_olin_runtime.py -q -k "hermes_t_exports_generic_runtime_api or hermes_t_does_not_expose_olin_named_store_in_generic_all"
```

Expected: FAIL — `ModuleNotFoundError: No module named 'hermes_t'`

**Step 3: Write minimal implementation**

- 新建 `hermes_t/__init__.py`
- 从现有实现 re-export：
  - `RuntimeProfile`
  - `DEFAULT_RUNTIME_PROFILE`
  - `TradingStateStore`
  - runtime 主链函数
- `__all__` 只暴露 generic 名称，不暴露 `OlinStateStore`

**Step 4: Run test to verify pass**

执行同一条 pytest，预期 PASS。

---

### Task 2: 为 generic CLI 入口写失败测试

**Objective:** 先用测试定义 `python -m hermes_t` 的默认行为和目录口径。

**Files:**
- Modify: `tests/test_hermes_olin_runtime.py`

**Step 1: Write failing tests**

新增测试覆盖：

```python
def test_hermes_t_cli_uses_generic_home_based_default_dir(monkeypatch, tmp_path, capsys):
    import hermes_t.__main__ as main_mod

    captured = {}
    home_dir = tmp_path / "home"
    home_dir.mkdir()

    def fake_run_runtime_cycle(store, **kwargs):
        captured["base_dir"] = str(store.base_dir)
        captured["profile_id"] = store.profile.profile_id
        return {"ok": True}

    monkeypatch.setattr(main_mod, "run_runtime_cycle", fake_run_runtime_cycle)
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setattr(
        "sys.argv",
        ["hermes_t", "--trade-date", "20260501", "--signal", "hold"],
    )

    main_mod.main()
    json.loads(capsys.readouterr().out)

    assert captured["base_dir"] == str(home_dir / ".hermes_t_runtime")
    assert captured["profile_id"] == DEFAULT_RUNTIME_PROFILE.profile_id
```
```

再补一个测试确认 generic CLI 不再走 `OlinStateStore`：

```python
def test_hermes_t_cli_uses_trading_state_store_even_for_default_profile(monkeypatch, tmp_path, capsys):
    import hermes_t.__main__ as main_mod

    captured = {}

    def fake_run_runtime_cycle(store, **kwargs):
        captured["store_class"] = type(store).__name__
        return {"ok": True}

    monkeypatch.setattr(main_mod, "run_runtime_cycle", fake_run_runtime_cycle)
    monkeypatch.setattr(
        "sys.argv",
        ["hermes_t", "--base-dir", str(tmp_path), "--trade-date", "20260501"],
    )

    main_mod.main()
    json.loads(capsys.readouterr().out)

    assert captured["store_class"] == "TradingStateStore"
```

**Step 2: Run test to verify failure**

Run:
```bash
cd /Users/wj/hermes-agent && uv run python3 -m pytest tests/test_hermes_olin_runtime.py -q -k "hermes_t_cli_uses_generic_home_based_default_dir or hermes_t_cli_uses_trading_state_store_even_for_default_profile"
```

Expected: FAIL — because `hermes_t.__main__` does not yet exist.

**Step 3: Write minimal implementation**

- 新建 `hermes_t/__main__.py`
- 复用现有 `RuntimeProfile` / `TradingStateStore` / `run_runtime_cycle`
- parser 描述改成 generic 文案，例如 `Run one Hermes generalized T-runtime cycle`
- 默认 `--base-dir` 改为 `Path.home() / ".hermes_t_runtime"`
- 即使是默认 profile，也统一实例化 `TradingStateStore`

**Step 4: Run test to verify pass**

执行同一条 pytest，预期 PASS。

---

### Task 3: 为 packaging 与文档口径补齐验证

**Objective:** 确保 `hermes_t` 会被打包发现，且 README 口径和阶段结论同步。

**Files:**
- Modify: `pyproject.toml`
- Create: `hermes_t/README.md`
- Modify: `hermes_olin/README.md`

**Step 1: Write/extend verification tests if needed**

如果已有 import 测试已足以覆盖包发现，可不额外增代码测试；至少要做人工验证：

- `pyproject.toml` 的 `[tool.setuptools.packages.find].include` 包含：
  - `hermes_t`
  - `hermes_t.*`

**Step 2: Implement docs/package updates**

- 在 `pyproject.toml` 加入 `hermes_t`, `hermes_t.*`
- 写 `hermes_t/README.md`：说明它是第二阶段起的通用入口 facade
- 在 `hermes_olin/README.md` 追加兼容说明：
  - 新通用入口优先 `python -m hermes_t`
  - `hermes_olin` 继续保留给欧林默认兼容口径

**Step 3: Run regression**

Run:
```bash
cd /Users/wj/hermes-agent && uv run python3 -m pytest tests/test_hermes_olin_runtime.py -q
```

Expected: PASS

---

## 4. 第二阶段完成判定（本批）

本批完成后，应满足：

- 存在正式 generic 包入口：`hermes_t`
- 存在正式 generic CLI：`python -m hermes_t`
- generic CLI 默认目录不再是 `.hermes_olin_runtime`
- generic facade 不把 `OlinStateStore` 作为主导出语义
- `hermes_olin` 仍可继续兼容既有调用和文档语义

---

## 5. 下一批预留方向

完成这批后，下一批可继续推进：

1. 抽 shared CLI builder，减少 `hermes_t` / `hermes_olin` 的重复解析逻辑
2. 引入 `signal policy` 层，把阈值/文本模板从 runtime 主文件剥离
3. 增加多 profile 编排入口（例如 `profiles.yaml` + orchestrator）
4. 继续把 `tests/test_hermes_olin_runtime.py` 拆成更通用的 `tests/test_trading_runtime.py`
