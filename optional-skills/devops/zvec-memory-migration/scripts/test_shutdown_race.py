#!/usr/bin/env python3
"""回归测试：shutdown() 与后台写入线程（sync_turn / on_session_end）的竞态。

背景：agent 关闭时会先触发 on_session_end（内部起 daemon 线程批写），紧接着
shutdown() 释放 collection 锁。若 shutdown 用 `del self._coll` 删除实例属性，
后台线程读 self._coll 就抛 AttributeError，整批记录丢失（被吞成一条 WARNING）。

用法：
    # 只验证当前（已部署）插件
    ~/.hermes/hermes-agent/venv/bin/python3 test_shutdown_race.py

    # A/B 对照：额外传入修复前的插件副本（任意文件名）
    ~/.hermes/hermes-agent/venv/bin/python3 test_shutdown_race.py \
        ~/.hermes/backups/<某次备份>/__init__.py.bak

期望：
    修复版 —— shutdown 后 hasattr(_coll)=True，批次 4/4 落库，0 告警
    旧版   —— hasattr(_coll)=False，4 条落库 0，告警 1（含 'has no attribute _coll'）
退出码 0 = 通过。
"""
import os
import sys
import threading
import time

HOME = os.path.expanduser("~/.hermes")
PLUGIN = os.path.join(HOME, "hermes-agent/plugins/memory/memory-zvec/__init__.py")

os.environ.setdefault("HERMES_HOME", HOME)
sys.path.insert(0, os.path.join(HOME, "hermes-agent"))


class FakeColl:
    """只记录 insert/flush 的假 collection（不碰真实 Zvec 数据）。"""

    def __init__(self):
        self.inserted = []
        self.lock = threading.Lock()

    def insert(self, doc):
        time.sleep(0.03)          # 放大竞态窗口
        with self.lock:
            self.inserted.append(doc)

    def flush(self):
        pass

    def optimize(self):
        pass


def load_module(path, name):
    from importlib.machinery import SourceFileLoader
    # 备份文件后缀不是 .py，必须显式给 loader
    loader = SourceFileLoader(name, path)
    import importlib.util
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod


def run(path, name, np):
    mod = load_module(path, name)
    warnings = []
    orig = mod.logger.warning

    def cap(msg, *a, **kw):
        try:
            text = msg % a if a else msg
        except Exception:
            text = str(msg)
        if "session_end batch store failed" in str(text):
            warnings.append(str(text))
        return orig(msg, *a, **kw)

    mod.logger.warning = cap
    mod._ollama_embed_single = lambda t, b, m: np.zeros(1024, dtype=np.float32)

    p = mod.ZvecMemoryProvider({})
    fake = FakeColl()
    p._coll = fake
    p._config = {"min_content_len": 50, "enable_hnsw_optimize": False}
    p._session_id = "race_test"
    p._read_only = False

    now = time.time()
    msgs = []
    for i in range(4):                      # 4 个 user/assistant 对 → 期望 4 条
        msgs.append({"role": "user", "content": f"user-{i}" * 30, "timestamp": now + i * 2})
        msgs.append({"role": "assistant", "content": f"asst-{i}" * 30, "timestamp": now + i * 2 + 1})

    p.on_session_end(msgs)          # 起后台批写线程
    p.shutdown()                    # 立刻关停（真实时序）
    time.sleep(1.5)
    batch_has_attr = hasattr(p, "_coll")
    batch_count = len(fake.inserted)
    batch_warns = list(warnings)
    batch_ok = (batch_has_attr and batch_count == 4 and not batch_warns)

    # 用例 2：sync_turn（每轮实时写入）与 shutdown 的竞态
    p._coll = fake
    before = len(fake.inserted)
    warnings.clear()
    p.sync_turn("U" * 200, "A" * 200, session_id="race_test")
    p.shutdown()
    time.sleep(1.0)
    turn_delta = len(fake.inserted) - before
    turn_ok = (turn_delta == 1 and hasattr(p, "_coll") and not warnings)

    print(f"--- {name}")
    print(f"    on_session_end: 写入 {batch_count}/4  "
          f"hasattr(_coll)={batch_has_attr}  告警={len(batch_warns)}")
    print(f"    sync_turn:      写入 {turn_delta}/1  告警={len(warnings)}")
    for w in batch_warns + list(warnings):
        print(f"      -> {w}")
    return batch_has_attr, batch_ok, turn_ok


def main():
    try:
        import numpy as np
    except ImportError:
        print("需要 numpy（hermes-agent venv 里已有）")
        return 2

    old_path = sys.argv[1] if len(sys.argv) > 1 else None
    failures = []

    if old_path:
        if not os.path.isfile(old_path):
            print(f"找不到旧版插件副本: {old_path}")
            return 2
        old_attr, old_batch, _ = run(old_path, "old_memzvec", np)
        if old_attr or old_batch:
            failures.append("旧版副本本应复现故障，但没有")
        print("=" * 60)

    new_attr, new_batch, new_turn = run(PLUGIN, "deployed_memzvec", np)
    print("=" * 60)
    if not new_attr:
        failures.append("shutdown 后 _coll 属性消失")
    if not new_batch:
        failures.append("on_session_end 批次未完整落库")
    if not new_turn:
        failures.append("sync_turn 与 shutdown 竞态下丢轮次")

    if old_path:
        print("A/B 对照:", "✅ 旧版必坏、修复版必好" if not failures else "❌ " + "; ".join(failures))
    else:
        print("结果:", "✅ PASS" if not failures else "❌ FAIL — " + "; ".join(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
