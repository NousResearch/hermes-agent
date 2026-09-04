"""اختبارات حزمة hermes_local — تحمي التكامل من أي تحديث لاحق للتطبيق.

تعمل بلا منظومة محلية مثبّتة وبلا شبكة: كل ما يلمس الجهاز يُستبدل بمزدوج
اختبار. الغرض هو عقد التكامل نفسه — التسجيل، المخططات، التوجيه، والتحلّل
الرشيق — لأن هذا بالضبط ما ينكسر بصمت عند تحديث التطبيق.

    python -m pytest tests/test_hermes_local.py -q
    python tests/test_hermes_local.py          # بلا pytest
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools.hermes_local_tool as hl          # noqa: E402
import tools.hermes_local_extra as hx         # noqa: E402
import toolsets                                # noqa: E402
from tools.registry import registry            # noqa: E402

TOOLS = ("hermes_local_ask", "hermes_knowledge", "hermes_market",
         "hermes_status", "hermes_task", "hermes_schedule", "hermes_ops",
         "hermes_codex", "hermes_web")


# ── مزدوج اختبار للمنظومة المحلية ────────────────────────────────────────────

class FakeEngine:
    local_model = "qwen2.5:1.5b"
    memory = type("M", (), {"db_path": "x.db"})()

    def _auto_detect_domain(self, t):
        return "trading" if "سعر" in t else "research"

    def turbo_call(self, q, d=None, fc=None):
        return f"محلي[{d}]"

    def run_agent(self, q, **k):
        return {"answer": "42", "steps": 1, "mode": "native",
                "trace": [{"tool": "calculator", "args": {}, "result": "42"}]}

    def run_big_task(self, task, steps=None, mp=3):
        ids = [f"s{i+1}" for i in range(len(steps or ["a"]))]
        return {"run_id": "run_test", "steps": len(ids),
                "results": {i: "تم:" + i for i in ids}}

    class _Sup:
        @staticmethod
        def run(q, d=None):
            return {"answer": "خبير", "domain": "trading", "seconds": 0.1}
    supervisor = _Sup()

    class _Rag:
        @staticmethod
        def query(q, d=None, k=5):
            return ["مقطع نصي"]
    rag = _Rag()

    class _Sem:
        @staticmethod
        def available():
            return True

        @staticmethod
        def query(q, d=None, k=5):
            return ["مقطع دلالي"]
    semantic_rag = _Sem()

    class _Okx:
        @staticmethod
        def get_prices(s=None):
            return {"BTC-USDT-SWAP": {"last": 80000.0}}

        @staticmethod
        def get_funding_rates(s=None):
            return {"BTC-USDT-SWAP": {"rate_pct": 0.01}}

        @staticmethod
        def scan_arbitrage(t=0.01):
            return []

        @staticmethod
        def market_report():
            return "تقرير"
    okx = _Okx()

    class _Sup2:
        restarts = 0

        @staticmethod
        def is_up():
            return True

        @staticmethod
        def list_models():
            return ["qwen2.5:1.5b", "nomic-embed-text"]

        @staticmethod
        def free_ram_gb():
            return 4.0

        @staticmethod
        def preflight(m, c):
            return {"ok": True, "reason": ""}

        @staticmethod
        def ensure_up(wait_secs=30):
            return True
    ollama_sup = _Sup2()

    class _Mon:
        @staticmethod
        def dashboard():
            return "لوحة"

        @staticmethod
        def tick():
            return {"overall": True, "ts": "now"}

        @staticmethod
        def history(n=10):
            return []

        @staticmethod
        def snapshot():
            return {"ollama": True, "free_ram_gb": 4.0}

        @staticmethod
        def repair(r):
            return []
    monitor = _Mon()

    class _Sel:
        @staticmethod
        def report():
            return "تقرير موديلات"

        @staticmethod
        def benchmarks():
            return {"qwen2.5:1.5b": {"tps": 13.7}}

        @staticmethod
        def is_chat_model(m):
            return "embed" not in m
    model_selector = _Sel()

    class _Sched:
        _jobs = []

        def every_minutes(self, n, t, d="research", label=""):
            self._jobs.append(label or f"{n}د")
            return self

        def daily_at(self, h, m, t, d="research", label=""):
            self._jobs.append(label or f"{h}:{m}")
            return self

        def add_preset_morning(self):
            self._jobs.append("صباحي")
            return self

        def add_preset_okx(self, m=30):
            self._jobs.append("okx")
            return self

        @staticmethod
        def run_due_now():
            return 0

        def status(self):
            return f"المهام: {len(self._jobs)}"
    scheduler = _Sched()

    class _Pipe:
        @staticmethod
        def send(p, topic="default", sender="x"):
            return 1

        @staticmethod
        def recv(topic="default", timeout_secs=0.0):
            return {"payload": {"text": "hi"}}

        @staticmethod
        def pending(topic=None):
            return 0
    pipe = _Pipe()

    class _Morning:
        @staticmethod
        def build(news=True, llm=False):
            return "موجز صباحي"
    morning = _Morning()

    class _Codex:
        @staticmethod
        def status():
            return {"installed": True, "version": "codex-cli 0.0"}
    codex = _Codex()

    class _Docs:
        @staticmethod
        def stats():
            return {"files": 3}

        @staticmethod
        def index_into(e, domain="codex"):
            return {"text": 3}
    codex_docs = _Docs()

    class _Web:
        @staticmethod
        def search(q, n=5):
            return [{"title": "t", "url": "https://x", "source": "wikipedia"}]

        @staticmethod
        def fetch(u, n=5000):
            return {"ok": True, "text": "نص"}
    web = _Web()

    @staticmethod
    def research(q, k=3):
        return "ملخص"

    class _UI:
        @staticmethod
        def start(open_browser=False):
            return "http://127.0.0.1:8713"

        @staticmethod
        def stop():
            return None
    webui = _UI()

    class _WD:
        @staticmethod
        def watch(p, cb=None, label=""):
            return None

        @staticmethod
        def scan_once():
            return []
    watchdog = _WD()

    class _Comp:
        @staticmethod
        def count_messages(m):
            return 10

        @staticmethod
        def compress(s, m):
            return s, m
    compressor = _Comp()

    class _Auto:
        @staticmethod
        def status():
            return {"startup": True}
    autostart = _Auto()


def _install_fake():
    """يحقن المزدوج ويتخطّى تبديل المجلد."""
    hl._ENGINE = FakeEngine()
    hl._ENGINE_ERR = None
    hl._in_dir = lambda fn, *a, **k: fn(*a, **k)
    hx._in_dir = hl._in_dir
    hx._engine = lambda: hl._ENGINE


def _reset():
    hl._ENGINE = None
    hl._ENGINE_ERR = None


def _call(name, args):
    return json.loads(registry.dispatch(name, args))


# ── الاختبارات ───────────────────────────────────────────────────────────────

def _entry(name):
    """الواجهة العامة للسجل — أمتن من _tools الخاصة عبر التحديثات."""
    return registry.get_entry(name)


def _known():
    return set(registry.get_all_tool_names())


def test_all_tools_registered():
    missing = [t for t in TOOLS if t not in _known()]
    assert not missing, f"أدوات غير مسجّلة: {missing}"


def test_toolset_matches_registry():
    resolved = toolsets.resolve_toolset("hermes_local")
    assert sorted(resolved) == sorted(TOOLS), resolved
    known = _known()
    assert all(t in known for t in resolved)


def test_not_in_core_tools():
    """لو عادت إلى القائمة الأساسية صار زر التفعيل/الإطفاء كذباً."""
    core = toolsets._HERMES_CORE_TOOLS
    assert not [t for t in core if t.startswith("hermes_")], core


def test_listed_in_configurator():
    from hermes_cli import tools_config
    names = [n for n, _, _ in tools_config.CONFIGURABLE_TOOLSETS]
    assert "hermes_local" in names


def test_schemas_are_valid():
    for t in TOOLS:
        sch = getattr(_entry(t), "schema", None)
        assert sch, f"{t} بلا مخطط"
        assert sch["name"] == t
        assert len(sch["description"]) > 80, f"{t} وصفه قصير"
        params = sch["parameters"]
        assert params["type"] == "object"
        for req in params.get("required", []):
            assert req in params["properties"], f"{t}: {req} مطلوب وغير معرّف"


def test_check_fn_gates_on_missing_stack():
    old = os.environ.get("HERMES_LOCAL_DIR")
    os.environ["HERMES_LOCAL_DIR"] = str(Path(__file__).parent / "_nope_")
    try:
        ok, msg = hl.check_hermes_local()
        assert ok is False and msg
    finally:
        if old is None:
            os.environ.pop("HERMES_LOCAL_DIR", None)
        else:
            os.environ["HERMES_LOCAL_DIR"] = old


def test_missing_stack_degrades_not_crashes():
    _reset()
    old = os.environ.get("HERMES_LOCAL_DIR")
    os.environ["HERMES_LOCAL_DIR"] = str(Path(__file__).parent / "_nope_")
    try:
        for name, args in (("hermes_local_ask", {"prompt": "س"}),
                           ("hermes_knowledge", {"query": "س"}),
                           ("hermes_market", {}),
                           ("hermes_ops", {"action": "dashboard"}),
                           ("hermes_web", {"query": "س"})):
            out = registry.dispatch(name, args)
            assert isinstance(out, str) and out, name
    finally:
        if old is None:
            os.environ.pop("HERMES_LOCAL_DIR", None)
        else:
            os.environ["HERMES_LOCAL_DIR"] = old
        _reset()


def test_required_args_rejected_when_empty():
    _install_fake()
    try:
        for name, args in (("hermes_local_ask", {"prompt": "  "}),
                           ("hermes_knowledge", {"query": ""}),
                           ("hermes_task", {"task": ""}),
                           ("hermes_web", {"query": ""})):
            assert "ERROR" in registry.dispatch(name, args).upper(), name
    finally:
        _reset()


def test_ask_modes_route_correctly():
    _install_fake()
    try:
        assert _call("hermes_local_ask", {"prompt": "س"})["answer"] == "محلي[research]"
        a = _call("hermes_local_ask", {"prompt": "س", "mode": "agent"})
        assert a["tools_used"] == ["calculator"]
        s = _call("hermes_local_ask", {"prompt": "س", "mode": "specialist"})
        assert s["domain"] == "trading"
    finally:
        _reset()


def test_knowledge_prefers_semantic():
    _install_fake()
    try:
        d = _call("hermes_knowledge", {"query": "س"})
        assert d["method"] == "semantic" and d["found"] == 1
    finally:
        _reset()


def test_market_actions():
    _install_fake()
    try:
        assert "prices" in _call("hermes_market", {})
        assert "funding" in _call("hermes_market", {"what": "funding"})
        assert "opportunities" in _call("hermes_market", {"what": "arbitrage"})
        assert "report" in _call("hermes_market", {"what": "report"})
    finally:
        _reset()


def test_task_splits_steps_on_pipe():
    _install_fake()
    try:
        d = _call("hermes_task", {"task": "t", "steps": "a|b|c"})
        assert d["steps"] == 3
    finally:
        _reset()


def test_schedule_actions():
    _install_fake()
    try:
        FakeEngine.scheduler._jobs = []
        assert _call("hermes_schedule", {"action": "presets"})["jobs"] == 2
        assert _call("hermes_schedule",
                     {"action": "add_interval", "task": "t", "minutes": 5})["jobs"] == 3
        assert "ERROR" in registry.dispatch(
            "hermes_schedule", {"action": "add_interval", "task": "t"}).upper()
        assert "ERROR" in registry.dispatch(
            "hermes_schedule", {"action": "add_daily", "task": "t", "hour": 99}).upper()
        assert _call("hermes_schedule", {"action": "morning"})["brief"]
        assert _call("hermes_schedule",
                     {"action": "pipe_send", "message": "m"})["sent"] == 1
        assert _call("hermes_schedule", {"action": "clear"})["cleared"] == 3
    finally:
        _reset()


def test_ops_actions():
    _install_fake()
    try:
        assert _call("hermes_ops", {"action": "dashboard"})["dashboard"] == "لوحة"
        assert _call("hermes_ops", {"action": "models"})["selected"]
        assert _call("hermes_ops", {"action": "repair"})["repaired"]
        assert _call("hermes_ops", {"action": "ui_start"})["url"]
        assert _call("hermes_ops", {"action": "ui_stop"})["stopped"]
        assert _call("hermes_ops", {"action": "autostart"})["status"]
        assert "ERROR" in registry.dispatch(
            "hermes_ops", {"action": "لا-يوجد"}).upper()
    finally:
        _reset()


def test_ops_set_model_rejects_embedding_and_unknown():
    _install_fake()
    try:
        assert "ERROR" in registry.dispatch(
            "hermes_ops", {"action": "set_model",
                           "model": "nomic-embed-text"}).upper()
        assert "ERROR" in registry.dispatch(
            "hermes_ops", {"action": "set_model", "model": "لا-يوجد"}).upper()
        assert _call("hermes_ops", {"action": "set_model",
                                    "model": "qwen2.5:1.5b"})["model"]
    finally:
        _reset()


def test_codex_actions():
    _install_fake()
    try:
        assert _call("hermes_codex", {"action": "status"})["codex"]["installed"]
        assert _call("hermes_codex", {"action": "index_docs"})["indexed"]
        assert "ERROR" in registry.dispatch(
            "hermes_codex", {"action": "لا-يوجد"}).upper()
    finally:
        _reset()


def test_web_actions():
    _install_fake()
    try:
        assert _call("hermes_web", {"query": "q"})["count"] == 1
        assert _call("hermes_web", {"query": "q", "action": "research"})["answer"]
        assert _call("hermes_web", {"query": "https://x",
                                    "action": "fetch"})["page"]["ok"]
    finally:
        _reset()


def _main():
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    passed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  [OK]   {name}")
            passed += 1
        except Exception as exc:
            print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")
    print(f"\n  {passed}/{len(fns)} passed")
    return 0 if passed == len(fns) else 1


if __name__ == "__main__":
    sys.exit(_main())
