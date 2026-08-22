"""work-sidebar snapshot 回归测试。

验证三种 id 形态（完整 stored id / stored id 前缀 / runtime sid）都解析到
同一 stored id，且 snapshot 返回一致的标题与消息数；覆盖 desktop 重启后
``_sessions`` 为空、依赖持久映射的恢复场景；验证 outputs 结构化白名单。
运行：python -m pytest tests/gateway/test_work_sidebar_snapshot.py -q
（或直接 python 运行本文件：python test_snapshot.py）
"""
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

PLUGIN_API = Path(__file__).resolve().parents[2] / "plugins" / "work-sidebar" / "dashboard" / "plugin_api.py"

spec = importlib.util.spec_from_file_location("work_sidebar_plugin_api", PLUGIN_API)
pa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pa)

from hermes_state import SessionDB  # noqa: E402


@pytest.fixture(scope="module")
def session_db():
    return SessionDB()


@pytest.fixture(autouse=True)
def _use_fixture_db(request, session_db, monkeypatch):
    """插件内部 _get_db() 一律指向 fixture 的种子 DB。

    conftest 的 per-test 隔离会把 HERMES_HOME 切到测试级临时目录，而
    module-scope 的 session_db 在隔离生效前初始化——两者不指向同一
    state.db。统一从这里走，避免 _get_db() 读到空库导致解析落空。
    """
    if request.node.name == "test_get_db_concurrent_singleton":
        return  # 该测试验证 _get_db 单例本身，不能用 fixture 替身
    monkeypatch.setattr(pa, "_get_db", lambda: session_db)


@pytest.fixture(scope="module")
def real_session_id(session_db):
    """播种一个真实格式的会话作为基准（conftest 已把 HERMES_HOME 隔离到
    临时目录，不触碰开发机 state.db）。"""
    import time as _time

    # 真实格式 stored id：%Y%m%d_%H%M%S_随机后缀 —— resolve_session_id 的
    # 前缀解析要求前缀唯一，不能用共享前缀的命名（多个种子会话会歧义 → None）
    sid = f"{_time.strftime('%Y%m%d_%H%M%S')}_{_time.time_ns() % 10**6:06d}"
    session_db.create_session(sid, source="test")
    # 显式标题：SessionDB 不自动从消息推标题，测试的 title 断言依赖它。
    # set_session_title 强制标题唯一（跨会话冲突抛 ValueError），加后缀避免
    # 前序 pytest 运行残留的同标题种子会话冲突。
    title = f"收编 work-sidebar 到 hermes 主仓库 {sid[-6:]}"
    session_db.set_session_title(sid, title)
    # 标题兜底来源：第一条 user 消息截断（_infer_title，插件后端路径）
    session_db.append_message(sid, "user", content=title)
    # todo 最新真值（_latest_session_todos 倒序找最后一条 todo tool 消息）
    session_db.append_message(
        sid,
        "tool",
        tool_name="todo",
        content=json.dumps(
            {
                "todos": [
                    {"id": "t1", "content": "转换前端为 bundled 插件", "status": "in_progress"},
                    {"id": "t2", "content": "适配后端路由", "status": "completed"},
                ]
            }
        ),
    )
    # 产物提取源：write_file 结构化结果（_extract_outputs 白名单）
    session_db.append_message(
        sid,
        "tool",
        tool_name="write_file",
        content=json.dumps({"resolved_path": r"C:\seed\out.txt", "verified": True}),
    )
    # 持久映射兜底校验要求 last_activity_at 新鲜（_RUNTIME_SID_FRESH_S 内），
    # SessionDB 不自动更新该字段——显式打点，否则映射兜底路径永远拒绝。
    session_db.touch_session_activity(sid, description="test seed", provenance="test")
    return sid


@pytest.fixture
def isolated_sid_map(tmp_path, monkeypatch):
    """把持久映射重定向到临时目录，避免测试污染真实插件目录。"""
    target = tmp_path / ".runtime_sid_map.json"
    monkeypatch.setattr(pa, "_RUNTIME_SID_MAP", target)
    return target


def _fake_sessions(fake_server, mapping):
    """把假的 _sessions 注入 sys.modules 的 tui_gateway.server。"""
    fake_server._sessions = {
        sid: {"session_key": stored} for sid, stored in mapping.items()
    }
    pkg = types.ModuleType("tui_gateway")
    pkg.__path__ = []
    sys.modules["tui_gateway"] = pkg
    sys.modules["tui_gateway.server"] = fake_server


def _cleanup_fake_sessions():
    sys.modules.pop("tui_gateway.server", None)
    sys.modules.pop("tui_gateway", None)


def test_resolve_stored_id_forms(session_db, real_session_id, isolated_sid_map):
    """完整 stored id 与唯一前缀 → 同一 stored id。"""
    full = real_session_id  # 20260805_154004_237e48
    # 取足够长的唯一前缀（resolve_session_id 对歧义前缀返回 None）
    prefix = full[:16]  # 20260805_154004_（含分节下划线，通常唯一）

    resolved_full = pa._resolve_stored_session_id(full)
    resolved_prefix = pa._resolve_stored_session_id(prefix)

    assert resolved_full == full
    assert resolved_prefix == full, f"prefix {prefix!r} 应解析为 {full!r}, got {resolved_prefix!r}"


def test_resolve_runtime_sid_mapping(real_session_id, isolated_sid_map):
    """runtime sid（8位 hex）→ stored id：_sessions 运行时映射 + 写回持久映射。"""
    fake_server = types.ModuleType("tui_gateway.server")
    _fake_sessions(fake_server, {"d5e60859": real_session_id})
    try:
        resolved = pa._resolve_stored_session_id("d5e60859")
        assert resolved == real_session_id
        # 命中后应写回持久映射，供下一次恢复场景使用
        mapping = json.loads(isolated_sid_map.read_text(encoding="utf-8"))
        assert mapping["d5e60859"] == real_session_id
    finally:
        _cleanup_fake_sessions()


def test_runtime_sid_recovery_via_persisted_map(real_session_id, isolated_sid_map):
    """恢复场景：serve 重启后 _sessions 为空，持久映射兜底解析。"""
    isolated_sid_map.write_text(
        json.dumps({"d5e60859": real_session_id}), encoding="utf-8"
    )
    # 不注入 _sessions（模拟空 _sessions）——直接依赖持久映射
    resolved = pa._resolve_stored_session_id("d5e60859")
    assert resolved == real_session_id


def test_three_forms_consistent(session_db, real_session_id, isolated_sid_map):
    """三种 id 形态 → 同一 stored id，且 title / 消息数一致。"""
    full = real_session_id
    prefix = full[:16]
    runtime = "d5e60859"
    fake_server = types.ModuleType("tui_gateway.server")
    _fake_sessions(fake_server, {runtime: full})

    def snapshot_core(sid):
        db = session_db
        stored = pa._resolve_stored_session_id(sid, db)
        messages = db.get_messages(stored, limit=400)
        session = db.get_session(stored)
        title = session.get("title") if session else ""
        return stored, title, len(messages)

    try:
        results = [snapshot_core(sid) for sid in (full, prefix, runtime)]
        stored_ids = {r[0] for r in results}
        titles = {r[1] for r in results}
        counts = {r[2] for r in results}
        assert len(stored_ids) == 1, f"三种 id 应解析到同一 stored id: {results}"
        assert len(titles) == 1, f"三种 id 的标题应一致: {results}"
        assert len(counts) == 1, f"三种 id 的消息数应一致: {results}"
        assert titles.pop(), "标题不应为空"
    finally:
        _cleanup_fake_sessions()


def test_snapshot_consistency(session_db, real_session_id, isolated_sid_map):
    """snapshot 核心逻辑：full 与 prefix 得到同一标题与同量消息。"""
    stored_id = pa._resolve_stored_session_id(real_session_id)
    messages = session_db.get_messages(stored_id, limit=400)
    session = session_db.get_session(stored_id)
    title = session.get("title") if session else ""

    prefix = real_session_id[:16]
    stored_id_2 = pa._resolve_stored_session_id(prefix)
    messages_2 = session_db.get_messages(stored_id_2, limit=400)
    session_2 = session_db.get_session(stored_id_2)
    title_2 = session_2.get("title") if session_2 else ""

    assert stored_id == stored_id_2
    assert len(messages) == len(messages_2)
    assert title == title_2
    assert title, "标题不应为空"


def test_todos_and_outputs_extraction(session_db, real_session_id, isolated_sid_map):
    """快照三件套可提取：todos / outputs / title 结构完整。"""
    stored_id = pa._resolve_stored_session_id(real_session_id)
    messages = session_db.get_messages(stored_id, limit=400)

    todos = pa._latest_session_todos(messages)
    outputs = pa._extract_outputs(messages)
    session = session_db.get_session(stored_id)
    title = session.get("title") if session else ""

    assert isinstance(todos, (list, type(None)))
    assert isinstance(outputs, list)
    assert isinstance(title, str)
    if todos:
        for t in todos:
            assert t.get("status") in ("pending", "in_progress", "completed", "cancelled")
    for o in outputs:
        assert o.get("kind") in ("image", "link", "file")
        assert o.get("value")
        assert o.get("label")


def test_structured_outputs_whitelist():
    """工具结果 JSON 的 *_path 白名单字段 → 结构化产物（正则兜底之外）。"""
    msgs = [
        {
            "role": "tool",
            "tool_name": "write_file",
            "content": json.dumps(
                {
                    "resolved_path": r"C:\Users\ASUS\workspace-test.txt",
                    "files_modified": [r"C:\Users\ASUS\check-ha2.py"],
                    "bytes_written": 127,
                    "verified": True,
                }
            ),
            "timestamp": 1000,
        },
        {
            "role": "tool",
            "tool_name": "browser_vision",
            "content": json.dumps({"screenshot_path": r"C:\shot\a.png"}),
            "timestamp": 2000,
        },
        # 嵌套结构也应命中；.py 不在扩展名白名单，结构化路径必须保留
        {
            "role": "tool",
            "tool_name": "some_tool",
            "content": json.dumps({"result": {"output_path": [r"C:\out\check-ha2.py"]}}),
            "timestamp": 3000,
        },
        # 附件线索：display_metadata 里的路径也应收
        {
            "role": "tool",
            "tool_name": "x",
            "content": json.dumps({"ok": True}),
            "display_metadata": {"attachment_path": r"C:\attachments\doc.pdf"},
            "timestamp": 4000,
        },
    ]
    outputs = pa._extract_outputs(msgs)
    values = {o["value"] for o in outputs}
    assert r"C:\Users\ASUS\workspace-test.txt" in values
    assert r"C:\Users\ASUS\check-ha2.py" in values
    assert r"C:\shot\a.png" in values
    assert r"C:\out\check-ha2.py" in values
    assert r"C:\attachments\doc.pdf" in values
    assert all(o["kind"] == "file" for o in outputs)


def test_structured_paths_reject_non_artifact():
    """白名单键但值不是路径/URL → 不误判为产物。"""
    msgs = [
        {
            "role": "tool",
            "tool_name": "x",
            "content": json.dumps({"resolved_path": "not-a-path", "screenshot_path": 42}),
            "timestamp": 1,
        }
    ]
    assert pa._extract_outputs(msgs) == []


def test_tail_window_gets_latest_todos():
    """截断策略：只取会话尾部（最新）消息 —— 旧 todo 不覆盖新 todo。

    get_messages 是插入正序 + LIMIT 从头取；快照改用 message_count +
    offset 分页取最后 400 条，否则长会话里最新 todo/产物会被截掉。
    """
    msgs = []
    for i in range(450):
        content = None
        if i == 10:
            content = json.dumps(
                {"todos": [{"id": "old", "content": "旧任务", "status": "completed"}]}
            )
        if i == 430:
            content = json.dumps(
                {"todos": [{"id": "new", "content": "新任务", "status": "in_progress"}]}
            )
        msgs.append(
            {
                "id": 1000 + i,
                "role": "tool",
                "tool_name": "todo",
                "content": content,
                "timestamp": i,
            }
        )
    head = msgs[:400]
    tail = msgs[-400:]
    # 旧实现（从头取 400）只能看到旧 todo；新策略（尾部 400）拿到新 todo
    head_todos = pa._latest_session_todos(head)
    tail_todos = pa._latest_session_todos(tail)
    assert head_todos and head_todos[0]["id"] == "old"
    assert tail_todos and tail_todos[0]["id"] == "new"


def test_snapshot_route_three_forms(session_db, real_session_id, isolated_sid_map):
    """直接调 /snapshot 路由核心：三种 id 形式返回同一快照（title/数量一致）。"""
    full = real_session_id
    prefix = full[:16]
    runtime = "d5e60859"
    fake_server = types.ModuleType("tui_gateway.server")
    _fake_sessions(fake_server, {runtime: full})
    try:
        results = [pa.snapshot(sid) for sid in (full, prefix, runtime)]
        assert all(isinstance(r, dict) for r in results)
        assert len({r["title"] for r in results}) == 1, results
        assert len({r["messageCount"] for r in results}) == 1, results
        assert results[0]["title"], "标题不应为空"
    finally:
        _cleanup_fake_sessions()


def test_snapshot_independent_of_tool_progress(session_db, real_session_id, isolated_sid_map):
    """tool_progress=off（tool 事件被抑制）时快照仍能从 state.db 回填。

    快照数据全部来自 state.db，不依赖事件流；tool_progress 只影响
    gateway 是否 emit tool 事件。这里模拟 _sessions 里该会话
    tool_progress_mode=off，snapshot 仍返回结构完整的数据。
    """
    full = real_session_id
    runtime = "d5e60859"
    fake_server = types.ModuleType("tui_gateway.server")
    _fake_sessions(fake_server, {runtime: full})
    fake_server._sessions[runtime]["tool_progress_mode"] = "off"
    try:
        result = pa.snapshot(runtime)
        assert set(result.keys()) == {
            "todos",
            "outputs",
            "activity",
            "title",
            "messageCount",
            "storedId",
            "snapshotVersion",
            "resolved",
        }, f"快照键集变化: {set(result.keys())}"
        assert result["resolved"] is True, "tool_progress=off 时应正常解析"
        assert isinstance(result["todos"], (list, type(None)))
        assert isinstance(result["outputs"], list)
        assert result["title"], "tool_progress=off 时标题也应回填"
    finally:
        _cleanup_fake_sessions()


def test_resolved_flag(session_db, isolated_sid_map):
    """resolved 标志：正常会话为 True，无法解析的 id 为 False（不吞失败）。"""
    rows = session_db.list_sessions_rich(limit=1)
    assert rows, "隔离 HERMES_HOME 里应有种子会话"
    ok = pa.snapshot(rows[0]["id"])
    assert ok["resolved"] is True

    # 不存在的 runtime sid：解析链全落空 → resolved=False，前端据此显示
    # “数据不可用”而不是“暂无待办/暂无产物”
    bad = pa.snapshot("deadbeef")
    assert bad["resolved"] is False
    assert bad["messageCount"] == 0


def test_todo_clear_semantics():
    """todo 显式清空：最后一条 todo 消息是空列表 → 返回 []（不闪回旧列表）。"""
    msgs = [
        {
            "id": 1,
            "role": "tool",
            "tool_name": "todo",
            "content": json.dumps(
                {"todos": [{"id": "a", "content": "任务A", "status": "in_progress"}]}
            ),
            "timestamp": 1,
        },
        {"id": 2, "role": "assistant", "content": "进度更新", "timestamp": 2},
        {
            "id": 3,
            "role": "tool",
            "tool_name": "todo",
            "content": json.dumps({"todos": []}),  # 显式清空
            "timestamp": 3,
        },
    ]
    # 旧实现：空列表被跳过 → 闪回"任务A"；新实现：尊重显式清空
    assert pa._latest_session_todos(msgs) == []

    # 没有 todo 消息 → None（区分"无数据"与"已清空"）
    assert pa._latest_session_todos([{"id": 9, "role": "assistant", "content": "hi"}]) is None


def test_windows_path_regex_extraction():
    """文本层正则也能抓到 Windows 盘符路径（不依赖结构化白名单）。"""
    content = (
        "我把文件写到 " + r"C:\Users\ASUS\out\report.txt" + "，截图在 " + r"C:\tmp\shot.png"
    )
    msgs = [{"id": 1, "role": "assistant", "content": content, "timestamp": 1}]
    values = {o["value"] for o in pa._extract_outputs(msgs)}
    assert r"C:\Users\ASUS\out\report.txt" in values
    assert r"C:\tmp\shot.png" in values


def test_infer_title():
    """无标题会话：从消息推断标题（user 优先，跳过 JSON/代码块/空消息）。"""
    msgs = [
        {"role": "system", "content": "ignore me"},
        {"role": "tool", "content": json.dumps({"x": 1})},
        {"role": "user", "content": '{"json": "not a title"}'},
        {"role": "user", "content": "帮我分析这个日志文件"},
        {"role": "assistant", "content": "好的，我来看看"},
    ]
    assert pa._infer_title(msgs) == "帮我分析这个日志文件"

    # 没有 user 文本时退回 assistant
    msgs2 = [
        {"role": "user", "content": "```python\nprint(1)\n```"},
        {"role": "assistant", "content": "已修复并验证。"},
    ]
    assert pa._infer_title(msgs2) == "已修复并验证。"

    # 超长截断
    long_text = "这是一条很长的消息，" * 50
    assert len(pa._infer_title([{"role": "user", "content": long_text}])) <= 40

    # 全空
    assert pa._infer_title([]) == ""


def test_extract_activity():
    """活动兜底提取：tool 消息 → 活动条目，排除 todo/assistant，时间戳秒→毫秒。"""
    msgs = [
        {
            "id": 1,
            "role": "tool",
            "tool_name": "todo",
            "content": json.dumps({"todos": []}),
            "timestamp": 100,
        },
        {"id": 2, "role": "assistant", "content": "我在干活", "timestamp": 200},
        {
            "id": 3,
            "role": "tool",
            "tool_name": "skill_view",
            "content": json.dumps({"file": r"C:\docs\guide.md"}),
            "timestamp": 300,
        },
        {
            "id": 4,
            "role": "tool",
            "tool_name": "terminal",
            "content": json.dumps({"output": "ps aux"}),
            "timestamp": 400,
        },
    ]
    acts = pa._extract_activity(msgs)
    # 倒序取：terminal 在前，skill_view 在后；todo 与 assistant 被排除
    assert [a["type"] for a in acts] == ["tool", "read"]
    assert acts[0]["text"] == "terminal"  # 无路径键 → 退回工具名
    assert acts[1]["text"] == "skill_view: C:\\docs\\guide.md"
    assert acts[1]["time"] == 300 * 1000  # 秒 → 毫秒


def test_extract_activity_limit_and_plain_content():
    """limit 截断 + 纯文本 content（非 JSON）也能给出上下文。"""
    msgs = [
        {"id": i, "role": "tool", "tool_name": f"tool_{i}", "content": "x", "timestamp": i}
        for i in range(20)
    ]
    acts = pa._extract_activity(msgs, limit=3)
    assert len(acts) == 3
    assert acts[0]["time"] == 19 * 1000  # 最新一条在前，毫秒
    assert acts[0]["text"] == "tool_19: x"


def test_windows_path_regex_spaced_paths():
    """B1：含空格/括号的 Windows 路径不再丢失。

    旧 _PATH_RE 字符类排除 \\s 与括号 → C:\\Program Files\\report.txt 只匹配出
    C:\\Program（无扩展名）→ 被扩展名白名单拦掉，整个路径丢失。扩展名用
    白名单内的 .txt/.png（.py 属结构化通道，正则兜底本就不收）。
    """
    content = (
        "写入到 " + r"C:\Program Files\report.txt" + "，括号路径 " + r"C:\Program Files (x86)\shot.png"
    )
    msgs = [{"id": 1, "role": "assistant", "content": content, "timestamp": 1}]
    values = {o["value"] for o in pa._extract_outputs(msgs)}
    assert r"C:\Program Files\report.txt" in values, f"空格路径应被提取: {values}"
    assert r"C:\Program Files (x86)\shot.png" in values, f"括号路径应被提取: {values}"
    # 正则层：.py 全路径也应被匹配（扩展名白名单拦截是另一道独立关卡）
    m = pa._PATH_RE_SPACED.search(r"C:\Program Files\foo.py")
    assert m and m.group(2) == r"C:\Program Files\foo.py"


def test_windows_path_regex_no_cjk_merge():
    """B1：空格路径不与相邻中文合并成一条假产物（CJK 排除段字符集）。"""
    content = r"C:\docs.txt 和 C:\notes.md 都生成完毕"
    msgs = [{"id": 1, "role": "assistant", "content": content, "timestamp": 1}]
    values = [o["value"] for o in pa._extract_outputs(msgs)]
    assert r"C:\docs.txt" in values
    assert r"C:\notes.md" in values
    assert r"C:\docs.txt 和 C:\notes.md" not in values


def test_get_db_concurrent_singleton(monkeypatch):
    """B3：并发 _get_db 只创建一个 SessionDB 实例（防双实例 + 连接泄漏）。

    旧实现检查-替换非原子：并发首次请求会各自 new 一个 SessionDB，先建的
    被覆盖且永不 close（连接/atexit hook 泄漏）。
    """
    import threading as _threading

    created = []
    orig_db = pa.SessionDB

    class CountingSessionDB(orig_db):
        def __init__(self, *args, **kwargs):
            created.append(1)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(pa, "SessionDB", CountingSessionDB)
    monkeypatch.setattr(pa, "_db_singleton", None)
    monkeypatch.setattr(pa, "_db_singleton_mtime", None)

    results = []

    def worker():
        results.append(pa._get_db())

    threads = [_threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    try:
        assert len(created) == 1, f"并发 _get_db 应只创建 1 个实例，实际 {len(created)}"
        assert len({id(r) for r in results}) == 1, "所有线程应拿到同一实例"
    finally:
        for db in results:
            try:
                db.close()
            except Exception:
                pass


def test_snapshot_includes_stored_id_and_version(session_db, real_session_id, isolated_sid_map):
    """B2/B4：快照响应携带 storedId（事件过滤漂移锚点）与 snapshotVersion（缓存失效键）。"""
    runtime = "d5e60859"
    fake_server = types.ModuleType("tui_gateway.server")
    _fake_sessions(fake_server, {runtime: real_session_id})
    try:
        result = pa.snapshot(runtime)
        assert result["storedId"] == real_session_id, "storedId 应为解析后的 stored id"
        assert isinstance(result["snapshotVersion"], str) and result["snapshotVersion"]
        assert result["snapshotVersion"].startswith(f"{result['messageCount']}|")
    finally:
        _cleanup_fake_sessions()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
