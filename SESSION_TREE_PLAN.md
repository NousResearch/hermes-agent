# Session Tree 重构计划

## 目标
在 `SessionDB` 中引入 session 关系树，一次 DB 查询构建完整树结构，替代当前散点式多次查询，统一 `/resume` 的候选过滤逻辑。

## 背景问题
1. `_handle_resume_command` 中候选过滤逻辑重复了两份（`/resume last` 和 `/resume` 列表模式）
2. 每个压缩过的候选要调 `_get_compression_chain_ids` + `_find_latest_leaf`，10个候选就 20+ 次 DB 查询
3. `get_ancestor_ids` 逐层查询（N+1 问题）

## 修改文件

### 1. `hermes_state.py` — 新增 SessionNode、SessionTree、build_session_tree()

在 `SessionDB` 类之前新增数据类，在 `SessionDB` 类内部新增方法。

#### 新增数据类（在 `class SessionDB:` 之前）

```python
from dataclasses import dataclass, field

@dataclass
class SessionNode:
    """Session 树中的一个节点"""
    id: str
    parent_id: Optional[str]
    end_reason: Optional[str]
    message_count: int
    started_at: float
    title: Optional[str]
    source: str
    children: List['SessionNode'] = field(default_factory=list)

    @property
    def is_compressed(self) -> bool:
        return self.end_reason == 'compression'

    @property
    def is_active(self) -> bool:
        return self.end_reason is None

    @property
    def is_independent_branch(self) -> bool:
        """session_reset / session_switch / branched — 独立分支，不是压缩链延续"""
        return self.end_reason in ('session_reset', 'session_switch', 'branched')


class SessionTree:
    """一次查询构建的完整 session 关系树，所有遍历操作在内存中完成"""

    def __init__(self, nodes: Dict[str, SessionNode], roots: List[SessionNode]):
        self.nodes = nodes   # id → SessionNode
        self.roots = roots   # 根节点列表

    def get_ancestor_ids(self, session_id: str) -> Set[str]:
        """O(depth) 内存遍历，0 次 DB 查询。返回不含 session_id 自身的祖先集合。"""
        ancestors = set()
        node = self.nodes.get(session_id)
        while node and node.parent_id:
            ancestors.add(node.parent_id)
            node = self.nodes.get(node.parent_id)
        return ancestors

    def find_compression_leaf(self, session_id: str) -> Optional[SessionNode]:
        """替代 _find_latest_leaf。沿 compression/NULL 子节点走到叶子。

        只跟踪 compression 或 active 的子节点（与原 _find_latest_leaf 行为一致），
        在 session_reset/session_switch 边界停止。
        """
        node = self.nodes.get(session_id)
        if not node:
            return None
        visited = set()
        while node.id not in visited:
            visited.add(node.id)
            comp_children = [
                c for c in node.children
                if c.is_compressed or c.is_active
            ]
            if not comp_children:
                return node
            node = max(comp_children, key=lambda c: c.started_at)
        return node  # 安全兜底

    def get_compression_chain_ids(self, session_id: str) -> Set[str]:
        """替代 _get_compression_chain_ids。BFS 收集 compression 链所有节点。"""
        chain = set()
        node = self.nodes.get(session_id)
        if not node:
            return chain
        stack = [node]
        while stack:
            n = stack.pop()
            if n.id in chain:
                continue
            chain.add(n.id)
            for c in n.children:
                if c.is_compressed or c.is_active:
                    stack.append(c)
        return chain

    def get_resume_candidates(
        self,
        current_sid: str,
        source: str = None,
        min_messages: int = 2,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """统一的候选列表构建逻辑，替代 _handle_resume_command 中重复的两份代码。

        返回 list of dict，每个 dict 包含 id, title, message_count, preview, last_active,
        end_reason 等字段（兼容现有的 candidates 格式）。

        注意：preview 和 last_active 需要 DB 查询，这里只返回节点基础信息，
        调用方需额外补充这些字段（或修改 list_sessions_rich 已有的数据）。
        """
        ancestor_ids = self.get_ancestor_ids(current_sid)
        candidates = []
        seen_ids = set()

        # 按时间倒序遍历所有节点
        all_nodes = sorted(self.nodes.values(), key=lambda n: n.started_at, reverse=True)

        for node in all_nodes:
            if node.id == current_sid:
                continue
            if node.id in ancestor_ids:
                continue
            if node.id in seen_ids:
                continue
            if node.message_count < min_messages:
                continue
            if source and node.source != source:
                continue

            display_node = node
            if node.is_compressed:
                chain_ids = self.get_compression_chain_ids(node.id)
                seen_ids.update(chain_ids)
                leaf = self.find_compression_leaf(node.id)
                if leaf and leaf.id != node.id:
                    if leaf.id == current_sid or leaf.id in seen_ids:
                        continue
                    display_node = leaf
                    seen_ids.add(leaf.id)

            seen_ids.add(node.id)
            candidates.append({
                "id": display_node.id,
                "title": display_node.title,
                "message_count": display_node.message_count,
                "end_reason": display_node.end_reason,
                "source": display_node.source,
                # preview 和 last_active 由调用方补充
                "preview": "",
                "last_active": None,
                # 保存原始节点引用，方便调用方获取原始节点的 preview
                "_original_node": node if display_node is not node else None,
            })
            if len(candidates) >= limit:
                break

        return candidates
```

#### 在 `SessionDB` 类内部新增方法（在 `get_ancestor_ids` 方法附近）

```python
def build_session_tree(self, source: str = None) -> SessionTree:
    """一次 SQL 查询构建完整 session 树。

    返回 SessionTree 对象，支持内存中的祖先链遍历、压缩链追踪、
    候选列表构建等操作，无需额外 DB 查询。
    """
    with self._lock:
        if source:
            cursor = self._conn.execute(
                "SELECT id, source, title, parent_session_id, end_reason, "
                "message_count, started_at FROM sessions WHERE source = ? "
                "ORDER BY started_at",
                (source,)
            )
        else:
            cursor = self._conn.execute(
                "SELECT id, source, title, parent_session_id, end_reason, "
                "message_count, started_at FROM sessions ORDER BY started_at"
            )
        rows = cursor.fetchall()

    # 构建节点字典
    nodes: Dict[str, SessionNode] = {}
    for row in rows:
        node = SessionNode(
            id=row["id"],
            parent_id=row["parent_session_id"],
            end_reason=row["end_reason"],
            message_count=row["message_count"] or 0,
            started_at=row["started_at"],
            title=row["title"],
            source=row["source"],
        )
        nodes[node.id] = node

    # 连接父子关系并收集根节点
    roots: List[SessionNode] = []
    for node in nodes.values():
        if node.parent_id and node.parent_id in nodes:
            nodes[node.parent_id].children.append(node)
        else:
            roots.append(node)

    return SessionTree(nodes, roots)
```

### 2. `gateway/run.py` — 重构 `_handle_resume_command`

用 `build_session_tree` + `SessionTree.get_resume_candidates` 替代当前两份重复的候选过滤逻辑。

#### 主要改动

1. 在方法开头调用 `self._session_db.build_session_tree(source=user_source)` 一次性构建树
2. 用 `tree.get_resume_candidates()` 统一获取候选列表
3. 用 `tree.find_compression_leaf()` 替代 `self._session_db._find_latest_leaf()`
4. 保留从 `list_sessions_rich` 获取的 preview/last_active 数据补充到候选中

#### 具体实现

需要把当前 `_handle_resume_command` 方法（约 6474-6752 行）重构为以下结构：

```python
async def _handle_resume_command(self, event: MessageEvent) -> str:
    """Handle /resume command — switch to a previously-named session."""
    if not self._session_db:
        return "Session database not available."

    source = event.source
    session_key = self._session_key_for_source(source)
    name = event.get_command_args().strip()
    user_source = source.platform.value if source.platform else None

    # ── 一次构建 session 树 ──
    tree = self._session_db.build_session_tree(source=user_source)
    current_entry = self.session_store.get_or_create_session(source)
    current_sid = current_entry.session_id

    # 获取配置
    resume_min_messages = 2
    try:
        if isinstance(self.config, dict):
            resume_min_messages = self.config.get("resume_min_messages", 2)
        else:
            resume_min_messages = getattr(self.config, "resume_min_messages", 2)
    except Exception:
        pass

    # ── 路由: /resume last | /resume <name/number> | /resume (列表) ──

    if name in ("last", "-"):
        # 取最近的 1 个候选
        candidates = tree.get_resume_candidates(
            current_sid, source=user_source,
            min_messages=resume_min_messages, limit=1
        )
        # 需要补充 preview/last_active，从 list_sessions_rich 取
        if candidates:
            candidates = self._enrich_candidates_with_display_info(candidates, tree)
            target_id = candidates[0].get("id")
        else:
            name = ""  # 降级到列表
    elif not name:
        # 列表模式
        candidates = tree.get_resume_candidates(
            current_sid, source=user_source,
            min_messages=resume_min_messages, limit=10
        )
        if not candidates:
            return (
                f"No resumable sessions found (need ≥ {resume_min_messages} messages).\n"
                "Use `/title My Session` to name your current session, "
                "then `/resume My Session` to return to it later."
            )
        candidates = self._enrich_candidates_with_display_info(candidates, tree)

        # 格式化显示
        lines = ["📋 **Recent Sessions**\n"]
        from datetime import datetime, timezone as dt_timezone
        now = datetime.now(dt_timezone.utc)

        for i, s in enumerate(candidates, 1):
            title = s.get("title") or "untitled"
            msg_count = s.get("message_count", 0)
            preview = s.get("preview", "")[:40]
            preview_part = f" — _{preview}_" if preview else ""
            time_part = ""
            last_active = s.get("last_active")
            if last_active:
                try:
                    if isinstance(last_active, str):
                        last_active_dt = datetime.fromisoformat(last_active.replace("Z", "+00:00"))
                    else:
                        last_active_dt = last_active
                    if last_active_dt.tzinfo is None:
                        last_active_dt = last_active_dt.replace(tzinfo=dt_timezone.utc)
                    delta = now - last_active_dt
                    days = delta.days
                    hours = delta.total_seconds() / 3600
                    minutes = delta.total_seconds() / 60
                    if days > 0:
                        time_part = f" • {days}d ago"
                    elif hours >= 1:
                        time_part = f" • {int(hours)}h ago"
                    elif minutes >= 1:
                        time_part = f" • {int(minutes)}m ago"
                    else:
                        time_part = " • just now"
                except Exception:
                    pass
            lines.append(f"`{i}`. **{title}** ({msg_count} msgs){time_part}{preview_part}")
        lines.append("\nUsage: `/resume <number or session name>`")

        if not hasattr(self, "_resume_candidates_map"):
            self._resume_candidates_map = {}
        self._resume_candidates_map[session_key] = candidates
        return "\n".join(lines)
    else:
        # 按名称或数字查找
        target_id = None
        if name.isdigit():
            idx = int(name)
            candidates_map = getattr(self, "_resume_candidates_map", {})
            candidates = candidates_map.get(session_key) if candidates_map else None
            if candidates and 1 <= idx <= len(candidates):
                target_id = candidates[idx - 1].get("id")
        if not target_id:
            target_id = self._session_db.resolve_session_by_title(name)
        if not target_id:
            return (
                f"No session found matching '**{name}**'.\n"
                "Use `/resume` with no arguments to see available sessions."
            )

        # 用树解析压缩叶子（0 次 DB 查询）
        leaf = tree.find_compression_leaf(target_id)
        if leaf and leaf.id != target_id:
            target_id = leaf.id

    # ── 执行 session 切换 ──

    # 检查是否已在目标 session
    if current_entry.session_id == target_id:
        return f"📌 Already on session **{name}**."

    # 后台 flush 记忆
    try:
        _flush_task = asyncio.create_task(
            self._async_flush_memories(current_entry.session_id, session_key)
        )
        self._background_tasks.add(_flush_task)
        _flush_task.add_done_callback(self._background_tasks.discard)
    except Exception as e:
        logger.debug("Memory flush on resume failed: %s", e)

    # 清理当前 agent
    if session_key in self._running_agents:
        del self._running_agents[session_key]
    _cache_lock = getattr(self, "_agent_cache_lock", None)
    if _cache_lock is not None:
        with _cache_lock:
            _cached = self._agent_cache.get(session_key)
            _old_agent = _cached[0] if isinstance(_cached, tuple) else _cached if _cached else None
    if _old_agent is not None:
        self._cleanup_agent_resources(_old_agent)
    self._evict_cached_agent(session_key)

    # 切换 session
    new_entry = self.session_store.switch_session(session_key, target_id)
    if not new_entry:
        return "Failed to switch session."

    title = self._session_db.get_session_title(target_id) or name
    msg_count = 0
    # 尝试从 tree 获取 message_count
    target_node = tree.nodes.get(target_id)
    if target_node:
        msg_count = target_node.message_count
    msg_part = f" ({msg_count} message{'s' if msg_count != 1 else ''})" if msg_count else ""

    return f"↻ Resumed session **{title}**{msg_part}. Conversation restored."
```

#### 新增辅助方法 `_enrich_candidates_with_display_info`

为候选补充 preview 和 last_active（仍需从 list_sessions_rich 获取）：

```python
def _enrich_candidates_with_display_info(
    self, candidates: List[Dict], tree: 'SessionTree'
) -> List[Dict]:
    """用 list_sessions_rich 的数据补充候选的 preview 和 last_active 字段。"""
    try:
        user_source = candidates[0].get("source") if candidates else None
        rich_sessions = self._session_db.list_sessions_rich(
            source=user_source, limit=100, include_children=True
        )
        rich_map = {s["id"]: s for s in rich_sessions}

        for c in candidates:
            cid = c.get("id")
            if cid in rich_map:
                rich = rich_map[cid]
                c["preview"] = rich.get("preview", "")
                c["last_active"] = rich.get("last_active")
            # 如果候选是压缩叶子但 preview 来自原始节点
            orig = c.get("_original_node")
            if orig and not c.get("preview") and orig.id in rich_map:
                c["preview"] = rich_map[orig.id].get("preview", "")
                c["last_active"] = rich_map[orig.id].get("last_active")
    except Exception:
        pass
    return candidates
```

## 注意事项

1. **保持向后兼容**：`_find_latest_leaf`、`_get_compression_chain_ids`、`get_ancestor_ids` 等旧方法保留不动（其他地方可能还在用），新增 `build_session_tree` 作为推荐接口
2. **线程安全**：`build_session_tree` 在 `_lock` 下读取数据，返回后树是纯内存不可变结构，无线程安全问题
3. **性能**：一次 SQL 查询替代 N+1 查询，树构建 O(N)，遍历 O(N)
4. **不要修改 `hermes_state.py` 的 `_init_schema` 或表结构** — 这是纯代码重构，无 schema 变更
5. **保留 `cli.py` 中的 `_handle_resume_command` 不动** — CLI 的 resume 逻辑不同（直接加载 transcript），不需要用树

## 验证

修改后运行现有测试：
```bash
cd /home/ubunutu/.hermes/hermes-agent
python -m pytest tests/ -x -q --tb=short 2>&1 | tail -20
```

手动验证 /resume 功能在飞书上正常工作。
