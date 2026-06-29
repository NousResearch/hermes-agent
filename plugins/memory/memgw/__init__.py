"""Memory Gateway memory plugin — MemoryProvider interface.

Connects Hermes to Dan's self-hosted Memory Gateway (Neo4j + Qdrant + Notion)
over its Streamable-HTTP MCP endpoint. Unlike sealed memory backends, the
gateway can fuse semantic + keyword + graph retrieval and ground answers in a
knowledge graph and Obsidian vault.

Tools exposed to the model:
  memgw_recall   — hybrid recall (semantic + keyword + graph fusion)
  memgw_retain   — store a durable memory
  memgw_reflect  — synthesize/return mental models (durable beliefs)

Auto behaviour:
  prefetch()      — background recall injected before each turn
  sync_turn()     — store completed turns (non-blocking)
  on_delegation() — record a subagent task+result as an Experience
  on_session_end()— store a session summary

Config via environment variables:
  MEMGW_API_URL  — gateway MCP URL (default: https://mcp.danizhaky.com/mcp)
  MEMGW_API_KEY  — bearer key for the gateway (required for cloud mode)

Or via $HERMES_HOME/memgw.json.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120
_DEFAULT_URL = 'https://mcp.danizhaky.com/mcp'


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config from env vars, with $HERMES_HOME/memgw.json overrides."""
    from hermes_constants import get_hermes_home

    config = {
        'api_url': os.environ.get('MEMGW_API_URL', _DEFAULT_URL),
        'api_key': os.environ.get('MEMGW_API_KEY', ''),
        'recall_limit': 5,
        'prefetch_method': 'recall',
    }
    config_path = get_hermes_home() / 'memgw.json'
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding='utf-8'))
            config.update({k: v for k, v in file_cfg.items() if v is not None and v != ''})
        except Exception:
            pass
    return config


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

RECALL_SCHEMA = {
    'name': 'memgw_recall',
    'description': (
        'Recall from long-term memory using hybrid retrieval — fuses semantic, '
        'exact-term (entity/ID), and knowledge-graph strategies. Use whenever '
        'context about the user, a project, a person, or a past decision would '
        'help. Especially strong for specific names, IDs, and linked context.'
    ),
    'parameters': {
        'type': 'object',
        'properties': {
            'query': {'type': 'string', 'description': 'What to recall.'},
            'limit': {'type': 'integer', 'description': 'Max results (default 5).'},
        },
        'required': ['query'],
    },
}

RETAIN_SCHEMA = {
    'name': 'memgw_retain',
    'description': (
        'Store a durable memory (fact, decision, preference, or learning). Use '
        'proactively when the user states something worth remembering across '
        'sessions. No permission needed.'
    ),
    'parameters': {
        'type': 'object',
        'properties': {
            'content': {'type': 'string', 'description': 'The memory content.'},
            'title': {'type': 'string', 'description': 'Short label.'},
            'memory_type': {
                'type': 'string',
                'description': "e.g. 'memory', 'decision', 'preference', 'fact', 'experience'.",
            },
        },
        'required': ['content', 'title'],
    },
}

REFLECT_SCHEMA = {
    'name': 'memgw_reflect',
    'description': (
        'Reflect on stored memories to surface durable beliefs / mental models '
        'about a topic — patterns synthesized across many raw memories. Use for '
        'higher-order questions ("what do we know about X", risk/opportunity).'
    ),
    'parameters': {
        'type': 'object',
        'properties': {
            'query': {'type': 'string', 'description': 'The topic to reflect on.'},
        },
        'required': ['query'],
    },
}


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class MemGatewayProvider(MemoryProvider):
    """Memory Gateway provider — hybrid recall, retain, reflect over MCP."""

    def __init__(self):
        self._config: dict | None = None
        self._client = None
        self._client_lock = threading.Lock()
        self._api_url = _DEFAULT_URL
        self._api_key = ''
        self._recall_limit = 5
        self._prefetch_method = 'recall'
        self._user_id = ''
        self._prefetch_result = ''
        self._prefetch_result_user: str = ''
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: threading.Thread | None = None
        # Monotonic generation: only the latest queued prefetch may store its
        # result, so a slow older worker can't overwrite a newer one.
        self._prefetch_gen = 0
        self._sync_thread: threading.Thread | None = None
        self._sync_threads: list[threading.Thread] = []
        self._sync_lock = threading.Lock()
        self._delegation_threads: list[threading.Thread] = []
        self._delegation_lock = threading.Lock()
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    @property
    def name(self) -> str:
        return 'memgw'

    def is_available(self) -> bool:
        import importlib.util
        if importlib.util.find_spec('mcp') is None:
            return False
        cfg = _load_config()
        url = cfg.get('api_url', '')
        if cfg.get('api_key'):
            return True
        # Keyless mode only for a true loopback *host* — a substring match on
        # 'localhost' would trust URLs like 'https://localhost@example.com/mcp'
        # whose hostname is example.com, leaking memory to a non-local endpoint
        # (Codex PR #30 review). Parse the URL and require an exact loopback host.
        from urllib.parse import urlparse

        try:
            host = (urlparse(url).hostname or '').lower()
        except ValueError:
            # Malformed URL (e.g. typoed IPv6 like "http://[::1:8081/mcp") —
            # treat as unavailable rather than propagating the exception.
            return False
        return host in ('localhost', '127.0.0.1', '::1')

    def get_config_schema(self):
        return [
            {
                'key': 'api_url',
                'description': 'Memory Gateway MCP URL',
                'default': _DEFAULT_URL,
                'env_var': 'MEMGW_API_URL',
            },
            {
                'key': 'api_key',
                'description': 'Gateway bearer key (required for cloud mode)',
                'secret': True,
                'required': False,
                'env_var': 'MEMGW_API_KEY',
            },
            {
                'key': 'prefetch_method',
                'description': 'Auto-recall method before each turn',
                'default': 'recall',
                'choices': ['recall', 'reflect', 'off'],
            },
        ]

    def save_config(self, values, hermes_home):
        from pathlib import Path

        config_path = Path(hermes_home) / 'memgw.json'
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(json.dumps(existing, indent=2))

    # -- lifecycle -----------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._api_url = self._config.get('api_url', _DEFAULT_URL)
        self._api_key = self._config.get('api_key', '')
        self._recall_limit = int(self._config.get('recall_limit', 5))
        self._prefetch_method = self._config.get('prefetch_method', 'recall')
        self._user_id = kwargs.get('user_id') or ''

    def _get_client(self):
        with self._client_lock:
            if self._client is not None:
                return self._client
            from .client import MemGatewayClient

            self._client = MemGatewayClient(self._api_url, self._api_key)
            return self._client

    # -- circuit breaker -----------------------------------------------------

    def _is_breaker_open(self) -> bool:
        if self._consecutive_failures < _BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._breaker_open_until:
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self) -> None:
        self._consecutive_failures = 0

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
            logger.warning(
                'Memory Gateway circuit breaker tripped after %d failures; '
                'pausing for %ds.',
                self._consecutive_failures,
                _BREAKER_COOLDOWN_SECS,
            )

    # -- prompt + recall -----------------------------------------------------

    def system_prompt_block(self) -> str:
        return (
            '# Memory Gateway\n'
            'Active (Neo4j + Qdrant hybrid memory).\n'
            'Use memgw_recall to find memories, memgw_retain to store facts, '
            'memgw_reflect for synthesized beliefs about a topic.'
        )

    @staticmethod
    def _format_recall(payload: dict) -> str:
        results = payload.get('results') or []
        lines = []
        for r in results:
            p = r.get('payload', r)
            title = p.get('title', '')
            content = p.get('content', '')
            snippet = f'{title}: {content}'.strip(': ').strip()
            if snippet:
                lines.append(f'- {snippet}')
        return '\n'.join(lines)

    def prefetch(self, query: str, *, session_id: str = '', user_id: str = '') -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            # Discard a result queued for a different user to prevent cross-user leak.
            if user_id and self._prefetch_result_user and self._prefetch_result_user != user_id:
                self._prefetch_result = ''
                self._prefetch_result_user = ''
            result = self._prefetch_result
            self._prefetch_result = ''
            self._prefetch_result_user = ''
        if not result:
            return ''
        return f'## Memory Gateway\n{result}'

    def on_session_switch(self, new_session_id: str, **kwargs) -> None:
        # Invalidate any in-flight prefetch from the previous session and drop
        # its cached result, so the new session can't be fed stale context.
        with self._prefetch_lock:
            self._prefetch_result = ''
            self._prefetch_result_user = ''
            self._prefetch_gen += 1

    def queue_prefetch(self, query: str, *, session_id: str = '', user_id: str = '') -> None:
        if self._is_breaker_open() or self._prefetch_method == 'off' or not query:
            return

        scope = self._user_scope(user_id)

        with self._prefetch_lock:
            self._prefetch_gen += 1
            my_gen = self._prefetch_gen

        def _run():
            try:
                client = self._get_client()
                if self._prefetch_method == 'reflect':
                    payload = client.call_tool('reflect', {'query': query, **scope})
                    text = self._format_reflect(payload)
                else:
                    payload = client.call_tool(
                        'recall', {'query': query, 'limit': self._recall_limit, **scope}
                    )
                    text = self._format_recall(payload)
                if text:
                    # Only store if no newer prefetch (or session switch) has
                    # superseded this worker — prevents a slow older query from
                    # clobbering a newer result.
                    with self._prefetch_lock:
                        if my_gen == self._prefetch_gen:
                            self._prefetch_result = text
                            self._prefetch_result_user = user_id
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug('memgw prefetch failed: %s', e)

        self._prefetch_thread = threading.Thread(
            target=_run, daemon=True, name='memgw-prefetch'
        )
        self._prefetch_thread.start()

    @staticmethod
    def _format_reflect(payload: dict) -> str:
        models = payload.get('mental_models') or []
        lines = []
        for m in models:
            stmt = m.get('statement', '')
            conf = m.get('confidence', '')
            if stmt:
                lines.append(f'- {stmt} (confidence {conf})')
        return '\n'.join(lines)

    # -- user scoping --------------------------------------------------------

    def _user_scope(self, user_id: str = '') -> dict:
        """Return user scoping metadata for multi-user gateway sessions."""
        uid = user_id or self._user_id
        if uid:
            return {'user_id': uid}
        return {}

    # -- writes --------------------------------------------------------------

    def _retain(self, content: str, title: str, memory_type: str = 'memory', *, user_id: str = '') -> dict:
        client = self._get_client()
        return client.call_tool(
            'write', {'content': content, 'title': title, 'memory_type': memory_type, **self._user_scope(user_id)}
        )

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = '', user_id: str = '') -> None:
        if self._is_breaker_open() or not user_content.strip():
            return

        scope = self._user_scope(user_id)

        def _sync():
            try:
                title = user_content.strip().splitlines()[0][:120]
                content = f'User: {user_content}\n\nAssistant: {assistant_content}'
                client = self._get_client()
                client.call_tool(
                    'write',
                    {'content': content[:4000], 'title': title or 'conversation turn', 'memory_type': 'memory', **scope},
                )
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug('memgw sync_turn failed: %s', e)

        # Track every writer so a still-in-flight write isn't dropped when the
        # next turn starts one; shutdown() joins all of them (not just the last).
        t = threading.Thread(target=_sync, daemon=True, name='memgw-sync')
        with self._sync_lock:
            self._sync_threads = [st for st in self._sync_threads if st.is_alive()]
            self._sync_threads.append(t)
        self._sync_thread = t
        t.start()

    def on_delegation(self, task: str, result: str, *, child_session_id: str = '', **kwargs) -> None:
        """Record a subagent task+result as a retrievable Experience."""
        if self._is_breaker_open() or not task.strip():
            return

        def _record():
            try:
                content = f'Task: {task}\n\nOutcome: {result}'
                self._retain(content[:4000], f'Experience: {task[:100]}', 'experience')
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug('memgw on_delegation failed: %s', e)

        t = threading.Thread(target=_record, daemon=True, name='memgw-exp')
        with self._delegation_lock:
            self._delegation_threads = [dt for dt in self._delegation_threads if dt.is_alive()]
            self._delegation_threads.append(t)
        t.start()

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Store a lightweight end-of-session summary."""
        if self._is_breaker_open() or not messages:
            return
        try:
            user_msgs = [
                m.get('content', '')
                for m in messages
                if isinstance(m, dict) and m.get('role') == 'user'
            ]
            if not user_msgs:
                return
            summary = ' | '.join(str(u)[:200] for u in user_msgs[-5:])
            self._retain(summary[:4000], 'Session summary', 'memory')
        except Exception as e:
            logger.debug('memgw on_session_end failed: %s', e)

    # -- model-facing tools --------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [RECALL_SCHEMA, RETAIN_SCHEMA, REFLECT_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._is_breaker_open():
            return json.dumps(
                {'error': 'Memory Gateway temporarily unavailable; will retry automatically.'}
            )
        try:
            client = self._get_client()
        except Exception as e:
            return tool_error(str(e))

        call_user_id = kwargs.get('user_id', '')
        try:
            if tool_name == 'memgw_recall':
                query = args.get('query', '')
                if not query:
                    return tool_error('Missing required parameter: query')
                limit = min(int(args.get('limit', self._recall_limit)), 50)
                payload = client.call_tool('recall', {'query': query, 'limit': limit, **self._user_scope(call_user_id)})
                self._record_success()
                return json.dumps(payload)

            if tool_name == 'memgw_retain':
                content = args.get('content', '')
                title = args.get('title', '')
                if not content or not title:
                    return tool_error('Missing required parameters: content, title')
                mtype = args.get('memory_type', 'memory')
                payload = self._retain(content, title, mtype, user_id=call_user_id)
                self._record_success()
                return json.dumps(payload)

            if tool_name == 'memgw_reflect':
                query = args.get('query', '')
                if not query:
                    return tool_error('Missing required parameter: query')
                payload = client.call_tool('reflect', {'query': query, **self._user_scope(call_user_id)})
                self._record_success()
                return json.dumps(payload)
        except Exception as e:
            self._record_failure()
            return tool_error(f'{tool_name} failed: {e}')

        return tool_error(f'Unknown tool: {tool_name}')

    def shutdown(self) -> None:
        with self._delegation_lock:
            pending = list(self._delegation_threads)
        with self._sync_lock:
            sync_writers = list(self._sync_threads)
        # Join every tracked writer/worker so no in-flight retain is dropped
        # when the client closes.
        for t in (self._prefetch_thread, *sync_writers, *pending):
            if t and t.is_alive():
                t.join(timeout=5.0)
        with self._client_lock:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None


def register(ctx) -> None:
    """Register the Memory Gateway as a memory provider plugin."""
    ctx.register_memory_provider(MemGatewayProvider())
