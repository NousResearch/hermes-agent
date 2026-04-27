from __future__ import annotations

import json
import re
import threading
import time
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.multimodal_recall_tool import _local_mmrag_connected
from tools.recall_with_artifacts_tool import recall_with_artifacts


def _local_mmrag_provider_ready() -> bool:
    try:
        return bool(_local_mmrag_connected())
    except Exception:
        return False


def _normalize_prefetch_query(query: str) -> str:
    return re.sub(r'\s+', ' ', (query or '').strip().lower())


def _multimodal_query_score(query: str) -> int:
    q = _normalize_prefetch_query(query)
    strong_terms = [
        'screenshot', 'pdf', 'attachment', 'image', 'ocr', 'evidence', 'artifact',
        '截圖', '附件', '圖片', '證據',
    ]
    weak_terms = [
        'report', 'slide', 'document', 'file', 'note', 'scan',
        '報告', '文件', '紀錄', '掃描',
    ]
    score = 0
    for term in strong_terms:
        if term in q:
            score += 3
    for term in weak_terms:
        if term in q:
            score += 1
    return score


def _recent_multimodal_signal_score(*signals: str) -> int:
    joined = ' '.join((s or '') for s in signals)
    return 2 if _multimodal_query_score(joined) >= 3 else 0


def _should_prefetch_multimodal(query: str, *signals: str) -> bool:
    query_score = _multimodal_query_score(query)
    if query_score >= 3:
        return True
    return (query_score + _recent_multimodal_signal_score(*signals)) >= 3


def _format_prefetch_result(raw: str) -> str:
    try:
        parsed = json.loads(raw)
    except Exception:
        return ''
    summary = (parsed.get('combined_summary') or '').strip()
    top = (((parsed.get('artifact_recall') or {}).get('top_evidence') or [])[:1])
    lines = []
    if summary:
        lines.append(summary)
    for item in top:
        path = item.get('source_path') or ''
        source_ref = item.get('source_ref') or ''
        collection_name = item.get('collection_name') or ''
        if path:
            lines.append(f'source: {path}')
        if source_ref:
            lines.append(f'source_ref: {source_ref}')
        if collection_name:
            lines.append(f'collection: {collection_name}')
    out = '\n'.join(lines).strip()
    return out[:780]


def _extract_session_end_signal(messages: List[Dict[str, Any]]) -> str:
    snippets = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = (msg.get('content') or '').strip()
        if not content:
            continue
        if _should_prefetch_multimodal(content):
            snippets.append(' '.join(content.split()))
        if len(snippets) >= 2:
            break
    if not snippets:
        return ''
    joined = ' | '.join(snippets)
    return joined[:380]


class MultimodalRecallMemoryProvider(MemoryProvider):
    def _recent_signal_text(self) -> str:
        return ' | '.join(
            s for s in [
                getattr(self, '_turn_signal', ''),
                getattr(self, '_session_end_signal', ''),
                getattr(self, '_memory_write_signal', ''),
            ]
            if s
        )

    def _should_prefetch(self, query: str) -> bool:
        return _should_prefetch_multimodal(query, self._recent_signal_text())

    def _fresh_recall_allowed(self, query: str, *, session_id: str = '') -> bool:
        normalized = _normalize_prefetch_query(query)
        key = (session_id or self._session_id, normalized)
        now = time.monotonic()
        last_key = getattr(self, '_last_prefetch_key', None)
        last_ts = getattr(self, '_last_prefetch_ts', 0.0)
        cooldown = getattr(self, '_prefetch_cooldown_seconds', 30.0)
        if last_key == key and (now - last_ts) < cooldown:
            return False
        self._last_prefetch_key = key
        self._last_prefetch_ts = now
        return True

    @property
    def name(self) -> str:
        return 'multimodal-recall'

    def is_available(self) -> bool:
        return _local_mmrag_provider_ready()

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._init_kwargs = kwargs
        self._prefetch_cache = {}
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        self._session_end_signal = ''
        self._turn_signal = ''
        self._memory_write_signal = ''
        self._last_prefetch_key = None
        self._last_prefetch_ts = 0.0
        self._prefetch_cooldown_seconds = 30.0

    def system_prompt_block(self) -> str:
        return 'An optional multimodal memory provider may prefetch compact artifact context when screenshots, PDFs, OCR, or evidence from prior work are relevant.'

    def prefetch(self, query: str, *, session_id: str = '') -> str:
        if not self._should_prefetch(query):
            return ''
        _cache_key = (session_id or self._session_id, _normalize_prefetch_query(query))
        if hasattr(self, '_prefetch_cache'):
            with self._prefetch_lock:
                cached = self._prefetch_cache.pop(_cache_key, '')
        else:
            cached = ''
        if cached:
            return cached
        if not _local_mmrag_provider_ready():
            return ''
        if not self._fresh_recall_allowed(query, session_id=session_id):
            return ''
        try:
            raw = recall_with_artifacts(query=query, session_limit=2, artifact_top_k=2)
        except Exception:
            return ''
        return _format_prefetch_result(raw)

    def queue_prefetch(self, query: str, *, session_id: str = '') -> None:
        if not self._should_prefetch(query):
            return
        if not _local_mmrag_provider_ready():
            return
        _cache_key = (session_id or self._session_id, _normalize_prefetch_query(query))

        def _run() -> None:
            try:
                raw = recall_with_artifacts(query=query, session_limit=2, artifact_top_k=2)
                formatted = _format_prefetch_result(raw)
                if not formatted:
                    return
                with self._prefetch_lock:
                    self._prefetch_cache[_cache_key] = formatted
            except Exception:
                return

        self._prefetch_thread = threading.Thread(
            target=_run,
            daemon=True,
            name='multimodal-recall-prefetch',
        )
        self._prefetch_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = '') -> None:
        messages = [
            {'role': 'user', 'content': user_content},
            {'role': 'assistant', 'content': assistant_content},
        ]
        self._turn_signal = _extract_session_end_signal(messages)

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        self._session_end_signal = _extract_session_end_signal(messages)

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        messages = [
            {'role': 'memory-write', 'content': content},
        ]
        self._memory_write_signal = _extract_session_end_signal(messages)

    def shutdown(self) -> None:
        thread = getattr(self, '_prefetch_thread', None)
        if thread and thread.is_alive():
            thread.join(timeout=5.0)
        self._prefetch_thread = None

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        return json.dumps({'error': f'{self.name} exposes no tools'})


def register(ctx) -> None:
    ctx.register_memory_provider(MultimodalRecallMemoryProvider())
