"""
Autolycus P6: ContextWriter — file-based LLM контекст через raw-wiki.

Сохраняет каждый turn в raw-wiki файл. В памяти — только active_window
последних 10 turn'ов. Старые turn'ы доступны через rg.

Зависимости: findings_to_wiki (уже установлен), rg (уже установлен).
Расширяет существующий findings_to_wiki MemoryProvider.
"""
from __future__ import annotations
import json, logging, os, subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _get_wiki_dir() -> Path:
    """Определить директорию wiki."""
    # Используем ту же, что и findings_to_wiki
    from hermes_constants import get_hermes_home
    wiki = get_hermes_home() / "wiki"
    wiki.mkdir(parents=True, exist_ok=True)
    return wiki


def _format_turn(turn_number: int, user_msg: str, assistant_msg: str,
                 tools: list[dict] | None = None) -> str:
    """Форматировать один turn для записи в файл."""
    lines = [
        f"## Turn {turn_number}",
        f"**Time:** {datetime.now().isoformat()[:19]}",
        "",
        "### User",
        user_msg,
        "",
        "### Assistant",
        assistant_msg,
    ]
    if tools:
        lines.extend(["", "### Tools"])
        for t in tools[-5:]:  # последние 5 tool calls
            name = t.get("name", "?")
            result = str(t.get("result", ""))[:200]
            lines.append(f"- `{name}`: {result}")
    return "\n".join(lines)


class ContextWriter:
    """Пишет turn'ы в raw-wiki, управляет active_window в памяти.
    
    Использование:
        cw = ContextWriter()
        cw.sync_turn(session_id="abc", turn=42, 
                     user_msg="hello", assistant_msg="world")
        
        # Active window (последние 10 turn'ов)
        context = cw.get_active_context(session_id)
        
        # Поиск по всей истории
        results = cw.search_context(session_id, "nginx")
    """
    
    def __init__(self, wiki_dir: str | Path | None = None,
                 window_size: int = 10):
        self.wiki_dir = Path(wiki_dir) if wiki_dir else _get_wiki_dir()
        self.window_size = window_size
        self._active_windows: dict[str, list[int]] = {}  # session_id -> [turn_numbers]
        logger.info("[ContextWriter] initialized: wiki=%s, window=%d turns",
                    self.wiki_dir, window_size)
    
    @property
    def _context_dir(self) -> Path:
        """Директория для контекстных файлов."""
        d = self.wiki_dir / "raw" / "context"
        d.mkdir(parents=True, exist_ok=True)
        return d
    
    def _session_dir(self, session_id: str) -> Path:
        """Директория для конкретной сессии."""
        d = self._context_dir / session_id
        d.mkdir(parents=True, exist_ok=True)
        return d
    
    def sync_turn(self, session_id: str, turn_number: int,
                  user_msg: str, assistant_msg: str,
                  tools: list[dict] | None = None,
                  metadata: dict | None = None) -> None:
        """Записать turn в файл и обновить active_window.
        
        Args:
            session_id: ID сессии
            turn_number: Номер turn'а (0, 1, 2...)
            user_msg: Сообщение пользователя
            assistant_msg: Ответ ассистента
            tools: Список tool call'ов (опционально)
            metadata: Дополнительные метаданные (опционально)
        """
        content = _format_turn(turn_number, user_msg, assistant_msg, tools)
        
        if metadata:
            meta_str = json.dumps(metadata, ensure_ascii=False)
            content += f"\n\n<!-- metadata: {meta_str} -->\n"
        
        # Запись в файл (stdlib, ~0.1ms)
        turn_file = self._session_dir(session_id) / f"turn_{turn_number:04d}.md"
        turn_file.write_text(content)
        
        # Обновление active_window
        if session_id not in self._active_windows:
            self._active_windows[session_id] = []
        window = self._active_windows[session_id]
        window.append(turn_number)
        while len(window) > self.window_size:
            window.pop(0)  # Удаляем самый старый
        
        logger.debug("[ContextWriter] turn %d written to %s (window: %d/%d)",
                     turn_number, turn_file, len(window), self.window_size)
    
    def get_active_context(self, session_id: str) -> list[str]:
        """Вернуть active_window контекста (последние N turn'ов).
        
        Args:
            session_id: ID сессии
            
        Returns:
            Список строк — содержимое turn'ов active_window
        """
        window = self._active_windows.get(session_id, [])
        session_dir = self._session_dir(session_id)
        
        result = []
        for turn_num in window[-self.window_size:]:
            turn_file = session_dir / f"turn_{turn_num:04d}.md"
            if turn_file.exists():
                result.append(turn_file.read_text())
        
        return result
    
    def search_context(self, session_id: str, query: str,
                       max_results: int = 5) -> list[dict]:
        """Поиск по всему контексту сессии через rg.
        
        Args:
            session_id: ID сессии
            query: Поисковый запрос (rg pattern)
            max_results: Максимум результатов
            
        Returns:
            Список dict'ов с path, preview, turn_number
        """
        session_dir = self._session_dir(session_id)
        if not session_dir.exists():
            return []
        
        result = subprocess.run(
            ["rg", "-l", query, str(session_dir)],
            capture_output=True, text=True, timeout=10
        )
        
        if not result.stdout.strip():
            return []
        
        files = result.stdout.strip().split("\n")[:max_results]
        results = []
        for f in files:
            try:
                turn_num = int(Path(f).stem.split("_")[1])
                preview = Path(f).read_text()[:200]
                results.append({
                    "turn": turn_num,
                    "path": f,
                    "preview": preview,
                })
            except (IndexError, ValueError):
                continue
        
        return results
    
    def get_summary(self, session_id: str) -> dict:
        """Статистика по сессии: сколько turn'ов, диапазон дат, размер."""
        session_dir = self._session_dir(session_id)
        if not session_dir.exists():
            return {"turns": 0, "files": 0, "size_kb": 0}
        
        files = sorted(session_dir.glob("turn_*.md"))
        total_size = sum(f.stat().st_size for f in files)
        
        return {
            "turns": len(self._active_windows.get(session_id, [])),
            "files": len(files),
            "archived_turns": len(files) - len(self._active_windows.get(session_id, [])),
            "size_kb": total_size // 1024,
        }
