from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider

GBRAIN_BIN = "/Users/fortune/.bun/bin/gbrain"
DEFAULT_QMD_LIMIT = 4
DEFAULT_GBRAIN_LIMIT = 4
DEFAULT_TIMEOUT = 8.0


class QmdGbrainMemoryProvider(MemoryProvider):
    def __init__(self) -> None:
        self._session_id = ""
        self._profile_name = ""
        self._hermes_home = Path.home() / ".hermes"
        self._backend_home = self._hermes_home
        self._prefetch_lock = threading.Lock()
        self._prefetch_result = ""
        self._prefetch_thread: threading.Thread | None = None
        self._last_prefetch_query = ""
        self._qmd_limit = DEFAULT_QMD_LIMIT
        self._gbrain_limit = DEFAULT_GBRAIN_LIMIT
        self._timeout = DEFAULT_TIMEOUT

    @property
    def name(self) -> str:
        return "qmd_gbrain"

    def is_available(self) -> bool:
        return shutil.which("qmd") is not None and Path(GBRAIN_BIN).exists()

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        hermes_home = Path(kwargs.get("hermes_home") or self._hermes_home)
        self._hermes_home = hermes_home
        self._backend_home = self._resolve_backend_home(hermes_home)
        self._profile_name = str(kwargs.get("agent_identity") or hermes_home.name or "default")

    def system_prompt_block(self) -> str:
        return (
            "# QMD+GBrain Memory\n"
            f"Active for profile '{self._profile_name}'. "
            "Profile-scoped QMD and GBrain recall is auto-injected before each turn as background context. "
            "Built-in Hermes memory remains the durable compact fact store."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive() and query == self._last_prefetch_query:
            self._prefetch_thread.join(timeout=2.0)
            with self._prefetch_lock:
                result = self._prefetch_result
                self._prefetch_result = ""
            if result:
                return result
        return self._build_prefetch(query)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        self._last_prefetch_query = query

        def _run() -> None:
            result = self._build_prefetch(query)
            with self._prefetch_lock:
                self._prefetch_result = result

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="qmd-gbrain-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        return None

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

    def shutdown(self) -> None:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=2.0)
        self._prefetch_thread = None

    def _resolve_backend_home(self, hermes_home: Path) -> Path:
        candidate = hermes_home / "home"
        return candidate if candidate.exists() else hermes_home

    def _build_prefetch(self, query: str) -> str:
        query = (query or "").strip()
        if not query:
            return ""

        qmd_rows = self._run_qmd_search(query)
        gbrain_rows = self._run_gbrain_search(query)

        sections: list[str] = []
        qmd_block = self._format_qmd_rows(qmd_rows)
        if qmd_block:
            sections.append("### QMD\n" + qmd_block)
        gbrain_block = self._format_gbrain_rows(gbrain_rows)
        if gbrain_block:
            sections.append("### GBrain\n" + gbrain_block)
        if not sections:
            return ""
        return "## QMD+GBrain Memory\n" + "\n\n".join(sections)

    def _run_qmd_search(self, query: str) -> list[dict[str, Any]]:
        cmd = ["qmd", "search", query, "--json", "-n", str(self._qmd_limit)]
        data = self._run_json_cmd(cmd)
        return data if isinstance(data, list) else []

    def _run_gbrain_search(self, query: str) -> list[dict[str, Any]]:
        payload = json.dumps({"query": query, "limit": self._gbrain_limit}, separators=(",", ":"))
        cmd = [GBRAIN_BIN, "call", "search", payload]
        data = self._run_json_cmd(cmd)
        return data if isinstance(data, list) else []

    def _run_json_cmd(self, cmd: list[str]) -> Any:
        env = os.environ.copy()
        env["HOME"] = str(self._backend_home)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                env=env,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []
        if result.returncode != 0:
            return []
        text = (result.stdout or "").strip()
        if not text:
            return []
        blob = self._extract_json_blob(text)
        if not blob:
            return []
        try:
            return json.loads(blob)
        except Exception:
            return []

    def _extract_json_blob(self, text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("[") or stripped.startswith("{"):
            return stripped
        for marker in ("\n[", "\n{"):
            idx = stripped.rfind(marker)
            if idx != -1:
                return stripped[idx + 1 :].strip()
        return ""

    def _format_qmd_rows(self, rows: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for row in rows[: self._qmd_limit]:
            title = row.get("title") or row.get("file") or "untitled"
            file_ref = row.get("file") or ""
            score = row.get("score")
            snippet = self._clip(row.get("snippet") or row.get("body") or "")
            head = f"- [{self._fmt_score(score)}] {title}"
            if file_ref:
                head += f" ({file_ref})"
            lines.append(head)
            if snippet:
                lines.append(f"  {snippet}")
        return "\n".join(lines)

    def _format_gbrain_rows(self, rows: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for row in rows[: self._gbrain_limit]:
            slug = row.get("slug") or row.get("title") or "untitled"
            score = row.get("score")
            snippet = self._clip(row.get("chunk_text") or "")
            lines.append(f"- [{self._fmt_score(score)}] {slug}")
            if snippet:
                lines.append(f"  {snippet}")
        return "\n".join(lines)

    def _fmt_score(self, value: Any) -> str:
        try:
            return f"{float(value):.2f}"
        except Exception:
            return "?"

    def _clip(self, text: str, limit: int = 220) -> str:
        text = " ".join(str(text).split())
        return text if len(text) <= limit else text[: limit - 3] + "..."


def register(ctx) -> None:
    ctx.register_memory_provider(QmdGbrainMemoryProvider())
