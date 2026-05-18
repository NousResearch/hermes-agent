from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from agent.memory_provider import MemoryProvider
from hermes_constants import get_hermes_home
from tools.registry import tool_error

logger = logging.getLogger(__name__)

PROFILE_SCHEMA = {
	"name": "mem0_profile",
	"description": "Return the user's stored long-tail semantic memories from self-hosted Mem0.",
	"parameters": {"type": "object", "properties": {}, "required": []},
}

SEARCH_SCHEMA = {
	"name": "mem0_search",
	"description": "Search self-hosted Mem0 for relevant personal context, prior decisions, workflows, and long-tail recall.",
	"parameters": {
		"type": "object",
		"properties": {
			"query": {"type": "string", "description": "Semantic search query."},
			"top_k": {"type": "integer", "description": "Maximum results to return (default 5, max 20)."},
		},
		"required": ["query"],
	},
}

CONCLUDE_SCHEMA = {
	"name": "mem0_conclude",
	"description": "Store a durable fact or correction in self-hosted Mem0 for future recall.",
	"parameters": {
		"type": "object",
		"properties": {
			"conclusion": {"type": "string", "description": "Durable fact to store."},
			"category": {"type": "string", "description": "Optional category hint such as preference, person, task, or general."},
		},
		"required": ["conclusion"],
	},
}


def _load_config() -> dict[str, Any]:
	config: dict[str, Any] = {
		"base_url": os.environ.get("MEM0_BASE_URL", "").rstrip("/"),
		"user_id": os.environ.get("MEM0_USER_ID", "hermes-user"),
		"agent_id": os.environ.get("MEM0_AGENT_ID", "hermes-brian"),
		"bearer_token": os.environ.get("MEM0_BEARER_TOKEN", ""),
		"prefetch_limit": 5,
		"prefetch_chars": 1200,
		"sync_turns": True,
		"request_timeout_seconds": 10,
	}

	config_path = get_hermes_home() / "pi_brian_mem0.json"
	if config_path.exists():
		try:
			file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
			for key, value in file_cfg.items():
				if value not in (None, ""):
					config[key] = value
		except Exception:
			logger.debug("Failed reading pi_brian_mem0.json", exc_info=True)

	return config


def _strip_html(text: str) -> str:
	return (
		text.replace("<b>", "")
		.replace("</b>", "")
		.replace("<i>", "")
		.replace("</i>", "")
		.replace("<code>", "")
		.replace("</code>", "")
		.replace("&lt;", "<")
		.replace("&gt;", ">")
		.replace("&amp;", "&")
	)


def _as_bool(value: Any) -> bool:
	if isinstance(value, bool):
		return value
	if isinstance(value, str):
		return value.strip().lower() not in {"", "0", "false", "no", "off"}
	return bool(value)


class PiBrianMem0MemoryProvider(MemoryProvider):
	def __init__(self) -> None:
		self._config: dict[str, Any] = {}
		self._base_url = ""
		self._user_id = "hermes-user"
		self._agent_id = "hermes-brian"
		self._bearer_token = ""
		self._prefetch_limit = 5
		self._prefetch_chars = 1200
		self._sync_turns = True
		self._request_timeout_seconds = 10
		self._sync_thread: Optional[threading.Thread] = None
		self._prefetch_thread: Optional[threading.Thread] = None
		self._memory_write_thread: Optional[threading.Thread] = None
		self._prefetch_lock = threading.Lock()
		self._prefetch_result = ""
		self._agent_context = "primary"

	@property
	def name(self) -> str:
		return "pi_brian_mem0"

	def is_available(self) -> bool:
		cfg = _load_config()
		return bool(str(cfg.get("base_url", "")).strip())

	def initialize(self, session_id: str, **kwargs) -> None:
		cfg = _load_config()
		self._config = cfg
		self._base_url = str(cfg.get("base_url", "")).rstrip("/")
		self._user_id = str(kwargs.get("user_id") or cfg.get("user_id") or "hermes-user")
		self._agent_id = str(cfg.get("agent_id") or "hermes-brian")
		self._bearer_token = str(cfg.get("bearer_token") or "")
		self._prefetch_limit = max(1, min(20, int(cfg.get("prefetch_limit", 5) or 5)))
		self._prefetch_chars = max(250, int(cfg.get("prefetch_chars", 1200) or 1200))
		self._sync_turns = _as_bool(cfg.get("sync_turns", True))
		self._request_timeout_seconds = max(1, int(cfg.get("request_timeout_seconds", 10) or 10))
		self._agent_context = str(kwargs.get("agent_context") or "primary")

	def system_prompt_block(self) -> str:
		return (
			"# Long-tail Memory\n"
			f"Self-hosted Mem0 active for user scope `{self._user_id}`.\n"
			"Use built-in USER.md and MEMORY.md for hot, always-on facts. "
			"Use mem0_search for prior conversations, personal context, recurring workflows, and long-tail recall. "
			"Store durable new facts with mem0_conclude; avoid storing ephemeral chatter."
		)

	def _headers(self) -> dict[str, str]:
		headers = {"Accept": "application/json"}
		if self._bearer_token:
			headers["Authorization"] = f"Bearer {self._bearer_token}"
		return headers

	def _request_json(
		self,
		method: str,
		path: str,
		*,
		payload: Optional[dict[str, Any]] = None,
		query: Optional[dict[str, Any]] = None,
	) -> Any:
		url = f"{self._base_url}{path}"
		if query:
			url = f"{url}?{urlencode(query)}"
		headers = self._headers()
		data = None
		if payload is not None:
			headers["Content-Type"] = "application/json"
			data = json.dumps(payload).encode("utf-8")
		request = Request(url, data=data, headers=headers, method=method)
		try:
			with urlopen(request, timeout=self._request_timeout_seconds) as response:
				raw = response.read().decode("utf-8")
				return json.loads(raw) if raw else {}
		except HTTPError as exc:
			body = exc.read().decode("utf-8", errors="replace")
			raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
		except URLError as exc:
			raise RuntimeError(str(exc.reason)) from exc

	@staticmethod
	def _unwrap_results(payload: Any) -> list[dict[str, Any]]:
		if isinstance(payload, dict):
			results = payload.get("results")
			return results if isinstance(results, list) else []
		if isinstance(payload, list):
			return payload
		return []

	def _format_results(self, results: list[dict[str, Any]], *, include_scores: bool) -> str:
		lines: list[str] = []
		budget = 0
		for index, item in enumerate(results[: self._prefetch_limit], 1):
			memory = _strip_html(str(item.get("memory") or item.get("text") or "")).strip()
			if not memory:
				continue
			if len(memory) > 260:
				memory = f"{memory[:257]}..."
			line = f"{index}. {memory}"
			if include_scores and item.get("score") is not None:
				try:
					line += f" (score={float(item['score']):.3f})"
				except Exception:
					pass
			projected = budget + len(line) + 1
			if projected > self._prefetch_chars:
				break
			lines.append(line)
			budget = projected
		return "\n".join(lines)

	def prefetch(self, query: str, *, session_id: str = "") -> str:
		with self._prefetch_lock:
			result = self._prefetch_result
			self._prefetch_result = ""
		if not result:
			return ""
		return f"## Relevant semantic memory\n{result}"

	def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
		query = (query or "").strip()
		if not query or query.startswith("/") or len(query) < 4:
			return

		def _run() -> None:
			try:
				results = self._unwrap_results(
					self._request_json(
						"POST",
						"/search",
						payload={"query": query, "user_id": self._user_id},
					)
				)
				formatted = self._format_results(results, include_scores=True)
				with self._prefetch_lock:
					self._prefetch_result = formatted
			except Exception:
				logger.debug("pi_brian_mem0 prefetch failed", exc_info=True)

		self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="pi-brian-mem0-prefetch")
		self._prefetch_thread.start()

	def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
		if not self._sync_turns or self._agent_context != "primary":
			return
		if not (user_content or "").strip() or not (assistant_content or "").strip():
			return

		def _run() -> None:
			try:
				self._request_json(
					"POST",
					"/memories",
					payload={
						"messages": [
							{"role": "user", "content": user_content},
							{"role": "assistant", "content": assistant_content},
						],
						"user_id": self._user_id,
						"metadata": {"source": "hermes_turn", "session_id": session_id or ""},
					},
				)
			except Exception:
				logger.debug("pi_brian_mem0 sync_turn failed", exc_info=True)

		if self._sync_thread and self._sync_thread.is_alive():
			self._sync_thread.join(timeout=5.0)
		self._sync_thread = threading.Thread(target=_run, daemon=True, name="pi-brian-mem0-sync")
		self._sync_thread.start()

	def get_tool_schemas(self) -> List[Dict[str, Any]]:
		return [PROFILE_SCHEMA, SEARCH_SCHEMA, CONCLUDE_SCHEMA]

	def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
		try:
			if tool_name == "mem0_profile":
				results = self._unwrap_results(self._request_json("GET", "/memories", query={"user_id": self._user_id}))
				formatted = self._format_results(results[:100], include_scores=False)
				return json.dumps({"count": len(results), "result": formatted or "No memories stored yet."})

			if tool_name == "mem0_search":
				query = str(args.get("query") or "").strip()
				if not query:
					return tool_error("Missing required parameter: query")
				try:
					top_k = max(1, min(20, int(args.get("top_k", self._prefetch_limit) or self._prefetch_limit)))
				except Exception:
					top_k = self._prefetch_limit
				results = self._unwrap_results(
					self._request_json("POST", "/search", payload={"query": query, "user_id": self._user_id})
				)[:top_k]
				return json.dumps({"count": len(results), "results": results})

			if tool_name == "mem0_conclude":
				conclusion = str(args.get("conclusion") or "").strip()
				if not conclusion:
					return tool_error("Missing required parameter: conclusion")
				category = str(args.get("category") or "general").strip() or "general"
				self._request_json(
					"POST",
					"/memories",
					payload={
						"messages": [{"role": "user", "content": conclusion}],
						"user_id": self._user_id,
						"metadata": {"source": "hermes_conclusion", "category": category},
					},
				)
				return json.dumps({"stored": True, "conclusion": conclusion, "category": category})
		except Exception as exc:
			return tool_error(f"Mem0 error: {exc}")
		return tool_error(f"Unknown tool: {tool_name}")

	def on_memory_write(self, action: str, target: str, content: str, metadata: Optional[dict[str, Any]] = None) -> None:
		if action != "add" or not (content or "").strip():
			return

		def _run() -> None:
			try:
				merged_metadata = {"source": "hermes_memory", "target": target}
				if isinstance(metadata, dict):
					merged_metadata.update(metadata)
				self._request_json(
					"POST",
					"/memories",
					payload={
						"messages": [{"role": "user", "content": f"[{target}] {content.strip()}"}],
						"user_id": self._user_id,
						"metadata": merged_metadata,
					},
				)
			except Exception:
				logger.debug("pi_brian_mem0 on_memory_write failed", exc_info=True)

		if self._memory_write_thread and self._memory_write_thread.is_alive():
			self._memory_write_thread.join(timeout=5.0)
		self._memory_write_thread = threading.Thread(target=_run, daemon=True, name="pi-brian-mem0-memory-write")
		self._memory_write_thread.start()

	def get_config_schema(self) -> List[Dict[str, Any]]:
		return [
			{
				"key": "base_url",
				"description": "Self-hosted Mem0 base URL",
				"required": True,
				"default": "http://127.0.0.1:8000",
			},
			{
				"key": "user_id",
				"description": "Fallback user identifier for CLI sessions",
				"default": "hermes-user",
			},
			{
				"key": "agent_id",
				"description": "Agent identifier for write attribution",
				"default": "hermes-brian",
			},
			{
				"key": "prefetch_limit",
				"description": "Maximum prefetch results to inject per turn",
				"default": "5",
			},
			{
				"key": "prefetch_chars",
				"description": "Maximum characters injected from Mem0 prefetch per turn",
				"default": "1200",
			},
			{
				"key": "sync_turns",
				"description": "Persist completed turns back into self-hosted Mem0",
				"default": "true",
				"choices": ["true", "false"],
			},
			{
				"key": "request_timeout_seconds",
				"description": "HTTP timeout for Mem0 requests",
				"default": "10",
			},
		]

	def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
		config_path = Path(hermes_home) / "pi_brian_mem0.json"
		existing: dict[str, Any] = {}
		if config_path.exists():
			try:
				existing = json.loads(config_path.read_text(encoding="utf-8"))
			except Exception:
				existing = {}
		existing.update(values)
		config_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")

	def shutdown(self) -> None:
		for thread in (self._prefetch_thread, self._sync_thread, self._memory_write_thread):
			if thread and thread.is_alive():
				thread.join(timeout=5.0)


def register(ctx) -> None:
	ctx.register_memory_provider(PiBrianMem0MemoryProvider())
