"""Thin Hermes-compatible provider wrapper."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from llmwiki_hermes.constants import PROVIDER_NAME
from llmwiki_hermes.provider.tools import ProviderTools
from llmwiki_hermes.provider.writeback import SessionWritebackService
from llmwiki_hermes.recall.search import RecallService
from llmwiki_hermes.settings import WikiSettings
from llmwiki_hermes.storage.vault import VaultService

if TYPE_CHECKING:

    class HermesMemoryProviderBase:
        """Typed fallback used during static analysis."""

        def system_prompt_block(self) -> str: ...

        def queue_prefetch(self, query: str, *, session_id: str = "") -> None: ...

        def on_turn_start(self, turn_number: int, message: str, **kwargs: Any) -> None: ...
else:
    try:
        from agent.memory_provider import MemoryProvider as HermesMemoryProviderBase
    except ImportError:

        class HermesMemoryProviderBase:
            """Fallback base class when Hermes is not installed."""

            def system_prompt_block(self) -> str:
                return ""

            def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
                return None

            def on_turn_start(self, turn_number: int, message: str, **kwargs: Any) -> None:
                return None


logger = logging.getLogger(__name__)


class WikiMemoryProvider(HermesMemoryProviderBase):
    """Runtime adapter with the shape Hermes expects."""

    def __init__(self) -> None:
        self.session_id: str | None = None
        self.settings: WikiSettings | None = None
        self.vault_service: VaultService | None = None
        self.recall_service: RecallService | None = None
        self.tools: ProviderTools | None = None
        self.writeback_service: SessionWritebackService | None = None

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        try:
            vault_path = kwargs.get("vault_path")
            config_path = kwargs.get("config_path")
            hermes_home = kwargs.get("hermes_home")
            if config_path is None and hermes_home:
                config_path = Path(str(hermes_home)) / PROVIDER_NAME / "config.yaml"
            self.session_id = session_id
            settings = WikiSettings.load(
                vault_path=Path(vault_path) if vault_path else None,
                config_path=Path(config_path) if config_path else None,
            )
            overrides = {
                key: kwargs[key]
                for key in ("top_k_semantic", "top_k_episodic", "auto_writeback")
                if key in kwargs
            }
            self.settings = settings.model_copy(update=overrides)
            self.vault_service = VaultService(self.settings.vault_path)
            self.vault_service.ensure_initialized()
            self.recall_service = RecallService.from_settings(self.settings)
            self.tools = ProviderTools(self.recall_service, self.vault_service)
            self.writeback_service = SessionWritebackService(self.vault_service)
            logger.info(
                "Initialized provider %s for session %s with vault %s.",
                self.name,
                session_id,
                self.settings.vault_path,
            )
        except Exception:
            logger.exception(
                "Provider %s failed to initialize for session %s.",
                self.name,
                session_id,
            )
            raise

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "wiki_recall",
                "description": "Search the local knowledge base.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "memory_type": {
                            "type": "string",
                            "enum": ["auto", "semantic", "episodic"],
                            "default": "auto",
                        },
                        "top_k": {"type": "integer", "default": 8, "minimum": 1, "maximum": 20},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "wiki_get_note",
                "description": "Read a full Markdown note by id or slug.",
                "parameters": {
                    "type": "object",
                    "properties": {"id_or_slug": {"type": "string"}},
                    "required": ["id_or_slug"],
                },
            },
        ]

    def handle_tool_call(self, name: str, args: dict[str, Any], **kwargs: Any) -> str:
        if self.tools is None:
            raise RuntimeError("Provider must be initialized before tool calls.")
        result: dict[str, Any]
        if name == "wiki_recall":
            result = self.tools.wiki_recall(
                query=str(args["query"]),
                memory_type=str(args.get("memory_type", "auto")),
                top_k=int(args.get("top_k", 8)),
            )
        elif name == "wiki_get_note":
            result = self.tools.wiki_get_note(id_or_slug=str(args["id_or_slug"]))
        else:
            raise ValueError(f"Unknown tool: {name}")
        return json.dumps(result, ensure_ascii=False)

    def get_config_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "vault_path",
                "description": "Path to the LLM-Wiki vault root.",
                "required": True,
            },
            {
                "key": "top_k_semantic",
                "description": "Default semantic recall result count.",
                "required": False,
                "default": 5,
            },
            {
                "key": "top_k_episodic",
                "description": "Default episodic recall result count.",
                "required": False,
                "default": 4,
            },
            {
                "key": "auto_writeback",
                "description": "Persist session summaries back into the vault.",
                "required": False,
                "default": False,
                "choices": [True, False],
            },
        ]

    def save_config(self, values: dict[str, Any], hermes_home: str | Path) -> None:
        config_dir = Path(hermes_home) / PROVIDER_NAME
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
        logger.info("Saved provider %s config to %s.", self.name, config_path)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self.recall_service is None:
            logger.warning("Prefetch requested before provider %s was initialized.", self.name)
            return ""
        try:
            result = self.recall_service.recall(query=query, memory_type="auto", top_k=8)
            logger.debug(
                "Prefetch for provider %s returned %s result(s).",
                self.name,
                len(result.results),
            )
            return result.recall_block
        except Exception:
            logger.exception("Recall prefetch failed for provider %s.", self.name)
            return ""

    def sync_turn(self, user: str, assistant: str, *, session_id: str = "") -> None:
        effective_session_id = session_id or self.session_id
        if self.writeback_service is None or effective_session_id is None:
            return
        try:
            self.writeback_service.sync_turn(effective_session_id, user, assistant)
            logger.debug("Synced turn for session %s.", effective_session_id)
        except Exception:
            logger.exception("Session sync failed for session %s.", effective_session_id)

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> None:
        if self.writeback_service is None or self.session_id is None:
            return
        try:
            self.writeback_service.on_pre_compress(self.session_id, messages)
            logger.debug("Stored pre-compress snapshot for session %s.", self.session_id)
        except Exception:
            logger.exception("Pre-compress snapshot failed for session %s.", self.session_id)

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        if (
            self.writeback_service is None
            or self.vault_service is None
            or self.settings is None
            or self.session_id is None
        ):
            return
        try:
            self.writeback_service.on_session_end(
                session_id=self.session_id,
                messages=messages,
                auto_writeback=self.settings.auto_writeback,
            )
            logger.debug("Completed session end writeback for session %s.", self.session_id)
        except Exception:
            logger.exception("Session writeback failed for session %s.", self.session_id)
            self.vault_service.write_session_summary(self.session_id, messages)

    def shutdown(self) -> None:
        self.writeback_service = None
        self.tools = None
        self.recall_service = None
        self.vault_service = None
        self.settings = None
        self.session_id = None

    def as_json(self) -> str:
        return json.dumps({"name": self.name})
