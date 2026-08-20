"""Knowledge subsystem configuration.

Reads ``knowledge:`` from Hermes config.yaml when available, falls back to
env vars, then to a fully offline local default. Secrets come from env
(``ANYTHINGLLM_API_KEY``) — never from config.yaml.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List


def _hermes_home() -> str:
    try:
        from hermes_constants import get_hermes_home  # type: ignore

        return str(get_hermes_home())
    except Exception:
        return os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")


@dataclass
class KnowledgeConfig:
    enabled: bool = True
    provider: str = "local"
    fallback_providers: List[str] = field(default_factory=list)
    workspace: str = "default"
    top_k: int = 5
    cache_ttl: float = 300.0
    cache_size: int = 256
    timeout: float = 30.0
    retries: int = 2
    retry_backoff: float = 0.4
    min_score: float = 0.05
    auto_retrieve: bool = True
    db_path: str = ""
    sync_sources: List[Dict[str, Any]] = field(default_factory=list)
    provider_options: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, overrides: Dict[str, Any] | None = None) -> "KnowledgeConfig":
        raw: Dict[str, Any] = {}
        try:
            from hermes_cli.config import load_config  # type: ignore

            raw = dict((load_config() or {}).get("knowledge") or {})
        except Exception:
            raw = {}

        env = os.environ
        if env.get("HERMES_KNOWLEDGE_PROVIDER"):
            raw["provider"] = env["HERMES_KNOWLEDGE_PROVIDER"]
        if env.get("HERMES_KNOWLEDGE_WORKSPACE"):
            raw["workspace"] = env["HERMES_KNOWLEDGE_WORKSPACE"]
        if env.get("HERMES_KNOWLEDGE_DB"):
            raw["db_path"] = env["HERMES_KNOWLEDGE_DB"]

        opts = dict(raw.get("provider_options") or {})
        allm = dict(opts.get("anythingllm") or {})
        allm.setdefault("base_url", env.get("ANYTHINGLLM_BASE_URL", "http://localhost:3001"))
        if env.get("ANYTHINGLLM_API_KEY"):
            allm["api_key"] = env["ANYTHINGLLM_API_KEY"]
        opts["anythingllm"] = allm
        raw["provider_options"] = opts

        cfg = cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})
        # `hermes config set` stores list/dict values typed at the CLI as JSON
        # strings; coerce them so the dataclass contract always holds.
        for attr in ("sync_sources", "fallback_providers", "provider_options"):
            val = getattr(cfg, attr)
            if isinstance(val, str):
                import json as _json

                try:
                    setattr(cfg, attr, _json.loads(val))
                except Exception:
                    setattr(cfg, attr, [] if attr != "provider_options" else {})
        if overrides:
            for k, v in overrides.items():
                if k in cls.__dataclass_fields__:
                    setattr(cfg, k, v)
        if not cfg.db_path:
            cfg.db_path = os.path.join(_hermes_home(), "knowledge", "knowledge.db")
        return cfg

    def options_for(self, provider: str) -> Dict[str, Any]:
        opts = dict(self.provider_options.get(provider) or {})
        opts.setdefault("default_workspace", self.workspace)
        if provider == "local":
            opts.setdefault("db_path", self.db_path)
            opts.pop("timeout", None)
        else:
            opts.setdefault("timeout", self.timeout)
        return opts
