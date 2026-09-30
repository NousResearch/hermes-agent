"""Hermes profile generator for SamAgent roles (05-final-plan.md §6, §12).

Writes role-specific Hermes profile configurations under ``<hermes_home>/profiles/<role>/config.yaml``
so ``hermes -p samagent-worker`` and ``kanban_db_dispatch`` route each role to its lean toolset
and model without any core patch to Hermes.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

from samagent.router.policy import LEAN_ROLE_TOOLS, TaskBoundaryRouter


def resolve_hermes_home(hermes_home: Optional[Path] = None) -> Path:
    if hermes_home is not None:
        return Path(hermes_home)
    env_home = os.environ.get("HERMES_HOME")
    if env_home:
        return Path(env_home)
    return Path.home() / ".hermes"


def generate_samagent_profiles(
    hermes_home: Optional[Path] = None,
    *,
    router_policy: str = "default",
    cloud_available: bool = True,
) -> Dict[str, Path]:
    """Materialize samagent-worker, samagent-orchestrator, samagent-verifier, and samagent-judge profiles."""
    home = resolve_hermes_home(hermes_home)
    profiles_dir = home / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    rt = TaskBoundaryRouter(policy=router_policy, cloud_available=cloud_available)
    role_specs = {
        "samagent-worker": ("module_impl", ["terminal", "file"], LEAN_ROLE_TOOLS["worker"]),
        "samagent-orchestrator": ("contract_freeze", ["terminal", "file", "delegation"], LEAN_ROLE_TOOLS["orchestrator"]),
        "samagent-verifier": ("browser_verify", ["terminal", "file", "browser"], LEAN_ROLE_TOOLS["verifier"]),
        "samagent-judge": ("judge", ["file"], LEAN_ROLE_TOOLS["verifier"]),
    }

    written: Dict[str, Path] = {}
    for profile_name, (task_kind, toolsets, eager_tools) in role_specs.items():
        decision = rt.route(task_kind)
        model_id = decision.model.model_id if decision.model else "qwen3.8-27b"
        provider = decision.model.provider if decision.model else "local"

        cfg: Dict[str, Any] = {
            "model": {
                "default": model_id,
                "provider": provider,
            },
            "toolsets": toolsets,
            "plugins": {
                "enabled": ["samagent"],
            },
            "agent": {
                "skip_memory": True,  # SamAgent injects <=2K Ledger facts into user turn via pre_llm_call
                "skip_background_review": True,  # Disable costly periodic reflection passes
                "max_verify_nudges": 2,
            },
            "tools": {
                "tool_search": {
                    "enabled": "auto",
                    "defer": [
                        "session_search",
                        "process_manage",
                        "cronjob_manage",
                        "image_generate",
                        "computer_use",
                    ],
                }
            },
            "samagent": {
                "role": profile_name,
                "router_policy": router_policy,
                "eager_tools": eager_tools,
            },
        }
        if decision.request_overrides:
            cfg["model"]["request_overrides"] = decision.request_overrides

        p_dir = profiles_dir / profile_name
        p_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = p_dir / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        written[profile_name] = cfg_path

    return written
