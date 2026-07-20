"""Brain Networks — simulated neural networks for agent cognition.

Implements simplified versions of:
- DMN (Default Mode Network) — idle reflection
- ECN (Executive Control Network) — task focus
- Limbic system — emotional salience tagging
- Dreaming — offline episode consolidation
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .dmn import DefaultModeNetwork
from .ecn import ExecutiveControlNetwork
from .limbic import LimbicSystem
from .dreaming import DreamEngine

logger = logging.getLogger(__name__)

__all__ = [
    "DefaultModeNetwork",
    "ExecutiveControlNetwork",
    "LimbicSystem",
    "DreamEngine",
    "BrainNetworkOrchestrator",
    "doctor_check",
    "normalize_brain_config",
]


def normalize_brain_config(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Map flat brain_networks keys into nested network configs."""
    from agent.brain_networks.runtime import normalize_brain_config as _norm

    return _norm(cfg)


class BrainNetworkOrchestrator:
    """Orchestrates all brain networks with session-persistent ECN focus."""

    def __init__(self):
        self.dmn = DefaultModeNetwork()
        self.ecn = ExecutiveControlNetwork()
        self.limbic = LimbicSystem()
        self.dream = DreamEngine()
        self._active = False
        self.session_id: str = ""

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize all networks (accepts flat or nested config)."""
        try:
            cfg = normalize_brain_config(config or {})
            self.dmn.initialize(cfg.get("dmn", {}))
            self.ecn.initialize(cfg.get("ecn", {}))
            self.limbic.initialize(cfg.get("limbic", {}))
            self.dream.initialize(cfg.get("dreaming", {}))
            self._active = True
            return True
        except Exception as e:
            logger.error("Failed to initialize brain networks: %s", e)
            return False

    def bind_session(self, session_id: str) -> None:
        """Bind orchestrator networks to a conversation session."""
        self.session_id = (session_id or "").strip()
        self.ecn.bind_session(self.session_id, load=True)

    def set_focus(self, task: str, *, pinned: bool = True) -> Dict[str, Any]:
        """Set standing ECN focus for the bound session."""
        if self.session_id:
            self.ecn.bind_session(self.session_id, load=True)
        return self.ecn.set_focus(task, pinned=pinned)

    def clear_focus(self) -> Dict[str, Any]:
        """Clear standing ECN focus for the bound session."""
        return self.ecn.clear_focus()

    def on_turn_start(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process turn start through all networks."""
        if not self._active:
            return {}

        ctx = dict(context or {})
        sid = str(ctx.get("session_id") or self.session_id or "").strip()
        if sid:
            self.bind_session(sid)
            ctx["session_id"] = sid

        results: Dict[str, Any] = {}

        # ECN evaluates task focus (persists across turns)
        results["ecn"] = self.ecn.evaluate_focus(ctx)

        # Limbic tags emotional salience
        results["limbic"] = self.limbic.tag_salience(ctx)

        # DMN contributes if idle or reflective query
        if self.dmn.should_reflect(ctx):
            results["dmn"] = self.dmn.generate_reflection(ctx)

        return results

    def format_turn_block(self, results: Optional[Dict[str, Any]] = None) -> str:
        """Format a cache-safe volatile context block from turn results."""
        if not self._active:
            return ""
        data = results if results is not None else {}
        parts: List[str] = []

        ecn = data.get("ecn") or {}
        reminder = str(ecn.get("reminder") or ecn.get("focus") or "").strip()
        if reminder:
            parts.append(f"[ECN focus] {reminder}")

        limbic = data.get("limbic") or {}
        if limbic.get("requires_attention"):
            emo = (limbic.get("emotional_state") or {}).get("dominant") or "alert"
            tone = (limbic.get("tone_adaptation") or {}).get("tone") or "neutral"
            parts.append(
                f"[Limbic] high salience emotion={emo} tone_hint={tone} "
                f"score={limbic.get('salience_score', 0)}"
            )

        dmn = data.get("dmn") or {}
        content = str(dmn.get("content") or "").strip()
        if content:
            parts.append(f"[DMN reflection] {content[:500]}")

        if not parts:
            return ""
        body = "\n".join(parts)
        return f"<brain-networks>\n{body}\n</brain-networks>"

    def on_idle(self, idle_seconds: float) -> Optional[Dict[str, Any]]:
        """Process idle time — trigger dreaming if sufficient."""
        if not self._active:
            return None

        if self.dream.should_dream(idle_seconds):
            return self.dream.generate_dream()

        return None

    def get_state(self) -> Dict[str, Any]:
        """Get current network states."""
        return {
            "active": self._active,
            "session_id": self.session_id,
            "dmn": self.dmn.get_state(),
            "ecn": self.ecn.get_state(),
            "limbic": self.limbic.get_state(),
            "dream": self.dream.get_state(),
        }


def doctor_check() -> Dict[str, Any]:
    """Health check for brain networks."""
    result: Dict[str, Any] = {"ok": True, "issues": [], "networks": {}}

    try:
        orchestrator = BrainNetworkOrchestrator()
        success = orchestrator.initialize({"enabled": True})

        result["initialized"] = success
        result["networks"] = orchestrator.get_state()

        # Persistence smoke
        try:
            from agent.brain_networks.persistence import dream_count, load_ecn_state

            result["persistence"] = {
                "ok": True,
                "dream_count": dream_count(),
                "ecn_probe": load_ecn_state("_doctor_probe") is None,
            }
        except Exception as pe:
            result["persistence"] = {"ok": False, "error": str(pe)[:160]}
            result["issues"].append(f"persistence: {pe}")

    except Exception as e:
        result["ok"] = False
        result["issues"].append(str(e))

    return result
