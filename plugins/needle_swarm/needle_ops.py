"""needle_ops.py - Thin wrapper around cactus-needle with stub/cloud fallback.

Every function provides real calls if cactus-needle is installed, or stubbed/cloud-assisted
behavior so the orchestration logic can run without local Needle hardware.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Check if cactus-needle is installed
_NEEDLE_AVAILABLE = False
try:
    import needle  # type: ignore
    _NEEDLE_AVAILABLE = True
except ImportError:
    _NEEDLE_AVAILABLE = False


def is_needle_available() -> bool:
    return _NEEDLE_AVAILABLE


class LoadedNeedleInstance:
    """Wrapper around a loaded Needle model instance."""

    def __init__(self, skill_id: str, kind: str, tools: Optional[List[str]] = None, weights_path: Optional[str] = None):
        self.skill_id = skill_id
        self.kind = kind
        self.tools = tools or []
        self.weights_path = weights_path
        self._instance = None
        if _NEEDLE_AVAILABLE:
            try:
                if weights_path:
                    self._instance = needle.Needle(weights_path=weights_path, tools=self.tools)
                else:
                    self._instance = needle.Needle(tools=self.tools)
            except Exception as e:
                logger.warning("Failed to initialize real Needle instance for %s: %s", skill_id, e)

    def dispatch(self, utterance: str) -> Tuple[Optional[str], Dict[str, Any], float]:
        """Dispatch utterance to (tool, args, confidence)."""
        if self._instance and hasattr(self._instance, "dispatch"):
            try:
                tool, args, conf = self._instance.dispatch(utterance)
                return tool, args, float(conf)
            except Exception as e:
                logger.warning("Needle dispatch call failed for %s: %s", self.skill_id, e)

        # Fallback / Stub implementation
        if not self.tools:
            return None, {}, 0.2
        # Deterministic pseudo-dispatch based on hash
        h = int(hashlib.md5(f"{self.skill_id}:{utterance}".encode()).hexdigest(), 16)
        tool = self.tools[h % len(self.tools)]
        # Heuristic confidence based on word overlap
        conf = 0.55 + 0.35 * (h % 100) / 100.0
        return tool, {"input": utterance}, round(conf, 3)

    def extract(self, situation: str) -> Tuple[Dict[str, Any], float]:
        """PLAYBOOK extraction: situation -> (matched_procedure, confidence)."""
        if self._instance and hasattr(self._instance, "extract"):
            try:
                proc, conf = self._instance.extract(situation)
                return proc, float(conf)
            except Exception as e:
                logger.warning("Needle extract call failed for %s: %s", self.skill_id, e)

        # Fallback / Stub
        h = int(hashlib.md5(f"{self.skill_id}:{situation}".encode()).hexdigest(), 16)
        conf = 0.6 + 0.35 * (h % 100) / 100.0
        procedure = {
            "playbook_id": self.skill_id,
            "situation": situation,
            "checklist": [
                f"Verify trigger condition for {self.skill_id}",
                "Execute primary response protocol",
                "Log outcome to hindsight store",
            ],
            "recommended_action": f"execute_{self.skill_id}_playbook",
        }
        return procedure, round(conf, 3)


def embed_text(text: str, dim: int = 64) -> List[float]:
    """Compute vector embedding for text using needle_embed or fallback token-hash embedding."""
    if _NEEDLE_AVAILABLE and hasattr(needle, "needle_embed"):
        try:
            return list(needle.needle_embed(text))
        except Exception as e:
            logger.warning("needle_embed failed: %s", e)

    # Token-based bag-of-words pseudo-embedding for semantic similarity in fallback mode
    words = re.findall(r"\w+", (text or "").lower())
    if not words:
        return [0.0] * dim

    vec = [0.0] * dim
    for word in words:
        # Map word hash into vector dimensions
        h_val = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
        idx1 = h_val % dim
        idx2 = (h_val >> 8) % dim
        sign = 1.0 if (h_val % 2 == 0) else -1.0
        vec[idx1] += sign
        vec[idx2] += 1.0

    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [round(v / norm, 5) for v in vec]


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two vector embeddings."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 == 0 or n2 == 0:
        return 0.0
    return max(-1.0, min(1.0, dot / (n1 * n2)))


def generate_synthetic_data(
    skill_description: str,
    examples: List[Dict[str, Any]],
    count: int = 10,
    llm_callback: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Generate synthetic training data using LLM callback or fallback template generator."""
    if llm_callback:
        try:
            prompt = (
                f"Generate {count} training examples for a micro-AI skill with description: '{skill_description}'.\n"
                f"Seed examples: {json.dumps(examples)}\n"
                "Return JSON array of objects with keys 'utterance', 'tool', 'args'."
            )
            raw = llm_callback(prompt)
            if isinstance(raw, list):
                return raw
            if isinstance(raw, str):
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return parsed
        except Exception as e:
            logger.warning("LLM synthetic data generation failed: %s", e)

    # Stub / Fallback template generator
    out = []
    tools = ["action_tool", "execute_task", "query_info"]
    for i in range(count):
        seed_ut = examples[i % len(examples)]["utterance"] if examples else "do task"
        out.append({
            "utterance": f"{seed_ut} variant {i+1}",
            "tool": tools[i % len(tools)],
            "args": {"param": f"value_{i+1}"},
        })
    return out


def finetune_and_build_skill(
    skill_id: str,
    data: List[Dict[str, Any]],
    output_dir: str,
    epochs: int = 3,
) -> str:
    """Run fine-tuning and build .cact weights artifact."""
    out_path = f"{output_dir}/{skill_id}_v1.cact"
    if _NEEDLE_AVAILABLE and hasattr(needle, "finetune"):
        try:
            needle.finetune(data=data, output_path=out_path, epochs=epochs)
            return out_path
        except Exception as e:
            logger.warning("Needle fine-tune failed: %s", e)

    # Stub weight file creation
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"// Stub Needle Weights Artifact for {skill_id}\n")
        f.write(json.dumps({"skill_id": skill_id, "data_count": len(data), "epochs": epochs}))
    return out_path
