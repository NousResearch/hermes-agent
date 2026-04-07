#!/usr/bin/env python3
"""
Inference-time adapter routing for the finetune pipeline.

Embeds incoming prompts, matches against cluster centroids, and selects
the best adapter. Designed to integrate with Hermes's provider pre-request hook.

Usage:
    python route.py "Your prompt text here"
    python route.py --test  # Run routing diagnostics
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from common import (
    ADAPTERS_DIR, CLUSTER_STATE_PATH, REGISTRY_PATH,
    load_config, load_json, logger,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class AdapterRouter:
    """Route prompts to the best fine-tuned adapter."""

    def __init__(self, config: dict = None):
        cfg = config or load_config()
        clustering_cfg = cfg.get("clustering", {})
        routing_cfg = cfg.get("routing", {})

        self.embedding_model_name = clustering_cfg.get("embedding_model", "all-MiniLM-L6-v2")
        self.confidence_threshold = clustering_cfg.get("confidence_threshold", 0.6)
        self.enabled = routing_cfg.get("enabled", True)
        self.allowed_providers = routing_cfg.get("providers", ["local", "llama-cpp", "custom"])

        self._embed_model = None
        self._cluster_state = None
        self._registry = None
        self._centroids = None

    def _get_embed_model(self):
        """Lazy-load the embedding model."""
        if self._embed_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                logger.debug("sentence-transformers not available for routing")
                return None
            self._embed_model = SentenceTransformer(self.embedding_model_name)
        return self._embed_model

    def _load_state(self):
        """Load cluster state and registry."""
        if self._cluster_state is None:
            self._cluster_state = load_json(CLUSTER_STATE_PATH, {})
        if self._registry is None:
            self._registry = load_json(REGISTRY_PATH, {"adapters": []})
        if self._centroids is None:
            raw = self._cluster_state.get("centroids", {})
            self._centroids = {
                cid: np.array(vec) for cid, vec in raw.items()
            }

    def _get_active_adapters(self) -> Dict[str, Dict]:
        """Get active adapters by cluster ID."""
        self._load_state()
        active = {}
        for entry in self._registry.get("adapters", []):
            if entry["status"] == "active":
                active[entry["cluster_id"]] = entry
        return active

    def route(self, prompt: str) -> Dict[str, Any]:
        """
        Route a prompt to the best adapter.

        Returns a dict with:
            - cluster_id: matched cluster (or None)
            - adapter_path: path to adapter (or None)
            - confidence: cosine similarity score
            - label: human-readable cluster label
            - fallback: True if using _general or base model
        """
        if not self.enabled:
            return {
                "cluster_id": None,
                "adapter_path": None,
                "confidence": 0.0,
                "label": "routing disabled",
                "fallback": True,
            }

        model = self._get_embed_model()
        if model is None:
            return {
                "cluster_id": None,
                "adapter_path": None,
                "confidence": 0.0,
                "label": "embedding model unavailable",
                "fallback": True,
            }

        self._load_state()
        active_adapters = self._get_active_adapters()

        if not active_adapters or not self._centroids:
            return {
                "cluster_id": None,
                "adapter_path": None,
                "confidence": 0.0,
                "label": "no active adapters",
                "fallback": True,
            }

        # Embed the prompt
        embedding = model.encode([prompt], normalize_embeddings=True)[0]

        # Compare against centroids
        best_cluster = None
        best_sim = -1.0

        for cid, centroid in self._centroids.items():
            if cid == "_general":
                continue
            if cid not in active_adapters:
                continue

            sim = float(np.dot(embedding, centroid) / (
                np.linalg.norm(embedding) * np.linalg.norm(centroid) + 1e-8
            ))
            if sim > best_sim:
                best_sim = sim
                best_cluster = cid

        # Check confidence threshold
        if best_cluster and best_sim >= self.confidence_threshold:
            adapter_entry = active_adapters[best_cluster]
            version = adapter_entry["version"]
            adapter_dir = ADAPTERS_DIR / best_cluster / version

            # Prefer merged GGUF, fall back to LoRA adapter
            gguf_path = adapter_dir / "merged.gguf"
            lora_path = adapter_dir / "adapter_model"

            if gguf_path.exists():
                adapter_path = str(gguf_path)
            elif lora_path.exists():
                adapter_path = str(lora_path)
            else:
                adapter_path = None

            cluster_info = self._cluster_state.get("clusters", {}).get(best_cluster, {})
            return {
                "cluster_id": best_cluster,
                "adapter_path": adapter_path,
                "confidence": round(best_sim, 4),
                "label": cluster_info.get("label", best_cluster),
                "fallback": False,
            }

        # Fall back to _general if available
        if "_general" in active_adapters:
            entry = active_adapters["_general"]
            version = entry["version"]
            adapter_dir = ADAPTERS_DIR / "_general" / version
            gguf_path = adapter_dir / "merged.gguf"
            lora_path = adapter_dir / "adapter_model"

            adapter_path = str(gguf_path) if gguf_path.exists() else (
                str(lora_path) if lora_path.exists() else None
            )

            return {
                "cluster_id": "_general",
                "adapter_path": adapter_path,
                "confidence": round(best_sim, 4) if best_sim > 0 else 0.0,
                "label": "general (fallback)",
                "fallback": True,
            }

        return {
            "cluster_id": None,
            "adapter_path": None,
            "confidence": round(best_sim, 4) if best_sim > 0 else 0.0,
            "label": "no match above threshold",
            "fallback": True,
        }

    def should_route(self, provider: str) -> bool:
        """Check if routing should activate for this provider."""
        if not self.enabled:
            return False
        return provider.lower() in [p.lower() for p in self.allowed_providers]


def _pre_llm_call_hook(**kwargs) -> Optional[Dict]:
    """
    Hook for Hermes's pre_llm_call system.

    This function is called before each LLM API call. It checks if the
    current provider supports adapter routing and, if so, selects the
    appropriate adapter.

    Returns a dict with adapter routing info, or None if routing doesn't apply.
    """
    platform = kwargs.get("platform", "")
    model = kwargs.get("model", "")
    user_message = kwargs.get("user_message", "")

    router = AdapterRouter()

    # Only route for local providers
    # (Cloud providers don't support LoRA loading)
    if not router.should_route(platform or "local"):
        return None

    result = router.route(user_message)

    if result.get("adapter_path"):
        logger.info(
            "Routing to adapter: %s (cluster=%s, confidence=%.3f)",
            result["label"], result["cluster_id"], result["confidence"],
        )
        return {
            "adapter_path": result["adapter_path"],
            "cluster_id": result["cluster_id"],
            "routing_confidence": result["confidence"],
        }

    return None


def main():
    parser = argparse.ArgumentParser(description="Test adapter routing")
    parser.add_argument("prompt", nargs="?", default=None, help="Prompt to route")
    parser.add_argument("--test", action="store_true", help="Run routing diagnostics")
    args = parser.parse_args()

    router = AdapterRouter()

    if args.test:
        print("Routing diagnostics:")
        print(f"  Enabled: {router.enabled}")
        print(f"  Embedding model: {router.embedding_model_name}")
        print(f"  Confidence threshold: {router.confidence_threshold}")
        print(f"  Allowed providers: {router.allowed_providers}")

        router._load_state()
        active = router._get_active_adapters()
        print(f"  Active adapters: {len(active)}")
        for cid, entry in active.items():
            print(f"    {cid}: {entry['version']}")

        centroids = router._centroids or {}
        print(f"  Centroids loaded: {len(centroids)}")

        if args.prompt:
            result = router.route(args.prompt)
            print(f"\n  Route result: {result}")

    elif args.prompt:
        result = router.route(args.prompt)
        print(f"Cluster:    {result['cluster_id'] or 'none'}")
        print(f"Label:      {result['label']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Adapter:    {result['adapter_path'] or 'base model'}")
        print(f"Fallback:   {result['fallback']}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
