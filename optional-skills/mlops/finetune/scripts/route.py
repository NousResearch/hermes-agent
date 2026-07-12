#!/usr/bin/env python3
"""
Inference-time adapter routing for the finetune pipeline.

Embeds incoming prompts, matches against cluster centroids, and selects
the best adapter. At inference time this runs as the finetune-routing
plugin (shipped in the skill's plugin/ directory), which registers
``llm_request`` middleware through the standard hermes plugin lifecycle.

Usage:
    python route.py "Your prompt text here"
    python route.py --test    # Run routing diagnostics
    python route.py enable    # Install the routing plugin into <hermes-home>/plugins/
    python route.py disable   # Remove the routing plugin
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


def _resolve_adapter_artifact(adapter_dir: Path) -> Optional[str]:
    """
    Resolve the servable artifact for an adapter version directory.

    llama-server can only load the GGUF LoRA that `manage.py redeploy`
    produces (<version_dir>/adapter.gguf via convert_lora_to_gguf.py).
    If only the PEFT safetensors dir exists, return None — handing
    llama-server a safetensors path would just fail at load time.
    """
    gguf_path = adapter_dir / "adapter.gguf"
    if gguf_path.exists():
        return str(gguf_path)

    if (adapter_dir / "adapter_model").exists():
        logger.warning(
            "Adapter %s has only PEFT safetensors (no adapter.gguf) — "
            "llama-server can't load that. Run '/finetune redeploy' to "
            "convert it.",
            adapter_dir,
        )
    return None


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
            adapter_path = _resolve_adapter_artifact(adapter_dir)

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
            adapter_path = _resolve_adapter_artifact(adapter_dir)

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


PLUGIN_NAME = "finetune-routing"
PLUGIN_SRC = Path(__file__).resolve().parent.parent / "plugin" / PLUGIN_NAME


def enable_routing_plugin() -> bool:
    """Copy the routing plugin into <hermes-home>/plugins/ so the standard
    plugin discovery (hermes_cli/plugins.py) loads it at session start."""
    import shutil

    from common import HERMES_HOME

    if not PLUGIN_SRC.is_dir():
        print(f"Routing plugin source not found: {PLUGIN_SRC}")
        return False

    dest = HERMES_HOME / "plugins" / PLUGIN_NAME
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(PLUGIN_SRC, dest)
    print(f"Routing plugin installed at {dest}")

    # User plugins are opt-in: they load only when listed in the
    # plugins.enabled allow-list. Add it through the official CLI helper so
    # the canonical key resolution stays in one place; fall back to printing
    # the command when hermes_cli isn't importable (bare repo checkout).
    try:
        from hermes_cli.plugins_cmd import cmd_enable
        cmd_enable(PLUGIN_NAME, allow_tool_override=False)
    except SystemExit:
        raise
    except Exception as e:
        logger.debug("cmd_enable unavailable: %s", e)
        print(
            "Now allow-list it (plugins are opt-in):\n"
            f"    hermes plugins enable {PLUGIN_NAME}"
        )

    print("It takes effect for new hermes sessions.")

    cfg = load_config()
    if not cfg.get("routing", {}).get("enabled", False):
        print(
            "Note: finetune.routing.enabled is false in config.yaml — the\n"
            "plugin loads but stays inactive until you set it to true."
        )
    return True


def disable_routing_plugin() -> bool:
    """Remove the routing plugin from <hermes-home>/plugins/."""
    import shutil

    from common import HERMES_HOME

    dest = HERMES_HOME / "plugins" / PLUGIN_NAME
    if not dest.exists():
        print("Routing plugin is not installed.")
        return True

    # Drop it from the plugins.enabled allow-list BEFORE removing the files
    # (cmd_disable resolves the plugin key from the installed directory).
    try:
        from hermes_cli.plugins_cmd import cmd_disable
        cmd_disable(PLUGIN_NAME)
    except SystemExit:
        pass
    except Exception as e:
        logger.debug("cmd_disable unavailable: %s", e)

    shutil.rmtree(dest)
    print(f"Routing plugin removed from {dest}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test adapter routing")
    parser.add_argument(
        "prompt", nargs="?", default=None,
        help="Prompt to route, or 'enable'/'disable' to manage the routing plugin",
    )
    parser.add_argument("--test", action="store_true", help="Run routing diagnostics")
    args = parser.parse_args()

    if args.prompt == "enable":
        raise SystemExit(0 if enable_routing_plugin() else 1)
    if args.prompt == "disable":
        raise SystemExit(0 if disable_routing_plugin() else 1)

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
