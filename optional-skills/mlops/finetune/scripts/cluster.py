#!/usr/bin/env python3
"""
Domain discovery via HDBSCAN clustering for the finetune pipeline.

Embeds user turns from scored sessions, clusters them to discover usage domains,
and manages cluster lifecycle (maturity, labeling, re-clustering).

Usage:
    python cluster.py [--min-cluster-size 30] [--embedding-model all-MiniLM-L6-v2]
"""

import argparse
import hashlib
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from common import (
    SCORED_DIR, CLUSTERS_DIR, CLUSTER_STATE_PATH,
    ensure_dirs, load_config, load_json, save_json, read_jsonl, logger,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _centroid_hash(centroid: np.ndarray) -> str:
    """Generate a content-addressed cluster ID from centroid embedding."""
    h = hashlib.sha256(centroid.tobytes()).hexdigest()[:6]
    return f"c-{h}"


def _cluster_maturity(good_turn_count: int) -> str:
    """Determine cluster maturity stage from good-bucket turn count."""
    if good_turn_count < 50:
        return "embryonic"
    elif good_turn_count < 150:
        return "nascent"
    elif good_turn_count < 500:
        return "established"
    else:
        return "mature"


class DomainClusterer:
    """Discover usage domains from scored session data."""

    def __init__(self, config: dict = None):
        cfg = config or load_config().get("clustering", {})
        self.embedding_model_name = cfg.get("embedding_model", "all-MiniLM-L6-v2")
        self.min_cluster_size = cfg.get("min_cluster_size", 30)
        self.confidence_threshold = cfg.get("confidence_threshold", 0.6)
        self._embed_model = None
        self._embeddings_cache: Dict[str, np.ndarray] = {}

    def _get_embed_model(self):
        """Lazy-load the sentence-transformers model."""
        if self._embed_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for clustering. "
                    "Install with: pip install sentence-transformers>=2.2"
                )
            logger.info("Loading embedding model: %s", self.embedding_model_name)
            self._embed_model = SentenceTransformer(self.embedding_model_name)
        return self._embed_model

    def _extract_user_text(self, session: Dict) -> str:
        """Extract concatenated user turns for embedding."""
        turns = session.get("turns", [])
        user_texts = [
            t.get("content", "") or ""
            for t in turns
            if t.get("role") == "user"
        ]
        return " ".join(user_texts).strip()

    def _embed_sessions(self, sessions: List[Dict]) -> Tuple[np.ndarray, List[str]]:
        """Embed all sessions, returning (embeddings_matrix, session_ids)."""
        model = self._get_embed_model()

        texts = []
        session_ids = []
        for s in sessions:
            text = self._extract_user_text(s)
            if text:
                texts.append(text)
                session_ids.append(s.get("session_id", ""))

        if not texts:
            return np.array([]), []

        logger.info("Embedding %d sessions...", len(texts))
        embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        return np.array(embeddings), session_ids

    def _run_hdbscan(self, embeddings: np.ndarray) -> np.ndarray:
        """Run HDBSCAN clustering, returning cluster labels."""
        try:
            import hdbscan
        except ImportError:
            raise ImportError(
                "hdbscan is required for clustering. "
                "Install with: pip install hdbscan>=0.8.33"
            )

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=5,
            metric="euclidean",  # Normalized embeddings → equivalent to cosine
            cluster_selection_method="eom",
            prediction_data=True,
        )

        labels = clusterer.fit_predict(embeddings)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        logger.info(
            "HDBSCAN: %d clusters, %d noise points (of %d total)",
            n_clusters, n_noise, len(labels),
        )
        return labels

    def _compute_centroids(
        self, embeddings: np.ndarray, labels: np.ndarray,
    ) -> Dict[int, np.ndarray]:
        """Compute centroid for each cluster."""
        centroids = {}
        for label in set(labels):
            if label == -1:
                continue
            mask = labels == label
            centroids[label] = embeddings[mask].mean(axis=0)
        return centroids

    def _match_previous_clusters(
        self, centroids: Dict[int, np.ndarray],
        prev_state: Dict,
    ) -> Dict[int, str]:
        """Match new clusters to previous ones by centroid similarity."""
        prev_centroids = prev_state.get("centroids", {})
        mapping = {}

        for label, centroid in centroids.items():
            best_sim = -1
            best_id = None

            for prev_id, prev_vec in prev_centroids.items():
                prev_arr = np.array(prev_vec)
                sim = float(np.dot(centroid, prev_arr) / (
                    np.linalg.norm(centroid) * np.linalg.norm(prev_arr) + 1e-8
                ))
                if sim > best_sim:
                    best_sim = sim
                    best_id = prev_id

            if best_sim > 0.9 and best_id:
                mapping[label] = best_id
            else:
                mapping[label] = _centroid_hash(centroid)

        return mapping

    def _generate_labels(
        self, sessions: List[Dict], cluster_assignments: Dict[str, str],
    ) -> Dict[str, str]:
        """Generate human-readable labels from TF-IDF top terms per cluster."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            # Fall back to simple word frequency
            return self._generate_labels_simple(sessions, cluster_assignments)

        # Group texts by cluster
        cluster_texts: Dict[str, List[str]] = {}
        for s in sessions:
            sid = s.get("session_id", "")
            cid = cluster_assignments.get(sid)
            if cid and cid != "_general":
                text = self._extract_user_text(s)
                cluster_texts.setdefault(cid, []).append(text)

        labels = {}
        for cid, texts in cluster_texts.items():
            corpus = " ".join(texts)
            try:
                vectorizer = TfidfVectorizer(max_features=5, stop_words="english")
                vectorizer.fit_transform([corpus])
                terms = vectorizer.get_feature_names_out()
                labels[cid] = "auto:" + "-".join(terms[:3])
            except Exception:
                labels[cid] = f"auto:cluster-{cid}"

        return labels

    def _generate_labels_simple(
        self, sessions: List[Dict], cluster_assignments: Dict[str, str],
    ) -> Dict[str, str]:
        """Simple frequency-based labeling fallback."""
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "to", "of", "in", "for", "on", "with", "at", "by", "from",
                     "it", "this", "that", "i", "you", "we", "they", "my", "me",
                     "can", "do", "how", "what", "and", "or", "but", "not", "so"}

        cluster_texts: Dict[str, List[str]] = {}
        for s in sessions:
            sid = s.get("session_id", "")
            cid = cluster_assignments.get(sid)
            if cid and cid != "_general":
                text = self._extract_user_text(s)
                cluster_texts.setdefault(cid, []).append(text)

        labels = {}
        for cid, texts in cluster_texts.items():
            words = " ".join(texts).lower().split()
            words = [w for w in words if len(w) > 3 and w not in stopwords]
            top = [w for w, _ in Counter(words).most_common(3)]
            labels[cid] = "auto:" + "-".join(top) if top else f"auto:cluster-{cid}"

        return labels

    def cluster(
        self, sessions: List[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Run the full clustering pipeline.

        Returns cluster state dict with assignments, centroids, labels, maturity.
        """
        ensure_dirs()

        if sessions is None:
            sessions = []
            for path in sorted(SCORED_DIR.glob("scored_*.jsonl")):
                sessions.extend(read_jsonl(path))

        if not sessions:
            logger.warning("No sessions to cluster.")
            return {}

        # Load previous state for cluster ID continuity
        prev_state = load_json(CLUSTER_STATE_PATH, {})

        # Graceful fallback: if clustering deps aren't installed, route
        # everything to the _general bucket instead of crashing. This lets
        # users run the rest of the pipeline (extract → score → train) on
        # the _general adapter without committing to the heavier embedding
        # stack. Train and route still work; only domain discovery is skipped.
        try:
            embeddings, session_ids = self._embed_sessions(sessions)
        except ImportError as e:
            logger.warning(
                "Clustering deps unavailable (%s). Routing all sessions to "
                "_general — install sentence-transformers, hdbscan, and "
                "scikit-learn to enable domain discovery.",
                e,
            )
            return self._fallback_to_general(sessions)
        if len(session_ids) == 0:
            logger.warning("No embeddable sessions found.")
            return {}

        # Build session lookup
        session_map = {s.get("session_id"): s for s in sessions}

        # Cluster
        if len(embeddings) < self.min_cluster_size:
            logger.info(
                "Only %d sessions — below min_cluster_size (%d). All go to _general.",
                len(embeddings), self.min_cluster_size,
            )
            labels = np.full(len(embeddings), -1)
        else:
            labels = self._run_hdbscan(embeddings)

        centroids = self._compute_centroids(embeddings, labels)

        # Match to previous cluster IDs
        id_mapping = self._match_previous_clusters(centroids, prev_state)

        # Build assignments: session_id -> cluster_id
        assignments: Dict[str, str] = {}
        for i, sid in enumerate(session_ids):
            label = int(labels[i])
            if label == -1:
                assignments[sid] = "_general"
            else:
                assignments[sid] = id_mapping.get(label, _centroid_hash(centroids[label]))

        # Generate labels
        text_labels = self._generate_labels(sessions, assignments)

        # Compute maturity per cluster
        cluster_info: Dict[str, Dict] = {}
        for cid in set(assignments.values()):
            cluster_sessions = [
                session_map[sid] for sid, c in assignments.items()
                if c == cid and sid in session_map
            ]
            good_turns = sum(
                len([t for t in s.get("turns", []) if t.get("role") == "assistant"])
                for s in cluster_sessions
                if s.get("scoring", {}).get("bucket") == "good"
            )
            cluster_info[cid] = {
                "session_count": len(cluster_sessions),
                "good_turns": good_turns,
                "maturity": _cluster_maturity(good_turns),
                "label": text_labels.get(cid, f"auto:{cid}"),
            }

        # Save cluster state
        state = {
            "algorithm": "hdbscan",
            "min_cluster_size": self.min_cluster_size,
            "embedding_model": self.embedding_model_name,
            "last_run": datetime.now().isoformat(),
            "total_sessions": len(sessions),
            "clusters_active": len([c for c in cluster_info if c != "_general"]),
            "noise_sessions": sum(1 for a in assignments.values() if a == "_general"),
            "centroids": {
                id_mapping.get(k, _centroid_hash(v)): v.tolist()
                for k, v in centroids.items()
            },
            "clusters": cluster_info,
            "assignments": assignments,
        }
        save_json(CLUSTER_STATE_PATH, state)
        logger.info("Cluster state saved to %s", CLUSTER_STATE_PATH)

        # Write per-cluster data splits
        from format import TrainingFormatter
        formatter = TrainingFormatter()
        good_threshold = load_config().get("scoring", {}).get("thresholds", {}).get("neutral", 0.4)

        for cid in set(assignments.values()):
            cluster_sessions = [
                session_map[sid] for sid, c in assignments.items()
                if c == cid and sid in session_map
            ]
            formatter.format_for_cluster(
                cluster_sessions, cid, min_score=good_threshold,
            )

        return state

    def _fallback_to_general(self, sessions: List[Dict]) -> Dict[str, Any]:
        """
        Fallback when clustering deps are unavailable.

        Routes every session to the _general bucket, writes a stub cluster
        state file (no centroids — routing falls back to base model), and
        formats the training data for the _general adapter only.
        """
        from format import TrainingFormatter

        assignments = {s.get("session_id"): "_general" for s in sessions}
        good_turns = sum(
            len([t for t in s.get("turns", []) if t.get("role") == "assistant"])
            for s in sessions
            if s.get("scoring", {}).get("bucket") == "good"
        )

        state = {
            "algorithm": "fallback-no-clustering",
            "min_cluster_size": self.min_cluster_size,
            "embedding_model": "(unavailable)",
            "last_run": datetime.now().isoformat(),
            "total_sessions": len(sessions),
            "clusters_active": 0,
            "noise_sessions": len(sessions),
            "centroids": {},
            "clusters": {
                "_general": {
                    "session_count": len(sessions),
                    "good_turns": good_turns,
                    "maturity": _cluster_maturity(good_turns),
                    "label": "general (clustering disabled)",
                }
            },
            "assignments": assignments,
        }
        save_json(CLUSTER_STATE_PATH, state)

        # Format _general training data
        good_threshold = load_config().get("scoring", {}).get("thresholds", {}).get("neutral", 0.4)
        TrainingFormatter().format_for_cluster(
            sessions, "_general", min_score=good_threshold,
        )

        logger.info(
            "Fallback complete: %d sessions routed to _general (%d good turns)",
            len(sessions), good_turns,
        )
        return state


def main():
    parser = argparse.ArgumentParser(description="Discover usage domains via clustering")
    parser.add_argument("--min-cluster-size", type=int, default=None,
                        help="HDBSCAN min_cluster_size (default: from config)")
    parser.add_argument("--embedding-model", type=str, default=None,
                        help="Sentence-transformers model name")
    args = parser.parse_args()

    config = load_config().get("clustering", {})
    if args.min_cluster_size is not None:
        config["min_cluster_size"] = args.min_cluster_size
    if args.embedding_model:
        config["embedding_model"] = args.embedding_model

    clusterer = DomainClusterer(config=config)
    state = clusterer.cluster()

    if state:
        print(f"\nClusters found: {state.get('clusters_active', 0)}")
        print(f"Noise sessions (→ _general): {state.get('noise_sessions', 0)}")
        for cid, info in state.get("clusters", {}).items():
            print(f"  {cid}: {info['label']} "
                  f"({info['session_count']} sessions, "
                  f"{info['good_turns']} good turns, "
                  f"maturity={info['maturity']})")
    else:
        print("No clusters found. Need more scored sessions.")


if __name__ == "__main__":
    main()
