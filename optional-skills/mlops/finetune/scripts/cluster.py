#!/usr/bin/env python3
"""
Domain discovery via HDBSCAN clustering for the finetune pipeline.

Embeds user turns from scored sessions, clusters them to discover usage domains,
and manages cluster lifecycle (maturity, labeling, re-clustering).

Usage:
    python cluster.py [--min-cluster-size 30] [--embedding-model all-MiniLM-L6-v2]
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# numpy is only needed on the real clustering path; the _general fallback
# must work without any of the optional deps (numpy, sentence-transformers,
# hdbscan), so the import failure is deferred to _get_embed_model.
try:
    import numpy as np
except ImportError:
    np = None

from common import (
    SCORED_DIR, CLUSTERS_DIR, CLUSTER_STATE_PATH,
    ensure_dirs, load_config, load_json, save_json,
    content_to_text, load_records_dedup, logger,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _centroid_hash(centroid: np.ndarray) -> str:
    """Generate a content-addressed cluster ID from centroid embedding."""
    h = hashlib.sha256(centroid.tobytes()).hexdigest()[:6]
    return f"c-{h}"


def _count_trainable_turns(sessions: List[Dict], min_turn_score: float) -> int:
    """Count assistant turns that would survive per-turn training filtering.

    A turn counts when its effective per-turn score (scoring.turn_scores,
    which already includes retro overrides) meets `min_turn_score` and it is
    not explicitly labeled bad (scoring.bad_turn_indices). Sessions without
    per-turn scores fall back to the composite score for every assistant
    turn — mirroring format.py::extract_training_turns.
    """
    total = 0
    for s in sessions:
        scoring = s.get("scoring", {})
        bad_indices = set(scoring.get("bad_turn_indices") or [])
        turn_scores = scoring.get("turn_scores") or []
        if turn_scores:
            for entry in turn_scores:
                if not (isinstance(entry, (list, tuple)) and len(entry) == 2):
                    continue
                try:
                    idx, score = int(entry[0]), float(entry[1])
                except (TypeError, ValueError):
                    continue
                if idx not in bad_indices and score >= min_turn_score:
                    total += 1
        else:
            composite = scoring.get("composite_score", 0.5)
            if composite >= min_turn_score:
                total += sum(
                    1 for idx, t in enumerate(s.get("turns", []))
                    if t.get("role") == "assistant" and idx not in bad_indices
                )
    return total


def _cluster_maturity(good_turn_count: int) -> str:
    """Determine cluster maturity stage from trainable-turn count."""
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

    def _get_embed_model(self):
        """Lazy-load the sentence-transformers model.

        Pinned to CPU regardless of CUDA availability. The clustering step
        only embeds at most a few thousand short text snippets per run, so
        CPU is fast enough (~30 seconds for 200 sessions on a modest
        machine), and pinning to CPU avoids competing with a running
        llama-server for GPU memory. With CUDA torch installed (the
        default after axolotl is added), sentence-transformers would
        otherwise auto-select cuda:0 and OOM the embedding load against
        an active inference server.
        """
        if self._embed_model is None:
            if np is None:
                raise ImportError(
                    "numpy is required for clustering. "
                    "Install with: pip install numpy"
                )
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for clustering. "
                    "Install with: pip install sentence-transformers>=2.2"
                )
            logger.info("Loading embedding model: %s (device=cpu)", self.embedding_model_name)
            self._embed_model = SentenceTransformer(
                self.embedding_model_name,
                device="cpu",
            )
        return self._embed_model

    def _extract_user_text(self, session: Dict) -> str:
        """Extract concatenated user turns for embedding."""
        turns = session.get("turns", [])
        user_texts = [
            content_to_text(t.get("content"))
            for t in turns
            if t.get("role") == "user"
        ]
        return " ".join(t for t in user_texts if t).strip()

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
        """Match new clusters to previous ones by centroid similarity.

        Assignment is greedy by descending similarity and strictly one-to-one:
        a previous cluster ID is consumed by at most one new cluster, so two
        new clusters can never share an ID (which would silently drop one of
        them via key collision downstream).

        If the embedding dimension changed since the previous run (a
        different clustering.embedding_model), the previous centroids are
        incomparable — warn and reset them instead of crashing on the
        similarity math. Every cluster then gets a fresh ID.
        """
        prev_centroids = prev_state.get("centroids", {})

        if centroids and prev_centroids:
            new_dim = len(next(iter(centroids.values())))
            prev_dim = len(next(iter(prev_centroids.values())))
            if new_dim != prev_dim:
                logger.warning(
                    "Embedding dimension changed (%d -> %d) — "
                    "clustering.embedding_model was likely switched "
                    "(previous run used %r). Resetting previous cluster "
                    "state: cluster IDs and adapter lineage will not carry "
                    "over.",
                    prev_dim, new_dim,
                    prev_state.get("embedding_model", "unknown"),
                )
                prev_centroids = {}

        # All candidate (similarity, label, prev_id) pairs above threshold
        candidates = []
        for label, centroid in centroids.items():
            for prev_id, prev_vec in prev_centroids.items():
                prev_arr = np.array(prev_vec)
                sim = float(np.dot(centroid, prev_arr) / (
                    np.linalg.norm(centroid) * np.linalg.norm(prev_arr) + 1e-8
                ))
                if sim > 0.9:
                    candidates.append((sim, label, prev_id))

        candidates.sort(key=lambda c: c[0], reverse=True)

        mapping: Dict[int, str] = {}
        used_ids = set()
        for sim, label, prev_id in candidates:
            if label in mapping or prev_id in used_ids:
                continue
            mapping[label] = prev_id
            used_ids.add(prev_id)

        # Unmatched clusters get fresh content-addressed IDs (collision-guarded)
        for label, centroid in centroids.items():
            if label in mapping:
                continue
            cid = _centroid_hash(centroid)
            if cid in used_ids:
                cid = f"{cid}-{label}"
            mapping[label] = cid
            used_ids.add(cid)

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
            # Dedupe by session_id — repeated score runs write full
            # snapshots; keep each session's record from the newest one.
            sessions = load_records_dedup(SCORED_DIR, "scored_*.jsonl")

        if not sessions:
            logger.warning("No sessions to cluster.")
            return {}

        # Load previous state for cluster ID continuity
        prev_state = load_json(CLUSTER_STATE_PATH, {})

        # Graceful fallback: if clustering deps aren't installed, route
        # everything to the _general bucket instead of crashing. This lets
        # users run the rest of the pipeline (extract → score → train) on
        # the _general adapter without committing to the heavier embedding
        # stack. Train and route still work; only domain discovery is
        # skipped. The try covers both _embed_sessions (numpy /
        # sentence-transformers) and _run_hdbscan (hdbscan) — any of the
        # three deps may be missing independently.
        try:
            embeddings, session_ids = self._embed_sessions(sessions)
            if len(session_ids) == 0:
                logger.warning("No embeddable sessions found.")
                return {}

            if len(embeddings) < self.min_cluster_size:
                logger.info(
                    "Only %d sessions — below min_cluster_size (%d). All go to _general.",
                    len(embeddings), self.min_cluster_size,
                )
                labels = np.full(len(embeddings), -1)
            else:
                labels = self._run_hdbscan(embeddings)
        except ImportError as e:
            logger.warning(
                "Clustering deps unavailable (%s). Routing all sessions to "
                "_general — install sentence-transformers, hdbscan, and "
                "scikit-learn to enable domain discovery.",
                e,
            )
            return self._fallback_to_general(sessions)

        # Build session lookup
        session_map = {s.get("session_id"): s for s in sessions}

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

        # Trainability threshold — the same per-turn cutoff format.py uses
        min_turn_score = float(
            load_config().get("training", {}).get("min_turn_score", 0.7)
        )

        # Compute maturity per cluster from *trainable* turns (per-turn
        # filtered, including retro overrides) — not raw assistant-turn
        # counts of good-bucket sessions.
        cluster_info: Dict[str, Dict] = {}
        for cid in set(assignments.values()):
            cluster_sessions = [
                session_map[sid] for sid, c in assignments.items()
                if c == cid and sid in session_map
            ]
            good_turns = _count_trainable_turns(cluster_sessions, min_turn_score)
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
            # id_mapping is total and one-to-one (see _match_previous_clusters)
            # so this can never collide keys and drop a centroid.
            "centroids": {
                id_mapping[k]: v.tolist() for k, v in centroids.items()
            },
            "clusters": cluster_info,
            "assignments": assignments,
        }
        save_json(CLUSTER_STATE_PATH, state)
        logger.info("Cluster state saved to %s", CLUSTER_STATE_PATH)

        # Write per-cluster data splits, filtered at the config training
        # threshold (training.min_turn_score), NOT the neutral scoring
        # threshold — otherwise mediocre turns leak into training data.
        from format import TrainingFormatter
        formatter = TrainingFormatter()

        for cid in set(assignments.values()):
            cluster_sessions = [
                session_map[sid] for sid, c in assignments.items()
                if c == cid and sid in session_map
            ]
            formatter.format_for_cluster(
                cluster_sessions, cid, min_score=min_turn_score,
            )

        self._truncate_stale_cluster_splits(
            formatter, prev_state, set(assignments.values()),
        )

        return state

    def _truncate_stale_cluster_splits(
        self, formatter, prev_state: Dict, active_ids: set,
    ) -> None:
        """Truncate train/eval splits of clusters that dissolved this run.

        A cluster present in the previous state but absent from the new
        assignments no longer exists; leaving its clusters/<cid>/train.jsonl
        behind would let a later train run pick up stale data. Formatting an
        empty session list reuses format.py's truncate-on-empty behavior.
        """
        for cid in set(prev_state.get("clusters", {})) - set(active_ids):
            if (CLUSTERS_DIR / cid).exists():
                logger.info(
                    "Cluster %s dissolved — truncating its stale train/eval split.",
                    cid,
                )
                formatter.format_for_cluster([], cid)

    def _fallback_to_general(self, sessions: List[Dict]) -> Dict[str, Any]:
        """
        Fallback when clustering deps are unavailable.

        Routes every session to the _general bucket, writes a stub cluster
        state file (no centroids — routing falls back to base model), and
        formats the training data for the _general adapter only.
        """
        from format import TrainingFormatter

        # Read the previous state BEFORE overwriting it so clusters that
        # existed under the full pipeline get their stale splits truncated.
        prev_state = load_json(CLUSTER_STATE_PATH, {})

        assignments = {s.get("session_id"): "_general" for s in sessions}
        min_turn_score = float(
            load_config().get("training", {}).get("min_turn_score", 0.7)
        )
        good_turns = _count_trainable_turns(sessions, min_turn_score)

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

        # Format _general training data at the config training threshold
        formatter = TrainingFormatter()
        formatter.format_for_cluster(
            sessions, "_general", min_score=min_turn_score,
        )
        self._truncate_stale_cluster_splits(formatter, prev_state, {"_general"})

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
