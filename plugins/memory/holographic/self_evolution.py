"""Self-Evolution Mechanism -- homeostatic plasticity for memory systems.

Biological neural circuits maintain optimal function through homeostatic
plasticity: when activity drops too low, synaptic strengths increase; when
it rises too high, they decrease.  This module applies the same principle
to the Hermes memory system.

Three feedback loops, inspired by cortical homeostasis:
  1. Retrieval hit rate  -- are searches finding relevant facts?
  2. Consolidation yield -- is dreaming/consolidation producing schemas?
  3. Prediction accuracy -- are hypotheses being verified?

When any metric drifts below threshold, parameters are adjusted to restore
optimal function.  All adjustments are logged for auditability.

Scientific basis:
- Turrigiano (1999) Neuron: homeostatic plasticity in cortical circuits
- Turrigiano & Nelson (2004) Nature Reviews: synaptic scaling
- Davis & Bhatt (2001): metaplasticity and synaptic homeostasis
- Buzsaki (2015): sleep replay maintains memory system equilibrium

All methods are best-effort: exceptions caught and logged, never blocking.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evolution-specific schema
# ---------------------------------------------------------------------------

_EVOLUTION_SCHEMA = """\
CREATE TABLE IF NOT EXISTS evolution_metrics (
    metric_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    period_start        TIMESTAMP,
    period_end          TIMESTAMP,
    retrieval_hit_rate  REAL,
    consolidation_yield REAL,
    prediction_accuracy REAL,
    user_feedback_score REAL,
    overall_health      REAL,
    trend               TEXT,
    timestamp           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS evolution_adjustments (
    adjustment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_id     INTEGER,
    parameter     TEXT NOT NULL,
    old_value     REAL,
    new_value     REAL,
    reason        TEXT,
    timestamp     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_evo_metrics_ts
    ON evolution_metrics(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_evo_adjustments_metric
    ON evolution_adjustments(metric_id);
"""


# ---------------------------------------------------------------------------
# Tuning constants (bounds for parameter clamping)
# ---------------------------------------------------------------------------

_WEIGHT_MIN = 0.01
_WEIGHT_MAX = 0.80
_THRESHOLD_MIN = 0.10
_THRESHOLD_MAX = 0.90
_CONFIDENCE_DECAY_MIN = 0.90
_CONFIDENCE_DECAY_MAX = 0.99


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EvolutionMetrics:
    """Computed health metrics for a single evaluation period."""
    period_start: str = ""
    period_end: str = ""
    retrieval_hit_rate: float = 0.0
    consolidation_yield: float = 0.0
    prediction_accuracy: float = 0.0
    user_feedback_score: float = 0.0
    overall_health: float = 0.0
    trend: str = "stable"
    metric_id: int | None = None


@dataclass
class AdjustmentRecord:
    """A single parameter adjustment made by auto_adjust."""
    parameter: str = ""
    old_value: float = 0.0
    new_value: float = 0.0
    reason: str = ""


# ---------------------------------------------------------------------------
# SelfEvolution
# ---------------------------------------------------------------------------
class SelfEvolution:
    """Homeostatic self-regulation for the memory system.

    Evaluates retrieval, consolidation, and prediction health over configurable
    periods, then automatically adjusts parameters to restore optimal function.

    Usage::

        evolution = SelfEvolution(conn, lock, config)
        evolution.init_tables()

        metrics = evolution.evaluate_period(period_hours=24)
        adjustments = evolution.auto_adjust(metrics)
        report = evolution.get_evolution_report()
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        lock: threading.RLock,
        config: dict | None = None,
        pipeline_conn: sqlite3.Connection | None = None,
    ) -> None:
        self._conn = conn               # memory_store.db (facts, feedback)
        self._pipeline_conn = pipeline_conn or conn  # pipeline_state.db (schemas, consolidation_runs, dream_hypotheses)
        self._lock = lock
        self._config = config or {}
        self._enabled = self._config.get("self_evolution_enabled", False)
        self._period_hours = self._config.get("evolution_period_hours", 24)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_tables(self) -> None:
        """Create evolution tables if they do not exist."""
        try:
            with self._lock:
                self._conn.executescript(_EVOLUTION_SCHEMA)
                self._conn.commit()
        except Exception as e:
            logger.debug("SelfEvolution init_tables failed: %s", e)

    # ------------------------------------------------------------------
    # Metric evaluation
    # ------------------------------------------------------------------

    def evaluate_period(self, period_hours: int = 24) -> EvolutionMetrics:
        """Compute health metrics for the given lookback window.

        Metrics:
          - retrieval_hit_rate:   fraction of retrievals with helpful feedback
          - consolidation_yield:  schemas created per fact consolidated
          - prediction_accuracy:  fraction of verified hypotheses
          - user_feedback_score:  net helpful-minus-unhelpful ratio
          - overall_health:       weighted average of the four metrics
          - trend:                improving / stable / declining vs prior period

        Returns an EvolutionMetrics dataclass.  On any failure returns a
        zero-valued instance so callers never crash.
        """
        metrics = EvolutionMetrics()
        if not self._enabled:
            return metrics

        try:
            now = datetime.now(timezone.utc)
            from datetime import timedelta
            period_start = now - timedelta(hours=period_hours)
            metrics.period_start = period_start.isoformat()
            metrics.period_end = now.isoformat()

            with self._lock:
                # 1. Retrieval hit rate
                metrics.retrieval_hit_rate = self._compute_retrieval_hit_rate(
                    period_start.isoformat())

                # 2. Consolidation yield
                metrics.consolidation_yield = self._compute_consolidation_yield(
                    period_start.isoformat())

                # 3. Prediction accuracy
                metrics.prediction_accuracy = self._compute_prediction_accuracy()

                # 4. User feedback score
                metrics.user_feedback_score = self._compute_user_feedback_score(
                    period_start.isoformat())

                # 5. Overall health (weighted average)
                metrics.overall_health = (
                    0.35 * metrics.retrieval_hit_rate
                    + 0.25 * metrics.consolidation_yield
                    + 0.20 * metrics.prediction_accuracy
                    + 0.20 * metrics.user_feedback_score
                )

                # 6. Trend (compare to prior period)
                metrics.trend = self._compute_trend(metrics.overall_health)

                # 7. Persist
                metrics.metric_id = self._persist_metrics(metrics)

        except Exception as e:
            logger.debug("evaluate_period failed: %s", e)

        return metrics

    def _compute_retrieval_hit_rate(self, since: str) -> float:
        """Fraction of retrieved facts that received helpful feedback.

        Uses helpful_count / retrieval_count across facts updated in window.
        Falls back to 0.5 if no retrieval activity (neutral prior).
        """
        try:
            row = self._conn.execute(
                """
                SELECT
                    COALESCE(SUM(helpful_count), 0) AS helpful,
                    COALESCE(SUM(retrieval_count), 0) AS total
                FROM facts
                WHERE updated_at >= ?
                """,
                (since,),
            ).fetchone()

            if row and row["total"] > 0:
                return min(1.0, row["helpful"] / row["total"])
        except Exception as e:
            logger.debug("_compute_retrieval_hit_rate failed: %s", e)
        return 0.5

    def _compute_consolidation_yield(self, since: str) -> float:
        """Schemas created per consolidation run in the period.

        Checks consolidation_runs table (from MemoryPipeline) and schemas table.
        Falls back to 0.3 if tables unavailable.
        """
        try:
            runs = self._pipeline_conn.execute(
                "SELECT COUNT(*) FROM consolidation_runs "
                "WHERE timestamp >= ?",
                (since,),
            ).fetchone()

            schemas = self._pipeline_conn.execute(
                "SELECT COUNT(*) FROM schemas "
                "WHERE created_at >= ?",
                (since,),
            ).fetchone()

            run_count = runs[0] if runs else 0
            schema_count = schemas[0] if schemas else 0

            if run_count > 0:
                return min(1.0, schema_count / run_count)
        except Exception as e:
            logger.debug("_compute_consolidation_yield failed: %s", e)
        return 0.3

    def _compute_prediction_accuracy(self) -> float:
        """Fraction of dream hypotheses that have been verified.

        Uses dream_hypotheses table (from DreamEngine).
        Falls back to 0.2 if no hypotheses exist.
        """
        try:
            row = self._pipeline_conn.execute(
                """
                SELECT
                    COUNT(*) FILTER (WHERE verified = 1) AS verified,
                    COUNT(*) AS total
                FROM dream_hypotheses
                """,
            ).fetchone()

            if row and row["total"] > 0:
                return row["verified"] / row["total"]
        except Exception as e:
            logger.debug("_compute_prediction_accuracy failed: %s", e)
        return 0.2

    def _compute_user_feedback_score(self, since: str) -> float:
        """Net feedback ratio normalized to [0,1].

        Uses helpful_count and retrieval_count from facts updated in the window.
        Falls back to 0.5 (neutral) if no feedback activity.
        """
        try:
            row = self._conn.execute(
                """
                SELECT
                    COALESCE(SUM(helpful_count), 0) AS helpful,
                    COALESCE(SUM(
                        CASE WHEN trust_score < 0.3 THEN 1 ELSE 0 END
                    ), 0) AS low_trust,
                    COUNT(*) AS total
                FROM facts
                WHERE updated_at >= ?
                """,
                (since,),
            ).fetchone()

            if row and row["total"] > 0:
                helpful_ratio = row["helpful"] / max(1, row["total"])
                penalty = row["low_trust"] / max(1, row["total"])
                score = 0.5 + (helpful_ratio * 0.5) - (penalty * 0.3)
                return max(0.0, min(1.0, score))
        except Exception as e:
            logger.debug("_compute_user_feedback_score failed: %s", e)
        return 0.5

    def _compute_trend(self, current_health: float) -> str:
        """Compare current health to the most recent prior evaluation.

        Returns 'improving', 'stable', or 'declining'.
        """
        try:
            row = self._conn.execute(
                "SELECT overall_health FROM evolution_metrics "
                "ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()

            if row is not None:
                prev = row["overall_health"]
                delta = current_health - prev
                if delta > 0.05:
                    return "improving"
                elif delta < -0.05:
                    return "declining"
        except Exception as e:
            logger.debug("_compute_trend failed: %s", e)
        return "stable"

    def _persist_metrics(self, metrics: EvolutionMetrics) -> int | None:
        """Store computed metrics and return metric_id."""
        try:
            cursor = self._conn.execute(
                "INSERT INTO evolution_metrics "
                "(period_start, period_end, retrieval_hit_rate, "
                "consolidation_yield, prediction_accuracy, "
                "user_feedback_score, overall_health, trend) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    metrics.period_start,
                    metrics.period_end,
                    metrics.retrieval_hit_rate,
                    metrics.consolidation_yield,
                    metrics.prediction_accuracy,
                    metrics.user_feedback_score,
                    metrics.overall_health,
                    metrics.trend,
                ),
            )
            self._conn.commit()
            return cursor.lastrowid
        except Exception as e:
            logger.debug("_persist_metrics failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Automatic parameter adjustment
    # ------------------------------------------------------------------

    def auto_adjust(self, metrics: EvolutionMetrics) -> List[AdjustmentRecord]:
        """Apply homeostatic corrections when metrics drift below thresholds.

        Rules (inspired by Turrigiano 1999):
          - hit_rate < 0.3   -> boost retrieval weights (signal is too weak)
          - yield < 0.1      -> lower consolidation threshold (too selective)
          - accuracy < 0.2   -> reduce confidence decay faster (hypotheses die)
          - trend = declining -> apply a modest global boost

        Returns list of AdjustmentRecords describing what changed.
        All adjustments are logged to evolution_adjustments table.
        """
        adjustments: List[AdjustmentRecord] = []
        if not self._enabled:
            return adjustments

        try:
            with self._lock:
                # Rule 1: low retrieval hit rate -> boost weights
                if metrics.retrieval_hit_rate < 0.3:
                    adj = self._adjust_retrieval_weights(
                        boost=0.05,
                        reason="hit_rate={:.3f} < 0.3".format(metrics.retrieval_hit_rate),
                        metric_id=metrics.metric_id,
                    )
                    if adj:
                        adjustments.extend(adj)

                # Rule 2: low consolidation yield -> lower threshold
                if metrics.consolidation_yield < 0.1:
                    adj = self._adjust_consolidation_threshold(
                        delta=-0.05,
                        reason="yield={:.3f} < 0.1".format(metrics.consolidation_yield),
                        metric_id=metrics.metric_id,
                    )
                    if adj:
                        adjustments.extend(adj)

                # Rule 3: low prediction accuracy -> reduce confidence decay
                if metrics.prediction_accuracy < 0.2:
                    adj = self._adjust_confidence_decay(
                        delta=-0.01,
                        reason="accuracy={:.3f} < 0.2".format(metrics.prediction_accuracy),
                        metric_id=metrics.metric_id,
                    )
                    if adj:
                        adjustments.extend(adj)

                # Rule 4: declining trend -> gentle global boost
                if metrics.trend == "declining" and metrics.overall_health < 0.4:
                    adj = self._adjust_global_boost(
                        boost=0.02,
                        reason="trend=declining, health={:.3f}".format(metrics.overall_health),
                        metric_id=metrics.metric_id,
                    )
                    if adj:
                        adjustments.extend(adj)

                self._conn.commit()

        except Exception as e:
            logger.debug("auto_adjust failed: %s", e)

        return adjustments

    def _adjust_retrieval_weights(
        self, boost: float, reason: str, metric_id: int | None,
    ) -> List[AdjustmentRecord]:
        """Boost fts_weight and hrr_weight in retrieval config."""
        adjustments = []
        try:
            for param in ["fts_weight", "hrr_weight", "embedding_weight"]:
                current = self._get_config_param(param)
                if current is None:
                    continue
                new_val = min(_WEIGHT_MAX, current + boost)
                if new_val != current:
                    self._set_config_param(param, new_val)
                    self._log_adjustment(
                        metric_id, param, current, new_val, reason)
                    adjustments.append(AdjustmentRecord(
                        parameter=param,
                        old_value=current,
                        new_value=new_val,
                        reason=reason,
                    ))
        except Exception as e:
            logger.debug("_adjust_retrieval_weights failed: %s", e)
        return adjustments

    def _adjust_consolidation_threshold(
        self, delta: float, reason: str, metric_id: int | None,
    ) -> List[AdjustmentRecord]:
        """Lower the consolidation salience threshold to be less selective."""
        adjustments = []
        try:
            param = "consolidation_threshold"
            current = self._get_config_param(param)
            if current is None:
                current = 0.5  # default
            new_val = max(_THRESHOLD_MIN, current + delta)
            if new_val != current:
                self._set_config_param(param, new_val)
                self._log_adjustment(metric_id, param, current, new_val, reason)
                adjustments.append(AdjustmentRecord(
                    parameter=param,
                    old_value=current,
                    new_value=new_val,
                    reason=reason,
                ))
        except Exception as e:
            logger.debug("_adjust_consolidation_threshold failed: %s", e)
        return adjustments

    def _adjust_confidence_decay(
        self, delta: float, reason: str, metric_id: int | None,
    ) -> List[AdjustmentRecord]:
        """Adjust hypothesis confidence decay rate.

        Lower decay multiplier means confidence fades faster, preventing
        stale hypotheses from persisting.
        """
        adjustments = []
        try:
            param = "confidence_decay_rate"
            current = self._get_config_param(param)
            if current is None:
                current = 0.97  # default
            new_val = max(_CONFIDENCE_DECAY_MIN,
                         min(_CONFIDENCE_DECAY_MAX, current + delta))
            if new_val != current:
                self._set_config_param(param, new_val)
                self._log_adjustment(metric_id, param, current, new_val, reason)
                adjustments.append(AdjustmentRecord(
                    parameter=param,
                    old_value=current,
                    new_value=new_val,
                    reason=reason,
                ))
        except Exception as e:
            logger.debug("_adjust_confidence_decay failed: %s", e)
        return adjustments

    def _adjust_global_boost(
        self, boost: float, reason: str, metric_id: int | None,
    ) -> List[AdjustmentRecord]:
        """Apply a modest boost to trust default when system is declining."""
        adjustments = []
        try:
            param = "default_trust"
            current = self._get_config_param(param)
            if current is None:
                current = 0.5
            new_val = min(_WEIGHT_MAX, current + boost)
            if new_val != current:
                self._set_config_param(param, new_val)
                self._log_adjustment(metric_id, param, current, new_val, reason)
                adjustments.append(AdjustmentRecord(
                    parameter=param,
                    old_value=current,
                    new_value=new_val,
                    reason=reason,
                ))
        except Exception as e:
            logger.debug("_adjust_global_boost failed: %s", e)
        return adjustments

    # ------------------------------------------------------------------
    # Config parameter helpers (adjustments table IS the config history)
    # ------------------------------------------------------------------

    def _get_config_param(self, name: str) -> float | None:
        """Read the most recent value of a parameter from adjustments."""
        try:
            row = self._conn.execute(
                "SELECT new_value FROM evolution_adjustments "
                "WHERE parameter = ? ORDER BY timestamp DESC LIMIT 1",
                (name,),
            ).fetchone()
            if row is not None:
                return float(row["new_value"])
        except Exception:
            pass
        return self._config.get(name)

    def _set_config_param(self, name: str, value: float) -> None:
        """Update in-memory config for immediate effect."""
        self._config[name] = value

    def _log_adjustment(
        self,
        metric_id: int | None,
        parameter: str,
        old_value: float,
        new_value: float,
        reason: str,
    ) -> None:
        """Persist an adjustment to the evolution_adjustments table."""
        try:
            self._conn.execute(
                "INSERT INTO evolution_adjustments "
                "(metric_id, parameter, old_value, new_value, reason) "
                "VALUES (?, ?, ?, ?, ?)",
                (metric_id, parameter, old_value, new_value, reason),
            )
        except Exception as e:
            logger.debug("_log_adjustment failed: %s", e)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_evolution_report(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Full history of metrics and their associated adjustments.

        Returns a list of dicts, each containing a metric row and its
        adjustments, ordered by most recent first.
        """
        try:
            with self._lock:
                metrics_rows = self._conn.execute(
                    "SELECT * FROM evolution_metrics "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()

                report = []
                for mrow in metrics_rows:
                    metric = dict(mrow)
                    metric_id = metric.get("metric_id")
                    adjustments = []
                    if metric_id is not None:
                        adj_rows = self._conn.execute(
                            "SELECT * FROM evolution_adjustments "
                            "WHERE metric_id = ? ORDER BY timestamp",
                            (metric_id,),
                        ).fetchall()
                        adjustments = [dict(r) for r in adj_rows]
                    metric["adjustments"] = adjustments
                    report.append(metric)

                return report

        except Exception as e:
            logger.debug("get_evolution_report failed: %s", e)
            return []

    def get_parameter_history(
        self, parameter: str | None = None, limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Timeline of all parameter changes.

        If *parameter* is given, filter to that parameter name only.
        Returns list of dicts ordered by timestamp descending.
        """
        try:
            with self._lock:
                if parameter:
                    rows = self._conn.execute(
                        "SELECT * FROM evolution_adjustments "
                        "WHERE parameter = ? ORDER BY timestamp DESC LIMIT ?",
                        (parameter, limit),
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        "SELECT * FROM evolution_adjustments "
                        "ORDER BY timestamp DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
                return [dict(r) for r in rows]

        except Exception as e:
            logger.debug("get_parameter_history failed: %s", e)
            return []

    def get_latest_metrics(self) -> EvolutionMetrics | None:
        """Return the most recent EvolutionMetrics evaluation, or None."""
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT * FROM evolution_metrics "
                    "ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
                if row is None:
                    return None
                return EvolutionMetrics(
                    period_start=row["period_start"] or "",
                    period_end=row["period_end"] or "",
                    retrieval_hit_rate=row["retrieval_hit_rate"] or 0.0,
                    consolidation_yield=row["consolidation_yield"] or 0.0,
                    prediction_accuracy=row["prediction_accuracy"] or 0.0,
                    user_feedback_score=row["user_feedback_score"] or 0.0,
                    overall_health=row["overall_health"] or 0.0,
                    trend=row["trend"] or "stable",
                    metric_id=row["metric_id"],
                )
        except Exception as e:
            logger.debug("get_latest_metrics failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Convenience: full cycle (evaluate + adjust)
    # ------------------------------------------------------------------

    def run_evolution_cycle(
        self, period_hours: int | None = None,
    ) -> tuple[EvolutionMetrics, List[AdjustmentRecord]]:
        """Run a full evolution cycle: evaluate then auto-adjust.

        Intended to be called during sleep cycles by MemoryPipeline.
        Returns (metrics, adjustments) tuple.
        """
        hours = period_hours or self._period_hours
        metrics = self.evaluate_period(period_hours=hours)
        adjustments = self.auto_adjust(metrics)
        if adjustments:
            logger.info(
                "Self-evolution cycle: health=%.3f trend=%s adjustments=%d",
                metrics.overall_health, metrics.trend, len(adjustments),
            )
        return metrics, adjustments
