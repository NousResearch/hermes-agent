"""
cosmos Health Analysis Engine

Advanced health data analysis including trend detection, anomaly detection,
correlation analysis, and wellness score calculation.
"""

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum

from .models import (
    MetricType,
    HealthMetricReading,
    DailySummary,
    HealthAlert,
    AlertSeverity,
    HealthRecommendation,
)

logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """Direction of a metric trend."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    FLUCTUATING = "fluctuating"


@dataclass
class TrendAnalysis:
    """Result of trend analysis for a metric."""
    metric_type: MetricType
    direction: TrendDirection
    change_percent: float
    period_days: int
    start_avg: float
    end_avg: float
    confidence: float
    description: str

    def to_dict(self) -> dict:
        return {
            "metric_type": self.metric_type.value,
            "direction": self.direction.value,
            "change_percent": round(self.change_percent, 2),
            "period_days": self.period_days,
            "start_avg": round(self.start_avg, 2),
            "end_avg": round(self.end_avg, 2),
            "confidence": round(self.confidence, 2),
            "description": self.description,
        }


@dataclass
class AnomalyDetection:
    """Result of anomaly detection."""
    reading: HealthMetricReading
    z_score: float
    expected_value: float
    deviation_percent: float
    severity: AlertSeverity

    def to_dict(self) -> dict:
        return {
            "metric_type": self.reading.metric_type.value,
            "value": self.reading.value,
            "timestamp": self.reading.timestamp.isoformat(),
            "z_score": round(self.z_score, 2),
            "expected_value": round(self.expected_value, 2),
            "deviation_percent": round(self.deviation_percent, 2),
            "severity": self.severity.value,
        }


@dataclass
class CorrelationResult:
    """Result of correlation analysis between two metrics."""
    metric_a: MetricType
    metric_b: MetricType
    correlation: float  # -1 to 1
    significance: float  # p-value equivalent
    sample_size: int
    description: str

    def to_dict(self) -> dict:
        return {
            "metric_a": self.metric_a.value,
            "metric_b": self.metric_b.value,
            "correlation": round(self.correlation, 3),
            "significance": round(self.significance, 3),
            "sample_size": self.sample_size,
            "description": self.description,
        }


@dataclass
class HealthInsight:
    """A natural language health insight."""
    category: str  # activity, sleep, recovery, nutrition
    title: str
    message: str
    priority: int  # 1-5
    related_metrics: List[str]
    actionable: bool

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "title": self.title,
            "message": self.message,
            "priority": self.priority,
            "related_metrics": self.related_metrics,
            "actionable": self.actionable,
        }


@dataclass
class WellnessScore:
    """Composite wellness score with breakdown."""
    overall: int  # 0-100
    activity_score: int
    sleep_score: int
    recovery_score: int
    heart_health_score: int
    stress_score: int
    components: Dict[str, int] = field(default_factory=dict)
    factors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "overall": self.overall,
            "activity_score": self.activity_score,
            "sleep_score": self.sleep_score,
            "recovery_score": self.recovery_score,
            "heart_health_score": self.heart_health_score,
            "stress_score": self.stress_score,
            "components": self.components,
            "factors": self.factors,
        }


class HealthAnalysisEngine:
    """
    Comprehensive health data analysis engine.

    Provides:
    - Trend detection over configurable time periods
    - Z-score based anomaly detection
    - Cross-metric correlation analysis
    - Natural language insight generation
    - Composite wellness score calculation
    """

    # Normal ranges for metrics (for anomaly detection)
    NORMAL_RANGES = {
        MetricType.HR: (50, 100),
        MetricType.HRV: (20, 120),
        MetricType.SPO2: (94, 100),
        MetricType.BP_SYSTOLIC: (90, 140),
        MetricType.BP_DIASTOLIC: (60, 90),
        MetricType.STRESS: (0, 70),
        MetricType.SLEEP_DURATION: (6, 9),
        MetricType.SLEEP_SCORE: (50, 100),
        MetricType.RECOVERY_SCORE: (30, 100),
        MetricType.BODY_FAT: (10, 30),
    }

    # Ideal targets for wellness scoring
    WELLNESS_TARGETS = {
        "steps": 10000,
        "active_minutes": 60,
        "sleep_hours": 8.0,
        "sleep_score": 85,
        "recovery_score": 75,
        "resting_hr": 60,
        "hrv": 50,
        "stress": 30,
    }

    def __init__(self):
        self._readings_cache: Dict[MetricType, List[HealthMetricReading]] = {}
        self._summaries_cache: List[DailySummary] = []

    def load_data(
        self,
        readings: List[HealthMetricReading],
        summaries: List[DailySummary],
    ):
        """Load data for analysis."""
        # Group readings by metric type
        self._readings_cache.clear()
        for reading in readings:
            if reading.metric_type not in self._readings_cache:
                self._readings_cache[reading.metric_type] = []
            self._readings_cache[reading.metric_type].append(reading)

        # Sort by timestamp
        for metric_type in self._readings_cache:
            self._readings_cache[metric_type].sort(key=lambda r: r.timestamp)

        self._summaries_cache = sorted(summaries, key=lambda s: s.date)

    # ============================================
    # Trend Analysis
    # ============================================

    def analyze_trends(
        self,
        metric_types: Optional[List[MetricType]] = None,
        days: int = 7,
    ) -> List[TrendAnalysis]:
        """
        Analyze trends for specified metrics over a time period.

        Args:
            metric_types: Metrics to analyze (None = all available)
            days: Number of days to analyze (7, 14, or 30)

        Returns:
            List of TrendAnalysis results
        """
        results = []
        target_metrics = metric_types or list(self._readings_cache.keys())

        for metric_type in target_metrics:
            if metric_type not in self._readings_cache:
                continue

            readings = self._readings_cache[metric_type]
            if len(readings) < 5:  # Need minimum data points
                continue

            # Filter to time period
            cutoff = datetime.now() - timedelta(days=days)
            period_readings = [r for r in readings if r.timestamp >= cutoff]

            if len(period_readings) < 5:
                continue

            trend = self._calculate_trend(metric_type, period_readings, days)
            if trend:
                results.append(trend)

        return results

    def _calculate_trend(
        self,
        metric_type: MetricType,
        readings: List[HealthMetricReading],
        days: int,
    ) -> Optional[TrendAnalysis]:
        """Calculate trend for a single metric."""
        values = [r.value for r in readings]

        # Split into first and second half
        mid = len(values) // 2
        first_half = values[:mid]
        second_half = values[mid:]

        if not first_half or not second_half:
            return None

        start_avg = statistics.mean(first_half)
        end_avg = statistics.mean(second_half)

        if start_avg == 0:
            change_percent = 0
        else:
            change_percent = ((end_avg - start_avg) / start_avg) * 100

        # Determine direction
        std_dev = statistics.stdev(values) if len(values) > 1 else 0
        threshold = std_dev * 0.5  # Significant if change > 0.5 std dev

        if abs(end_avg - start_avg) < threshold:
            direction = TrendDirection.STABLE
        elif end_avg > start_avg:
            direction = TrendDirection.INCREASING
        else:
            direction = TrendDirection.DECREASING

        # Check for fluctuation
        if std_dev > 0:
            cv = std_dev / statistics.mean(values)  # Coefficient of variation
            if cv > 0.3:  # High variability
                direction = TrendDirection.FLUCTUATING

        # Generate description
        description = self._generate_trend_description(
            metric_type, direction, change_percent, days
        )

        # Calculate confidence based on sample size and variance
        confidence = min(1.0, len(readings) / 50) * (1 - min(cv, 1.0) if std_dev > 0 else 1.0)

        return TrendAnalysis(
            metric_type=metric_type,
            direction=direction,
            change_percent=change_percent,
            period_days=days,
            start_avg=start_avg,
            end_avg=end_avg,
            confidence=confidence,
            description=description,
        )

    def _generate_trend_description(
        self,
        metric_type: MetricType,
        direction: TrendDirection,
        change_percent: float,
        days: int,
    ) -> str:
        """Generate natural language description of a trend."""
        metric_name = metric_type.value.replace("_", " ").title()

        if direction == TrendDirection.STABLE:
            return f"Your {metric_name} has been stable over the past {days} days."
        elif direction == TrendDirection.INCREASING:
            return f"Your {metric_name} increased by {abs(change_percent):.1f}% over the past {days} days."
        elif direction == TrendDirection.DECREASING:
            return f"Your {metric_name} decreased by {abs(change_percent):.1f}% over the past {days} days."
        else:
            return f"Your {metric_name} has been fluctuating significantly over the past {days} days."

    # ============================================
    # Anomaly Detection
    # ============================================

    def detect_anomalies(
        self,
        metric_types: Optional[List[MetricType]] = None,
        z_threshold: float = 2.5,
        lookback_days: int = 14,
    ) -> List[AnomalyDetection]:
        """
        Detect anomalies using Z-score based outlier detection.

        Args:
            metric_types: Metrics to check (None = all)
            z_threshold: Z-score threshold for anomaly (default 2.5)
            lookback_days: Days of historical data for baseline

        Returns:
            List of detected anomalies
        """
        anomalies = []
        target_metrics = metric_types or list(self._readings_cache.keys())
        cutoff = datetime.now() - timedelta(days=lookback_days)

        for metric_type in target_metrics:
            if metric_type not in self._readings_cache:
                continue

            readings = self._readings_cache[metric_type]
            historical = [r for r in readings if r.timestamp >= cutoff]

            if len(historical) < 10:
                continue

            values = [r.value for r in historical]
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0

            if std == 0:
                continue

            # Check recent readings for anomalies
            recent_cutoff = datetime.now() - timedelta(days=1)
            recent = [r for r in historical if r.timestamp >= recent_cutoff]

            for reading in recent:
                z_score = abs((reading.value - mean) / std)
                if z_score >= z_threshold:
                    deviation = ((reading.value - mean) / mean) * 100 if mean != 0 else 0

                    # Determine severity
                    if z_score >= 4.0:
                        severity = AlertSeverity.CRITICAL
                    elif z_score >= 3.0:
                        severity = AlertSeverity.WARNING
                    else:
                        severity = AlertSeverity.INFO

                    # Check against known normal ranges
                    if metric_type in self.NORMAL_RANGES:
                        low, high = self.NORMAL_RANGES[metric_type]
                        if reading.value < low or reading.value > high:
                            severity = AlertSeverity.WARNING

                    anomalies.append(
                        AnomalyDetection(
                            reading=reading,
                            z_score=z_score,
                            expected_value=mean,
                            deviation_percent=deviation,
                            severity=severity,
                        )
                    )

        return anomalies

    def create_alerts_from_anomalies(
        self, anomalies: List[AnomalyDetection]
    ) -> List[HealthAlert]:
        """Convert anomaly detections to health alerts."""
        alerts = []
        for anomaly in anomalies:
            metric_name = anomaly.reading.metric_type.value.replace("_", " ").title()

            if anomaly.reading.value > anomaly.expected_value:
                direction = "higher"
            else:
                direction = "lower"

            alert = HealthAlert(
                title=f"Unusual {metric_name}",
                message=f"Your {metric_name} ({anomaly.reading.value:.1f}) is {abs(anomaly.deviation_percent):.1f}% {direction} than your usual average ({anomaly.expected_value:.1f}).",
                severity=anomaly.severity,
                metric_type=anomaly.reading.metric_type,
                triggered_value=anomaly.reading.value,
                threshold_value=anomaly.expected_value,
            )
            alerts.append(alert)

        return alerts

    # ============================================
    # Correlation Analysis
    # ============================================

    def calculate_correlations(
        self,
        metric_pairs: Optional[List[Tuple[MetricType, MetricType]]] = None,
        min_samples: int = 20,
    ) -> List[CorrelationResult]:
        """
        Calculate correlations between metric pairs.

        Args:
            metric_pairs: Specific pairs to analyze (None = common pairs)
            min_samples: Minimum data points required

        Returns:
            List of correlation results
        """
        # Default interesting pairs
        if metric_pairs is None:
            metric_pairs = [
                (MetricType.SLEEP_DURATION, MetricType.RECOVERY_SCORE),
                (MetricType.SLEEP_SCORE, MetricType.HRV),
                (MetricType.STEPS, MetricType.CALORIES_BURNED),
                (MetricType.STRESS, MetricType.HRV),
                (MetricType.ACTIVE_MINUTES, MetricType.SLEEP_SCORE),
                (MetricType.HR, MetricType.STRESS),
            ]

        results = []
        for metric_a, metric_b in metric_pairs:
            if metric_a not in self._readings_cache or metric_b not in self._readings_cache:
                continue

            correlation = self._calculate_correlation(metric_a, metric_b, min_samples)
            if correlation:
                results.append(correlation)

        return results

    def _calculate_correlation(
        self,
        metric_a: MetricType,
        metric_b: MetricType,
        min_samples: int,
    ) -> Optional[CorrelationResult]:
        """Calculate Pearson correlation between two metrics."""
        readings_a = self._readings_cache.get(metric_a, [])
        readings_b = self._readings_cache.get(metric_b, [])

        # Align readings by date
        values_a = {}
        values_b = {}

        for r in readings_a:
            day = r.timestamp.date()
            if day not in values_a:
                values_a[day] = []
            values_a[day].append(r.value)

        for r in readings_b:
            day = r.timestamp.date()
            if day not in values_b:
                values_b[day] = []
            values_b[day].append(r.value)

        # Get common days and average values
        common_days = set(values_a.keys()) & set(values_b.keys())
        if len(common_days) < min_samples:
            return None

        aligned_a = [statistics.mean(values_a[d]) for d in sorted(common_days)]
        aligned_b = [statistics.mean(values_b[d]) for d in sorted(common_days)]

        # Calculate Pearson correlation
        n = len(aligned_a)
        mean_a = statistics.mean(aligned_a)
        mean_b = statistics.mean(aligned_b)

        numerator = sum((a - mean_a) * (b - mean_b) for a, b in zip(aligned_a, aligned_b))
        denom_a = sum((a - mean_a) ** 2 for a in aligned_a) ** 0.5
        denom_b = sum((b - mean_b) ** 2 for b in aligned_b) ** 0.5

        if denom_a == 0 or denom_b == 0:
            return None

        correlation = numerator / (denom_a * denom_b)

        # Simple significance estimate (not a true p-value)
        significance = 1 - min(1.0, abs(correlation) * (n ** 0.5) / 2)

        description = self._generate_correlation_description(metric_a, metric_b, correlation)

        return CorrelationResult(
            metric_a=metric_a,
            metric_b=metric_b,
            correlation=correlation,
            significance=significance,
            sample_size=n,
            description=description,
        )

    def _generate_correlation_description(
        self,
        metric_a: MetricType,
        metric_b: MetricType,
        correlation: float,
    ) -> str:
        """Generate description of correlation."""
        name_a = metric_a.value.replace("_", " ")
        name_b = metric_b.value.replace("_", " ")

        strength = ""
        if abs(correlation) >= 0.7:
            strength = "strong"
        elif abs(correlation) >= 0.4:
            strength = "moderate"
        elif abs(correlation) >= 0.2:
            strength = "weak"
        else:
            return f"No significant correlation found between {name_a} and {name_b}."

        direction = "positive" if correlation > 0 else "negative"

        return f"There is a {strength} {direction} correlation between {name_a} and {name_b}."

    # ============================================
    # Insight Generation
    # ============================================

    def generate_insights(
        self,
        summaries: Optional[List[DailySummary]] = None,
        max_insights: int = 5,
    ) -> List[HealthInsight]:
        """
        Generate natural language health insights.

        Args:
            summaries: Daily summaries to analyze (uses cache if None)
            max_insights: Maximum number of insights to generate

        Returns:
            List of insights sorted by priority
        """
        summaries = summaries or self._summaries_cache
        if len(summaries) < 3:
            return []

        insights = []
        recent = summaries[-7:] if len(summaries) >= 7 else summaries

        # Activity insights
        avg_steps = statistics.mean([s.total_steps for s in recent])
        if avg_steps < 5000:
            insights.append(
                HealthInsight(
                    category="activity",
                    title="Low Activity Level",
                    message=f"Your average daily steps ({int(avg_steps)}) is below the recommended 10,000. Try adding a short walk to your routine.",
                    priority=4,
                    related_metrics=["steps", "active_minutes"],
                    actionable=True,
                )
            )
        elif avg_steps >= 10000:
            insights.append(
                HealthInsight(
                    category="activity",
                    title="Great Activity Level",
                    message=f"Excellent! You're averaging {int(avg_steps)} steps per day, meeting the recommended goal.",
                    priority=2,
                    related_metrics=["steps"],
                    actionable=False,
                )
            )

        # Sleep insights
        sleep_data = [s for s in recent if s.sleep_duration_hours > 0]
        if sleep_data:
            avg_sleep = statistics.mean([s.sleep_duration_hours for s in sleep_data])
            if avg_sleep < 6:
                insights.append(
                    HealthInsight(
                        category="sleep",
                        title="Insufficient Sleep",
                        message=f"You're averaging only {avg_sleep:.1f} hours of sleep. Adults need 7-9 hours for optimal health.",
                        priority=5,
                        related_metrics=["sleep_duration", "sleep_score"],
                        actionable=True,
                    )
                )
            elif avg_sleep >= 7:
                avg_score = statistics.mean([s.sleep_score for s in sleep_data if s.sleep_score])
                if avg_score and avg_score >= 80:
                    insights.append(
                        HealthInsight(
                            category="sleep",
                            title="Quality Sleep",
                            message=f"You're getting good quality sleep with an average score of {int(avg_score)}.",
                            priority=1,
                            related_metrics=["sleep_score"],
                            actionable=False,
                        )
                    )

        # Recovery insights
        recovery_data = [s for s in recent if s.recovery_score]
        if recovery_data:
            avg_recovery = statistics.mean([s.recovery_score for s in recovery_data])
            if avg_recovery < 50:
                insights.append(
                    HealthInsight(
                        category="recovery",
                        title="Low Recovery",
                        message=f"Your recovery score is averaging {int(avg_recovery)}. Consider reducing training intensity or improving sleep.",
                        priority=4,
                        related_metrics=["recovery_score", "hrv"],
                        actionable=True,
                    )
                )

        # Heart health insights
        hr_data = [s for s in recent if s.resting_heart_rate]
        if hr_data:
            avg_rhr = statistics.mean([s.resting_heart_rate for s in hr_data])
            if avg_rhr > 80:
                insights.append(
                    HealthInsight(
                        category="heart",
                        title="Elevated Resting Heart Rate",
                        message=f"Your resting heart rate ({int(avg_rhr)} bpm) is elevated. Consider stress management and regular exercise.",
                        priority=4,
                        related_metrics=["resting_heart_rate", "stress"],
                        actionable=True,
                    )
                )

        # Sort by priority and limit
        insights.sort(key=lambda i: i.priority, reverse=True)
        return insights[:max_insights]

    # ============================================
    # Wellness Score Calculation
    # ============================================

    def calculate_wellness_score(
        self,
        summary: Optional[DailySummary] = None,
    ) -> WellnessScore:
        """
        Calculate composite wellness score from daily summary.

        Components:
        - Activity (steps, active minutes)
        - Sleep (duration, quality)
        - Recovery (score, HRV)
        - Heart Health (resting HR, HRV)
        - Stress
        """
        summary = summary or (self._summaries_cache[-1] if self._summaries_cache else None)
        if not summary:
            return WellnessScore(
                overall=0,
                activity_score=0,
                sleep_score=0,
                recovery_score=0,
                heart_health_score=0,
                stress_score=0,
            )

        factors = []
        components = {}

        # Activity Score (0-100)
        steps_score = min(100, (summary.total_steps / self.WELLNESS_TARGETS["steps"]) * 100)
        active_score = min(100, (summary.active_minutes / self.WELLNESS_TARGETS["active_minutes"]) * 100)
        activity_score = int((steps_score + active_score) / 2)
        components["steps"] = int(steps_score)
        components["active_minutes"] = int(active_score)
        if activity_score < 50:
            factors.append("Low physical activity")

        # Sleep Score (0-100)
        if summary.sleep_duration_hours > 0:
            duration_score = self._score_in_range(
                summary.sleep_duration_hours, 6, 9, optimal=8
            )
            quality_score = summary.sleep_score or 70
            sleep_score = int((duration_score + quality_score) / 2)
            components["sleep_duration"] = int(duration_score)
            components["sleep_quality"] = quality_score
            if sleep_score < 60:
                factors.append("Poor sleep quality or duration")
        else:
            sleep_score = 50
            factors.append("No sleep data")

        # Recovery Score (0-100)
        if summary.recovery_score:
            recovery_score = summary.recovery_score
        else:
            recovery_score = 70  # Default
        components["recovery"] = recovery_score
        if recovery_score < 50:
            factors.append("Low recovery score")

        # Heart Health Score (0-100)
        hr_score = 70
        if summary.resting_heart_rate:
            hr_score = self._score_in_range(
                summary.resting_heart_rate, 50, 80, optimal=60, inverse=True
            )
            components["resting_hr"] = int(hr_score)

        hrv_score = 70
        if summary.hrv_avg:
            hrv_score = self._score_in_range(summary.hrv_avg, 20, 100, optimal=50)
            components["hrv"] = int(hrv_score)

        heart_health_score = int((hr_score + hrv_score) / 2)
        if heart_health_score < 50:
            factors.append("Heart health metrics need attention")

        # Stress Score (0-100, higher = less stressed = better)
        stress_score = 70  # Default
        # Would need stress metric data from readings

        # Calculate overall score (weighted average)
        weights = {
            "activity": 0.25,
            "sleep": 0.30,
            "recovery": 0.20,
            "heart": 0.15,
            "stress": 0.10,
        }

        overall = int(
            activity_score * weights["activity"]
            + sleep_score * weights["sleep"]
            + recovery_score * weights["recovery"]
            + heart_health_score * weights["heart"]
            + stress_score * weights["stress"]
        )

        return WellnessScore(
            overall=overall,
            activity_score=activity_score,
            sleep_score=sleep_score,
            recovery_score=recovery_score,
            heart_health_score=heart_health_score,
            stress_score=stress_score,
            components=components,
            factors=factors,
        )

    def _score_in_range(
        self,
        value: float,
        min_val: float,
        max_val: float,
        optimal: Optional[float] = None,
        inverse: bool = False,
    ) -> float:
        """
        Score a value based on a range.

        Args:
            value: The value to score
            min_val: Minimum acceptable value
            max_val: Maximum acceptable value
            optimal: Optimal value (if different from max)
            inverse: If True, lower is better
        """
        if inverse:
            # Lower is better (e.g., resting HR)
            if value <= min_val:
                return 100
            elif value >= max_val:
                return 30
            else:
                return 100 - ((value - min_val) / (max_val - min_val)) * 70
        else:
            # Higher is better, up to optimal
            optimal = optimal or max_val
            if value >= optimal:
                return 100
            elif value <= min_val:
                return 30
            else:
                return 30 + ((value - min_val) / (optimal - min_val)) * 70
