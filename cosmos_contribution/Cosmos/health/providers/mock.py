"""
cosmos Mock Health Provider

Simulated health data provider for testing without real hardware or APIs.
Generates realistic health metrics for development and demonstration.
"""

import asyncio
import random
import math
from datetime import datetime, date, timedelta
from typing import List, Optional, AsyncIterator

from .base import (
    HealthProvider,
    OAuthCredentials,
    BioDataPacket,
)
from ..models import (
    MetricType,
    HealthMetricReading,
    DailySummary,
)


class MockHealthProvider(HealthProvider):
    """
    Mock health provider for testing.

    Generates realistic simulated data for all metric types.
    Useful for development and demonstration without real devices.
    """

    def __init__(self):
        super().__init__(name="mock", client_id=None, client_secret=None)
        self._streaming = False
        self._base_hr = random.randint(60, 75)  # Base heart rate
        self._steps_today = 0
        self._last_step_update = datetime.now()

    @property
    def supported_metrics(self) -> List[MetricType]:
        return [
            MetricType.HR,
            MetricType.HRV,
            MetricType.SPO2,
            MetricType.STEPS,
            MetricType.CALORIES_BURNED,
            MetricType.ACTIVE_MINUTES,
            MetricType.DISTANCE,
            MetricType.SLEEP_DURATION,
            MetricType.SLEEP_SCORE,
            MetricType.SLEEP_DEEP,
            MetricType.SLEEP_REM,
            MetricType.SLEEP_LIGHT,
            MetricType.RECOVERY_SCORE,
            MetricType.STRESS,
            MetricType.WEIGHT,
            MetricType.BODY_FAT,
            MetricType.EEG_ALPHA,
            MetricType.EEG_BETA,
            MetricType.EEG_THETA,
            MetricType.GSR,
        ]

    @property
    def requires_oauth(self) -> bool:
        return False  # Mock provider doesn't need OAuth

    async def _connect_impl(self) -> bool:
        """Mock connection always succeeds."""
        await asyncio.sleep(0.1)  # Simulate connection delay
        return True

    async def _disconnect_impl(self):
        """Mock disconnection."""
        self._streaming = False

    async def _stream_impl(self) -> AsyncIterator[BioDataPacket]:
        """Stream simulated real-time data."""
        self._streaming = True
        while self._streaming:
            await asyncio.sleep(1.0)  # 1Hz update

            # Heart Rate
            hr = self._generate_heart_rate()
            yield BioDataPacket(
                source_device="MockWatch",
                signal_type="HR",
                processed_value=hr,
                metadata={"unit": "bpm"},
            )

            # HRV (every 5 seconds)
            if random.random() < 0.2:
                hrv = self._generate_hrv()
                yield BioDataPacket(
                    source_device="MockWatch",
                    signal_type="HRV",
                    processed_value=hrv,
                    metadata={"unit": "ms"},
                )

            # Steps (accumulate)
            if random.random() < 0.5:
                steps = random.randint(0, 20)
                self._steps_today += steps
                yield BioDataPacket(
                    source_device="MockWatch",
                    signal_type="STEPS",
                    processed_value=float(self._steps_today),
                    metadata={"unit": "steps"},
                )

            # Stress level (occasionally)
            if random.random() < 0.1:
                stress = self._generate_stress()
                yield BioDataPacket(
                    source_device="MockWatch",
                    signal_type="STRESS",
                    processed_value=stress,
                    metadata={"unit": "level"},
                )

    async def _fetch_metrics_impl(
        self,
        metric_types: List[MetricType],
        start_date: date,
        end_date: date,
    ) -> List[HealthMetricReading]:
        """Generate historical mock data."""
        readings = []
        current = start_date

        while current <= end_date:
            for metric_type in metric_types:
                day_readings = self._generate_day_readings(metric_type, current)
                readings.extend(day_readings)
            current += timedelta(days=1)

        return readings

    async def _fetch_daily_summary_impl(self, target_date: date) -> Optional[DailySummary]:
        """Generate a mock daily summary."""
        return self._generate_daily_summary(target_date)

    async def _fetch_latest_impl(
        self, metric_type: MetricType
    ) -> Optional[HealthMetricReading]:
        """Generate a single latest reading."""
        value, unit = self._generate_metric_value(metric_type)
        return HealthMetricReading(
            metric_type=metric_type,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            source=self.name,
            quality=0.95,
        )

    # ============================================
    # Data Generation Methods
    # ============================================

    def _generate_heart_rate(self) -> float:
        """Generate realistic heart rate with natural variation."""
        # Add circadian variation
        hour = datetime.now().hour
        circadian_offset = -5 if 2 <= hour <= 6 else (5 if 10 <= hour <= 14 else 0)

        # Add random variation
        variation = random.gauss(0, 3)

        return max(50, min(180, self._base_hr + circadian_offset + variation))

    def _generate_hrv(self) -> float:
        """Generate HRV (RMSSD) value."""
        # Higher HRV indicates better recovery
        base_hrv = 45 + random.gauss(0, 15)
        return max(10, min(150, base_hrv))

    def _generate_stress(self) -> float:
        """Generate stress level (0-100)."""
        hour = datetime.now().hour
        # Higher stress during work hours
        base_stress = 40 if 9 <= hour <= 17 else 25
        return max(0, min(100, base_stress + random.gauss(0, 15)))

    def _generate_metric_value(self, metric_type: MetricType) -> tuple:
        """Generate a value for any metric type."""
        generators = {
            MetricType.HR: (self._generate_heart_rate(), "bpm"),
            MetricType.HRV: (self._generate_hrv(), "ms"),
            MetricType.SPO2: (random.uniform(95, 100), "%"),
            MetricType.STEPS: (random.randint(0, 15000), "steps"),
            MetricType.CALORIES_BURNED: (random.randint(1500, 3000), "kcal"),
            MetricType.ACTIVE_MINUTES: (random.randint(10, 120), "min"),
            MetricType.DISTANCE: (random.uniform(0.5, 15.0), "km"),
            MetricType.SLEEP_DURATION: (random.uniform(5.0, 9.0), "hours"),
            MetricType.SLEEP_SCORE: (random.randint(50, 100), "score"),
            MetricType.SLEEP_DEEP: (random.uniform(0.5, 2.5), "hours"),
            MetricType.SLEEP_REM: (random.uniform(1.0, 2.5), "hours"),
            MetricType.SLEEP_LIGHT: (random.uniform(2.0, 4.0), "hours"),
            MetricType.RECOVERY_SCORE: (random.randint(30, 100), "score"),
            MetricType.READINESS_SCORE: (random.randint(40, 100), "score"),
            MetricType.STRESS: (self._generate_stress(), "level"),
            MetricType.WEIGHT: (random.uniform(60.0, 90.0), "kg"),
            MetricType.BODY_FAT: (random.uniform(10.0, 30.0), "%"),
            MetricType.EEG_ALPHA: (random.uniform(0.3, 0.8), "normalized"),
            MetricType.EEG_BETA: (random.uniform(0.2, 0.6), "normalized"),
            MetricType.EEG_THETA: (random.uniform(0.2, 0.5), "normalized"),
            MetricType.EEG_DELTA: (random.uniform(0.1, 0.4), "normalized"),
            MetricType.EEG_GAMMA: (random.uniform(0.1, 0.3), "normalized"),
            MetricType.GSR: (random.uniform(0.5, 5.0), "microsiemens"),
            MetricType.BP_SYSTOLIC: (random.randint(100, 140), "mmHg"),
            MetricType.BP_DIASTOLIC: (random.randint(60, 90), "mmHg"),
        }

        return generators.get(metric_type, (0.0, ""))

    def _generate_day_readings(
        self, metric_type: MetricType, day: date
    ) -> List[HealthMetricReading]:
        """Generate readings throughout a day."""
        readings = []

        # Determine reading frequency based on metric type
        if metric_type in [MetricType.HR, MetricType.HRV, MetricType.GSR]:
            # Frequent readings (every 15 minutes during waking hours)
            for hour in range(7, 23):
                for minute in [0, 15, 30, 45]:
                    ts = datetime.combine(day, datetime.min.time()).replace(
                        hour=hour, minute=minute
                    )
                    value, unit = self._generate_metric_value(metric_type)
                    readings.append(
                        HealthMetricReading(
                            metric_type=metric_type,
                            value=value,
                            unit=unit,
                            timestamp=ts,
                            source=self.name,
                            quality=random.uniform(0.85, 1.0),
                        )
                    )

        elif metric_type in [MetricType.STEPS, MetricType.CALORIES_BURNED]:
            # Hourly cumulative readings
            cumulative = 0
            for hour in range(24):
                ts = datetime.combine(day, datetime.min.time()).replace(hour=hour)
                # More steps during day, less at night
                if 7 <= hour <= 22:
                    increment = random.randint(200, 1500)
                else:
                    increment = random.randint(0, 50)
                cumulative += increment

                readings.append(
                    HealthMetricReading(
                        metric_type=metric_type,
                        value=cumulative if metric_type == MetricType.STEPS else cumulative * 0.04,
                        unit="steps" if metric_type == MetricType.STEPS else "kcal",
                        timestamp=ts,
                        source=self.name,
                        quality=0.95,
                    )
                )

        elif metric_type in [
            MetricType.SLEEP_DURATION,
            MetricType.SLEEP_SCORE,
            MetricType.SLEEP_DEEP,
            MetricType.SLEEP_REM,
        ]:
            # Single daily reading (morning)
            ts = datetime.combine(day, datetime.min.time()).replace(hour=7, minute=30)
            value, unit = self._generate_metric_value(metric_type)
            readings.append(
                HealthMetricReading(
                    metric_type=metric_type,
                    value=value,
                    unit=unit,
                    timestamp=ts,
                    source=self.name,
                    quality=0.9,
                )
            )

        elif metric_type in [MetricType.WEIGHT, MetricType.BODY_FAT]:
            # Single daily reading (morning)
            ts = datetime.combine(day, datetime.min.time()).replace(hour=7, minute=0)
            value, unit = self._generate_metric_value(metric_type)
            readings.append(
                HealthMetricReading(
                    metric_type=metric_type,
                    value=value,
                    unit=unit,
                    timestamp=ts,
                    source=self.name,
                    quality=0.95,
                )
            )

        else:
            # Default: a few readings throughout the day
            for hour in [8, 12, 18, 22]:
                ts = datetime.combine(day, datetime.min.time()).replace(hour=hour)
                value, unit = self._generate_metric_value(metric_type)
                readings.append(
                    HealthMetricReading(
                        metric_type=metric_type,
                        value=value,
                        unit=unit,
                        timestamp=ts,
                        source=self.name,
                        quality=random.uniform(0.8, 1.0),
                    )
                )

        return readings

    def _generate_daily_summary(self, target_date: date) -> DailySummary:
        """Generate a complete daily summary."""
        # Generate consistent data for the day
        random.seed(target_date.toordinal())  # Reproducible per date

        sleep_hours = random.uniform(5.5, 9.0)
        deep_ratio = random.uniform(0.15, 0.25)
        rem_ratio = random.uniform(0.2, 0.3)

        summary = DailySummary(
            date=target_date,
            avg_heart_rate=random.uniform(60, 80),
            resting_heart_rate=random.uniform(55, 70),
            max_heart_rate=random.uniform(100, 160),
            hrv_avg=random.uniform(30, 80),
            total_steps=random.randint(3000, 15000),
            total_calories_burned=random.randint(1800, 3500),
            active_minutes=random.randint(15, 120),
            distance_km=random.uniform(2.0, 12.0),
            floors_climbed=random.randint(0, 30),
            sleep_duration_hours=sleep_hours,
            sleep_score=random.randint(50, 95),
            deep_sleep_hours=sleep_hours * deep_ratio,
            rem_sleep_hours=sleep_hours * rem_ratio,
            recovery_score=random.randint(40, 95),
            readiness_score=random.randint(50, 95),
            wellness_score=random.randint(55, 95),
            calories_consumed=random.randint(1500, 2800),
            protein_g=random.uniform(40, 150),
            carbs_g=random.uniform(150, 350),
            fat_g=random.uniform(40, 120),
            water_ml=random.randint(1000, 3000),
        )

        # Reset random seed
        random.seed()

        return summary

    def reset_daily_counters(self):
        """Reset daily counters (call at midnight)."""
        self._steps_today = 0
        self._last_step_update = datetime.now()
