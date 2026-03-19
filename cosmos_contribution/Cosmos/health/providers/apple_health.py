"""
cosmos Apple Health Provider

Integration with Apple Health via exported XML files.
Since Apple Health doesn't have a direct API, this parses export.xml files.
"""

import os
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

from .base import HealthProvider
from ..models import MetricType, HealthMetricReading, DailySummary

logger = logging.getLogger(__name__)


class AppleHealthProvider(HealthProvider):
    """
    Apple Health integration via XML export.

    To export data from Apple Health:
    1. Open Health app on iPhone
    2. Tap profile picture > Export All Health Data
    3. Unzip and provide path to export.xml

    Supports:
    - Heart rate and HRV
    - Steps, distance, flights climbed
    - Sleep analysis
    - Active energy burned
    - Body measurements
    - Blood oxygen
    """

    # Apple Health type identifiers
    TYPE_MAPPING = {
        "HKQuantityTypeIdentifierHeartRate": MetricType.HR,
        "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": MetricType.HRV,
        "HKQuantityTypeIdentifierStepCount": MetricType.STEPS,
        "HKQuantityTypeIdentifierDistanceWalkingRunning": MetricType.DISTANCE,
        "HKQuantityTypeIdentifierFlightsClimbed": MetricType.FLOORS,
        "HKQuantityTypeIdentifierActiveEnergyBurned": MetricType.CALORIES_BURNED,
        "HKQuantityTypeIdentifierBasalEnergyBurned": MetricType.CALORIES_BURNED,
        "HKQuantityTypeIdentifierAppleExerciseTime": MetricType.ACTIVE_MINUTES,
        "HKQuantityTypeIdentifierOxygenSaturation": MetricType.SPO2,
        "HKQuantityTypeIdentifierBodyMass": MetricType.WEIGHT,
        "HKQuantityTypeIdentifierBodyFatPercentage": MetricType.BODY_FAT,
        "HKQuantityTypeIdentifierBodyMassIndex": MetricType.BMI,
        "HKQuantityTypeIdentifierBloodPressureSystolic": MetricType.BP_SYSTOLIC,
        "HKQuantityTypeIdentifierBloodPressureDiastolic": MetricType.BP_DIASTOLIC,
        "HKCategoryTypeIdentifierSleepAnalysis": MetricType.SLEEP_DURATION,
    }

    # Unit conversions
    UNIT_CONVERSIONS = {
        "count/min": ("bpm", 1.0),  # Heart rate
        "ms": ("ms", 1.0),  # HRV
        "count": ("steps", 1.0),
        "mi": ("km", 1.60934),  # Miles to km
        "km": ("km", 1.0),
        "kcal": ("kcal", 1.0),
        "Cal": ("kcal", 1.0),
        "min": ("min", 1.0),
        "%": ("%", 1.0),
        "lb": ("kg", 0.453592),  # Pounds to kg
        "kg": ("kg", 1.0),
        "mmHg": ("mmHg", 1.0),
    }

    def __init__(
        self,
        export_path: Optional[str] = None,
    ):
        """
        Initialize Apple Health provider.

        Args:
            export_path: Path to export.xml file (or APPLE_HEALTH_EXPORT env var)
        """
        super().__init__(name="apple_health", client_id=None, client_secret=None)

        self.export_path = Path(
            export_path or os.getenv("APPLE_HEALTH_EXPORT", "")
        )
        self._parsed_data: Dict[MetricType, List[HealthMetricReading]] = {}
        self._last_parse_time: Optional[datetime] = None

    @property
    def supported_metrics(self) -> List[MetricType]:
        return list(set(self.TYPE_MAPPING.values()))

    @property
    def requires_oauth(self) -> bool:
        return False  # Uses file-based import

    async def _connect_impl(self) -> bool:
        """Verify export file exists and is parseable."""
        if not self.export_path or not self.export_path.exists():
            logger.warning(f"Apple Health export not found: {self.export_path}")
            return False

        try:
            # Test parsing first few records
            for event, elem in ET.iterparse(str(self.export_path), events=["end"]):
                if elem.tag == "Record":
                    # Successfully parsed at least one record
                    elem.clear()
                    return True

            return False

        except Exception as e:
            logger.error(f"Apple Health parse error: {e}")
            return False

    async def _fetch_metrics_impl(
        self,
        metric_types: List[MetricType],
        start_date: date,
        end_date: date,
    ) -> List[HealthMetricReading]:
        """Fetch metrics from Apple Health export."""
        # Parse if not cached or cache is old
        if not self._parsed_data or self._should_reparse():
            await self._parse_export()

        readings = []
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time())

        for metric_type in metric_types:
            if metric_type in self._parsed_data:
                for reading in self._parsed_data[metric_type]:
                    if start_dt <= reading.timestamp <= end_dt:
                        readings.append(reading)

        return readings

    def _should_reparse(self) -> bool:
        """Check if we should re-parse the export file."""
        if not self._last_parse_time:
            return True
        # Re-parse if file was modified since last parse
        if self.export_path.stat().st_mtime > self._last_parse_time.timestamp():
            return True
        return False

    async def _parse_export(self):
        """Parse the Apple Health export.xml file."""
        self._parsed_data.clear()
        count = 0

        try:
            logger.info(f"Parsing Apple Health export: {self.export_path}")

            for event, elem in ET.iterparse(str(self.export_path), events=["end"]):
                if elem.tag == "Record":
                    reading = self._parse_record(elem)
                    if reading:
                        if reading.metric_type not in self._parsed_data:
                            self._parsed_data[reading.metric_type] = []
                        self._parsed_data[reading.metric_type].append(reading)
                        count += 1

                    # Clear element to save memory
                    elem.clear()

            self._last_parse_time = datetime.now()
            logger.info(f"Parsed {count} records from Apple Health export")

        except Exception as e:
            logger.error(f"Error parsing Apple Health export: {e}")

    def _parse_record(self, elem: ET.Element) -> Optional[HealthMetricReading]:
        """Parse a single Record element."""
        try:
            record_type = elem.get("type", "")

            # Check if we handle this type
            if record_type not in self.TYPE_MAPPING:
                return None

            metric_type = self.TYPE_MAPPING[record_type]

            # Get value
            value_str = elem.get("value", "0")
            try:
                value = float(value_str)
            except ValueError:
                # Handle categorical values (e.g., sleep analysis)
                if record_type == "HKCategoryTypeIdentifierSleepAnalysis":
                    # Sleep values: 0=InBed, 1=Asleep, 2=Awake
                    value = 1.0 if value_str in ["HKCategoryValueSleepAnalysisAsleep",
                                                  "HKCategoryValueSleepAnalysisAsleepCore",
                                                  "HKCategoryValueSleepAnalysisAsleepDeep",
                                                  "HKCategoryValueSleepAnalysisAsleepREM"] else 0.0
                else:
                    return None

            # Get and convert unit
            unit = elem.get("unit", "")
            if unit in self.UNIT_CONVERSIONS:
                unit, factor = self.UNIT_CONVERSIONS[unit]
                value *= factor

            # Parse timestamp
            start_date = elem.get("startDate", "")
            if not start_date:
                return None

            timestamp = self._parse_timestamp(start_date)
            if not timestamp:
                return None

            # Get source
            source = elem.get("sourceName", "Apple Health")

            return HealthMetricReading(
                metric_type=metric_type,
                value=value,
                unit=unit,
                timestamp=timestamp,
                source=f"apple_health:{source}",
                metadata={
                    "device": elem.get("device", ""),
                    "creation_date": elem.get("creationDate", ""),
                },
            )

        except Exception as e:
            logger.debug(f"Error parsing record: {e}")
            return None

    def _parse_timestamp(self, date_str: str) -> Optional[datetime]:
        """Parse Apple Health timestamp format."""
        try:
            # Format: 2024-01-15 08:30:00 -0500
            # Remove timezone offset for simpler parsing
            if " " in date_str and len(date_str) > 19:
                date_str = date_str[:19]
            return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            try:
                return datetime.fromisoformat(date_str.replace(" ", "T"))
            except:
                return None

    async def _fetch_daily_summary_impl(self, target_date: date) -> Optional[DailySummary]:
        """Generate daily summary from parsed data."""
        if not self._parsed_data:
            await self._parse_export()

        summary = DailySummary(date=target_date)

        start_dt = datetime.combine(target_date, datetime.min.time())
        end_dt = datetime.combine(target_date, datetime.max.time())

        # Aggregate steps
        if MetricType.STEPS in self._parsed_data:
            steps = sum(
                r.value for r in self._parsed_data[MetricType.STEPS]
                if start_dt <= r.timestamp <= end_dt
            )
            summary.total_steps = int(steps)

        # Aggregate calories
        if MetricType.CALORIES_BURNED in self._parsed_data:
            cals = sum(
                r.value for r in self._parsed_data[MetricType.CALORIES_BURNED]
                if start_dt <= r.timestamp <= end_dt
            )
            summary.total_calories_burned = int(cals)

        # Active minutes
        if MetricType.ACTIVE_MINUTES in self._parsed_data:
            mins = sum(
                r.value for r in self._parsed_data[MetricType.ACTIVE_MINUTES]
                if start_dt <= r.timestamp <= end_dt
            )
            summary.active_minutes = int(mins)

        # Distance
        if MetricType.DISTANCE in self._parsed_data:
            dist = sum(
                r.value for r in self._parsed_data[MetricType.DISTANCE]
                if start_dt <= r.timestamp <= end_dt
            )
            summary.distance_km = dist

        # Floors climbed
        if MetricType.FLOORS in self._parsed_data:
            floors = sum(
                r.value for r in self._parsed_data[MetricType.FLOORS]
                if start_dt <= r.timestamp <= end_dt
            )
            summary.floors_climbed = int(floors)

        # Heart rate average
        if MetricType.HR in self._parsed_data:
            hr_readings = [
                r.value for r in self._parsed_data[MetricType.HR]
                if start_dt <= r.timestamp <= end_dt
            ]
            if hr_readings:
                summary.avg_heart_rate = sum(hr_readings) / len(hr_readings)
                summary.max_heart_rate = max(hr_readings)
                # Approximate resting HR as minimum during sleep hours
                night_readings = [
                    r.value for r in self._parsed_data[MetricType.HR]
                    if start_dt <= r.timestamp <= end_dt
                    and 2 <= r.timestamp.hour <= 6
                ]
                if night_readings:
                    summary.resting_heart_rate = min(night_readings)

        # HRV average
        if MetricType.HRV in self._parsed_data:
            hrv_readings = [
                r.value for r in self._parsed_data[MetricType.HRV]
                if start_dt <= r.timestamp <= end_dt
            ]
            if hrv_readings:
                summary.hrv_avg = sum(hrv_readings) / len(hrv_readings)

        # Sleep duration (simplified - count hours of sleep records)
        if MetricType.SLEEP_DURATION in self._parsed_data:
            # For proper sleep analysis, would need to aggregate sleep session durations
            sleep_readings = [
                r for r in self._parsed_data[MetricType.SLEEP_DURATION]
                if start_dt <= r.timestamp <= end_dt and r.value == 1.0
            ]
            # Approximate sleep hours based on record count (assuming 5-min intervals)
            summary.sleep_duration_hours = len(sleep_readings) * 5 / 60

        # Latest weight
        if MetricType.WEIGHT in self._parsed_data:
            weight_readings = [
                r for r in self._parsed_data[MetricType.WEIGHT]
                if start_dt <= r.timestamp <= end_dt
            ]
            if weight_readings:
                # Take latest reading
                weight_readings.sort(key=lambda r: r.timestamp, reverse=True)
                # Store in metadata since DailySummary doesn't have weight field

        return summary

    def get_available_date_range(self) -> Optional[tuple]:
        """Get the date range available in the export."""
        if not self._parsed_data:
            return None

        all_timestamps = []
        for readings in self._parsed_data.values():
            for reading in readings:
                all_timestamps.append(reading.timestamp)

        if not all_timestamps:
            return None

        return (min(all_timestamps).date(), max(all_timestamps).date())
