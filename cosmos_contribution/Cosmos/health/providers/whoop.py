"""
cosmos WHOOP Health Provider

Integration with WHOOP API for strain, recovery, and sleep tracking.
"""

import os
import logging
from datetime import datetime, date, timedelta
from typing import List, Optional

import httpx

from .base import HealthProvider, OAuthCredentials
from ..models import MetricType, HealthMetricReading, DailySummary

logger = logging.getLogger(__name__)


class WHOOPProvider(HealthProvider):
    """
    WHOOP API integration.

    Supports:
    - Strain score
    - Recovery score
    - Sleep tracking
    - Heart rate and HRV
    - Respiratory rate
    - Workout detection
    """

    API_BASE = "https://api.prod.whoop.com/developer"
    AUTH_URL = "https://api.prod.whoop.com/oauth/oauth2/auth"
    TOKEN_URL = "https://api.prod.whoop.com/oauth/oauth2/token"

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ):
        """
        Initialize WHOOP provider.

        Args:
            client_id: WHOOP OAuth client ID (or WHOOP_CLIENT_ID env var)
            client_secret: WHOOP OAuth client secret (or WHOOP_CLIENT_SECRET env var)
        """
        super().__init__(
            name="whoop",
            client_id=client_id or os.getenv("WHOOP_CLIENT_ID"),
            client_secret=client_secret or os.getenv("WHOOP_CLIENT_SECRET"),
        )
        self.client = httpx.AsyncClient(timeout=30.0)
        self.user_id = None

    @property
    def supported_metrics(self) -> List[MetricType]:
        return [
            MetricType.HR,
            MetricType.HRV,
            MetricType.STRAIN,
            MetricType.RECOVERY_SCORE,
            MetricType.SLEEP_DURATION,
            MetricType.SLEEP_SCORE,
            MetricType.SLEEP_DEEP,
            MetricType.SLEEP_REM,
            MetricType.SLEEP_LIGHT,
            MetricType.SPO2,
            MetricType.CALORIES_BURNED,
        ]

    async def _connect_impl(self) -> bool:
        """Verify WHOOP connection and get user ID."""
        if not self.credentials:
            return False

        try:
            response = await self._api_request("GET", "/v1/user/profile/basic")
            if response and "user_id" in response:
                self.user_id = response["user_id"]
                return True
            return False
        except Exception as e:
            logger.error(f"WHOOP connection test failed: {e}")
            return False

    def _get_auth_url_impl(self, redirect_uri: str, state: str) -> str:
        """Generate WHOOP OAuth URL."""
        scope = "read:recovery%20read:cycles%20read:sleep%20read:workout%20read:profile%20read:body_measurement"
        return (
            f"{self.AUTH_URL}?"
            f"response_type=code"
            f"&client_id={self.client_id}"
            f"&redirect_uri={redirect_uri}"
            f"&scope={scope}"
            f"&state={state}"
        )

    async def _exchange_code_impl(
        self,
        code: str,
        redirect_uri: str,
    ) -> Optional[OAuthCredentials]:
        """Exchange authorization code for tokens."""
        try:
            response = await self.client.post(
                self.TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
            )

            if response.status_code != 200:
                logger.error(f"Token exchange failed: {response.text}")
                return None

            data = response.json()

            return OAuthCredentials(
                access_token=data["access_token"],
                refresh_token=data.get("refresh_token"),
                expires_at=datetime.now() + timedelta(seconds=data.get("expires_in", 86400)),
            )

        except Exception as e:
            logger.error(f"Token exchange error: {e}")
            return None

    async def _refresh_token_impl(self) -> Optional[OAuthCredentials]:
        """Refresh the access token."""
        try:
            response = await self.client.post(
                self.TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self.credentials.refresh_token,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
            )

            if response.status_code != 200:
                return None

            data = response.json()

            return OAuthCredentials(
                access_token=data["access_token"],
                refresh_token=data.get("refresh_token", self.credentials.refresh_token),
                expires_at=datetime.now() + timedelta(seconds=data.get("expires_in", 86400)),
            )

        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return None

    async def _api_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict] = None,
    ) -> Optional[dict]:
        """Make authenticated API request."""
        if not self.credentials:
            return None

        headers = {
            "Authorization": f"Bearer {self.credentials.access_token}",
        }

        try:
            response = await self.client.request(
                method,
                f"{self.API_BASE}{endpoint}",
                headers=headers,
                params=params,
            )

            if response.status_code == 401:
                if await self.refresh_token():
                    return await self._api_request(method, endpoint, params)
                return None

            if response.status_code == 429:
                self.status.rate_limited = True
                return None

            if response.status_code != 200:
                logger.error(f"WHOOP API error: {response.status_code}")
                return None

            return response.json()

        except Exception as e:
            logger.error(f"WHOOP API request error: {e}")
            return None

    async def _fetch_metrics_impl(
        self,
        metric_types: List[MetricType],
        start_date: date,
        end_date: date,
    ) -> List[HealthMetricReading]:
        """Fetch historical metrics from WHOOP."""
        readings = []

        # WHOOP uses cycle-based data
        need_recovery = any(m in [MetricType.RECOVERY_SCORE, MetricType.HRV,
                                  MetricType.HR] for m in metric_types)
        need_sleep = any(m in [MetricType.SLEEP_DURATION, MetricType.SLEEP_SCORE,
                               MetricType.SLEEP_DEEP, MetricType.SLEEP_REM,
                               MetricType.SLEEP_LIGHT] for m in metric_types)
        need_strain = MetricType.STRAIN in metric_types or \
                      MetricType.CALORIES_BURNED in metric_types

        if need_recovery:
            readings.extend(await self._fetch_recovery(start_date, end_date, metric_types))

        if need_sleep:
            readings.extend(await self._fetch_sleep(start_date, end_date, metric_types))

        if need_strain:
            readings.extend(await self._fetch_strain(start_date, end_date, metric_types))

        return readings

    async def _fetch_recovery(
        self,
        start_date: date,
        end_date: date,
        metric_types: List[MetricType],
    ) -> List[HealthMetricReading]:
        """Fetch recovery data."""
        readings = []

        data = await self._api_request(
            "GET",
            "/v1/recovery",
            params={
                "start": f"{start_date.isoformat()}T00:00:00.000Z",
                "end": f"{end_date.isoformat()}T23:59:59.999Z",
            },
        )

        if not data or "records" not in data:
            return readings

        for record in data["records"]:
            ts = datetime.fromisoformat(record["created_at"].replace("Z", "+00:00"))

            if MetricType.RECOVERY_SCORE in metric_types:
                score = record.get("score", {})
                if "recovery_score" in score:
                    readings.append(HealthMetricReading(
                        metric_type=MetricType.RECOVERY_SCORE,
                        value=score["recovery_score"],
                        unit="%",
                        timestamp=ts,
                        source="whoop",
                    ))

            if MetricType.HRV in metric_types:
                score = record.get("score", {})
                if "hrv_rmssd_milli" in score:
                    readings.append(HealthMetricReading(
                        metric_type=MetricType.HRV,
                        value=score["hrv_rmssd_milli"],
                        unit="ms",
                        timestamp=ts,
                        source="whoop",
                    ))

            if MetricType.HR in metric_types:
                score = record.get("score", {})
                if "resting_heart_rate" in score:
                    readings.append(HealthMetricReading(
                        metric_type=MetricType.HR,
                        value=score["resting_heart_rate"],
                        unit="bpm",
                        timestamp=ts,
                        source="whoop",
                        metadata={"type": "resting"},
                    ))

        return readings

    async def _fetch_sleep(
        self,
        start_date: date,
        end_date: date,
        metric_types: List[MetricType],
    ) -> List[HealthMetricReading]:
        """Fetch sleep data."""
        readings = []

        data = await self._api_request(
            "GET",
            "/v1/activity/sleep",
            params={
                "start": f"{start_date.isoformat()}T00:00:00.000Z",
                "end": f"{end_date.isoformat()}T23:59:59.999Z",
            },
        )

        if not data or "records" not in data:
            return readings

        for record in data["records"]:
            ts = datetime.fromisoformat(record["start"].replace("Z", "+00:00"))

            if MetricType.SLEEP_DURATION in metric_types:
                # Duration in milliseconds
                duration_ms = record.get("end", 0) - record.get("start", 0)
                # Parse timestamps to calculate duration
                try:
                    start_ts = datetime.fromisoformat(record["start"].replace("Z", "+00:00"))
                    end_ts = datetime.fromisoformat(record["end"].replace("Z", "+00:00"))
                    duration_hours = (end_ts - start_ts).total_seconds() / 3600
                    readings.append(HealthMetricReading(
                        metric_type=MetricType.SLEEP_DURATION,
                        value=duration_hours,
                        unit="hours",
                        timestamp=ts,
                        source="whoop",
                    ))
                except:
                    pass

            score = record.get("score", {})

            if MetricType.SLEEP_SCORE in metric_types and "sleep_performance_percentage" in score:
                readings.append(HealthMetricReading(
                    metric_type=MetricType.SLEEP_SCORE,
                    value=score["sleep_performance_percentage"],
                    unit="%",
                    timestamp=ts,
                    source="whoop",
                ))

            # Sleep stages
            stage_summary = score.get("stage_summary", {})

            if MetricType.SLEEP_DEEP in metric_types and "total_slow_wave_sleep_time_milli" in stage_summary:
                readings.append(HealthMetricReading(
                    metric_type=MetricType.SLEEP_DEEP,
                    value=stage_summary["total_slow_wave_sleep_time_milli"] / 3600000,
                    unit="hours",
                    timestamp=ts,
                    source="whoop",
                ))

            if MetricType.SLEEP_REM in metric_types and "total_rem_sleep_time_milli" in stage_summary:
                readings.append(HealthMetricReading(
                    metric_type=MetricType.SLEEP_REM,
                    value=stage_summary["total_rem_sleep_time_milli"] / 3600000,
                    unit="hours",
                    timestamp=ts,
                    source="whoop",
                ))

            if MetricType.SLEEP_LIGHT in metric_types and "total_light_sleep_time_milli" in stage_summary:
                readings.append(HealthMetricReading(
                    metric_type=MetricType.SLEEP_LIGHT,
                    value=stage_summary["total_light_sleep_time_milli"] / 3600000,
                    unit="hours",
                    timestamp=ts,
                    source="whoop",
                ))

        return readings

    async def _fetch_strain(
        self,
        start_date: date,
        end_date: date,
        metric_types: List[MetricType],
    ) -> List[HealthMetricReading]:
        """Fetch strain/cycle data."""
        readings = []

        data = await self._api_request(
            "GET",
            "/v1/cycle",
            params={
                "start": f"{start_date.isoformat()}T00:00:00.000Z",
                "end": f"{end_date.isoformat()}T23:59:59.999Z",
            },
        )

        if not data or "records" not in data:
            return readings

        for record in data["records"]:
            ts = datetime.fromisoformat(record["start"].replace("Z", "+00:00"))

            score = record.get("score", {})

            if MetricType.STRAIN in metric_types and "strain" in score:
                readings.append(HealthMetricReading(
                    metric_type=MetricType.STRAIN,
                    value=score["strain"],
                    unit="strain",
                    timestamp=ts,
                    source="whoop",
                ))

            if MetricType.CALORIES_BURNED in metric_types and "kilojoule" in score:
                # Convert kJ to kcal
                readings.append(HealthMetricReading(
                    metric_type=MetricType.CALORIES_BURNED,
                    value=score["kilojoule"] / 4.184,
                    unit="kcal",
                    timestamp=ts,
                    source="whoop",
                ))

        return readings

    async def _fetch_daily_summary_impl(self, target_date: date) -> Optional[DailySummary]:
        """Fetch daily summary from WHOOP."""
        summary = DailySummary(date=target_date)

        # Fetch all data types
        recovery_readings = await self._fetch_recovery(
            target_date, target_date,
            [MetricType.RECOVERY_SCORE, MetricType.HRV, MetricType.HR]
        )
        sleep_readings = await self._fetch_sleep(
            target_date, target_date,
            [MetricType.SLEEP_DURATION, MetricType.SLEEP_SCORE,
             MetricType.SLEEP_DEEP, MetricType.SLEEP_REM]
        )
        strain_readings = await self._fetch_strain(
            target_date, target_date,
            [MetricType.STRAIN, MetricType.CALORIES_BURNED]
        )

        # Populate summary from readings
        for reading in recovery_readings:
            if reading.metric_type == MetricType.RECOVERY_SCORE:
                summary.recovery_score = int(reading.value)
            elif reading.metric_type == MetricType.HRV:
                summary.hrv_avg = reading.value
            elif reading.metric_type == MetricType.HR:
                summary.resting_heart_rate = reading.value

        for reading in sleep_readings:
            if reading.metric_type == MetricType.SLEEP_DURATION:
                summary.sleep_duration_hours = reading.value
            elif reading.metric_type == MetricType.SLEEP_SCORE:
                summary.sleep_score = int(reading.value)
            elif reading.metric_type == MetricType.SLEEP_DEEP:
                summary.deep_sleep_hours = reading.value
            elif reading.metric_type == MetricType.SLEEP_REM:
                summary.rem_sleep_hours = reading.value

        for reading in strain_readings:
            if reading.metric_type == MetricType.CALORIES_BURNED:
                summary.total_calories_burned = int(reading.value)

        return summary
