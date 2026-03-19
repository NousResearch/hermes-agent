"""
cosmos Fitbit Health Provider

Integration with Fitbit Web API for health metrics.
Requires OAuth 2.0 authentication.
"""

import os
import logging
from datetime import datetime, date, timedelta
from typing import List, Optional, AsyncIterator

import httpx

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

logger = logging.getLogger(__name__)


class FitbitProvider(HealthProvider):
    """
    Fitbit Web API integration.

    Supports:
    - Heart rate (real-time and historical)
    - Steps, distance, calories
    - Sleep tracking
    - Activity minutes
    - Body measurements

    Requires Fitbit Developer App credentials.
    """

    # Fitbit API endpoints
    AUTH_URL = "https://www.fitbit.com/oauth2/authorize"
    TOKEN_URL = "https://api.fitbit.com/oauth2/token"
    API_BASE = "https://api.fitbit.com"

    # Required OAuth scopes
    SCOPES = [
        "activity",
        "heartrate",
        "sleep",
        "weight",
        "profile",
    ]

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ):
        """
        Initialize Fitbit provider.

        Args:
            client_id: Fitbit OAuth client ID (or FITBIT_CLIENT_ID env var)
            client_secret: Fitbit OAuth client secret (or FITBIT_CLIENT_SECRET env var)
        """
        super().__init__(
            name="fitbit",
            client_id=client_id or os.getenv("FITBIT_CLIENT_ID"),
            client_secret=client_secret or os.getenv("FITBIT_CLIENT_SECRET"),
        )
        self.client = httpx.AsyncClient(timeout=30.0)

    @property
    def supported_metrics(self) -> List[MetricType]:
        return [
            MetricType.HR,
            MetricType.HRV,
            MetricType.STEPS,
            MetricType.CALORIES_BURNED,
            MetricType.ACTIVE_MINUTES,
            MetricType.DISTANCE,
            MetricType.FLOORS,
            MetricType.SLEEP_DURATION,
            MetricType.SLEEP_DEEP,
            MetricType.SLEEP_REM,
            MetricType.SLEEP_LIGHT,
            MetricType.SLEEP_SCORE,
            MetricType.WEIGHT,
            MetricType.BODY_FAT,
            MetricType.BMI,
            MetricType.SPO2,
        ]

    async def _connect_impl(self) -> bool:
        """Verify Fitbit connection."""
        if not self.credentials:
            return False

        try:
            # Test API access
            response = await self._api_request("GET", "/1/user/-/profile.json")
            return response is not None
        except Exception as e:
            logger.error(f"Fitbit connection test failed: {e}")
            return False

    async def _disconnect_impl(self):
        """Cleanup on disconnect."""
        pass

    def _get_auth_url_impl(self, redirect_uri: str, state: str) -> str:
        """Generate Fitbit OAuth URL."""
        scope = "%20".join(self.SCOPES)
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
        import base64

        auth = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()

        try:
            response = await self.client.post(
                self.TOKEN_URL,
                headers={
                    "Authorization": f"Basic {auth}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                },
            )

            if response.status_code != 200:
                logger.error(f"Token exchange failed: {response.text}")
                return None

            data = response.json()

            return OAuthCredentials(
                access_token=data["access_token"],
                refresh_token=data.get("refresh_token"),
                token_type=data.get("token_type", "Bearer"),
                expires_at=datetime.now() + timedelta(seconds=data.get("expires_in", 3600)),
                scope=data.get("scope", ""),
            )

        except Exception as e:
            logger.error(f"Token exchange error: {e}")
            return None

    async def _refresh_token_impl(self) -> Optional[OAuthCredentials]:
        """Refresh the access token."""
        import base64

        auth = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()

        try:
            response = await self.client.post(
                self.TOKEN_URL,
                headers={
                    "Authorization": f"Basic {auth}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self.credentials.refresh_token,
                },
            )

            if response.status_code != 200:
                logger.error(f"Token refresh failed: {response.text}")
                return None

            data = response.json()

            return OAuthCredentials(
                access_token=data["access_token"],
                refresh_token=data.get("refresh_token", self.credentials.refresh_token),
                token_type=data.get("token_type", "Bearer"),
                expires_at=datetime.now() + timedelta(seconds=data.get("expires_in", 3600)),
                scope=data.get("scope", ""),
            )

        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return None

    async def _api_request(
        self,
        method: str,
        endpoint: str,
        **kwargs,
    ) -> Optional[dict]:
        """Make authenticated API request."""
        if not self.credentials:
            return None

        headers = {
            "Authorization": f"Bearer {self.credentials.access_token}",
            "Accept": "application/json",
        }

        try:
            response = await self.client.request(
                method,
                f"{self.API_BASE}{endpoint}",
                headers=headers,
                **kwargs,
            )

            if response.status_code == 401:
                # Token expired, try refresh
                if await self.refresh_token():
                    return await self._api_request(method, endpoint, **kwargs)
                return None

            if response.status_code == 429:
                # Rate limited
                self.status.rate_limited = True
                logger.warning("Fitbit API rate limited")
                return None

            if response.status_code != 200:
                logger.error(f"Fitbit API error: {response.status_code} - {response.text}")
                return None

            return response.json()

        except Exception as e:
            logger.error(f"Fitbit API request error: {e}")
            return None

    async def _fetch_metrics_impl(
        self,
        metric_types: List[MetricType],
        start_date: date,
        end_date: date,
    ) -> List[HealthMetricReading]:
        """Fetch historical metrics from Fitbit."""
        readings = []

        for metric_type in metric_types:
            try:
                if metric_type == MetricType.HR:
                    readings.extend(await self._fetch_heart_rate(start_date, end_date))
                elif metric_type == MetricType.STEPS:
                    readings.extend(await self._fetch_activity_metric("steps", start_date, end_date))
                elif metric_type == MetricType.CALORIES_BURNED:
                    readings.extend(await self._fetch_activity_metric("calories", start_date, end_date))
                elif metric_type == MetricType.DISTANCE:
                    readings.extend(await self._fetch_activity_metric("distance", start_date, end_date))
                elif metric_type == MetricType.FLOORS:
                    readings.extend(await self._fetch_activity_metric("floors", start_date, end_date))
                elif metric_type in [MetricType.SLEEP_DURATION, MetricType.SLEEP_DEEP,
                                     MetricType.SLEEP_REM, MetricType.SLEEP_LIGHT]:
                    readings.extend(await self._fetch_sleep(start_date, end_date))
                elif metric_type == MetricType.WEIGHT:
                    readings.extend(await self._fetch_weight(start_date, end_date))
            except Exception as e:
                logger.error(f"Error fetching {metric_type}: {e}")

        return readings

    async def _fetch_heart_rate(
        self,
        start_date: date,
        end_date: date,
    ) -> List[HealthMetricReading]:
        """Fetch heart rate data."""
        readings = []
        date_str = start_date.isoformat()
        end_str = end_date.isoformat()

        # Fitbit limits to 1 month of intraday data
        data = await self._api_request(
            "GET",
            f"/1/user/-/activities/heart/date/{date_str}/{end_str}.json"
        )

        if not data or "activities-heart" not in data:
            return readings

        for day_data in data["activities-heart"]:
            day_date = date.fromisoformat(day_data["dateTime"])

            # Resting heart rate
            if "restingHeartRate" in day_data.get("value", {}):
                readings.append(HealthMetricReading(
                    metric_type=MetricType.HR,
                    value=day_data["value"]["restingHeartRate"],
                    unit="bpm",
                    timestamp=datetime.combine(day_date, datetime.min.time()),
                    source="fitbit",
                    metadata={"type": "resting"},
                ))

        return readings

    async def _fetch_activity_metric(
        self,
        resource: str,
        start_date: date,
        end_date: date,
    ) -> List[HealthMetricReading]:
        """Fetch activity metrics (steps, calories, distance, floors)."""
        readings = []

        data = await self._api_request(
            "GET",
            f"/1/user/-/activities/{resource}/date/{start_date.isoformat()}/{end_date.isoformat()}.json"
        )

        if not data:
            return readings

        resource_key = f"activities-{resource}"
        if resource_key not in data:
            return readings

        metric_map = {
            "steps": (MetricType.STEPS, "steps"),
            "calories": (MetricType.CALORIES_BURNED, "kcal"),
            "distance": (MetricType.DISTANCE, "km"),
            "floors": (MetricType.FLOORS, "floors"),
        }

        metric_type, unit = metric_map.get(resource, (MetricType.STEPS, ""))

        for entry in data[resource_key]:
            readings.append(HealthMetricReading(
                metric_type=metric_type,
                value=float(entry["value"]),
                unit=unit,
                timestamp=datetime.fromisoformat(entry["dateTime"]),
                source="fitbit",
            ))

        return readings

    async def _fetch_sleep(
        self,
        start_date: date,
        end_date: date,
    ) -> List[HealthMetricReading]:
        """Fetch sleep data."""
        readings = []

        data = await self._api_request(
            "GET",
            f"/1.2/user/-/sleep/date/{start_date.isoformat()}/{end_date.isoformat()}.json"
        )

        if not data or "sleep" not in data:
            return readings

        for sleep in data["sleep"]:
            sleep_date = datetime.fromisoformat(sleep["startTime"])

            # Total duration
            if "duration" in sleep:
                readings.append(HealthMetricReading(
                    metric_type=MetricType.SLEEP_DURATION,
                    value=sleep["duration"] / 3600000,  # ms to hours
                    unit="hours",
                    timestamp=sleep_date,
                    source="fitbit",
                ))

            # Sleep stages if available
            if "levels" in sleep and "summary" in sleep["levels"]:
                summary = sleep["levels"]["summary"]

                if "deep" in summary:
                    readings.append(HealthMetricReading(
                        metric_type=MetricType.SLEEP_DEEP,
                        value=summary["deep"]["minutes"] / 60,
                        unit="hours",
                        timestamp=sleep_date,
                        source="fitbit",
                    ))

                if "rem" in summary:
                    readings.append(HealthMetricReading(
                        metric_type=MetricType.SLEEP_REM,
                        value=summary["rem"]["minutes"] / 60,
                        unit="hours",
                        timestamp=sleep_date,
                        source="fitbit",
                    ))

                if "light" in summary:
                    readings.append(HealthMetricReading(
                        metric_type=MetricType.SLEEP_LIGHT,
                        value=summary["light"]["minutes"] / 60,
                        unit="hours",
                        timestamp=sleep_date,
                        source="fitbit",
                    ))

        return readings

    async def _fetch_weight(
        self,
        start_date: date,
        end_date: date,
    ) -> List[HealthMetricReading]:
        """Fetch weight data."""
        readings = []

        data = await self._api_request(
            "GET",
            f"/1/user/-/body/log/weight/date/{start_date.isoformat()}/{end_date.isoformat()}.json"
        )

        if not data or "weight" not in data:
            return readings

        for entry in data["weight"]:
            readings.append(HealthMetricReading(
                metric_type=MetricType.WEIGHT,
                value=entry["weight"],
                unit="kg",
                timestamp=datetime.fromisoformat(f"{entry['date']}T{entry.get('time', '00:00:00')}"),
                source="fitbit",
            ))

            if "fat" in entry:
                readings.append(HealthMetricReading(
                    metric_type=MetricType.BODY_FAT,
                    value=entry["fat"],
                    unit="%",
                    timestamp=datetime.fromisoformat(f"{entry['date']}T{entry.get('time', '00:00:00')}"),
                    source="fitbit",
                ))

            if "bmi" in entry:
                readings.append(HealthMetricReading(
                    metric_type=MetricType.BMI,
                    value=entry["bmi"],
                    unit="",
                    timestamp=datetime.fromisoformat(f"{entry['date']}T{entry.get('time', '00:00:00')}"),
                    source="fitbit",
                ))

        return readings

    async def _fetch_daily_summary_impl(self, target_date: date) -> Optional[DailySummary]:
        """Fetch daily summary from Fitbit."""
        summary = DailySummary(date=target_date)

        # Fetch activity summary
        activity_data = await self._api_request(
            "GET",
            f"/1/user/-/activities/date/{target_date.isoformat()}.json"
        )

        if activity_data and "summary" in activity_data:
            s = activity_data["summary"]
            summary.total_steps = s.get("steps", 0)
            summary.total_calories_burned = s.get("caloriesOut", 0)
            summary.active_minutes = s.get("veryActiveMinutes", 0) + s.get("fairlyActiveMinutes", 0)
            summary.distance_km = sum(d["distance"] for d in s.get("distances", []) if d["activity"] == "total")
            summary.floors_climbed = s.get("floors", 0)
            summary.resting_heart_rate = s.get("restingHeartRate")

        # Fetch sleep summary
        sleep_data = await self._api_request(
            "GET",
            f"/1.2/user/-/sleep/date/{target_date.isoformat()}.json"
        )

        if sleep_data and "summary" in sleep_data:
            s = sleep_data["summary"]
            summary.sleep_duration_hours = s.get("totalMinutesAsleep", 0) / 60
            summary.sleep_score = s.get("overallScore")

            if "stages" in s:
                summary.deep_sleep_hours = s["stages"].get("deep", 0) / 60
                summary.rem_sleep_hours = s["stages"].get("rem", 0) / 60

        return summary
