"""
cosmos Oura Ring Health Provider

Integration with Oura Ring API v2 for comprehensive health metrics.
"""

import os
import logging
from datetime import datetime, date, timedelta
from typing import List, Optional

import httpx

from .base import HealthProvider, OAuthCredentials
from ..models import MetricType, HealthMetricReading, DailySummary

logger = logging.getLogger(__name__)


class OuraProvider(HealthProvider):
    """
    Oura Ring API v2 integration.

    Supports:
    - Sleep tracking (duration, stages, score)
    - Readiness score
    - Activity tracking
    - Heart rate and HRV
    - Body temperature deviation
    - SpO2
    """

    API_BASE = "https://api.ouraring.com/v2"
    AUTH_URL = "https://cloud.ouraring.com/oauth/authorize"
    TOKEN_URL = "https://api.ouraring.com/oauth/token"

    def __init__(
        self,
        access_token: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ):
        """
        Initialize Oura provider.

        Args:
            access_token: Personal access token (or OURA_ACCESS_TOKEN env var)
            client_id: OAuth client ID (for OAuth flow)
            client_secret: OAuth client secret
        """
        super().__init__(
            name="oura",
            client_id=client_id or os.getenv("OURA_CLIENT_ID"),
            client_secret=client_secret or os.getenv("OURA_CLIENT_SECRET"),
        )

        # Support direct token access (simpler for personal use)
        token = access_token or os.getenv("OURA_ACCESS_TOKEN")
        if token:
            self.credentials = OAuthCredentials(access_token=token)

        self.client = httpx.AsyncClient(timeout=30.0)

    @property
    def supported_metrics(self) -> List[MetricType]:
        return [
            MetricType.HR,
            MetricType.HRV,
            MetricType.SPO2,
            MetricType.SLEEP_DURATION,
            MetricType.SLEEP_DEEP,
            MetricType.SLEEP_REM,
            MetricType.SLEEP_LIGHT,
            MetricType.SLEEP_SCORE,
            MetricType.READINESS_SCORE,
            MetricType.RECOVERY_SCORE,
            MetricType.STEPS,
            MetricType.CALORIES_BURNED,
            MetricType.ACTIVE_MINUTES,
        ]

    @property
    def requires_oauth(self) -> bool:
        # Can use personal access token directly
        return self.credentials is None

    async def _connect_impl(self) -> bool:
        """Verify Oura connection."""
        if not self.credentials:
            return False

        try:
            # Test API access
            response = await self._api_request("GET", "/usercollection/personal_info")
            return response is not None
        except Exception as e:
            logger.error(f"Oura connection test failed: {e}")
            return False

    def _get_auth_url_impl(self, redirect_uri: str, state: str) -> str:
        """Generate Oura OAuth URL."""
        return (
            f"{self.AUTH_URL}?"
            f"response_type=code"
            f"&client_id={self.client_id}"
            f"&redirect_uri={redirect_uri}"
            f"&state={state}"
            f"&scope=daily%20heartrate%20workout%20sleep%20personal"
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
                # Token expired
                if self.credentials.refresh_token and await self.refresh_token():
                    return await self._api_request(method, endpoint, params)
                return None

            if response.status_code == 429:
                self.status.rate_limited = True
                return None

            if response.status_code != 200:
                logger.error(f"Oura API error: {response.status_code}")
                return None

            return response.json()

        except Exception as e:
            logger.error(f"Oura API request error: {e}")
            return None

    async def _fetch_metrics_impl(
        self,
        metric_types: List[MetricType],
        start_date: date,
        end_date: date,
    ) -> List[HealthMetricReading]:
        """Fetch historical metrics from Oura."""
        readings = []

        # Check which data types we need
        need_sleep = any(m in [MetricType.SLEEP_DURATION, MetricType.SLEEP_DEEP,
                               MetricType.SLEEP_REM, MetricType.SLEEP_LIGHT,
                               MetricType.SLEEP_SCORE] for m in metric_types)
        need_hr = any(m in [MetricType.HR, MetricType.HRV] for m in metric_types)
        need_activity = any(m in [MetricType.STEPS, MetricType.CALORIES_BURNED,
                                  MetricType.ACTIVE_MINUTES] for m in metric_types)
        need_readiness = MetricType.READINESS_SCORE in metric_types or \
                         MetricType.RECOVERY_SCORE in metric_types

        if need_sleep:
            readings.extend(await self._fetch_sleep(start_date, end_date, metric_types))

        if need_hr:
            readings.extend(await self._fetch_heart_rate(start_date, end_date, metric_types))

        if need_activity:
            readings.extend(await self._fetch_activity(start_date, end_date, metric_types))

        if need_readiness:
            readings.extend(await self._fetch_readiness(start_date, end_date))

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
            "/usercollection/sleep",
            params={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
        )

        if not data or "data" not in data:
            return readings

        for sleep in data["data"]:
            sleep_date = datetime.fromisoformat(sleep["day"])

            # Sleep duration
            if MetricType.SLEEP_DURATION in metric_types and "total_sleep_duration" in sleep:
                readings.append(HealthMetricReading(
                    metric_type=MetricType.SLEEP_DURATION,
                    value=sleep["total_sleep_duration"] / 3600,  # seconds to hours
                    unit="hours",
                    timestamp=sleep_date,
                    source="oura",
                ))

            # Deep sleep
            if MetricType.SLEEP_DEEP in metric_types and "deep_sleep_duration" in sleep:
                readings.append(HealthMetricReading(
                    metric_type=MetricType.SLEEP_DEEP,
                    value=sleep["deep_sleep_duration"] / 3600,
                    unit="hours",
                    timestamp=sleep_date,
                    source="oura",
                ))

            # REM sleep
            if MetricType.SLEEP_REM in metric_types and "rem_sleep_duration" in sleep:
                readings.append(HealthMetricReading(
                    metric_type=MetricType.SLEEP_REM,
                    value=sleep["rem_sleep_duration"] / 3600,
                    unit="hours",
                    timestamp=sleep_date,
                    source="oura",
                ))

            # Light sleep
            if MetricType.SLEEP_LIGHT in metric_types and "light_sleep_duration" in sleep:
                readings.append(HealthMetricReading(
                    metric_type=MetricType.SLEEP_LIGHT,
                    value=sleep["light_sleep_duration"] / 3600,
                    unit="hours",
                    timestamp=sleep_date,
                    source="oura",
                ))

        # Fetch sleep scores separately
        if MetricType.SLEEP_SCORE in metric_types:
            score_data = await self._api_request(
                "GET",
                "/usercollection/daily_sleep",
                params={
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                },
            )

            if score_data and "data" in score_data:
                for day in score_data["data"]:
                    if "score" in day:
                        readings.append(HealthMetricReading(
                            metric_type=MetricType.SLEEP_SCORE,
                            value=day["score"],
                            unit="score",
                            timestamp=datetime.fromisoformat(day["day"]),
                            source="oura",
                        ))

        return readings

    async def _fetch_heart_rate(
        self,
        start_date: date,
        end_date: date,
        metric_types: List[MetricType],
    ) -> List[HealthMetricReading]:
        """Fetch heart rate data."""
        readings = []

        data = await self._api_request(
            "GET",
            "/usercollection/heartrate",
            params={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
        )

        if not data or "data" not in data:
            return readings

        for hr in data["data"]:
            ts = datetime.fromisoformat(hr["timestamp"])

            if MetricType.HR in metric_types:
                readings.append(HealthMetricReading(
                    metric_type=MetricType.HR,
                    value=hr["bpm"],
                    unit="bpm",
                    timestamp=ts,
                    source="oura",
                ))

        # HRV from sleep data
        if MetricType.HRV in metric_types:
            sleep_data = await self._api_request(
                "GET",
                "/usercollection/sleep",
                params={
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                },
            )

            if sleep_data and "data" in sleep_data:
                for sleep in sleep_data["data"]:
                    if "average_hrv" in sleep:
                        readings.append(HealthMetricReading(
                            metric_type=MetricType.HRV,
                            value=sleep["average_hrv"],
                            unit="ms",
                            timestamp=datetime.fromisoformat(sleep["day"]),
                            source="oura",
                        ))

        return readings

    async def _fetch_activity(
        self,
        start_date: date,
        end_date: date,
        metric_types: List[MetricType],
    ) -> List[HealthMetricReading]:
        """Fetch activity data."""
        readings = []

        data = await self._api_request(
            "GET",
            "/usercollection/daily_activity",
            params={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
        )

        if not data or "data" not in data:
            return readings

        for activity in data["data"]:
            day_date = datetime.fromisoformat(activity["day"])

            if MetricType.STEPS in metric_types and "steps" in activity:
                readings.append(HealthMetricReading(
                    metric_type=MetricType.STEPS,
                    value=activity["steps"],
                    unit="steps",
                    timestamp=day_date,
                    source="oura",
                ))

            if MetricType.CALORIES_BURNED in metric_types and "total_calories" in activity:
                readings.append(HealthMetricReading(
                    metric_type=MetricType.CALORIES_BURNED,
                    value=activity["total_calories"],
                    unit="kcal",
                    timestamp=day_date,
                    source="oura",
                ))

            if MetricType.ACTIVE_MINUTES in metric_types:
                active = activity.get("high_activity_time", 0) + \
                         activity.get("medium_activity_time", 0)
                readings.append(HealthMetricReading(
                    metric_type=MetricType.ACTIVE_MINUTES,
                    value=active // 60,  # seconds to minutes
                    unit="min",
                    timestamp=day_date,
                    source="oura",
                ))

        return readings

    async def _fetch_readiness(
        self,
        start_date: date,
        end_date: date,
    ) -> List[HealthMetricReading]:
        """Fetch readiness scores."""
        readings = []

        data = await self._api_request(
            "GET",
            "/usercollection/daily_readiness",
            params={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
        )

        if not data or "data" not in data:
            return readings

        for day in data["data"]:
            if "score" in day:
                readings.append(HealthMetricReading(
                    metric_type=MetricType.READINESS_SCORE,
                    value=day["score"],
                    unit="score",
                    timestamp=datetime.fromisoformat(day["day"]),
                    source="oura",
                ))
                # Also add as recovery score (similar concept)
                readings.append(HealthMetricReading(
                    metric_type=MetricType.RECOVERY_SCORE,
                    value=day["score"],
                    unit="score",
                    timestamp=datetime.fromisoformat(day["day"]),
                    source="oura",
                ))

        return readings

    async def _fetch_daily_summary_impl(self, target_date: date) -> Optional[DailySummary]:
        """Fetch daily summary from Oura."""
        summary = DailySummary(date=target_date)

        # Get all data for the day
        date_str = target_date.isoformat()

        # Sleep
        sleep_data = await self._api_request(
            "GET",
            "/usercollection/daily_sleep",
            params={"start_date": date_str, "end_date": date_str},
        )
        if sleep_data and "data" in sleep_data and sleep_data["data"]:
            s = sleep_data["data"][0]
            summary.sleep_score = s.get("score")

        # Activity
        activity_data = await self._api_request(
            "GET",
            "/usercollection/daily_activity",
            params={"start_date": date_str, "end_date": date_str},
        )
        if activity_data and "data" in activity_data and activity_data["data"]:
            a = activity_data["data"][0]
            summary.total_steps = a.get("steps", 0)
            summary.total_calories_burned = a.get("total_calories", 0)
            summary.active_minutes = (a.get("high_activity_time", 0) +
                                      a.get("medium_activity_time", 0)) // 60

        # Readiness
        readiness_data = await self._api_request(
            "GET",
            "/usercollection/daily_readiness",
            params={"start_date": date_str, "end_date": date_str},
        )
        if readiness_data and "data" in readiness_data and readiness_data["data"]:
            r = readiness_data["data"][0]
            summary.readiness_score = r.get("score")
            summary.recovery_score = r.get("score")

        return summary
