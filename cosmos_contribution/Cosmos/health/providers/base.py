"""
cosmos Health Provider Base Classes

Abstract base class for health data providers, extending BioInterfaceProvider
with OAuth support, historical data fetching, and real-time streaming.
"""

import asyncio
import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, AsyncIterator

from ..models import (
    HealthMetricReading,
    DailySummary,
    MetricType,
)

try:
    from Cosmos.integration.bio.interface import (
        BioInterfaceProvider,
        BioDataPacket,
    )
except ImportError:
    # Fallback if bio interface not available
    from abc import ABC

    class BioInterfaceProvider(ABC):
        """Fallback base class."""
        pass

    @dataclass
    class BioDataPacket:
        timestamp: datetime = field(default_factory=datetime.now)
        source_device: str = "unknown"
        signal_type: str = "unknown"
        raw_value: Any = None
        processed_value: Optional[float] = None
        metadata: Dict[str, Any] = field(default_factory=dict)


logger = logging.getLogger(__name__)


@dataclass
class OAuthCredentials:
    """OAuth credentials for health provider authentication."""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "Bearer"
    expires_at: Optional[datetime] = None
    scope: str = ""

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() >= self.expires_at

    def to_dict(self) -> dict:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_type": self.token_type,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "scope": self.scope,
        }


@dataclass
class ProviderStatus:
    """Status information for a health provider."""
    name: str
    connected: bool = False
    last_sync: Optional[datetime] = None
    error: Optional[str] = None
    supported_metrics: List[MetricType] = field(default_factory=list)
    rate_limited: bool = False
    rate_limit_reset: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "connected": self.connected,
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "error": self.error,
            "supported_metrics": [m.value for m in self.supported_metrics],
            "rate_limited": self.rate_limited,
        }


class HealthProvider(BioInterfaceProvider):
    """
    Abstract base class for health data providers.

    Extends BioInterfaceProvider with:
    - OAuth authentication
    - Historical data fetching
    - Daily summary generation
    - Real-time streaming where supported
    """

    def __init__(
        self,
        name: str,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ):
        self.name = name
        self.client_id = client_id
        self.client_secret = client_secret
        self.credentials: Optional[OAuthCredentials] = None
        self.status = ProviderStatus(name=name)
        self._connected = False
        self._streaming = False

    @property
    @abstractmethod
    def supported_metrics(self) -> List[MetricType]:
        """List of metric types this provider can supply."""
        pass

    @property
    def requires_oauth(self) -> bool:
        """Whether this provider requires OAuth authentication."""
        return True

    # ============================================
    # BioInterfaceProvider Implementation
    # ============================================

    async def connect(self) -> bool:
        """Connect to the health provider."""
        try:
            if self.requires_oauth and not self.credentials:
                logger.warning(f"{self.name}: No credentials provided")
                return False

            if self.credentials and self.credentials.is_expired:
                if self.credentials.refresh_token:
                    await self.refresh_token()
                else:
                    logger.warning(f"{self.name}: Token expired and no refresh token")
                    return False

            # Provider-specific connection logic
            success = await self._connect_impl()

            if success:
                self._connected = True
                self.status.connected = True
                self.status.error = None
                self.status.supported_metrics = self.supported_metrics
                logger.info(f"{self.name}: Connected successfully")
            else:
                self.status.error = "Connection failed"

            return success

        except Exception as e:
            logger.error(f"{self.name}: Connection error - {e}")
            self.status.error = str(e)
            return False

    async def disconnect(self):
        """Disconnect from the health provider."""
        self._connected = False
        self._streaming = False
        self.status.connected = False
        await self._disconnect_impl()
        logger.info(f"{self.name}: Disconnected")

    async def get_stream(self) -> AsyncIterator[BioDataPacket]:
        """Get real-time data stream if supported."""
        if not self._connected:
            logger.warning(f"{self.name}: Not connected, cannot stream")
            return

        self._streaming = True
        try:
            async for packet in self._stream_impl():
                yield packet
        finally:
            self._streaming = False

    def health_check(self) -> bool:
        """Check if provider is healthy."""
        return self._connected and not self.status.rate_limited

    # ============================================
    # OAuth Methods
    # ============================================

    def get_auth_url(self, redirect_uri: str, state: str = "") -> str:
        """Get OAuth authorization URL."""
        return self._get_auth_url_impl(redirect_uri, state)

    async def exchange_code(self, code: str, redirect_uri: str) -> bool:
        """Exchange authorization code for tokens."""
        try:
            self.credentials = await self._exchange_code_impl(code, redirect_uri)
            return self.credentials is not None
        except Exception as e:
            logger.error(f"{self.name}: Token exchange error - {e}")
            return False

    async def refresh_token(self) -> bool:
        """Refresh the access token."""
        if not self.credentials or not self.credentials.refresh_token:
            return False

        try:
            self.credentials = await self._refresh_token_impl()
            return self.credentials is not None
        except Exception as e:
            logger.error(f"{self.name}: Token refresh error - {e}")
            return False

    def set_credentials(self, credentials: OAuthCredentials):
        """Set OAuth credentials directly."""
        self.credentials = credentials

    # ============================================
    # Data Fetching Methods
    # ============================================

    async def get_metrics(
        self,
        metric_types: List[MetricType],
        start_date: date,
        end_date: date,
    ) -> List[HealthMetricReading]:
        """
        Fetch historical metrics for a date range.

        Args:
            metric_types: List of metric types to fetch
            start_date: Start of date range
            end_date: End of date range

        Returns:
            List of HealthMetricReading objects
        """
        if not self._connected:
            logger.warning(f"{self.name}: Not connected")
            return []

        # Filter to supported metrics
        supported = [m for m in metric_types if m in self.supported_metrics]
        if not supported:
            return []

        try:
            readings = await self._fetch_metrics_impl(supported, start_date, end_date)
            self.status.last_sync = datetime.now()
            return readings
        except Exception as e:
            logger.error(f"{self.name}: Error fetching metrics - {e}")
            self.status.error = str(e)
            return []

    async def get_daily_summary(self, target_date: date) -> Optional[DailySummary]:
        """
        Get aggregated daily summary for a specific date.

        Args:
            target_date: Date to get summary for

        Returns:
            DailySummary or None
        """
        if not self._connected:
            return None

        try:
            return await self._fetch_daily_summary_impl(target_date)
        except Exception as e:
            logger.error(f"{self.name}: Error fetching daily summary - {e}")
            return None

    async def get_latest(self, metric_type: MetricType) -> Optional[HealthMetricReading]:
        """Get the most recent reading for a metric type."""
        if not self._connected or metric_type not in self.supported_metrics:
            return None

        try:
            return await self._fetch_latest_impl(metric_type)
        except Exception as e:
            logger.error(f"{self.name}: Error fetching latest - {e}")
            return None

    # ============================================
    # Abstract Implementation Methods
    # ============================================

    @abstractmethod
    async def _connect_impl(self) -> bool:
        """Provider-specific connection logic."""
        pass

    async def _disconnect_impl(self):
        """Provider-specific disconnection logic."""
        pass

    async def _stream_impl(self) -> AsyncIterator[BioDataPacket]:
        """Provider-specific streaming implementation."""
        # Default: no streaming supported
        return
        yield  # Make this a generator

    def _get_auth_url_impl(self, redirect_uri: str, state: str) -> str:
        """Provider-specific OAuth URL generation."""
        return ""

    async def _exchange_code_impl(
        self, code: str, redirect_uri: str
    ) -> Optional[OAuthCredentials]:
        """Provider-specific code exchange."""
        return None

    async def _refresh_token_impl(self) -> Optional[OAuthCredentials]:
        """Provider-specific token refresh."""
        return None

    @abstractmethod
    async def _fetch_metrics_impl(
        self,
        metric_types: List[MetricType],
        start_date: date,
        end_date: date,
    ) -> List[HealthMetricReading]:
        """Provider-specific metrics fetching."""
        pass

    async def _fetch_daily_summary_impl(self, target_date: date) -> Optional[DailySummary]:
        """Provider-specific daily summary fetching."""
        return None

    async def _fetch_latest_impl(
        self, metric_type: MetricType
    ) -> Optional[HealthMetricReading]:
        """Provider-specific latest reading fetching."""
        return None


class HealthProviderManager:
    """
    Manages multiple health providers and aggregates data.

    Handles:
    - Provider registration and lifecycle
    - Data aggregation across providers
    - Conflict resolution for overlapping data
    - Real-time streaming aggregation
    """

    def __init__(self):
        self.providers: Dict[str, HealthProvider] = {}
        self._callbacks: List[callable] = []
        self._streaming_tasks: List[asyncio.Task] = []

    def register_provider(self, provider: HealthProvider):
        """Register a health provider."""
        self.providers[provider.name] = provider
        logger.info(f"Registered health provider: {provider.name}")

    def unregister_provider(self, name: str):
        """Unregister a health provider."""
        if name in self.providers:
            del self.providers[name]
            logger.info(f"Unregistered health provider: {name}")

    def add_subscriber(self, callback: callable):
        """Add a callback for real-time data."""
        self._callbacks.append(callback)

    async def connect_all(self) -> Dict[str, bool]:
        """Connect to all registered providers."""
        results = {}
        tasks = [
            self._connect_provider(name, provider)
            for name, provider in self.providers.items()
        ]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for (name, _), result in zip(self.providers.items(), completed):
            if isinstance(result, Exception):
                results[name] = False
                logger.error(f"Error connecting to {name}: {result}")
            else:
                results[name] = result

        return results

    async def _connect_provider(self, name: str, provider: HealthProvider) -> bool:
        """Connect to a single provider."""
        return await provider.connect()

    async def disconnect_all(self):
        """Disconnect from all providers."""
        # Stop streaming tasks
        for task in self._streaming_tasks:
            task.cancel()
        self._streaming_tasks.clear()

        # Disconnect providers
        for provider in self.providers.values():
            await provider.disconnect()

    async def start_streaming(self):
        """Start streaming from all providers that support it."""
        for name, provider in self.providers.items():
            if provider.health_check():
                task = asyncio.create_task(self._stream_provider(name, provider))
                self._streaming_tasks.append(task)

    async def _stream_provider(self, name: str, provider: HealthProvider):
        """Stream data from a provider."""
        try:
            async for packet in provider.get_stream():
                self._broadcast(packet)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Streaming error from {name}: {e}")

    def _broadcast(self, packet: BioDataPacket):
        """Broadcast data packet to all subscribers."""
        for callback in self._callbacks:
            try:
                callback(packet)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def get_status(self) -> Dict[str, dict]:
        """Get status of all providers."""
        return {name: p.status.to_dict() for name, p in self.providers.items()}

    def get_supported_metrics(self) -> List[MetricType]:
        """Get all metrics supported by any connected provider."""
        metrics = set()
        for provider in self.providers.values():
            if provider.status.connected:
                metrics.update(provider.supported_metrics)
        return list(metrics)

    async def get_metrics(
        self,
        metric_types: List[MetricType],
        start_date: date,
        end_date: date,
        providers: Optional[List[str]] = None,
    ) -> List[HealthMetricReading]:
        """
        Fetch metrics from providers, with conflict resolution.

        Args:
            metric_types: Types of metrics to fetch
            start_date: Start of date range
            end_date: End of date range
            providers: Specific providers to query (None = all)

        Returns:
            Deduplicated list of readings
        """
        all_readings: List[HealthMetricReading] = []
        target_providers = (
            [self.providers[p] for p in providers if p in self.providers]
            if providers
            else list(self.providers.values())
        )

        # Fetch from all providers in parallel
        tasks = [
            p.get_metrics(metric_types, start_date, end_date)
            for p in target_providers
            if p.status.connected
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_readings.extend(result)

        # Resolve conflicts (prefer higher quality readings)
        return self._resolve_conflicts(all_readings)

    def _resolve_conflicts(
        self, readings: List[HealthMetricReading]
    ) -> List[HealthMetricReading]:
        """
        Resolve conflicts when multiple providers report same metric.

        Strategy: Group by (timestamp, metric_type), keep highest quality.
        """
        if not readings:
            return []

        # Group readings by time window (within 1 minute) and metric type
        groups: Dict[str, List[HealthMetricReading]] = {}
        for reading in readings:
            # Round timestamp to minute for grouping
            minute_ts = reading.timestamp.replace(second=0, microsecond=0)
            key = f"{minute_ts.isoformat()}_{reading.metric_type.value}"
            if key not in groups:
                groups[key] = []
            groups[key].append(reading)

        # Select best reading from each group
        resolved = []
        for group in groups.values():
            # Sort by quality descending, take first
            group.sort(key=lambda r: r.quality, reverse=True)
            resolved.append(group[0])

        # Sort by timestamp
        resolved.sort(key=lambda r: r.timestamp)
        return resolved

    async def get_daily_summary(
        self,
        target_date: date,
        providers: Optional[List[str]] = None,
    ) -> DailySummary:
        """
        Get aggregated daily summary from all providers.

        Merges data from multiple providers into a single summary.
        """
        summary = DailySummary(date=target_date)

        target_providers = (
            [self.providers[p] for p in providers if p in self.providers]
            if providers
            else list(self.providers.values())
        )

        # Fetch summaries from all providers
        tasks = [
            p.get_daily_summary(target_date)
            for p in target_providers
            if p.status.connected
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge summaries (take first non-None value for each field)
        for result in results:
            if isinstance(result, DailySummary):
                summary = self._merge_summaries(summary, result)

        return summary

    def _merge_summaries(
        self, base: DailySummary, other: DailySummary
    ) -> DailySummary:
        """Merge two daily summaries, preferring non-None values."""
        # For numeric fields, take non-None/non-zero values from other
        if other.avg_heart_rate and not base.avg_heart_rate:
            base.avg_heart_rate = other.avg_heart_rate
        if other.resting_heart_rate and not base.resting_heart_rate:
            base.resting_heart_rate = other.resting_heart_rate
        if other.max_heart_rate and not base.max_heart_rate:
            base.max_heart_rate = other.max_heart_rate
        if other.hrv_avg and not base.hrv_avg:
            base.hrv_avg = other.hrv_avg

        # Take higher values for cumulative metrics
        base.total_steps = max(base.total_steps, other.total_steps)
        base.total_calories_burned = max(
            base.total_calories_burned, other.total_calories_burned
        )
        base.active_minutes = max(base.active_minutes, other.active_minutes)
        base.distance_km = max(base.distance_km, other.distance_km)
        base.floors_climbed = max(base.floors_climbed, other.floors_climbed)

        # Sleep data
        if other.sleep_duration_hours > base.sleep_duration_hours:
            base.sleep_duration_hours = other.sleep_duration_hours
            base.deep_sleep_hours = other.deep_sleep_hours
            base.rem_sleep_hours = other.rem_sleep_hours
        if other.sleep_score and not base.sleep_score:
            base.sleep_score = other.sleep_score

        # Scores
        if other.recovery_score and not base.recovery_score:
            base.recovery_score = other.recovery_score
        if other.readiness_score and not base.readiness_score:
            base.readiness_score = other.readiness_score

        return base
