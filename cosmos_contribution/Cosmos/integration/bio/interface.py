from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import logging

logger = logging.getLogger(__name__)

@dataclass
class BioDataPacket:
    timestamp: datetime = field(default_factory=datetime.now)
    source_device: str = "unknown"
    signal_type: str = "unknown" # e.g., "EEG", "HRV", "GSR"
    raw_value: Any = None
    processed_value: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
class BioInterfaceProvider(ABC):
    """
    Abstract Base Class for Biological Interface Providers (e.g., Muse, OpenBCI, Apple Health).
    """
    @abstractmethod
    async def connect(self) -> bool:
        pass

    @abstractmethod
    async def disconnect(self):
        pass

    @abstractmethod
    async def get_stream(self) -> Any:
        # Returns an async generator or stream
        pass

    @abstractmethod
    def health_check(self) -> bool:
        pass

class BioIntegrationManager:
    """
    Manages connections to multiple biological interfaces and normalizes data streams.
    """
    def __init__(self):
        self.providers: Dict[str, BioInterfaceProvider] = {}
        self.active_streams: List[asyncio.Task] = []
        self._callbacks: List[Callable[[BioDataPacket], None]] = []

    def register_provider(self, name: str, provider: BioInterfaceProvider):
        self.providers[name] = provider
        logger.info(f"Registered BioProvider: {name}")

    def add_subscriber(self, callback: Callable[[BioDataPacket], None]):
        self._callbacks.append(callback)

    async def start_streaming(self):
        logger.info("Starting Bio-Integration Streams...")
        for name, provider in self.providers.items():
            if await provider.connect():
                task = asyncio.create_task(self._monitor_stream(name, provider))
                self.active_streams.append(task)
            else:
                logger.error(f"Failed to connect to provider: {name}")

    async def _monitor_stream(self, name: str, provider: BioInterfaceProvider):
        try:
            async for packet in provider.get_stream():
                self._broadcast(packet)
        except Exception as e:
            logger.error(f"Error in stream {name}: {e}")

    def _broadcast(self, packet: BioDataPacket):
        for cb in self._callbacks:
            try:
                cb(packet)
            except Exception as e:
                logger.error(f"Error in bio callback: {e}")

    async def stop_all(self):
        for task in self.active_streams:
            task.cancel()
        for provider in self.providers.values():
            await provider.disconnect()

class MockBioProvider(BioInterfaceProvider):
    """
    Simulated Bio-Interface for testing without hardware.
    Generates random HRV and Alpha wave data.
    """
    def __init__(self):
        self.connected = False

    async def connect(self) -> bool:
        self.connected = True
        logger.info("MockBioProvider Connected (Simulated)")
        return True

    async def disconnect(self):
        self.connected = False
        logger.info("MockBioProvider Disconnected")

    async def get_stream(self):
        import random
        while self.connected:
            await asyncio.sleep(1.0) # 1Hz update
            
            # Simulate Heart Rate
            yield BioDataPacket(
                source_device="MockHealth",
                signal_type="HR",
                processed_value=60 + random.random() * 20, # 60-80 BPM
                metadata={"unit": "bpm"}
            )

            # Simulate Focus (Beta Waves)
            yield BioDataPacket(
                source_device="MockEEG",
                signal_type="EEG_BETA",
                processed_value=random.random(), # 0.0 - 1.0
                metadata={"quality": "good"}
            )

    def health_check(self) -> bool:
        return True
