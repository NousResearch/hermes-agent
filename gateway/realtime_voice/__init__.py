"""Provider-neutral realtime voice primitives for gateway integrations."""

from .config import DEFAULT_REALTIME_TOOLS, RealtimeVoiceConfig
from .providers import (
    RealtimeVoiceProviderError,
    create_realtime_voice_session,
    is_realtime_voice_provider_supported,
    register_realtime_voice_provider,
)
from .session import (
    RealtimeAudioDelta,
    RealtimeToolCall,
    RealtimeTranscriptDelta,
    RealtimeVoiceSession,
)
from .xai import XAIRealtimeVoiceSession

__all__ = [
    "DEFAULT_REALTIME_TOOLS",
    "RealtimeVoiceConfig",
    "RealtimeVoiceProviderError",
    "create_realtime_voice_session",
    "is_realtime_voice_provider_supported",
    "register_realtime_voice_provider",
    "RealtimeAudioDelta",
    "RealtimeToolCall",
    "RealtimeTranscriptDelta",
    "RealtimeVoiceSession",
    "XAIRealtimeVoiceSession",
]
