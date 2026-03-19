# Cosmos VTuber Streaming System
# Real-time AI avatar streaming to Twitter/X with swarm collective

"""
Cosmos VTuber Module
========================

A complete AI VTuber streaming system that integrates:
- Multi-backend avatar rendering (Live2D, VTube Studio, Neural, Image Sequence)
- Real-time lip sync from audio/text
- Emotion/expression mapping from AI responses
- RTMPS streaming to Twitter/X
- Chat reading and AI response generation
- Swarm collective integration for multi-agent responses

Quick Start:
    from Cosmos.integration.vtuber import CosmosVTuber, VTuberConfig

    config = VTuberConfig(
        stream_key="your_twitter_stream_key",
        simulate_chat=True,  # For testing
    )

    vtuber = CosmosVTuber(config)
    await vtuber.start()

Or use the CLI:
    python scripts/start_vtuber.py --stream-key YOUR_KEY
    python scripts/start_vtuber.py --test  # Simulated mode
"""

from .avatar_controller import (
    AvatarController,
    AvatarState,
    AvatarConfig,
    AvatarBackend,
)
from .lip_sync import (
    LipSyncEngine,
    LipSyncMethod,
    LipSyncData,
    Viseme,
    VisemeEvent,
)
from .expression_engine import (
    ExpressionEngine,
    ExpressionState,
    Emotion,
)
from .stream_manager import (
    StreamManager,
    StreamConfig,
    StreamQuality,
    StreamPlatform,
    StreamStats,
    OverlayRenderer,
)
from .chat_reader import (
    TwitterChatReader,
    ChatReaderConfig,
    ChatMessage,
    SimulatedChatReader,
)
from .vtuber_core import (
    CosmosVTuber,
    VTuberConfig,
    VTuberState,
    start_vtuber_stream,
)
from .local_animation import (
    LocalAnimationBackend,
    LocalAnimationConfig,
)
from .neural_avatar import (
    NeuralAvatarConfig,
    NeuralAvatarManager,
    MuseTalkAvatar,
)
from .musetalk_backend import (
    MuseTalkBackend,
    MuseTalkConfig,
)
from .server_integration import (
    router as vtuber_router,
    register_vtuber_routes,
    get_vtuber,
)

__version__ = "1.0.0"

__all__ = [
    # Core
    'CosmosVTuber',
    'VTuberConfig',
    'VTuberState',
    'start_vtuber_stream',

    # Avatar
    'AvatarController',
    'AvatarState',
    'AvatarConfig',
    'AvatarBackend',

    # Lip Sync
    'LipSyncEngine',
    'LipSyncMethod',
    'LipSyncData',
    'Viseme',
    'VisemeEvent',

    # Expressions
    'ExpressionEngine',
    'ExpressionState',
    'Emotion',

    # Streaming
    'StreamManager',
    'StreamConfig',
    'StreamQuality',
    'StreamPlatform',
    'StreamStats',
    'OverlayRenderer',

    # Chat
    'TwitterChatReader',
    'ChatReaderConfig',
    'ChatMessage',
    'SimulatedChatReader',

    # Local Animation
    'LocalAnimationBackend',
    'LocalAnimationConfig',

    # Neural Avatar
    'NeuralAvatarConfig',
    'NeuralAvatarManager',
    'MuseTalkAvatar',

    # MuseTalk Backend
    'MuseTalkBackend',
    'MuseTalkConfig',

    # Server Integration
    'vtuber_router',
    'register_vtuber_routes',
    'get_vtuber',
]
