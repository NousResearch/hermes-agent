"""Realtime speech: thin OpenAI Realtime client + file-queue speaker for the Meet bot."""

from .openai_client import RealtimeSession, RealtimeSpeaker  # noqa: F401

__all__ = ["RealtimeSession", "RealtimeSpeaker"]
