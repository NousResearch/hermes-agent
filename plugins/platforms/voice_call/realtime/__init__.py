"""Realtime voice for the voice_call platform.

Plugin-local speech-to-speech support (Hermes has no realtime provider
registry): carrier media-stream frame adapters, OpenAI Realtime / Gemini
Live websocket clients, and the bridge that splices a phone call's µ-law
audio into a realtime model session.
"""
