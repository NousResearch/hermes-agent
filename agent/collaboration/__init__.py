"""Collaboration learning framework — observes user interaction patterns,
extracts preferences, adapts behavior, and validates improvements.

All storage uses the fleet graph API at graph.xentropy.ai as the backend.
"""
from agent.collaboration.graph_client import GraphClient, get_client, reset_client
from agent.collaboration.observation import capture_observation, detect_correction, classify_response_format
from agent.collaboration.extraction import CollaborationExtractor, PreferenceModel
from agent.collaboration.adaptation import load_preference_summary, format_preference_entries
from agent.collaboration.feedback import CorrectionRateTracker, DriftDetector, validate_preference_impact
