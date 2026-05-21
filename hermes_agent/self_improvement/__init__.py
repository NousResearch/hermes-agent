"""
hermes_agent.self_improvement — Plan 004 self-improvement module.

Provides four sub-components built on the Plan 001-D Neon telemetry foundation:

  004-A: feedback_capture  — Slack reaction events → skill_feedback table
  004-A: skill_scorer      — Per-skill score aggregation from skill_feedback
  004-B: promotion_proposer — Daily job: surface promotion candidates for Blake's approval
  004-C: drift_detector    — Regression alerts when thumbs_rate drops
  004-D: recommender       — Weekly LLM-driven skill-gap analysis

Imports are lazy to avoid circular dependency issues at module init time.
"""

__all__ = [
    "feedback_capture",
    "skill_scorer",
    "promotion_proposer",
    "drift_detector",
    "recommender",
]
