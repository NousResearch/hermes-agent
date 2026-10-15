"""
REHOBOAM Data Schemas
Pydantic models for all JSON data structures used in the system.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import json
import uuid


def gen_id(prefix: str = "") -> str:
    return f"{prefix}{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


@dataclass
class OceanScores:
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5


@dataclass
class DarkTriad:
    narcissism: float = 0.0
    machiavellianism: float = 0.0
    psychopathy: float = 0.0


@dataclass
class MoralFoundations:
    care: float = 0.5
    fairness: float = 0.5
    loyalty: float = 0.5
    authority: float = 0.5
    sanctity: float = 0.5
    liberty: float = 0.5


@dataclass
class Psychometrics:
    ocean: OceanScores = field(default_factory=OceanScores)
    mbti_estimate: str = ""
    dark_triad: DarkTriad = field(default_factory=DarkTriad)
    moral_foundations: MoralFoundations = field(default_factory=MoralFoundations)
    confidence: float = 0.0
    sample_size: int = 0


@dataclass
class VoiceFingerprint:
    vocabulary_tier: str = ""
    avg_sentence_length: float = 0.0
    exclamation_rate: float = 0.0
    question_rate: float = 0.0
    emoji_rate: float = 0.0
    slang_index: float = 0.0
    formality_score: float = 0.5
    humor_style: str = ""
    signature_phrases: list[str] = field(default_factory=list)
    topics_vocabulary: dict[str, float] = field(default_factory=dict)
    cadence_pattern: str = ""


@dataclass
class Stance:
    position: str = ""
    intensity: float = 0.0
    last_seen: str = ""


@dataclass
class Influence:
    score: float = 0.0
    reach: str = "micro"
    engagement_rate: float = 0.0
    amplification_power: float = 0.0
    thought_leadership_domains: list[str] = field(default_factory=list)


@dataclass
class PostingPatterns:
    avg_posts_per_day: float = 0.0
    peak_hours_utc: list[int] = field(default_factory=list)
    weekend_ratio: float = 0.5
    reply_ratio: float = 0.0
    repost_ratio: float = 0.0
    thread_frequency: float = 0.0
    controversy_rate: float = 0.0


@dataclass
class Relationships:
    allies: list[str] = field(default_factory=list)
    rivals: list[str] = field(default_factory=list)
    frequent_interactions: list[str] = field(default_factory=list)
    mentioned_by_frequently: list[str] = field(default_factory=list)


@dataclass
class ProfileMeta:
    data_sources: list[str] = field(default_factory=list)
    computation_time_sec: float = 0.0
    model_used: str = ""
    last_full_rebuild: str = ""
    last_incremental: str = ""


@dataclass
class Identity:
    bio: str = ""
    location: str = ""
    verified: bool = False
    follower_count: int = 0
    following_count: int = 0
    account_created: str = ""


@dataclass
class Profile:
    schema_version: str = "7.0"
    handle: str = ""
    platform: str = "x"
    display_name: str = ""
    created_at: str = ""
    last_updated: str = ""
    update_count: int = 0
    staleness_score: float = 1.0
    identity: Identity = field(default_factory=Identity)
    psychometrics: Psychometrics = field(default_factory=Psychometrics)
    voice_fingerprint: VoiceFingerprint = field(default_factory=VoiceFingerprint)
    stances: dict[str, Stance] = field(default_factory=dict)
    community_membership: list[str] = field(default_factory=list)
    influence: Influence = field(default_factory=Influence)
    posting_patterns: PostingPatterns = field(default_factory=PostingPatterns)
    relationships: Relationships = field(default_factory=Relationships)
    star_thread_ref: str = "star_thread.json"
    raw_data_refs: list[str] = field(default_factory=list)
    _meta: ProfileMeta = field(default_factory=ProfileMeta)

    def to_dict(self) -> dict:
        """Recursively convert to dict for JSON serialization."""
        import dataclasses
        def _convert(obj):
            if dataclasses.is_dataclass(obj):
                return {k: _convert(v) for k, v in dataclasses.asdict(obj).items()}
            elif isinstance(obj, list):
                return [_convert(i) for i in obj]
            elif isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            return obj
        return _convert(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class StarThread:
    handle: str = ""
    computed_at: str = ""
    based_on_profile_version: str = ""
    thread_version: int = 1
    core_compression: str = ""
    key_drives: list[str] = field(default_factory=list)
    predictive_axioms: list[str] = field(default_factory=list)
    voice_template: dict = field(default_factory=dict)
    anti_slop_markers: list[str] = field(default_factory=list)
    _meta: dict = field(default_factory=dict)


@dataclass
class Prediction:
    pred_id: str = ""
    created_at: str = ""
    sim_id: str = ""
    handle: str = ""
    prediction_type: str = ""  # statement, career, alliance, content, network_reaction
    prediction_text: str = ""
    confidence: float = 0.5
    calibrated_confidence: float = 0.5
    timeframe_days: int = 30
    resolved_at: Optional[str] = None
    outcome: Optional[str] = None  # correct, partially_correct, incorrect
    outcome_evidence: Optional[str] = None
    accuracy_score: Optional[float] = None


@dataclass
class WatchConfig:
    watch_id: str = ""
    handle: str = ""
    platform: str = "x"
    enabled: bool = True
    check_interval_minutes: int = 120
    watch_for: list[dict] = field(default_factory=list)
    alert_severity_minimum: str = "notable"
    created_at: str = ""


@dataclass
class PopulationDefinition:
    group_id: str = ""
    name: str = ""
    description: str = ""
    created_at: str = ""
    last_updated: str = ""
    explicit_members: list[str] = field(default_factory=list)
    criteria: dict = field(default_factory=dict)
    resolved_members: list[str] = field(default_factory=list)
    sampling_strategy: str = "representative"
    default_sample_size: int = 12
