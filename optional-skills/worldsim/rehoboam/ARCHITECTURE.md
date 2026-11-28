
# REHOBOAM: Persistent Intelligence Architecture for Hermes WorldSim v7
# =====================================================================
# Author: Hermes Systems Architect
# Date: 2026-04-07
# Status: Design Document — Implementable Specification

## OVERVIEW

REHOBOAM transforms WorldSim from a one-shot simulation tool into a persistent
intelligence system. It maintains durable profiles, caches computations, tracks
predictions, models populations, and monitors reality in real-time.

All data lives under: ~/.hermes/rehoboam/
The skill code lives under: ~/.hermes/skills/creative/hermes-simulator/

The system is file-based (JSON + SQLite) — no external databases, no cloud
dependencies. Everything runs locally.

---

## 1. DIRECTORY STRUCTURE

```
~/.hermes/rehoboam/
├── db/
│   ├── rehoboam.db              # Main SQLite database (analytics, logs, predictions)
│   └── social_graph.db          # Dedicated graph database (SQLite with graph queries)
├── profiles/
│   ├── {handle}/
│   │   ├── profile.json         # Current composite profile
│   │   ├── star_thread.json     # Cached star thread
│   │   ├── history/
│   │   │   ├── profile_2026-04-07.json   # Snapshot archive
│   │   │   └── ...
│   │   ├── raw/
│   │   │   ├── osint_2026-04-07.json     # Raw OSINT data snapshots
│   │   │   ├── posts_2026-04-07.json     # Raw post collections
│   │   │   └── ...
│   │   └── predictions/
│   │       ├── pred_2026-04-07_abc123.json
│   │       └── ...
│   └── _index.json              # Fast lookup: handle -> last_updated, staleness
├── populations/
│   ├── {group_id}/
│   │   ├── definition.json      # Group definition (members, criteria)
│   │   ├── aggregate.json       # Aggregate psychometric model
│   │   └── history/
│   │       └── ...
│   └── _index.json
├── simulations/
│   ├── {sim_id}/
│   │   ├── config.json          # Simulation parameters
│   │   ├── output.json          # Full simulation output
│   │   ├── analytics.json       # Extracted analytics
│   │   └── audit.json           # Audit trail
│   └── _index.json
├── monitoring/
│   ├── watches.json             # Active watch configurations
│   ├── alerts/
│   │   └── {alert_id}.json
│   └── cron_state.json          # State for cron-based monitoring
└── config/
    ├── rehoboam.json            # Global configuration
    └── staleness_policy.json    # Staleness thresholds per data type
```

---

## 2. PERSISTENT PROFILES

### 2.1 Profile Schema (profile.json)

```json
{
  "schema_version": "7.0",
  "handle": "@elonmusk",
  "platform": "x",
  "display_name": "Elon Musk",
  "created_at": "2026-01-15T10:30:00Z",
  "last_updated": "2026-04-07T20:00:00Z",
  "update_count": 7,
  "staleness_score": 0.23,

  "identity": {
    "bio": "...",
    "location": "...",
    "verified": true,
    "follower_count": 180000000,
    "following_count": 800,
    "account_created": "2009-06-02"
  },

  "psychometrics": {
    "ocean": {
      "openness": 0.85,
      "conscientiousness": 0.45,
      "extraversion": 0.78,
      "agreeableness": 0.30,
      "neuroticism": 0.55
    },
    "mbti_estimate": "INTJ",
    "dark_triad": {
      "narcissism": 0.72,
      "machiavellianism": 0.65,
      "psychopathy": 0.31
    },
    "moral_foundations": {
      "care": 0.40,
      "fairness": 0.55,
      "loyalty": 0.60,
      "authority": 0.70,
      "sanctity": 0.30,
      "liberty": 0.90
    },
    "confidence": 0.78,
    "sample_size": 342
  },

  "voice_fingerprint": {
    "vocabulary_tier": "technical-casual",
    "avg_sentence_length": 12.3,
    "exclamation_rate": 0.15,
    "question_rate": 0.08,
    "emoji_rate": 0.22,
    "slang_index": 0.45,
    "formality_score": 0.35,
    "humor_style": "deadpan-absurdist",
    "signature_phrases": ["the thing is", "absolutely based", "..."],
    "topics_vocabulary": {"AI": 0.30, "crypto": 0.15, "space": 0.25},
    "cadence_pattern": "short-burst-with-occasional-thread"
  },

  "stances": {
    "ai_safety": {"position": "accelerationist", "intensity": 0.8, "last_seen": "2026-04-01"},
    "crypto": {"position": "bullish-selective", "intensity": 0.6, "last_seen": "2026-03-15"},
    "politics": {"position": "libertarian-populist", "intensity": 0.7, "last_seen": "2026-04-05"}
  },

  "community_membership": ["tech-twitter", "ai-twitter", "space-twitter", "meme-culture"],

  "influence": {
    "score": 0.97,
    "reach": "mega",
    "engagement_rate": 0.034,
    "amplification_power": 0.95,
    "thought_leadership_domains": ["EVs", "space", "AI"]
  },

  "posting_patterns": {
    "avg_posts_per_day": 15.2,
    "peak_hours_utc": [14, 15, 16, 2, 3],
    "weekend_ratio": 0.85,
    "reply_ratio": 0.40,
    "repost_ratio": 0.15,
    "thread_frequency": 0.08,
    "controversy_rate": 0.35
  },

  "relationships": {
    "allies": ["@handle1", "@handle2"],
    "rivals": ["@handle3"],
    "frequent_interactions": ["@handle4", "@handle5"],
    "mentioned_by_frequently": ["@handle6"]
  },

  "star_thread_ref": "star_thread.json",
  "raw_data_refs": ["raw/osint_2026-04-07.json", "raw/posts_2026-04-07.json"],

  "_meta": {
    "data_sources": ["x_api", "web_scrape", "osint_tools"],
    "computation_time_sec": 45.2,
    "model_used": "claude-opus-4-20250514",
    "last_full_rebuild": "2026-03-01T00:00:00Z",
    "last_incremental": "2026-04-07T20:00:00Z"
  }
}
```

### 2.2 Staleness Policy (staleness_policy.json)

```json
{
  "thresholds": {
    "fresh": {"max_age_hours": 72, "description": "No update needed"},
    "stale": {"max_age_hours": 336, "description": "14 days — incremental update recommended"},
    "expired": {"max_age_hours": 2160, "description": "90 days — full rebuild recommended"},
    "archived": {"max_age_hours": 8760, "description": "1 year — treat as historical only"}
  },
  "per_field_decay": {
    "psychometrics": {"half_life_days": 180, "note": "Personality is stable"},
    "stances": {"half_life_days": 30, "note": "Opinions shift"},
    "posting_patterns": {"half_life_days": 60, "note": "Habits change slowly"},
    "relationships": {"half_life_days": 45, "note": "Alliances shift"},
    "influence": {"half_life_days": 90, "note": "Clout changes slowly"},
    "voice_fingerprint": {"half_life_days": 365, "note": "Voice is very stable"}
  },
  "auto_refresh_on_simulation": true,
  "auto_refresh_threshold": "stale"
}
```

### 2.3 Incremental Update Algorithm

When a profile needs updating:

1. CHECK STALENESS: Compare `last_updated` against policy thresholds.
2. FETCH DELTA: Pull only new posts since `last_updated` timestamp.
3. MERGE STRATEGY per field:
   - `psychometrics`: Bayesian update — new data weighted by sample size, blended
     with existing scores. Never throw away old data, just downweight it.
   - `stances`: Check new posts for stance signals. If found, update position and
     timestamp. If not found, decay confidence on old stance.
   - `voice_fingerprint`: Running averages. New metrics blended at 30% weight
     (exponential moving average).
   - `relationships`: Union of old and new, with recency weighting.
   - `posting_patterns`: Rolling 90-day window statistics.
4. SNAPSHOT: Save old profile to `history/profile_{date}.json` before overwriting.
5. BUMP METADATA: Update `last_updated`, increment `update_count`, recompute
   `staleness_score`.

Incremental update is triggered by:
- Simulation request involving a stale profile
- Explicit `rehoboam update @handle` command
- Cron-scheduled refresh for monitored profiles

### 2.4 Profile Index (_index.json)

```json
{
  "elonmusk": {
    "platform": "x",
    "last_updated": "2026-04-07T20:00:00Z",
    "staleness": "fresh",
    "has_star_thread": true,
    "simulation_count": 12,
    "populations": ["tech-twitter", "ai-twitter"]
  }
}
```

---

## 3. AUTO-THREADING (Star Thread Cache)

### 3.1 Star Thread Schema (star_thread.json)

```json
{
  "handle": "@elonmusk",
  "computed_at": "2026-04-07T20:00:00Z",
  "based_on_profile_version": "2026-04-07T20:00:00Z",
  "thread_version": 2,

  "core_compression": "Elon Musk is a first-principles engineer-showman who
    genuinely believes he's saving humanity while enjoying the chaos he creates.
    His communication style oscillates between CEO-speak and shitposting with no
    middle ground. He processes criticism as either useful signal or enemy action
    with nothing in between...",

  "key_drives": [
    "Legacy as species-level savior",
    "Technical problem-solving as identity",
    "Dominance through ownership of platforms/companies",
    "Meme culture as authentic self-expression"
  ],

  "predictive_axioms": [
    "Will always escalate rather than de-escalate in public conflicts",
    "Will frame business decisions in civilizational terms",
    "Will shitpost more when under stress",
    "Will align with whoever gives him the most operational freedom"
  ],

  "voice_template": {
    "register": "oscillates CEO-formal / 4chan-casual",
    "favorite_moves": ["rhetorical question", "meme reply", "one-word agreement", "thread-rant"],
    "never_does": ["apologize unconditionally", "cite academic sources", "use corporate PR language"]
  },

  "anti_slop_markers": [
    "Never says 'I think it's important to consider...'",
    "Never uses therapy-speak",
    "Never hedges with 'some might say...'",
    "Always: sentence fragments OK, ALL CAPS for emphasis, '!!' for excitement"
  ],

  "_meta": {
    "computation_time_sec": 12.5,
    "input_post_count": 342,
    "model_used": "claude-opus-4-20250514"
  }
}
```

### 3.2 Auto-Threading Flow

When a simulation starts with participants ["@alice", "@bob", "@carol"]:

```
FOR each participant:
  1. CHECK profiles/{handle}/star_thread.json EXISTS?
     YES -> CHECK staleness:
       - If star_thread.based_on_profile_version < profile.last_updated:
         RECOMPUTE (profile changed since thread was made)
       - If star_thread.computed_at older than 30 days:
         RECOMPUTE (time-based expiry)
       - Else: LOAD cached thread
     NO -> CHECK profile.json EXISTS?
       YES -> COMPUTE star thread from existing profile
       NO -> FULL PIPELINE: OSINT -> Profile -> Star Thread
  2. INJECT star thread into simulation context
```

### 3.3 Thread Computation

Star thread computation is an LLM call with a specific prompt template:

```
Input: profile.json (full profile)
Prompt: "Given this comprehensive profile of {handle}, produce a STAR THREAD —
  a compressed personality model that enables accurate behavioral prediction and
  voice simulation. Include: core compression (1 paragraph), key drives (3-5),
  predictive axioms (3-7 IF/THEN rules), voice template, and anti-slop markers."
Output: star_thread.json
```

The computation is cached because it's expensive (full LLM call with large context).
Invalidation is version-based: if the underlying profile changes, the thread is stale.

---

## 4. ANALYTICS ENGINE

### 4.1 SQLite Schema (rehoboam.db)

```sql
-- Core tables
CREATE TABLE profiles (
    handle TEXT PRIMARY KEY,
    platform TEXT NOT NULL,
    display_name TEXT,
    last_updated TEXT NOT NULL,
    staleness TEXT NOT NULL,
    profile_path TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE simulations (
    sim_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    scenario TEXT NOT NULL,
    participant_count INTEGER,
    duration_sec REAL,
    model_used TEXT,
    config_path TEXT,
    output_path TEXT
);

CREATE TABLE sim_participants (
    sim_id TEXT REFERENCES simulations(sim_id),
    handle TEXT REFERENCES profiles(handle),
    role TEXT,  -- 'active', 'observer', 'catalyst'
    PRIMARY KEY (sim_id, handle)
);

-- Per-simulation analytics
CREATE TABLE sim_dynamics (
    sim_id TEXT REFERENCES simulations(sim_id),
    handle TEXT,
    post_count INTEGER,
    word_count INTEGER,
    avg_sentiment REAL,
    dominance_score REAL,       -- 0-1, share of conversation
    agreement_score REAL,       -- avg agreement with others
    controversy_score REAL,     -- how much disagreement they generated
    ratio_score REAL,           -- got ratio'd metric
    influence_in_sim REAL,      -- did others respond to them
    PRIMARY KEY (sim_id, handle)
);

CREATE TABLE sim_interactions (
    sim_id TEXT REFERENCES simulations(sim_id),
    from_handle TEXT,
    to_handle TEXT,
    interaction_type TEXT,       -- 'reply', 'agree', 'disagree', 'quote', 'ratio'
    count INTEGER,
    avg_sentiment REAL,
    PRIMARY KEY (sim_id, from_handle, to_handle, interaction_type)
);

-- Predictions tracking
CREATE TABLE predictions (
    pred_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    sim_id TEXT REFERENCES simulations(sim_id),
    handle TEXT,                 -- NULL for network-level predictions
    prediction_type TEXT,        -- 'statement', 'career', 'alliance', 'content', 'reaction'
    prediction_text TEXT NOT NULL,
    confidence REAL NOT NULL,    -- 0.0 to 1.0
    timeframe_days INTEGER,     -- when should this resolve?
    resolved_at TEXT,
    outcome TEXT,                -- 'correct', 'partially_correct', 'incorrect', 'unresolved'
    outcome_evidence TEXT,
    accuracy_score REAL          -- 0.0 to 1.0 granular accuracy
);

-- Social graph
CREATE TABLE social_edges (
    from_handle TEXT,
    to_handle TEXT,
    relationship_type TEXT,      -- 'ally', 'rival', 'neutral', 'follows', 'interacts'
    weight REAL,                 -- strength 0-1
    first_observed TEXT,
    last_observed TEXT,
    observation_count INTEGER,
    source TEXT,                 -- 'osint', 'simulation', 'monitoring'
    PRIMARY KEY (from_handle, to_handle, relationship_type)
);

CREATE TABLE social_clusters (
    cluster_id TEXT PRIMARY KEY,
    name TEXT,
    description TEXT,
    member_handles TEXT,         -- JSON array
    computed_at TEXT,
    cohesion_score REAL
);

-- Monitoring & alerts
CREATE TABLE monitoring_events (
    event_id TEXT PRIMARY KEY,
    handle TEXT,
    detected_at TEXT NOT NULL,
    event_type TEXT,             -- 'prediction_match', 'model_violation', 'stance_shift', 'anomaly'
    description TEXT,
    related_prediction_id TEXT,
    severity TEXT,               -- 'info', 'notable', 'significant', 'critical'
    acknowledged INTEGER DEFAULT 0
);

-- Audit log
CREATE TABLE audit_log (
    log_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    sim_id TEXT,
    action TEXT NOT NULL,        -- 'profile_create', 'profile_update', 'simulation_run',
                                 -- 'prediction_made', 'prediction_resolved', 'thread_computed',
                                 -- 'monitoring_alert', 'population_analysis'
    handle TEXT,
    details TEXT,                -- JSON blob
    duration_sec REAL,
    model_used TEXT,
    token_count INTEGER,
    error TEXT
);

-- Indexes
CREATE INDEX idx_predictions_handle ON predictions(handle);
CREATE INDEX idx_predictions_type ON predictions(prediction_type);
CREATE INDEX idx_predictions_unresolved ON predictions(outcome) WHERE outcome IS NULL;
CREATE INDEX idx_audit_action ON audit_log(action);
CREATE INDEX idx_audit_sim ON audit_log(sim_id);
CREATE INDEX idx_social_edges_from ON social_edges(from_handle);
CREATE INDEX idx_social_edges_to ON social_edges(to_handle);
CREATE INDEX idx_monitoring_handle ON monitoring_events(handle);
CREATE INDEX idx_monitoring_unack ON monitoring_events(acknowledged) WHERE acknowledged = 0;
```

### 4.2 Analytics Extraction

After each simulation, an analytics pass extracts structured data:

```python
# Pseudocode for post-simulation analytics extraction
def extract_analytics(simulation_output):
    """
    Input: Raw simulation output (the generated conversation/posts)
    Output: Populated sim_dynamics, sim_interactions rows
    
    Process:
    1. Parse each simulated post: author, content, reply_to, timestamp
    2. Per author: count posts, words, run sentiment analysis
    3. Dominance = author_posts / total_posts
    4. For each reply pair: classify as agree/disagree/neutral
    5. Ratio detection: if a reply gets more engagement than parent
    6. Influence: how many replies did this person's posts generate
    7. Store all to SQLite
    """
```

This extraction is an LLM call with structured output:

```
Input: Full simulation output
Prompt: "Analyze this simulated conversation. For each participant, score:
  dominance (share of conversation), agreement patterns, controversy generated.
  For each pair of participants, classify their interactions.
  Output as JSON matching the sim_dynamics and sim_interactions schemas."
Output: analytics.json -> written to DB
```

### 4.3 Cross-Simulation Analytics Queries

Built-in analytical queries:

```sql
-- Prediction accuracy over time
SELECT prediction_type,
       COUNT(*) as total,
       AVG(CASE WHEN outcome='correct' THEN 1.0 WHEN outcome='partially_correct' THEN 0.5 ELSE 0.0 END) as accuracy
FROM predictions WHERE outcome IS NOT NULL
GROUP BY prediction_type;

-- Most simulated people
SELECT handle, COUNT(*) as sim_count
FROM sim_participants GROUP BY handle ORDER BY sim_count DESC LIMIT 20;

-- Relationship strength between two people across simulations
SELECT sim_id, interaction_type, count, avg_sentiment
FROM sim_interactions
WHERE from_handle = ? AND to_handle = ?
ORDER BY sim_id;

-- Social graph: find clusters (people who frequently co-occur in sims and agree)
SELECT a.handle as person_a, b.handle as person_b,
       COUNT(DISTINCT a.sim_id) as co_simulations,
       AVG(si.avg_sentiment) as avg_sentiment
FROM sim_participants a
JOIN sim_participants b ON a.sim_id = b.sim_id AND a.handle < b.handle
LEFT JOIN sim_interactions si ON si.sim_id = a.sim_id
  AND si.from_handle = a.handle AND si.to_handle = b.handle
GROUP BY a.handle, b.handle
HAVING co_simulations > 2
ORDER BY avg_sentiment DESC;
```

---

## 5. PREDICTION FRAMEWORK (Beyond Tweets)

### 5.1 Prediction Types

```json
{
  "prediction_types": {
    "statement": {
      "description": "What will this person say about topic X?",
      "inputs": ["handle", "topic", "context_event"],
      "output": "predicted_statement with confidence",
      "timeframe": "1-7 days",
      "resolution": "Compare predicted vs actual statement"
    },
    "career": {
      "description": "Career move prediction",
      "inputs": ["handle", "current_role", "signals"],
      "output": "predicted_move with confidence and timeframe",
      "timeframe": "30-180 days",
      "resolution": "Check if move happened",
      "signals": [
        "decreasing post frequency about current company",
        "increasing engagement with people at other companies",
        "tone shift: from 'we' to 'I' language",
        "subtle distancing from company controversies",
        "networking pattern changes in social graph"
      ]
    },
    "alliance": {
      "description": "Will two people clash or align?",
      "inputs": ["handle_a", "handle_b", "context"],
      "output": "predicted_dynamic with confidence",
      "timeframe": "7-90 days",
      "resolution": "Check for public interactions",
      "model": "Compare psychometric profiles, stances, community overlap,
                past interaction history, and current incentive structures"
    },
    "content_strategy": {
      "description": "What will this person post about?",
      "inputs": ["handle", "timeframe"],
      "output": "topic_distribution with confidence per topic",
      "timeframe": "7-30 days",
      "resolution": "Compare predicted vs actual topic distribution"
    },
    "network_reaction": {
      "description": "If event X happens, how does the network react?",
      "inputs": ["event_description", "network_handles"],
      "output": "per_person_reaction + aggregate dynamics",
      "timeframe": "1-3 days",
      "resolution": "Compare predicted vs actual reactions"
    }
  }
}
```

### 5.2 Prediction Pipeline

```
PREDICTION REQUEST
       |
       v
[1. GATHER CONTEXT]
  - Load profiles for all relevant handles
  - Load star threads
  - Load social graph edges
  - Load recent monitoring data (if available)
  - Load current events context (web search)
       |
       v
[2. BUILD PREDICTION PROMPT]
  - Inject all profile data
  - Inject relationship data
  - Inject historical predictions and their outcomes (calibration data)
  - Inject the specific prediction request
  - Request structured output: prediction + confidence + reasoning
       |
       v
[3. GENERATE PREDICTION]
  - LLM call with full context
  - Output: prediction_text, confidence, reasoning, key_assumptions
       |
       v
[4. MECHANICAL VALIDATION]
  - Sanity checks: is confidence calibrated? (compare to historical accuracy)
  - Assumption checks: are key assumptions still valid?
  - Contradiction checks: does this contradict known facts?
  - If issues found: regenerate with feedback
       |
       v
[5. STORE PREDICTION]
  - Write to predictions table in DB
  - Write detailed prediction file to profiles/{handle}/predictions/
  - Log in audit trail
       |
       v
[6. SCHEDULE RESOLUTION CHECK]
  - Create a cron job or monitoring rule to check resolution
  - At timeframe expiry: prompt user to resolve OR auto-check via monitoring
```

### 5.3 Second-Order Effects Modeling

For network-level predictions ("if X happens, how does everyone react?"):

```
1. Define the event/stimulus
2. For each person in the network:
   a. Generate their FIRST-ORDER reaction (direct response to event)
   b. Confidence-weight by how likely they are to respond at all
3. For each person, inject everyone else's first-order reactions:
   a. Generate SECOND-ORDER reactions (responses to others' responses)
   b. Identify emergent dynamics: pile-ons, counter-narratives, unlikely alliances
4. Synthesize: aggregate narrative, sentiment distribution, key inflection points
5. Output: timeline of predicted unfolding, key turning points, confidence intervals
```

This is essentially running a simulation but structured as a prediction with
explicit tracking and later resolution.

### 5.4 Prediction Confidence Calibration

Track prediction accuracy to calibrate future confidence scores:

```python
def calibrate_confidence(prediction_type, raw_confidence):
    """
    Pull historical accuracy for this prediction type.
    If we've been overconfident, apply shrinkage.
    If we've been underconfident, apply boost.
    """
    historical = query("""
        SELECT confidence, accuracy_score
        FROM predictions
        WHERE prediction_type = ? AND outcome IS NOT NULL
    """, prediction_type)
    
    if len(historical) < 10:
        return raw_confidence  # Not enough data to calibrate
    
    # Compute calibration curve
    # If avg(confidence) = 0.8 but avg(accuracy) = 0.6, apply 0.75x shrinkage
    avg_conf = mean(h.confidence for h in historical)
    avg_acc = mean(h.accuracy_score for h in historical)
    calibration_ratio = avg_acc / avg_conf if avg_conf > 0 else 1.0
    
    return min(0.99, max(0.01, raw_confidence * calibration_ratio))
```

---

## 6. LOGGING & AUDIT TRAIL

### 6.1 Simulation Audit File (audit.json)

Every simulation produces this alongside its output:

```json
{
  "sim_id": "sim_20260407_203000_abc123",
  "timestamp": "2026-04-07T20:30:00Z",
  "duration_sec": 34.7,

  "config": {
    "scenario": "How would AI Twitter react to OpenAI open-sourcing GPT-5?",
    "participants": ["@elonmusk", "@sama", "@kabortz", "@EMostaque"],
    "simulation_type": "group_reaction",
    "model": "claude-opus-4-20250514",
    "temperature": 0.7,
    "max_turns": 20
  },

  "profile_states": {
    "@elonmusk": {"version": "2026-04-07T20:00:00Z", "staleness": "fresh", "action": "loaded_cached"},
    "@sama": {"version": "2026-03-20T10:00:00Z", "staleness": "stale", "action": "incremental_update"},
    "@kabortz": {"version": null, "staleness": "missing", "action": "full_build"},
    "@EMostaque": {"version": "2026-04-01T15:00:00Z", "staleness": "fresh", "action": "loaded_cached"}
  },

  "thread_states": {
    "@elonmusk": {"action": "loaded_cached", "version": 2},
    "@sama": {"action": "recomputed", "reason": "profile_updated"},
    "@kabortz": {"action": "computed_new"},
    "@EMostaque": {"action": "loaded_cached", "version": 1}
  },

  "mechanical_checks": [
    {"check": "anti_slop", "handle": "@elonmusk", "pass": true, "details": "No corporate speak detected"},
    {"check": "anti_slop", "handle": "@sama", "pass": false, "details": "Removed 2 slopped phrases, regenerated", "before": "...", "after": "..."},
    {"check": "voice_consistency", "handle": "@kabortz", "pass": true, "details": "Voice fingerprint match: 0.87"},
    {"check": "factual_grounding", "pass": true, "details": "No anachronisms detected"}
  ],

  "predictions_generated": [
    {
      "pred_id": "pred_20260407_abc123_001",
      "type": "statement",
      "handle": "@elonmusk",
      "prediction": "Would post something mocking OpenAI's timing",
      "confidence": 0.82,
      "calibrated_confidence": 0.78
    }
  ],

  "token_usage": {
    "profile_building": 12500,
    "thread_computation": 8200,
    "simulation": 45000,
    "analytics_extraction": 6000,
    "total": 71700
  },

  "errors": [],
  "warnings": ["@kabortz had limited OSINT data (47 posts found)"]
}
```

### 6.2 Audit Log Database Entries

Every significant action writes to the audit_log table. This creates a
queryable history:

```sql
-- What did we do today?
SELECT timestamp, action, handle, json_extract(details, '$.summary')
FROM audit_log
WHERE date(timestamp) = '2026-04-07'
ORDER BY timestamp;

-- How much compute have we used on a specific person?
SELECT action, COUNT(*), SUM(token_count), SUM(duration_sec)
FROM audit_log
WHERE handle = '@elonmusk'
GROUP BY action;

-- Find all errors
SELECT * FROM audit_log WHERE error IS NOT NULL ORDER BY timestamp DESC;
```

---

## 7. POPULATION-LEVEL ANALYSIS

### 7.1 Population Definition (definition.json)

```json
{
  "group_id": "ai-twitter",
  "name": "AI Twitter",
  "description": "People who primarily post about artificial intelligence on X",
  "created_at": "2026-04-07T20:00:00Z",
  "last_updated": "2026-04-07T20:00:00Z",

  "membership": {
    "method": "explicit+criteria",
    "explicit_members": ["@ylecun", "@kabortz", "@sama", "@EMostaque", "@ClementDelworker"],
    "criteria": {
      "min_ai_topic_ratio": 0.3,
      "min_followers": 1000,
      "must_be_profiled": true
    },
    "resolved_members": ["@ylecun", "@kabortz", "@sama", "@EMostaque", "@ClementDelworker",
                          "@drjimfan", "@svpino", "@_akhaliq"],
    "member_count": 8
  },

  "sampling": {
    "strategy": "representative",
    "description": "When group is too large for full simulation, sample N members
                    weighted by influence and diversity of viewpoints",
    "default_sample_size": 12,
    "stratify_by": ["influence.score", "stances.ai_safety.position"]
  }
}
```

### 7.2 Aggregate Model (aggregate.json)

```json
{
  "group_id": "ai-twitter",
  "computed_at": "2026-04-07T20:00:00Z",
  "member_count": 8,

  "aggregate_psychometrics": {
    "ocean_distribution": {
      "openness": {"mean": 0.82, "std": 0.08, "min": 0.70, "max": 0.95},
      "conscientiousness": {"mean": 0.55, "std": 0.15, "min": 0.30, "max": 0.80},
      "extraversion": {"mean": 0.65, "std": 0.20, "min": 0.30, "max": 0.90},
      "agreeableness": {"mean": 0.45, "std": 0.18, "min": 0.20, "max": 0.75},
      "neuroticism": {"mean": 0.40, "std": 0.12, "min": 0.25, "max": 0.60}
    }
  },

  "stance_distribution": {
    "ai_safety": {
      "accelerationist": 0.35,
      "cautious_optimist": 0.30,
      "doomer": 0.15,
      "pragmatist": 0.20
    },
    "open_source_ai": {
      "strongly_for": 0.45,
      "mixed": 0.35,
      "against": 0.20
    }
  },

  "voice_centroid": {
    "avg_formality": 0.55,
    "avg_humor_level": 0.4,
    "common_vocabulary": ["model", "training", "scale", "alignment", "benchmark"],
    "dominant_register": "technical-casual"
  },

  "influence_distribution": {
    "mega": 2,
    "macro": 3,
    "mid": 2,
    "micro": 1
  },

  "internal_factions": [
    {
      "name": "Accelerationists",
      "members": ["@sama", "@EMostaque"],
      "cohesion": 0.7,
      "key_stance": "Build fast, safety is overblown"
    },
    {
      "name": "Safety-conscious",
      "members": ["@ylecun", "@kabortz"],
      "cohesion": 0.6,
      "key_stance": "Need guardrails but not pause"
    }
  ]
}
```

### 7.3 Population Query Interface

```
rehoboam population "ai-twitter" react "OpenAI open-sources GPT-5"
```

Pipeline:
1. Load population definition
2. Check all member profiles (update stale ones)
3. If group > 15 members, sample representative subset
4. For each member, generate individual reaction using their star thread
5. Aggregate: sentiment distribution, faction responses, predicted viral posts
6. Synthesize narrative: "AI Twitter would largely celebrate but with faction splits..."

```
rehoboam population "ai-twitter" sentiment "government AI regulation"
```

Pipeline:
1. Load population + aggregate model
2. Use stance_distribution + psychometrics to estimate sentiment without
   running full simulation (fast path)
3. Output: sentiment distribution + confidence interval
4. If user wants detail: run full simulation (slow path)

### 7.4 Population Composition Operations

```
# Create population from social graph cluster
rehoboam population create "rationalists" --from-cluster "rationalist-adjacent"

# Create by query
rehoboam population create "ai-doomers" --stance "ai_safety=doomer" --min-influence 0.3

# Merge populations
rehoboam population merge "ai-twitter" "crypto-twitter" --name "tech-twitter"

# Compare populations
rehoboam population compare "ai-twitter" "crypto-twitter" --on "regulation"
```

---

## 8. REAL-TIME MONITORING

### 8.1 Watch Configuration (watches.json)

```json
{
  "watches": [
    {
      "watch_id": "watch_001",
      "handle": "@elonmusk",
      "platform": "x",
      "enabled": true,
      "check_interval_minutes": 60,
      "watch_for": [
        {"type": "prediction_match", "prediction_ids": ["pred_001", "pred_002"]},
        {"type": "stance_shift", "topics": ["ai_safety", "crypto"]},
        {"type": "anomaly", "description": "Unusual posting pattern or tone"},
        {"type": "keyword", "keywords": ["leaving", "stepping down", "new role"]},
        {"type": "relationship_change", "handles": ["@sama", "@kabortz"]}
      ],
      "alert_severity_minimum": "notable",
      "created_at": "2026-04-07T20:00:00Z"
    }
  ]
}
```

### 8.2 Monitoring Architecture

```
CRON JOB (every N minutes per watch)
       |
       v
[1. FETCH NEW POSTS]
  - Use X API / web scrape to get posts since last check
  - Store raw posts in profiles/{handle}/raw/
       |
       v
[2. QUICK ANALYSIS] (lightweight, no LLM for most checks)
  - Keyword matching: regex against watch keywords
  - Posting pattern: compare frequency/timing to baseline
  - Sentiment: basic sentiment scoring
  - If any trigger fires -> proceed to deep analysis
       |
       v
[3. DEEP ANALYSIS] (LLM-powered, only when triggered)
  - Load the person's star thread
  - Compare new posts against predictive axioms
  - Check: does this match any open predictions?
  - Check: does this violate the model? (unexpected behavior)
  - Check: stance shift detected?
       |
       v
[4. GENERATE ALERT]
  - Write to monitoring_events table
  - Write alert file to monitoring/alerts/
  - If critical: trigger notification (write to a notification file
    that the Hermes agent checks, or send via configured webhook)
       |
       v
[5. UPDATE PROFILE]
  - Merge new data into profile (incremental update)
  - Update staleness score
```

### 8.3 Cron Job Setup

```bash
# Hermes agent schedules these via its cron capability

# High-priority watches: every 30 minutes
*/30 * * * * ~/.hermes/skills/creative/hermes-simulator/rehoboam/monitor.py --priority high

# Standard watches: every 2 hours
0 */2 * * * ~/.hermes/skills/creative/hermes-simulator/rehoboam/monitor.py --priority standard

# Daily analytics refresh
0 6 * * * ~/.hermes/skills/creative/hermes-simulator/rehoboam/daily_refresh.py

# Weekly prediction resolution check
0 9 * * 1 ~/.hermes/skills/creative/hermes-simulator/rehoboam/resolve_predictions.py

# Weekly social graph recomputation
0 3 * * 0 ~/.hermes/skills/creative/hermes-simulator/rehoboam/rebuild_graph.py
```

### 8.4 Notification File

```json
// ~/.hermes/rehoboam/monitoring/pending_notifications.json
{
  "notifications": [
    {
      "id": "notif_001",
      "timestamp": "2026-04-07T21:15:00Z",
      "severity": "significant",
      "summary": "@elonmusk posted about leaving X board — matches prediction pred_003",
      "watch_id": "watch_001",
      "event_id": "evt_001",
      "read": false
    }
  ]
}
```

When Hermes agent starts or periodically checks, it reads this file and surfaces
unread notifications to the user.

---

## 9. COMMAND INTERFACE

All commands are invoked through the Hermes agent's skill system:

```
# Profile Management
rehoboam profile @handle                    # View/create profile
rehoboam profile @handle --update           # Force incremental update
rehoboam profile @handle --rebuild          # Force full rebuild
rehoboam profile @handle --history          # View profile evolution
rehoboam profile list                       # List all profiles
rehoboam profile list --stale               # List profiles needing update

# Star Threads
rehoboam thread @handle                     # View/compute star thread
rehoboam thread @handle --recompute         # Force recomputation

# Simulation (enhanced v6)
rehoboam simulate "scenario" --with @h1 @h2 @h3
rehoboam simulate "scenario" --population "ai-twitter"
rehoboam simulate "scenario" --with @h1 @h2 --predict  # Generate explicit predictions

# Predictions
rehoboam predict statement @handle --topic "AI regulation"
rehoboam predict career @handle
rehoboam predict alliance @handle1 @handle2
rehoboam predict content @handle --timeframe 30d
rehoboam predict reaction "event description" --network @h1 @h2 @h3
rehoboam predict resolve pred_id --outcome correct --evidence "they posted X"
rehoboam predict accuracy                   # Show calibration stats
rehoboam predict open                       # List unresolved predictions

# Analytics
rehoboam analytics @handle                  # Person analytics summary
rehoboam analytics simulation sim_id        # Simulation analytics
rehoboam analytics graph                    # Social graph overview
rehoboam analytics graph @handle            # Person's graph neighborhood
rehoboam analytics accuracy                 # Prediction accuracy dashboard

# Populations
rehoboam population create "name" --members @h1 @h2 @h3
rehoboam population "name" react "event"
rehoboam population "name" sentiment "topic"
rehoboam population compare "pop1" "pop2" --on "topic"

# Monitoring
rehoboam watch @handle --for prediction_match,stance_shift --interval 60m
rehoboam watch list
rehoboam watch pause watch_id
rehoboam watch alerts                       # Show unread alerts
rehoboam watch alerts --ack                 # Acknowledge all

# System
rehoboam status                             # System health, counts, staleness overview
rehoboam audit --today                      # Today's audit log
rehoboam audit --sim sim_id                 # Specific simulation audit
rehoboam export @handle --format json       # Export a profile
rehoboam import profile.json                # Import a profile
```

---

## 10. IMPLEMENTATION PLAN

### Phase 1: Foundation (Core Storage + Profiles)
Files to create:
- rehoboam/db.py           — SQLite setup, migrations, query helpers
- rehoboam/profiles.py     — Profile CRUD, staleness checks, incremental updates
- rehoboam/storage.py      — Directory management, index maintenance
- rehoboam/config.py       — Configuration loading
- rehoboam/schemas.py      — Pydantic models for all JSON schemas

### Phase 2: Auto-Threading + Enhanced Simulation
Files to create:
- rehoboam/threading.py    — Star thread computation, caching, invalidation
- rehoboam/simulation.py   — Enhanced simulation runner (auto-profile, auto-thread)
- rehoboam/analytics.py    — Post-simulation analytics extraction

### Phase 3: Prediction Engine
Files to create:
- rehoboam/predictions.py  — Prediction generation, storage, resolution
- rehoboam/calibration.py  — Confidence calibration from historical data
- rehoboam/network.py      — Second-order effects modeling

### Phase 4: Population Modeling
Files to create:
- rehoboam/populations.py  — Population definition, aggregation, sampling
- rehoboam/aggregate.py    — Aggregate model computation

### Phase 5: Monitoring
Files to create:
- rehoboam/monitor.py      — Watch management, cron-based checking
- rehoboam/alerts.py       — Alert generation and notification

### Phase 6: CLI Interface
Files to create:
- rehoboam/cli.py          — Command routing and argument parsing
- rehoboam/__init__.py     — Package init

### Dependencies
- Python 3.10+
- sqlite3 (stdlib)
- pydantic (for schema validation)
- No external database dependencies
- LLM access via Hermes agent's existing model integration

---

## 11. DATA FLOW DIAGRAM

```
USER COMMAND
     |
     v
[CLI ROUTER] ─────────────────────────────────────────────────┐
     |                                                          |
     v                                                          v
[PROFILE MANAGER] ──> [OSINT PIPELINE] ──> [PROFILE STORE]   [AUDIT LOG]
     |                      |                    |               ^
     v                      v                    v               |
[THREAD COMPUTER] ──> [LLM ENGINE] ──> [THREAD CACHE]          |
     |                      |                    |               |
     v                      v                    v               |
[SIMULATION RUNNER] ──────────────────> [SIM OUTPUT]            |
     |                                       |                   |
     v                                       v                   |
[ANALYTICS EXTRACTOR] ──────────────> [ANALYTICS DB] ──────────┘
     |                                       |
     v                                       v
[PREDICTION ENGINE] ──────────────> [PREDICTIONS DB]
     |                                       |
     v                                       v
[POPULATION MODELER] ──────────────> [POPULATION STORE]
     |                                       |
     v                                       v
[MONITOR] ──> [CRON JOBS] ──> [ALERTS] ──> [NOTIFICATIONS]
```

---

## 12. SECURITY & ETHICS NOTES

1. ALL DATA IS LOCAL. No cloud sync, no telemetry, no sharing.
2. Profiles are derived from PUBLIC data only (OSINT from public posts).
3. Predictions are probabilistic models, not surveillance tools.
4. The system should include a `rehoboam ethics` command that reminds users
   of responsible use guidelines.
5. Profile deletion: `rehoboam profile @handle --delete` removes all data
   including history, predictions, and monitoring.
6. No automated actions based on predictions — humans make decisions.

---

## 13. PERFORMANCE CONSIDERATIONS

1. PROFILE LOADING: Profiles are JSON files loaded on demand. The _index.json
   provides O(1) lookup without scanning directories.
2. SQLITE: Single-writer, multiple-reader. Fine for local single-user system.
   WAL mode enabled for concurrent reads during writes.
3. LLM CALLS: The expensive part. Minimize by caching aggressively:
   - Star threads cached until profile changes
   - Profiles incrementally updated (not rebuilt)
   - Population analyses cached with TTL
4. MONITORING: Lightweight checks first (keyword, pattern), LLM only on trigger.
5. SOCIAL GRAPH: Recomputed weekly, not on every query. Cached in SQLite.

---

END OF ARCHITECTURE DOCUMENT
