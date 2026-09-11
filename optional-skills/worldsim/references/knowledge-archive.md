# Knowledge Archive — Per-Person Source Library + Expert Synthesis

## The Problem With Profiles

A profile is a SNAPSHOT. It says "this person believes X" but doesn't
show you WHERE they said it, WHEN, in WHAT context, or HOW their
thinking evolved. You can't cite a profile. You can't trace a claim
back to a source. And when you're simulating a conversation about
topic Z, the profile gives you everything about the person equally
weighted — their views on AI and their views on cooking and their
views on politics all crammed into the same context window.

## The Archive

For every person the system touches, build a LIBRARY:

```
~/.hermes/rehoboam/archives/{handle}/
├── index.json              ← master index: all entries, metadata, embeddings
├── sources/
│   ├── x_tweets.jsonl      ← every tweet pulled, with ID, timestamp, URL, metrics
│   ├── x_replies.jsonl     ← their replies (different voice register)
│   ├── bluesky_posts.jsonl ← bluesky posts
│   ├── blog_posts.jsonl    ← full text of blog posts with URLs
│   ├── podcast_quotes.jsonl ← attributed quotes from transcripts
│   ├── interviews.jsonl    ← quotes from news articles/interviews
│   ├── reddit_comments.jsonl
│   ├── github_comments.jsonl
│   ├── goodreads_reviews.jsonl
│   ├── threads_posts.jsonl
│   └── other.jsonl         ← anything else (HN, Quora, etc.)
├── topics/
│   ├── ai_safety.jsonl     ← auto-clustered by topic
│   ├── open_source.jsonl
│   ├── consciousness.jsonl
│   └── ...
└── embeddings/
    └── all_embeddings.npy  ← sentence-transformer vectors for semantic search
```

### Entry Format (every entry in every source file)

```json
{
  "id": "unique_id",
  "handle": "teknium",
  "platform": "x",
  "type": "tweet|reply|blog|podcast|interview|comment|review",
  "text": "the actual text they said",
  "url": "https://x.com/Teknium/status/1234567890",
  "timestamp": "2026-04-05T21:40:48Z",
  "context": {
    "replying_to": "@otheruser's tweet about X",
    "thread_position": 3,
    "topic": "open source AI",
    "source_title": "Lex Fridman Podcast #412"
  },
  "metrics": {
    "likes": 234,
    "retweets": 45,
    "replies": 12
  },
  "topics": ["open_source", "ai_models", "hermes"],
  "embedding_id": 42
}
```

Every entry has a URL. Everything is traceable. Nothing is paraphrased
without the original alongside it.

## Collection Pipeline

When `worldsim> profile @handle` or `worldsim> archive @handle` runs:

### Step 1: Pull Everything
Use every verified access method to collect raw materials:
- X API: get max tweets (paginate with next_token to get hundreds)
- nitter.cz: timeline content
- ThreadReaderApp: historical threads
- Bluesky: full post history
- GitHub: issue comments, PR reviews, gists, README
- Reddit: comment history
- Blog/Substack: full posts (web_extract)
- Podcast transcripts: attributed quotes
- Interviews: quotes with attribution
- Goodreads: reviews
- Medium: RSS feed full text

### Step 2: Deduplicate
Same content appears across platforms (cross-posted tweets, syndicated
blog posts). Deduplicate by content similarity, keep the richest version
(the one with most metadata/context).

### Step 3: Topic Cluster
Run lightweight topic classification on each entry:
- Use the LLM or a simple keyword matcher to assign 1-3 topic tags
- Cluster into topic files for fast retrieval
- Topics are dynamic — new topics emerge from the data

### Step 4: Embed
Generate sentence-transformer embeddings for every entry.
Store in numpy array for fast cosine similarity search.
This enables semantic retrieval: "find everything @handle said about
consciousness" even if they never used the word "consciousness."

### Step 5: Index
Build the master index.json with entry count, topic distribution,
timestamp range, platform coverage, and quality metrics.

## Context-Aware Retrieval

This is the key. The archive might have 500 entries for a person.
The context window can hold maybe 30-50 of them alongside all the
other simulation context. You MUST retrieve selectively.

### For Simulation
When simulating @handle talking about topic X:

```
1. Semantic search: embed the current conversation context
2. Retrieve top 10-15 entries by cosine similarity to context
3. Also retrieve: 5 highest-engagement entries (their "greatest hits")
4. Also retrieve: 3 most recent entries (freshness)
5. Also retrieve: 2 entries that CONTRADICT the expected position
   (prevents confirmation bias in the simulation)
6. Deduplicate. Cap at 25-30 entries total.
7. These become the "voice anchors" for generation.
```

The simulation draws from SPECIFIC REAL QUOTES relevant to the current
conversation. Not a generic profile. Not everything they've ever said.
The 25 most relevant things they've said about THIS topic.

### For Expert Synthesis
When the user asks "who are the best minds on X and what have they said?":

```
1. Search ALL archived people's entries for topic X
2. Rank by: entry quality × person expertise × relevance to query
3. Return a synthesis with CITATIONS:

   On the topic of AI consciousness:

   @repligate argues that LLMs exhibit "simulacra of consciousness"
   rather than consciousness itself, distinguishing between the
   model's behavior and its substrate:
     > "the question isn't whether GPT is conscious but whether the
     > character it's simulating is conscious within the fiction"
     — tweet, 2025-03-15 (2.4K likes)
     https://x.com/repligate/status/...

   @nickcammarata approaches it from a meditation/first-person
   perspective, noting parallels between introspective practice
   and interpretability:
     > "observation changes the system being observed, in meditation
     > and in interp"
     — tweet, 2026-04-05 (2.9K likes)
     https://x.com/nickcammarata/status/...

   @tszzl is skeptical of the framing entirely:
     > "consciousness discourse is philosophy cosplaying as engineering"
     — tweet, 2025-11-22 (5.1K likes)
     https://x.com/tszzl/status/...
```

Every claim attributed. Every quote sourced. Every link clickable.

### For Grounding Predictions
When predicting what @handle would say about event Y:

```
1. Retrieve all archive entries related to Y or adjacent topics
2. Identify their PATTERN of response to similar events
3. Ground the prediction in specific past statements:

   PREDICTION: @handle would likely frame event Y through the lens
   of [topic Z], based on:
   - tweet [url]: "quote about Z" (2025-06-15)
   - blog post [url]: "longer quote about Z" (2025-09-20)
   - podcast [url]: "verbal quote about Z" (2026-01-10)
   CONFIDENCE: 78% (3 consistent sources over 7 months)
```

## Incremental Updates

The archive grows over time. Each time the person is profiled:
1. Pull new content since last archive timestamp
2. Append to source files
3. Re-embed new entries only
4. Update topic clusters
5. Update index

Don't rebuild from scratch. Append and re-index.

## Expert Table

When you have 20+ archived people, build an expert table:

```
worldsim> experts "open source AI"

EXPERT TABLE: open source AI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  @Teknium | 47 entries | voice: builder/practitioner
    "we can prove that open approaches build better, more
    trustworthy systems" — tweet, 2026-04-05
    Latest: 2 hours ago | Stance: STRONG ADVOCATE

  @repligate | 12 entries | voice: philosophical/theoretical
    "open weights = accountability. you can't audit a black box"
    — tweet, 2025-11-30
    Latest: 3 days ago | Stance: ADVOCATE (principled)

  @eigenrobot | 8 entries | voice: statistical/contrarian
    "the open source premium is largely downstream of selection
    effects in who contributes" — tweet, 2025-08-14
    Latest: 1 week ago | Stance: SKEPTICAL OF FRAMING

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  3 experts found | 67 total entries | synthesize? (y/n)
```

The table shows: who knows about this, what they've said, how recently,
and what their stance is. All grounded in archived quotes with sources.

## Integration With Simulation

When the star thread + dossier + archive work together:

```
STAR THREAD: drives the core generation (what they're DOING)
DOSSIER: provides constraints (psychometrics, voice metrics, baselines)
ARCHIVE: provides GROUNDING (specific real quotes for this context)
MECHANICAL CHECKS: verifies surface features (emoji, length, slop)
```

The archive prevents the simulation from drifting into generic territory.
Instead of "this person would probably say something about open source,"
it's "this person said THIS SPECIFIC THING about open source 3 weeks ago,
and their simulation should be consistent with that while also being fresh."

## The Overfitting Problem

"Without overfitting to a particular material the new context doesn't call for."

The retrieval system MUST be selective. If someone said 47 things about
open source AI, and the current conversation is about AI regulation,
don't dump all 47 open source quotes into context. Maybe 3 are relevant
because they connect open source to regulation. Retrieve THOSE 3.

The cosine similarity search handles this naturally — it matches the
CURRENT conversation context against the archive and returns what's
actually relevant, not everything tagged with a nearby topic.

The anti-overfitting checklist:
- Never load more than 25-30 archive entries per person into context
- Weight by relevance to CURRENT conversation, not by general importance
- Include at least 2 entries that contradict the expected position
- Include at least 3 recent entries regardless of topic relevance (freshness)
- If the conversation shifts topic mid-simulation, RE-RETRIEVE for new context
- The archive is a LIBRARY you consult, not a script you follow
