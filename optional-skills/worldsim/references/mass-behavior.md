# Mass Behavior Modeling — Communities, Clusters, Cascades

Understanding individual behavior requires understanding the social
ecosystem they exist in. This reference covers the macro layer:
community detection, influence networks, audience modeling, and
predicting how groups respond to events.

## Why This Matters For Simulation

Individual prediction accuracy: ~56-60%
Individual-in-context prediction: significantly higher

A person's behavior is constrained by their community. Knowing WHICH
community they belong to, WHO influences them, and WHAT information
ecosystem they're in makes individual predictions much sharper.

Lewin's equation: B = f(P, E). This reference is about the E.

## The Ecosystem Stack

```
Layer 5: AUDIENCE REACTION    — How would this person's audience respond?
Layer 4: STANCE & SENTIMENT   — What positions do clusters hold?
Layer 3: INFLUENCE NETWORKS   — Who spreads ideas to whom?
Layer 2: COMMUNITY CLUSTERS   — Who groups together?
Layer 1: SOCIAL GRAPH         — Who follows/interacts with whom?
```

## Layer 1: Social Graph Construction

### Data Sources (by accessibility)

| Source | Access | Quality | Tools |
|--------|--------|---------|-------|
| Bluesky AT Protocol | FREE, open, no auth | Excellent | atproto (pip) |
| X/Twitter API | Bearer token, limited | Good but restricted | curl, tweepy |
| Reddit | API with limits | Good for comments | PRAW (pip) |
| GitHub | Free API | Great for tech people | PyGithub (pip) |
| Web scraping | Fragile, TOS issues | Variable | Last resort |

### Bluesky: The Open Gold Mine
```python
# pip install atproto
from atproto import Client
client = Client()
# No auth needed for public data

# Get follower graph
followers = client.get_followers(actor="handle.bsky.social")
following = client.get_follows(actor="handle.bsky.social")

# Real-time firehose (no auth!)
# wss://jetstream1.us-east.bsky.network/subscribe
```

### Graph Types
- **Follow graph**: who follows whom (directed, static-ish)
- **Interaction graph**: who replies to / retweets whom (directed, dynamic)
- **Mention graph**: who mentions whom (directed, weighted by frequency)
- **Co-engagement graph**: who engages with the same content (undirected)

Interaction graphs are more informative than follow graphs for predicting
actual behavioral alignment.

### Tools
```
pip install networkx python-igraph
```
NetworkX for prototyping (<100K nodes), igraph for production (millions).

## Layer 2: Community Detection

### Algorithms (ranked by quality)

| Algorithm | Quality | Speed | Notes |
|-----------|---------|-------|-------|
| Leiden | Best | Fast | Guarantees connected communities |
| Louvain | Good | Fastest | Can produce disconnected communities |
| Infomap | Excellent | Medium | Based on information theory |
| Label Propagation | Decent | Very fast | Non-deterministic |

### The Meta-Library: CDLib
```
pip install cdlib
```
Wraps 50+ community detection algorithms in a unified API.
Works on top of networkx/igraph. Highly recommended.

```python
import cdlib
from cdlib import algorithms
import networkx as nx

G = nx.karate_club_graph()
communities = algorithms.leiden(G)
# Also: louvain, infomap, label_propagation, angel, demon, etc.
```

### What Communities Tell Us
Each community in a social graph typically shares:
- Ideological orientation
- Topic interests
- Information sources
- Language patterns and in-group vocabulary
- Reaction patterns to events

Knowing which community someone belongs to immediately constrains
predictions about their likely positions and reactions.

## Layer 3: Influence Networks

### Key Insight (Zhou et al., National Science Review 2024)
Network centrality alone is INSUFFICIENT for predicting influence.
Must combine structural position with behavioral features:
- Posting frequency
- Historical content virality
- Response rate / engagement ratio
- Content originality (original vs repost ratio)

### Centrality Measures
```python
import networkx as nx
G = nx.DiGraph()  # directed social graph

# Who has the most connections?
degree = nx.degree_centrality(G)

# Who bridges different communities?
betweenness = nx.betweenness_centrality(G)

# Who's connected to other well-connected people?
eigenvector = nx.eigenvector_centrality(G)

# Adapted from web — directed influence flow
pagerank = nx.pagerank(G)
```

### Superspreader Identification (DeVerna et al., PLOS ONE 2024)
Superspreaders of content fall into three categories:
1. **Pundits**: large following, high authority, original content
2. **Media outlets**: institutional accounts, news organizations
3. **Affiliated personal accounts**: connected to pundits/outlets

For simulation: knowing who the superspreaders are in a person's
network tells you what information they're likely exposed to.

### Information Cascade Modeling
```
pip install ndlib  # Network Diffusion Library
```

NDlib models how information spreads through networks:
- Independent Cascade Model
- Linear Threshold Model
- SIR/SIS epidemiological models adapted for info spread
- Voter Model (opinion dynamics)
- Sznajd Model (social influence)

## Layer 4: Stance & Sentiment Analysis

### Ready-To-Use Models (HuggingFace)

**Tweet Sentiment** (most reliable):
```
cardiffnlp/twitter-roberta-base-sentiment-latest
# Labels: positive / negative / neutral
```

**Political Stance**:
```
kornosk/bert-election2020-twitter-stance-biden-KE-MLM
kornosk/bert-election2020-twitter-stance-trump-KE-MLM
launch/POLITICS  # left / center / right
```

**All-in-One Tweet NLP**:
```
pip install tweetnlp
# Sentiment, emotion, hate speech, NER, topic classification
```

### Topic-Level Stance Tracking
Combine BERTopic (dynamic topic modeling) with stance classifiers:
1. Cluster posts into topics over time windows
2. Classify stance per topic per community
3. Track stance shifts over time
4. Detect divergence between communities on emerging topics

### PRISM Framework (ACL 2025)
First framework for interpretable political bias embeddings.
Two-stage: mine bias indicators → cross-encoder assigns structured scores.
```
github.com/dukesun99/ACL-PRISM
```

## Layer 5: Audience Modeling & Crowd Prediction

### The Frontier: Predicting How Groups React

Key papers and findings:

**CReAM (WWW 2024)**: Predicts which of two posts gets more engagement.
Uses LLM-generated features + FLANG-RoBERTa cross-encoder.
Demonstrates crowd reaction IS predictable from content alone.

**PopSim (Dec 2025)**: LLM multi-agent social network sandbox.
Simulates content propagation dynamics using "Social Mean Field"
for individual-population interaction. Reduces prediction error 8.82%.

**Conditioned Comment Prediction (EACL 2026)**:
KEY FINDING: behavioral traces (past posts) are BETTER than
descriptive personas for conditioning LLMs to predict user behavior.
This validates our OSINT approach: real data > personality labels.

**DEBATE Benchmark (Oct 2025)**:
WARNING: LLM agents converge opinions TOO QUICKLY vs real humans.
SFT + DPO helps but gap remains. Real communities maintain
disagreement longer than simulated ones.

**Distributional vs Individual Prediction (PMC 2025)**:
Group-level predictions are more reliable than individual ones.
Predicting "65% of this community will react negatively" is more
accurate than predicting "this specific person will react negatively."

### Application to Simulation

When simulating @person talking about event X, consider:
1. What community does @person belong to?
2. How is that community reacting to X? (distributional prediction)
3. Where does @person sit within that community? (conformist vs contrarian)
4. Who influences @person? What are THEY saying?
5. How does @person's audience react to their take? (engagement prediction)

This context makes individual predictions sharper.

## Echo Chamber & Filter Bubble Detection

### Technique
1. Build interaction graph
2. Run Leiden community detection
3. For each community, aggregate stance on key issues
4. Measure ideological homogeneity within communities
5. Compare cross-community vs within-community content similarity
6. High within + low cross = echo chamber

### Tools
```
github.com/mminici/Echo-Chamber-Detection  # Cascade-based, CIKM 2022
# Includes Brexit and VaxNoVax datasets
```

### What It Tells Us
Knowing someone's echo chamber tells you:
- What information they're exposed to
- What they're NOT exposed to
- How extreme their positions might be (isolation → radicalization)
- Whether they'll encounter pushback or only agreement
- How they'll react to information from outside their bubble

## User Embeddings: "Find People Like @person"

### Strategy
1. Embed each user's recent N posts with sentence-transformers
2. Average embeddings → user vector
3. Use FAISS for similarity search
4. Cluster users with HDBSCAN in embedding space

### Best Models for Social Media Text
```
# General purpose (good baseline)
sentence-transformers/all-mpnet-base-v2

# Tweet-specific (better domain fit)
cardiffnlp/twitter-roberta-base
vinai/bertweet-base  # pretrained on 850M tweets
```

### Graph + Text Hybrid Embeddings
```
pip install karateclub
```
KarateClub provides Node2Vec, DeepWalk, Graph2Vec — embed users
based on graph position. Combine with text embeddings for hybrid
vectors that capture BOTH what someone says AND where they sit
in the social network.

## Practical Application to Simulation

### For Individual Simulation (what we already do)
Add ecosystem context to each dossier:
- Which community cluster they belong to
- Who their top influencers are (who do they retweet/amplify most)
- What echo chamber are they in (information environment)
- How does their community view the simulation topic

### For Audience Simulation (new capability)
When user asks "what would @person's audience say":
1. Identify @person's follower community
2. Sample representative voices from that community
3. Model the DISTRIBUTION of responses, not just one response
4. Include: cheerleaders, critics, joke-makers, lurkers
5. Weight by typical engagement patterns

### For Cascade Prediction (new capability)
When user asks "how would this take spread":
1. Model the initial tweet and its immediate network
2. Predict which nodes amplify (based on stance alignment + influence)
3. Estimate reach and engagement range
4. Predict quote-tweet ratio (agreement vs dunking)

## Recommended Minimal Stack

```bash
pip install networkx python-igraph leidenalg cdlib karateclub
pip install sentence-transformers transformers tweetnlp
pip install ndlib faiss-cpu hdbscan atproto
```

This gives you: graph construction, community detection, user embeddings,
stance/sentiment analysis, diffusion simulation, similarity search,
clustering, and Bluesky data access. All open source, all pip-installable.
