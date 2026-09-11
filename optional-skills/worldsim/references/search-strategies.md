# Search Strategies — Finding Anyone Across Platforms

The hardest part of simulation is building an accurate model of a real person. This doc
covers how to systematically discover and profile someone across every platform we care about.

## General Principles

1. **Start broad, go narrow.** First establish WHO they are, then drill into HOW they talk.
2. **Cross-reference.** Someone's Reddit persona may differ wildly from their Twitter persona. That's signal, not noise.
3. **Recency matters.** People's views evolve. Weight recent posts (last 6 months) over older ones.
4. **Interactions > monologues.** How someone replies reveals more about their voice than their prepared posts.
5. **Controversy is gold.** People are most themselves when arguing. Search for debates and disagreements.

## Platform-Specific Discovery

### X / Twitter

Twitter is the richest source for most public figures in tech/AI. Multiple approaches:

#### With x-cli (if API keys available)
```bash
# Recent timeline — best single source of voice data
x-cli user timeline {handle} --max 30 -j

# Their replies — how they interact, argue, joke
x-cli tweet search "from:{handle}" --max 30 -j

# What others say about/to them
x-cli tweet search "to:{handle}" --max 20 -j

# On specific topics
x-cli tweet search "from:{handle} open source" --max 10 -j
```

#### Without API (web_search + web_extract)
```
# Identity + role
web_search("{handle} twitter bio role company")

# Voice + opinions
web_search("{handle} twitter hot takes opinions")
web_search("site:x.com {handle}")

# Topic-specific positions
web_search("{handle} twitter {topic}")
web_search("{handle} {topic} opinion take")

# Interviews / longform (reveals deeper thinking)
web_search("{handle} interview podcast AI")
web_search("{handle} blog post essay")

# Beefs and debates (reveals personality under pressure)
web_search("{handle} twitter debate disagree controversial")
web_search("{handle} vs {other_person}")

# Newsletter aggregators that index tweets
web_search("site:buttondown.com/ainews {handle}")
web_search("site:news.smol.ai {handle}")
web_search("site:techmeme.com {handle}")
web_search("site:latent.space {handle}")
```

#### AI Twitter Aggregator Sites (high value)
These sites index AI Twitter conversations daily:
- `buttondown.com/ainews` — swyx's AI News, indexes hundreds of AI Twitter accounts
- `news.smol.ai` — smol AI news aggregator
- `techmeme.com` — tech news, includes tweet citations
- `latent.space` — AI podcast/newsletter with Twitter references

Search pattern: `site:{aggregator} "{handle}"` to find indexed tweets and discussions.

#### IMPORTANT: web_extract does NOT work on x.com
web_extract returns "Website Not Supported" for all x.com/twitter.com URLs.
Do NOT attempt it — it wastes a tool call every time.

#### Verified Fallback Access Methods (tested April 2026)

**PRIMARY: X API v2 Bearer Token** (confirmed working)
- Profiles, timelines, search — 300-10K requests/15min
- See scripts/x_api.py

**FALLBACK 1: nitter.cz via web_extract** (WORKS)
```
web_extract(["https://nitter.cz/{handle}"])
```
Returns full profile + recent timeline. Direct curl gets Cloudflare-blocked
but web_extract bypasses it. Rich data: bio, stats, pinned tweets, full text.
NOTE: Most other nitter instances are DEAD (nitter.net, xcancel.com, etc.)

**FALLBACK 2: ThreadReaderApp** (WORKS — excellent for historical threads)
```
web_extract(["https://threadreaderapp.com/user/{handle}"])
```
Returns unrolled historical threads with full text. Found threads back to 2023.
Gold for longform voice samples.

**FALLBACK 3: GitHub API** (WORKS — excellent for tech people)
```
curl -s https://api.github.com/users/{handle}
curl -s https://api.github.com/users/{handle}/repos?sort=updated
curl -s https://api.github.com/users/{handle}/events
curl -s https://api.github.com/users/{handle}/gists
```
No auth needed (60 req/hr). Profile READMEs are voice profiling gold.
Events API shows recent activity with comment text.

**FALLBACK 4: Reddit JSON API** (WORKS)
```
curl -s -H 'User-Agent: hermes-sim/1.0' 'https://www.reddit.com/user/{username}.json'
curl -s -H 'User-Agent: hermes-sim/1.0' 'https://www.reddit.com/user/{username}/comments.json'
curl -s -H 'User-Agent: hermes-sim/1.0' 'https://www.reddit.com/r/{sub}/search.json?q={query}&restrict_sr=on'
```
MUST include User-Agent header or get 429. Reddit voice is often more
candid/detailed than Twitter voice — high value for personality profiling.

**FALLBACK 5: HackerNews Algolia API** (WORKS — fully open)
```
curl -s 'https://hn.algolia.com/api/v1/search?query={name}&tags=comment'
```
No auth, no rate limits visible. Great for finding what others say about
someone + their own HN comments if they have an account.

**FALLBACK 6: YouTube via web_extract** (WORKS)
Search for interviews/talks, then web_extract the video pages.
Returns rich summaries with attributed quotes from specific speakers.

**NOT VIABLE** (tested, confirmed blocked):
- Google Cache of Twitter → empty results
- Wayback Machine for tweets → sparse captures, no JS content
- Twitter Syndication API → rate limited / broken
- All Instagram viewers (imginn, picuki, dumpoir, gramhir) → 403
- LinkedIn → fully blocked for scraping
- Archive.today → rate limited + CAPTCHA
- Most nitter instances → dead or 403

#### Best approach without x-cli
The most reliable path is: web_search with aggregator sites (ainews, smol.ai,
techmeme, latent.space). These index AI Twitter daily and return actual tweet
text in search descriptions. Stack multiple aggregator searches to build a
composite picture. This was validated in practice — it returns enough signal
to build solid dossiers for anyone active in AI Twitter.

### Reddit

Reddit profiles are public and indexable. Reddit users often have very different 
personas from their Twitter selves — more detailed, more argumentative, more honest.

```
# Find their Reddit username (often different from Twitter)
web_search("{real_name} reddit account")
web_search("{twitter_handle} reddit username")

# Profile and post history
web_search("site:reddit.com/user/{reddit_username}")
web_search("site:reddit.com {reddit_username} {topic}")

# Subreddit-specific behavior
web_search("site:reddit.com/r/LocalLLaMA {username}")
web_search("site:reddit.com/r/MachineLearning {username}")

# Extract actual posts
web_extract(["https://www.reddit.com/user/{username}/comments/"])
web_extract(["https://www.reddit.com/user/{username}/submitted/"])
```

Key subreddits for AI people:
- r/LocalLLaMA — open source LLM community
- r/MachineLearning — academic ML
- r/singularity — AGI speculation  
- r/ChatGPT, r/ClaudeAI, r/OpenAI — product-focused
- r/StableDiffusion — image gen community

### Discord

Discord is hardest — most servers aren't publicly indexed. Strategies:

```
# Find what servers they're in
web_search("{name} discord server")
web_search("{name} discord community")

# Some Discord logs are public via indexers
web_search("site:discordchats.net {username}")

# AI News indexes some Discord channels
web_search("site:buttondown.com/ainews discord {name}")
```

Discord personality notes:
- People are MUCH more casual on Discord than Twitter
- More profanity, more shitposting, more stream-of-consciousness
- Server context matters hugely (same person behaves differently in different servers)
- Harder to research but very valuable if you can find logs

### Blogs / Newsletters / Long-form

These reveal deeper thinking that tweets can't capture:

```
web_search("{name} blog substack medium")
web_search("{name} essay AI opinion")
web_search("{name} substack newsletter")

# Personal sites
web_search("{name} personal website about")

# Extract full posts
web_extract(["https://{their-substack}.substack.com/"])
```

### YouTube / Podcasts

Interview appearances reveal speaking style, humor, and unscripted thinking:

```
web_search("{name} podcast interview AI YouTube")
web_search("{name} YouTube talk presentation")

# Use youtube-content skill if available to pull transcripts
```

### GitHub

For technical people, their GitHub activity reveals priorities and communication style:

```
web_search("site:github.com {username} issues comments")
web_search("site:github.com {username}")

# Issue comments and PR reviews show how they communicate technically
web_extract(["https://github.com/{username}"])
```

## Cross-Platform Identity Resolution

People use different handles across platforms. Resolution strategies:

1. **Bio links**: Twitter bios often link to personal sites with other handles
2. **Name search**: `web_search("{real_name} {platform}")` 
3. **Email/domain**: personal domains often connect identities
4. **Aggregator profiles**: sites like Linktree, bio.link collect handles
5. **Conference talks**: speaker bios list multiple handles
6. **Direct search**: `web_search("{twitter_handle} reddit OR github OR discord")`

## Confidence Scoring

After research, rate confidence for each person:

- **HIGH (80-100%)**: 20+ indexed tweets/posts found, clear voice patterns, known positions on multiple topics, interviews/longform available
- **MEDIUM (50-79%)**: 5-20 indexed posts, general voice sense but some gaps, positions on some topics unclear
- **LOW (20-49%)**: <5 posts found, voice is guesswork, mostly inferring from role/org
- **INSUFFICIENT (<20%)**: can't find enough to simulate accurately. Tell the user.

Always be honest about confidence. A low-confidence simulation should be flagged as such.

## Research Optimization

For fidelity levels:

**Low (1-30)**: 2 searches per person max
- web_search("{handle} twitter") — identity
- web_search("{handle} {topic}") — position on topic if specified

**Medium (31-70)**: 4-6 searches per person
- Identity search
- Voice/opinions search  
- Topic-specific search
- One aggregator site search
- Optional: one web_extract on a blog/interview

**High (71-100)**: 8-12+ searches per person
- All medium searches
- Multiple aggregator sites
- web_extract on 2-3 longform pieces
- Cross-platform search (Reddit, GitHub)
- Debate/controversy search
- Recent vs historical position comparison
- Browser fallback if needed
