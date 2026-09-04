# OSINT Pipeline — Deep Intelligence Gathering

Full-spectrum open source intelligence for building personality models.
This goes beyond social media posts into visual identity, cross-platform
footprints, and behavioral analysis.

## Tool Arsenal

| Tool | Use Case | Strength |
|------|----------|----------|
| `web_search` | Find anything, initial discovery | Fast, broad, indexed content |
| `web_extract` | Pull full page content | Blogs, articles, profiles, PDFs |
| `browser_navigate` + `browser_snapshot` | View live pages | Dynamic content, login walls |
| `browser_vision` | Analyze what a page looks like | Layouts, visual identity, screenshots |
| `vision_analyze` | Analyze any image by URL/path | Profile pics, post images, aesthetics |
| `browser_get_images` | List all images on a page | Find images to feed to vision_analyze |
| Yandex reverse image search | Find where an image appears | Identity verification, alt accounts |
| `x-cli` (if available) | Direct Twitter API | Timelines, search, metadata |

## Instagram Intelligence

Instagram is CRITICAL for personality modeling — it reveals:
- Visual identity and aesthetic preferences
- Real-life social circles (tagged people, group photos)
- Lifestyle signals (travel, food, hobbies, pets)
- Caption voice (often different from Twitter voice)
- Story highlights (curated self-image)
- Bio links (cross-platform connections)

### Viewing Instagram Profiles (VERIFIED APRIL 2026)

**METHOD 1 — Instagram Private Web API (BEST, returns full JSON)**
```bash
curl -s -H 'User-Agent: Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X)' \
  -H 'x-ig-app-id: 936619743392459' \
  'https://i.instagram.com/api/v1/users/web_profile_info/?username={handle}'
```
Returns ~500KB of JSON: full profile + last 12 posts with captions, likes,
comments, CDN image URLs, timestamps. No auth needed.

**METHOD 2 — Instagram oEmbed API (for individual posts)**
```bash
curl -s 'https://www.instagram.com/api/v1/oembed/?url=https://www.instagram.com/p/{SHORTCODE}/'
```
Returns: caption text, author_name, thumbnail URL. No auth.

**METHOD 3 — Pixwox via web_extract (profile viewer)**
```python
web_extract(["https://pixwox.com/profile/{username}"])
```
Returns 12+ recent posts with captions, engagement stats. Cloudflare blocks
curl but web_extract bypasses it.

**METHOD 4 — SocialBlade via web_extract (analytics)**
```python
web_extract(["https://socialblade.com/instagram/user/{handle}"])
```
Returns follower count, engagement rate, 14-day tracking.

**METHOD 5 — CDN direct download (images from API responses)**
Image URLs from API responses (scontent-*.cdninstagram.com) download
directly with no auth. Feed them to vision_analyze for visual profiling.

**METHOD 6 — Google indexed content**
```
web_search("site:instagram.com {username}")
```
Returns bio text, follower count, recent post captions from search snippets.

**WHAT DOESN'T WORK:** direct web_extract on instagram.com, ?__a=1 trick,
graph.instagram.com (needs OAuth), imginn/picuki/dumpoir/gramhir (403)

### Instagram Discovery (finding someone's handle)
```
web_search("{real_name} instagram")
web_search("{twitter_handle} instagram account")
web_search("site:instagram.com {real_name}")

# Check their Twitter/X bio for IG links
# Check their personal website for social links
# Check Linktree / bio.link pages
```

### Extracting Signal from Instagram

**Profile Picture**: Reveals self-presentation style
- Professional headshot vs casual vs meme/avatar
- Analyze with vision_analyze for clothing, setting, expression

**Bio Text**: Compressed self-identity
- Role/title claims
- Emoji usage patterns
- Link destinations
- Location claims

**Post Grid**: Visual identity fingerprint
- Color palette tendencies
- Content categories (food/travel/tech/selfies/memes)
- Posting frequency
- Professional vs personal ratio

**Captions**: Voice sample different from Twitter
- Usually longer, more personal
- Hashtag usage patterns
- Emoji patterns
- Tone (inspirational vs casual vs funny)

**Tagged Photos**: Real social graph
- Who they hang out with IRL
- Events they attend
- Social circles outside tech/AI

## Visual Identity Analysis

Use vision tools to analyze HOW someone presents visually:

### Profile Pictures Across Platforms
```
# Collect profile pics from multiple platforms
# Twitter, Instagram, LinkedIn, GitHub, Discord

# Analyze each
vision_analyze(image_url="{pic_url}", 
    question="Describe this profile picture in detail: person's appearance, clothing style, setting, expression, professional vs casual, any notable elements")

# Cross-reference: do they use the same pic everywhere? Different personas?
```

### Reverse Image Search (Yandex Pipeline)
From memory — Google Lens blocks Browserbase IPs, use Yandex:

```
# For images behind auth/CDN, upload to catbox first
terminal("curl -F 'reqtype=fileupload' -F 'fileToUpload=@{local_path}' https://catbox.moe/user/api.php")

# Then Yandex reverse image search
browser_navigate("https://yandex.com/images/search?rpt=imageview&url={encoded_public_url}")

# Or via web_extract (slower but automatable)
web_extract(["https://yandex.com/images/search?rpt=imageview&url={encoded_url}"])
```

Yandex provides:
- Similar images (find the same person elsewhere)
- Site matches (where this image appears)
- OCR text extraction (text in images)
- Image tags (what's in the image)
- Knowledge panels (identified entities)

### Screenshot Analysis
When you can see a page but can't extract text:
```
browser_vision(question="Read all text on this page. List usernames, post content, dates, engagement numbers")
browser_vision(annotate=true, question="What interactive elements are on this page?")
```

## LinkedIn Intelligence

**STATUS: BLOCKED for automated access** (tested April 2026).
web_extract returns "Website Not Supported". Direct browsing triggers auth walls.

**Workarounds:**
```
# LinkedIn content IS indexed by search engines
web_search("{real_name} linkedin {company}")
web_search("site:linkedin.com/in {name}")
# These return snippets with headline, role, company — useful even without full profile

# Google sometimes caches LinkedIn profiles
web_search("{name} site:linkedin.com headline")
```

**METHOD 1 — Google indexed snippets (always works)**
```
web_search("site:linkedin.com/in {name} {company}")
```
Returns: name, headline, company, location, connection count, bio snippet.

**METHOD 2 — Crunchbase (EXCELLENT for founders/execs)**
```python
web_extract(["https://www.crunchbase.com/person/{slug}"])
```
Returns: full career history, education, investments, board positions,
social links. Best source for professional identity of startup people.

**METHOD 3 — Corporate press pages**
```
web_search("{person} {company} site:{company}.com bio OR press")
```
Official bios from company newsrooms. High quality, curated but factual.

**METHOD 4 — Third-party aggregators**
- RocketReach, SignalHire — job title + company from web_search snippets
- rootdata.com — good for crypto/AI people
- Crunchbase — best all-round for tech executives

**METHOD 5 — Paid LinkedIn API wrappers** (if budget allows)
- LinkdAPI, Proxycurl: $0.07-0.15 per profile, full structured data
- No OAuth needed, just API key

LinkedIn reveals (from combined methods):
- Career trajectory (Crunchbase full history)
- Current role and headline (search snippets)
- Education (Crunchbase or search snippets)
- Professional self-presentation (company bio pages)
- Investment/board activity (Crunchbase)

## Podcast Transcripts (HIGHEST VALUE for voice profiling)

Podcast interviews are THE gold mine for personality modeling. Hours of
unscripted speech, natural conversation, real personality showing through.

**Discovery:**
```
web_search("{name} podcast transcript interview")
web_search("{name} lex fridman OR tyler cowen OR joe rogan OR dwarkesh")
```

**Extraction — verified working transcript sources:**
```python
# Lex Fridman (full verbatim transcripts)
web_extract(["https://lexfridman.com/EPISODE_URL/transcript"])

# Conversations with Tyler (Tyler Cowen — full transcripts)
web_extract(["https://conversationswithtyler.com/episodes/..."])

# TED Talks transcripts
web_extract(["https://www.ted.com/talks/.../transcript"])

# Sequoia Capital podcast
web_extract(["https://www.sequoiacap.com/podcast/..."])
```

Podcast transcripts reveal:
- Natural speech patterns (filler words, pacing, sentence structure)
- Unguarded opinions (less curated than tweets)
- How they respond to pushback (interviewer challenges)
- Humor style in conversation (different from written humor)
- Depth of knowledge on specific topics
- Personality under pressure

## YouTube / Video Intelligence

```
web_search("{name} youtube talk keynote interview")
web_search("{name} podcast appearance")
```

web_extract on YouTube pages returns rich summaries with attributed quotes.
Use youtube-content skill for full transcripts if available.

## Personal Blogs & Substacks (HIGH VALUE)

Personal writing is curated self-expression — how someone WANTS to be
seen intellectually. Very different signal from social media.

```
web_search("{name} blog substack essay")
# Extract full posts
web_extract(["https://{blog-url}/"])
# Wayback Machine works for archived blog posts
web_extract(["https://web.archive.org/web/2024/{blog-url}"])
```

## GitHub Intelligence

For technical people:

```
web_search("site:github.com {handle}")
web_extract(["https://github.com/{handle}"])

# Issue comments reveal communication style under technical pressure
web_search("site:github.com {handle} issue comment")

# README style reveals documentation personality
# Commit messages reveal terseness vs verbosity
```

## General Web Footprint

```
# Personal website / blog
web_search("{name} personal website blog about")

# Conference talks / speaker bios
web_search("{name} speaker conference talk bio")

# News mentions
web_search("{name} {company} news interview profile")

# Academic papers (for researchers)
web_search("{name} arxiv paper author")
web_search("site:scholar.google.com {name}")

# Podcast appearances
web_search("{name} podcast guest appearance")

# Forum posts (HN, specific communities)
web_search("site:news.ycombinator.com {handle} OR {name}")
```

## Cross-Platform Identity Resolution

### Handle Mapping Strategy
1. Start from known handle (usually Twitter)
2. Check bio links — most people link to other platforms
3. Search "{known_handle} {platform}" for each platform
4. Check personal website for social links
5. Reverse image search profile pic to find matching accounts
6. Search unique phrases they use across platforms

### Identity Verification
When you find a potential match on another platform:
- Same profile picture? (reverse image search)
- Same bio keywords?
- Same name/handle pattern?
- Cross-references (do they mention each other?)
- Writing style match?

## Search Space Narrowing

### The Jiggle Technique
When broad searches return noise, narrow progressively:

1. **Start broad**: `"{name}" AI` 
2. **Add role**: `"{name}" {company} {role}`
3. **Add context**: `"{name}" {company} {specific_project_or_topic}`
4. **Add platform**: `site:{platform} "{name}" {context}`
5. **Add time**: `"{name}" {topic} 2025 OR 2026`
6. **Quote unique phrases**: if you found a distinctive phrase they use, search for that exact phrase to find more of their content

### Disambiguation
Common names need extra signals:
- Add their company/org
- Add their specific domain (AI, crypto, etc.)
- Use their unique handle as anchor
- Search for combinations of their known associates
- Use image search to verify you have the right person

### Signal vs Noise Heuristics
- **High signal**: direct quotes, interview transcripts, personal blog posts, long-form content
- **Medium signal**: mentions in aggregator sites, conference bios, LinkedIn summaries
- **Low signal**: generic news mentions, third-party profiles, directory listings
- **Noise**: same-name different person, outdated info (>2 years), scraped/regurgitated content

## Confidence Calibration

After full OSINT sweep, rate data quality:

| Confidence | Data Available | Simulation Quality |
|-----------|---------------|-------------------|
| 95-100% | 50+ posts, longform, video, visual, cross-platform | Near-perfect voice replication |
| 80-94% | 20-50 posts, some longform, basic visual | Very good, occasional educated guesses |
| 60-79% | 10-20 posts, mostly short-form | Good general sense, some gaps |
| 40-59% | 5-10 posts, limited platforms | Broad strokes only, flag uncertainty |
| 20-39% | <5 posts, single platform | Sketch at best, heavy disclaimers |
| <20% | Almost nothing found | Decline to simulate, ask user for context |

## Privacy & Ethics Note

All research uses publicly available information only. We don't:
- Access private/locked accounts
- Bypass authentication
- Use leaked/hacked data
- Dox or expose private information
- Simulate in ways designed to deceive or impersonate

The goal is personality MODELING for creative simulation, grounded in
what people choose to share publicly.
