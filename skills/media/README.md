# Media Skills

Media processing, music generation, audio analysis, and content transformation tools for Hermes Agent.

## Overview

This category contains 5 skills for working with media — from searching GIFs to generating music, analyzing audio features, managing Spotify playlists, and transforming YouTube content. Whether you're creating content, analyzing audio, or automating media workflows, these skills provide powerful media operations.

## Available Skills

### Visual Media

#### **gif-search**
Search and download GIFs from Tenor using curl and jq.

**Use when:** Finding reaction GIFs, visual content, or animated illustrations for presentations and communications.

**Key features:**
- Search Tenor GIF library
- Keyword-based queries
- Direct GIF downloads
- CLI-based workflow (curl + jq)
- Fast integration into scripts

**Use cases:**
- Find reaction GIFs for messaging
- Download GIFs for presentations
- Automate GIF collection
- Build GIF libraries

---

### Music & Audio Generation

#### **heartmula** (HeartMuLa)
Suno-style song generation from lyrics and style tags.

**Use when:** Generating complete songs from text descriptions and lyrics.

**Key features:**
- Text-to-music generation
- Lyrics-based composition
- Style tag control (genre, mood, instruments)
- Suno-inspired quality
- Complete song generation

**Example tags:**
- Genre: pop, rock, jazz, electronic, classical
- Mood: upbeat, melancholic, energetic, calm
- Instruments: guitar, piano, synth, drums

---

### Audio Analysis

#### **songsee**
Generate audio spectrograms and extract features (mel, chroma, MFCC) via CLI.

**Use when:** Analyzing audio characteristics, visualizing sound, or extracting audio features for ML.

**Key features:**
- Spectrogram generation (mel-spectrogram)
- Chroma features (pitch class profiles)
- MFCC extraction (Mel-frequency cepstral coefficients)
- Audio visualization
- CLI-based analysis
- Feature extraction for ML models

**Use cases:**
- Audio analysis and research
- Music information retrieval (MIR)
- Sound visualization
- ML feature extraction
- Audio quality analysis

---

### Music Streaming & Playback

#### **spotify**
Control Spotify playback, search tracks, manage playlists, and handle devices.

**Use when:** Automating Spotify workflows, building music bots, or integrating music into applications.

**Key features:**
- Playback control (play, pause, skip, seek)
- Search tracks, albums, artists, playlists
- Queue management
- Playlist creation and modification
- Device management and switching
- Currently playing information
- Spotify API integration

**Use cases:**
- Music discovery automation
- Playlist generation
- DJ bots and automation
- Listening statistics
- Multi-room audio control

---

### Content Transformation

#### **youtube-content**
Transform YouTube video transcripts into summaries, Twitter threads, blog posts, and more.

**Use when:** Repurposing YouTube content, creating written content from videos, or analyzing video transcripts.

**Key features:**
- Transcript extraction from YouTube videos
- Video-to-summary conversion
- Twitter thread generation
- Blog post creation
- Key takeaways extraction
- Quote extraction
- Multi-format content transformation

**Output formats:**
- Summaries (brief, detailed)
- Twitter/X threads
- Blog posts
- Bullet-point notes
- Quote collections
- Timestamped highlights

---

## Quick Start

### Example: Content Creation

```bash
# 1. Find GIFs for your presentation
/gif-search "Search for 'success celebration' GIFs"

# 2. Generate background music
/heartmula "Create upbeat electronic song with lyrics about productivity"

# 3. Analyze the audio
/songsee "Generate spectrogram and extract MFCC features"
```

### Example: Music Workflow

```bash
# 1. Search Spotify
/spotify "Find playlists similar to 'Deep Focus'"

# 2. Create custom playlist
/spotify "Create playlist 'Coding Flow' and add lo-fi hip hop tracks"

# 3. Play on device
/spotify "Play 'Coding Flow' on office speakers"
```

### Example: YouTube Content Repurposing

```bash
# 1. Get video transcript
/youtube-content "Extract transcript from [YouTube URL]"

# 2. Create summary
/youtube-content "Summarize in 5 key points with timestamps"

# 3. Generate Twitter thread
/youtube-content "Convert to engaging Twitter thread with quotes"

# 4. Write blog post
/youtube-content "Transform into 800-word blog post"
```

## Skill Combinations

**Content Creation Pipeline:**
1. Use `youtube-content` to extract insights from research videos
2. Use `gif-search` to find visual elements
3. Use `heartmula` to generate background music
4. Combine into presentation or video

**Music Production Workflow:**
1. Use `heartmula` to generate initial composition
2. Use `songsee` to analyze audio characteristics
3. Use `spotify` to create and share playlists

**Media Analysis:**
1. Use `youtube-content` to extract transcripts
2. Use `songsee` for audio feature analysis
3. Use `gif-search` to find related visual content

**Automation & Bots:**
1. Use `spotify` for music playback automation
2. Use `gif-search` for reaction GIF bots
3. Use `youtube-content` for content summarization bots

## Choosing the Right Tool

**For visual content:**
- Animated GIFs → `gif-search`
- Static images → Use creative/ category skills

**For music:**
- Generate new music → `heartmula`
- Stream existing music → `spotify`
- Analyze audio → `songsee`

**For content transformation:**
- YouTube videos → `youtube-content`
- Other video formats → Combine with transcription tools

**For audio analysis:**
- Feature extraction → `songsee`
- Music generation → `heartmula`
- Playback control → `spotify`

## Common Workflows

### Podcast to Blog Workflow

```bash
# 1. If podcast is on YouTube
/youtube-content "Extract transcript from podcast episode URL"

# 2. Transform to blog post
/youtube-content "Create structured blog post with introduction, key sections, quotes"

# 3. Add GIFs for engagement
/gif-search "Find relevant reaction GIFs for key moments"

# 4. Generate theme music
/heartmula "Create podcast intro music - upbeat, professional"
```

### Music Playlist Curation

```bash
# 1. Search for mood
/spotify "Find energetic workout songs from 2024"

# 2. Create playlist
/spotify "Create 'Morning Energy' playlist"

# 3. Analyze characteristics
/songsee "Extract audio features from playlist tracks"

# 4. Add similar tracks
/spotify "Find songs with similar tempo and energy"
```

### Content Summarization Bot

```bash
# 1. Extract transcript
/youtube-content "Get transcript from educational video"

# 2. Create multiple formats
/youtube-content "Generate: brief summary, detailed notes, Twitter thread"

# 3. Add visual elements
/gif-search "Find educational/lightbulb moment GIFs"

# 4. Share and distribute
```

### Audio Research Project

```bash
# 1. Generate test audio
/heartmula "Create jazz composition for audio analysis study"

# 2. Extract features
/songsee "Generate mel-spectrogram, chroma, and MFCC features"

# 3. Compare with reference
/spotify "Find similar jazz tracks for comparison"

# 4. Analyze differences
/songsee "Extract features from Spotify reference tracks"
```

## Best Practices

**GIF Search:**
- Use specific, descriptive keywords
- Consider context and tone
- Preview before using in professional settings
- Respect copyright and usage terms

**Music Generation:**
- Be specific with style tags and lyrics
- Iterate on prompts for better results
- Consider licensing for commercial use
- Experiment with different combinations

**Audio Analysis:**
- Use appropriate sample rates
- Consider file format (WAV, MP3, FLAC)
- Normalize audio before analysis
- Document analysis parameters

**Spotify Integration:**
- Authenticate once, reuse tokens
- Handle rate limits gracefully
- Cache search results when possible
- Respect user privacy and preferences

**YouTube Content:**
- Verify transcript accuracy
- Maintain attribution to original creator
- Check video availability (not deleted/private)
- Respect copyright when repurposing

## Integration Tips

**API Keys & Authentication:**
- Spotify requires OAuth flow
- Tenor (GIFs) requires API key
- YouTube Data API for extended features
- Configure in Hermes settings

**Rate Limits:**
- Spotify: 180 requests/min
- Tenor: Varies by tier
- YouTube: 10,000 units/day (basic quota)
- Implement exponential backoff

**Output Formats:**
- GIFs: Save as .gif files
- Audio: WAV for quality, MP3 for size
- Spectrograms: PNG or SVG
- Text: Markdown for summaries

## Use Case Examples

**Content Creator:**
```bash
# Research topic on YouTube
/youtube-content "Summarize top 5 videos on [topic]"

# Create complementary content
/heartmula "Generate intro music matching video mood"

# Find visual elements
/gif-search "Relevant reaction GIFs for key points"
```

**Music Researcher:**
```bash
# Generate test compositions
/heartmula "Create songs in different genres for analysis"

# Extract audio features
/songsee "Analyze all generated tracks"

# Compare with commercial music
/spotify "Find commercial examples, analyze with songsee"
```

**Social Media Manager:**
```bash
# Repurpose long-form content
/youtube-content "Convert CEO interview to Twitter thread"

# Add engaging GIFs
/gif-search "Professional, on-brand reaction GIFs"

# Schedule music for live streams
/spotify "Create branded playlist for weekly stream"
```

## Contributing

Found a bug or have an enhancement idea?

1. Open an issue describing the improvement
2. Fork the repository
3. Make changes to the relevant `SKILL.md`
4. Submit a pull request

## Related Categories

- **creative/** - Visual design, video creation, generative art
- **productivity/** - Content management and organization
- **research/** - Content analysis and knowledge management
- **mlops/** - Audio ML models and feature extraction

---

**Questions?** Check the [Hermes Agent documentation](https://hermes-agent.nousresearch.com/docs/) or ask in the [Discord community](https://discord.gg/nousresearch).
