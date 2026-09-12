# Voice and Pacing

Narration-first design for young learners. The voice carries 80% of the teaching. On-screen text is for reinforcement, not information delivery.

## Voice Character

The narrator is warm, curious, and slightly playful. Not a teacher voice. Not a kids' TV host voice. Think "cool older sibling explaining something they find genuinely interesting."

**Tone attributes**:
- Conversational, not instructional
- Genuinely curious ("I wonder what happens if...")
- Slightly conspiratorial ("Here's the cool part...")
- Never condescending ("Great job!" is banned)
- Allows silence (pauses are part of the voice)

## Script Writing

### Structure Per Scene

```
[VISUAL CUE] Narration text. (timing note)

[Objects appear] "Look at this." (1s pause)
[Pip enters] "Pip sees three groups." (point at groups)
[Question] "How many is that altogether?" (3s PAUSE)
[Reveal] "Twelve! Three groups of four makes twelve." (celebration)
```

### Language Rules

- **Short sentences.** Max 10 words. Usually 5-7.
- **Active voice.** "We have three groups" not "Three groups are present."
- **Present tense.** "Look, here come the circles" not "The circles will appear."
- **Concrete nouns.** "Three apples" not "three items" or "three objects."
- **Questions.** At least 2 per video. Always followed by a 3-second pause.
- **Sound words.** "Pop!" "Whoosh!" "Boing!" -- subcaption cues for the animation.

### Vocabulary by Grade

| Grade | Math vocabulary to use | Avoid |
|-------|----------------------|-------|
| K-1 | add, take away, more, less, same, group | sum, difference, combine |
| 2 | plus, minus, equal, skip count | operation, equation |
| 3 | times, multiply, groups of, divide, share, fraction, equal parts | product, quotient, numerator, denominator (introduce LATE) |

Introduce formal terms AFTER the concept is visually established, never before.

## Pacing Table

| Moment | Narration speed | On-screen | Duration |
|--------|----------------|-----------|----------|
| Hook | Normal | Visual surprise | 3-5s |
| Setup | Slow, clear | Objects appearing | 5-8s |
| Build | Normal | Animations playing | 10-15s |
| Question | Slower, rising inflection | Pip thinking, question mark | 2s narration |
| Pause | SILENCE | Still frame, Pip waiting | 3s |
| Reveal | Normal, slightly excited | Answer + reward | 3-5s |
| Celebration | Quick, upbeat | Confetti, Pip spin | 2s |
| Transition | Normal | FadeOut cleanup | 1-2s |

**Total scene: 25-40 seconds.** Never exceed 45s.

## Subcaption Strategy

Since Manim renders video without audio, all narration is written as subcaptions. These serve three purposes:

1. **TTS script**: If voiceover is added later, subcaptions become the TTS input
2. **Subtitle track**: Auto-generated .srt file for accessibility
3. **Timing guide**: Ensures animations sync with intended narration

```python
# Narration via subcaptions
self.add_subcaption("Look at these three groups.", duration=2)
self.play(FadeIn(groups), run_time=1.0)
self.wait(1.0)

self.add_subcaption("Each group has four circles.", duration=2.5)
self.play(Indicate(groups[0], color=GOLD), run_time=0.5)
self.wait(2.0)

self.add_subcaption("How many circles is that altogether?", duration=3.5)
self.play(pip.think())
self.wait(3.0)  # Sacred question pause

self.add_subcaption("Let's count!", duration=1.5)
self.play(pip.excited())
```

## TTS Integration (Optional)

For full narrated videos, use ElevenLabs or edge-tts:

### ElevenLabs (higher quality)

```python
# Extract subcaptions to narration script
# Then generate audio segments matching scene timing

import requests

def generate_narration(text, voice_id="pNInz6obpgDQGcFmaJgB"):
    """Generate narration clip for one subcaption."""
    resp = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        headers={"xi-api-key": os.environ["ELEVENLABS_KEY"]},
        json={
            "text": text,
            "model_id": "eleven_turbo_v2",
            "voice_settings": {
                "stability": 0.6,        # Slightly varied for warmth
                "similarity_boost": 0.8,
                "style": 0.3,            # Subtle expressiveness
            }
        },
    )
    return resp.content  # MP3 bytes
```

### Edge-TTS (free, decent quality)

```bash
# Generate narration from subcaption text
edge-tts --voice "en-US-AnaNeural" --text "Look at these three groups." --write-media scene1.mp3
```

`en-US-AnaNeural` is a warm, clear female voice that works well for kids' content. `en-US-GuyNeural` for a male alternative.

### Audio Muxing

```bash
# Combine narration with rendered video
ffmpeg -i final.mp4 -i narration.mp3 \
  -c:v copy -c:a aac -shortest \
  final_narrated.mp4
```

## Rhythm and Flow

A well-paced kids' video has a musical quality. The rhythm alternates between tension (question, build) and release (answer, celebration):

```
BUILD   ... BUILD   ... QUESTION ... pause ... REVEAL!  ... celebrate ...
BUILD   ... BUILD   ... QUESTION ... pause ... REVEAL!  ... celebrate ...
FINAL BUILD ... FINAL QUESTION ... long pause ... BIG REVEAL!! ... BIG CELEBRATE!
```

Each cycle is one scene. The final cycle gets more time and a bigger celebration. The kid knows the pattern by scene 2 and starts anticipating -- anticipation IS engagement.

## Video Length Guide

| Content | Scenes | Total time | Notes |
|---------|--------|-----------|-------|
| Single concept intro | 4-5 | 2-3 min | One ladder traversal |
| Concept with practice | 6-8 | 3-5 min | Ladder + 2-3 practice examples |
| Comparison (e.g., mult vs add) | 6-8 | 3-5 min | Two short ladders + contrast |

Never exceed 5 minutes. If the concept needs more time, split into two videos ("Part 1: What is multiplication?" "Part 2: Multiplication tricks").

## Silence Is Not Empty

The pauses in these videos are the most important parts. During silence:
- The kid is counting
- The kid is guessing the answer
- The kid is connecting what they see to what they know
- The kid is talking to the screen

**Never fill silence with music, sound effects, or narration during a question pause.** The silence IS the learning tool.
