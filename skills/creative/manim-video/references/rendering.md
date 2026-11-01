# Rendering Reference

## Prerequisites

```bash
manim --version       # Manim CE
pdflatex --version    # LaTeX
ffmpeg -version       # ffmpeg
```

## CLI Reference

```bash
manim -ql script.py Scene1 Scene2    # draft (480p 15fps)
manim -qm script.py Scene1           # medium (720p 30fps)
manim -qh script.py Scene1           # production (1080p 60fps)
manim -ql --format=png -s script.py Scene1  # preview still (last frame)
manim -ql --format=gif script.py Scene1     # GIF output
```

## Quality Presets

| Flag | Resolution | FPS | Use case |
|------|-----------|-----|----------|
| `-ql` | 854x480 | 15 | Draft iteration (layout, timing) |
| `-qm` | 1280x720 | 30 | Preview (use for text-heavy scenes) |
| `-qh` | 1920x1080 | 60 | Production |

**Text rendering quality:** `-ql` (480p15) produces noticeably poor text kerning and readability. For scenes with significant text, preview stills at `-qm` to catch issues invisible at 480p. Use `-ql` only for testing layout and animation timing.

## Output Structure

```
media/videos/script/480p15/Scene1_Intro.mp4
media/images/script/Scene1_Intro.png  (from -s flag)
```

## Stitching with ffmpeg

**CRITICAL: Always re-encode when stitching Manim scenes. Never use `-c copy` for the concat step.**

Manim's partial movie files embed non-monotonic DTS timestamps. Copying streams directly causes `ffmpeg` to silently truncate the video stream (you'll see a 74s output when you expected 104s). The fix is always to re-encode:

```bash
# Step 1: Add silent audio track to each scene (required for concat with mixed audio)
for scene in Scene1 Scene2 Scene3; do
  ffmpeg -y -i "media/videos/script/480p15/${scene}.mp4" \
    -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=44100 \
    -c:v copy -c:a aac -shortest \
    "media/videos/script/480p15/${scene}_s.mp4" 2>/dev/null
done

# Step 2: Stitch with re-encode (NOT -c copy)
cat > concat.txt << 'EOF'
file 'media/videos/script/480p15/Scene1_s.mp4'
file 'media/videos/script/480p15/Scene2_s.mp4'
EOF
ffmpeg -y -f concat -safe 0 -i concat.txt \
  -vf "fps=15" -c:v libx264 -preset fast -crf 22 -c:a aac \
  stitched.mp4
```

**Verifying duration:** After stitching, always check both format and video stream duration separately — they can disagree when timestamps are broken:
```bash
ffprobe -i stitched.mp4 -show_entries format=duration -v quiet -of csv="p=0"          # format duration
ffprobe -i stitched.mp4 -select_streams v:0 -show_entries stream=duration -v quiet -of csv="p=0"  # video stream
```
If they differ, the concat had timestamp issues. Re-encode.

## Tail Pad (when voiceover is longer than video)

When the voiceover duration exceeds the stitched video duration, extend with a freeze-frame tail:

```bash
# Grab last frame of stitched video
ffmpeg -y -sseof -0.1 -i stitched.mp4 -update 1 -vframes 1 last_frame.png 2>/dev/null

# Create freeze-frame tail at matching fps (e.g., 15fps for -ql renders)
ffmpeg -y -loop 1 -i last_frame.png \
  -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=44100 \
  -t 40 -c:v libx264 -r 15 -pix_fmt yuv420p -c:a aac -ar 44100 \
  tail_pad.mp4

# Concat scenes + tail, re-encode
cat > concat_full.txt << 'EOF'
file 'stitched.mp4'
file 'tail_pad.mp4'
EOF
ffmpeg -y -f concat -safe 0 -i concat_full.txt \
  -vf "fps=15" -c:v libx264 -preset fast -crf 22 -c:a aac \
  full_video.mp4
```

**Tail pad fps must match scene fps** (15 for `-ql`, 30 for `-qm`, 60 for `-qh`). Mismatched fps causes the tail to be silently dropped during concat.

## Add Voiceover

```bash
# Mux voiceover onto final video (works with .ogg, .mp3, .wav)
# Use -t to set explicit duration (voiceover length in seconds) rather than -shortest
# -shortest clips to video end if video is shorter than audio
AUDIO_DUR=$(ffprobe -i voiceover.ogg -show_entries format=duration -v quiet -of csv="p=0")
ffmpeg -y -i full_video.mp4 -i voiceover.ogg \
  -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 \
  -t "$AUDIO_DUR" \
  final_with_audio.mp4
```

**Note:** `text_to_speech` tool outputs `.ogg` regardless of requested extension. ffmpeg handles `.ogg` input fine with the `-c:a aac` output flag.

## Add Background Music

```bash
ffmpeg -y -i final.mp4 -i music.mp3 \
  -filter_complex "[1:a]volume=0.15[bg];[0:a][bg]amix=inputs=2:duration=shortest" \
  -c:v copy final_with_music.mp4
```

## GIF Export

```bash
ffmpeg -y -i scene.mp4 \
  -vf "fps=15,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
  output.gif
```

## Aspect Ratios

```bash
manim -ql --resolution 1080,1920 script.py Scene  # 9:16 vertical
manim -ql --resolution 1080,1080 script.py Scene  # 1:1 square
```

## Render Workflow

1. Draft render all scenes at `-ql`
2. Preview stills at key moments (`-s`)
3. Fix and re-render only broken scenes
4. Stitch with ffmpeg
5. Review stitched output
6. Production render at `-qh`
7. Re-stitch + add audio

## manim.cfg — Project Configuration

Create `manim.cfg` in the project directory for per-project defaults:
```ini
[CLI]
quality = low_quality
preview = True
media_dir = ./media

[renderer]
background_color = #0D1117

[tex]
tex_template_file = custom_template.tex
```

This eliminates repetitive CLI flags and `self.camera.background_color` in every scene.

## Sections — Chapter Markers

Mark sections within a scene for organized output:
```python
class LongVideo(Scene):
    def construct(self):
        self.next_section("Introduction")
        # ... intro content ...

        self.next_section("Main Concept")
        # ... main content ...

        self.next_section("Conclusion")
        # ... closing ...
```

Render individual sections: `manim --save_sections script.py LongVideo`
This outputs separate video files per section — useful for long videos where you want to re-render only one part.

## manim-voiceover Plugin (Recommended for Narrated Videos)

The official `manim-voiceover` plugin integrates TTS directly into scene code, auto-syncing animation duration to voiceover length. This is significantly cleaner than the manual ffmpeg muxing approach above.

### Installation
```bash
pip install "manim-voiceover[elevenlabs]"
# Or for free/local TTS:
pip install "manim-voiceover[gtts]"    # Google TTS (free, lower quality)
pip install "manim-voiceover[azure]"   # Azure Cognitive Services
```

### Usage
```python
from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.elevenlabs import ElevenLabsService

class NarratedScene(VoiceoverScene):
    def construct(self):
        self.set_speech_service(ElevenLabsService(
            voice_name="Alice",
            model_id="eleven_multilingual_v2"
        ))

        with self.voiceover(text="Here is a circle being drawn.") as tracker:
            self.play(Create(Circle()), run_time=tracker.duration)

        with self.voiceover(text="Now let's transform it into a square.") as tracker:
            self.play(Transform(circle, Square()), run_time=tracker.duration)
```

### Key Features

- `tracker.duration` — total voiceover duration in seconds
- `tracker.time_until_bookmark("mark1")` — sync specific animations to specific words
- Auto-generates subtitle `.srt` files
- Caches audio locally — re-renders don't re-generate TTS
- Works with: ElevenLabs, Azure, Google TTS, pyttsx3 (offline), and custom services

### Bookmarks for Precise Sync
```python
with self.voiceover(text='This is a <bookmark mark="circle"/>circle.') as tracker:
    self.wait_until_bookmark("circle")
    self.play(Create(Circle()), run_time=tracker.time_until_bookmark("circle", limit=1))
```

This is the recommended approach for any video with narration. The manual ffmpeg muxing workflow above is still useful for adding background music or post-production audio mixing.
