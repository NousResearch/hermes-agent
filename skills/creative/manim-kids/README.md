# manim-kids

Animated math and science videos for elementary-age kids (ages 6-10) using [Manim Community Edition](https://www.manim.community/).

Built for visual learners and neurodivergent kids who need more than worksheets. Produces short, vivid, mascot-driven explainer clips that make abstract concepts physical and obvious.

## What it does

- Generates Manim Python scripts that produce 2-5 minute animated math videos
- Features Pip, a friendly circle mascot who learns alongside the viewer
- Uses a concrete-to-abstract ladder: real objects -> shapes -> diagrams -> equations
- Color-coded operations (blue=addition, green=multiplication, etc.) for consistent visual language
- Reward animations (confetti, star bursts) after key reveals
- Voice-first pacing with 3-second question pauses for active thinking

## Target concepts

Primarily third grade math: multiplication as groups, division as sharing, fractions, area, perimeter, rounding, telling time. Full K-5 concept map included in references.

## Dependencies

Python 3.10+, Manim CE, LaTeX, ffmpeg. Run `scripts/setup.sh` to verify.

## References

| File | Contents |
|------|----------|
| `references/mascot.md` | Pip implementation, expressions, positioning |
| `references/rewards.md` | Celebration animations, confetti, star burst |
| `references/concrete-to-abstract.md` | The 4-rung pedagogical ladder with examples |
| `references/elementary-math.md` | K-5 concept map with visual metaphors |
| `references/engagement.md` | Hooks, humor, question pauses, surprise patterns |
| `references/voice-and-pacing.md` | Narration workflow, TTS integration, pacing |
