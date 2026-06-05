# Reel post-production QA (no-avatar, voiceover-driven)

## Trigger
Use this when producing IG Reels from generated clips + TTS voiceover (especially when user asks for clean output, no UGC avatar, or no subtitles).

## Hard QA gates before delivery
1. **Aspect ratio must be vertical from generation step**
   - For Higgsfield CLI, use `--aspect_ratio 9:16` (underscore).
   - Do not assume `--aspect-ratio` works.
2. **No subtitle artifacts unless explicitly requested**
   - If user asks "sin subtítulos", remove subtitle filter entirely.
   - Do not ship text overlays from prior render presets.
3. **Audio/timeline integrity (avoid accidental truncation)**
   - Probe both durations (`ffprobe`): base timeline video and voiceover.
   - Do **not** blindly use `-shortest` in final mux when timeline includes CTA or tail visuals; `-shortest` will cut output to the shortest stream (often VO) and can silently remove the ending.
   - If video is shorter than VO, extend last frame via `tpad=stop_mode=clone:stop_duration=<delta>`.
   - If VO is shorter than video and you still need ambience under the tail, pad/mix audio (`apad`, `amix`) and `atrim` to full video length.
4. **Final format check**
   - Verify output with `ffprobe`: duration, width/height, fps.
   - Reel target: `1080x1920`, stable fps, no truncation at end.

## Common pitfalls observed
- Generated JSON from `higgsfield generate create ... --json` is an array; parsers expecting object paths fail silently and skip downloads.
- Subtitle styling can destroy legibility if line length is not constrained; for subtitle versions, cap at 1–2 short lines and anchor to bottom safe area.
- Concatenated clips may have mixed timestamps; re-encode concat stage (`libx264`) when needed for stable downstream timing.

## Recommended render sequence (no subtitles)
1. Generate all scenes in 9:16.
2. Download and validate scene file sizes (>100KB sanity check).
3. Concat/re-encode to base timeline.
4. Probe voice duration.
5. Extend video tail if needed.
6. Final mux with voiceover, no subtitle filter.
7. Run ffprobe QA and only then deliver.
