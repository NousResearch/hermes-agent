# Long-form no-lipsync mini-documentary pipeline (Higgsfield)

Use when user asks for 45–60s vertical reel without avatar/lipsync, but with cinematic visuals + VO + captions.

## Proven flow
1. Write VO-first script (10–14 lines, emotionally connected hook + factual core + reflective close).
2. Generate TTS once (target ~50–60s).
3. Create 10–12 scene prompts (5s each) with one dominant concept per scene.
4. Run Higgsfield scene generation per prompt (`cinematic_studio_video_v2`, duration=5, wait=true, json=true).
5. Parse JSON robustly: CLI output can be a **list** of jobs, not a single dict.
   - Read `result_url` from `output[0].result_url` when output is list.
6. Download all completed scenes; skip/replace blocked scenes (e.g., NSFW status).
7. Concatenate scenes, retime to VO duration, conform to 1080x1920 @ 30fps.
8. Burn readable subtitles with outline + safe bottom margin.
9. Final mux: video + VO (`-shortest`) and verify duration/resolution via ffprobe.

## Important pitfalls
- Prompting "vertical 9:16" in text is not enough; model params may still default to 16:9 unless explicit aspect ratio option is supported/passed.
- Some scenes can return `status=nsfw` with empty URL. Keep extra scene prompts ready and replace blocked shots.
- Batch generation can partially succeed. Build pipeline to continue with available clips and preserve narrative continuity.

## Quality notes for social retention
- First 2 seconds must carry curiosity hook.
- Use short visual beats (5s) and avoid concept overload per shot.
- Captions: large size, high contrast, outline, mobile-safe margin.
