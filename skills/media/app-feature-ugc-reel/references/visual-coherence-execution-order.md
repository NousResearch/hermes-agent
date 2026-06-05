# Visual Coherence Execution Order (BeSoul/ITSU-style storytelling UGC)

Use this exact order when the user wants coherent character + product UI across clips.

1. Generate base `character` still (`imagegen_2_0`, 3:4).
2. Generate base `app_screen` still (`imagegen_2_0`, 3:4) with exact feature wording.
3. Generate storyboard boards sequentially (`imagegen_2_0`, 16:9):
   - B1 from `[character, app_screen]`
   - B2 from `[character, app_screen, B1]`
   - B3 from `[character, app_screen, B2]` (and B4 similarly when 60s)
4. Generate Seedance clips from board images (9:16), one per board.
5. Stitch clips after verifying each exists and has valid duration.

## Anti-pattern to avoid
- Starting directly with Seedance text prompts for all clips (without image anchors/boards) when continuity is required.

## Recovery rule
- If user flags inconsistency, cancel running generation, acknowledge, and restart from step 1 with corrected prompts and anchors.
