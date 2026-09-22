# Recovered multi-view workflow

Verified from the September 4 session record and preserved asset README on September 7.

The previous workflow used Hunyuan3D-2mv neural shape generation. It split the 2x2 card, removed backgrounds, supplied front/left/back elevations, generated a mesh through diffusion and volumetric decoding, and exported GLB for Blender cleanup. The plan crop was retained for visual checking; it is not a supported elevation label in the recovered interface.

The initial leader script is recovered as historical-leader-2mv-generator.py.txt. Do not execute it unchanged: its paths no longer exist, its first view dictionary included invalid labels, and subsequent runs reduced inference quality due to CPU performance. The session records a later correction to front/left/back mapping.

Historical records report initial building runs at 20 steps and resolution 256 with MPS, followed by CPU fallback. Cat Arts was regenerated from background-cleaned cards after a procedural replacement was rejected. Leader runs were reduced to one step at resolution 64; the first owl output was reported as 460 vertices and 908 faces. That is historical reference geometry, not proof of a finished leader or current file availability.

Current checks: the original /private/tmp/hunyuan3d-2 checkout, hunyuan3d-2-venv, and hunyuan-cache are absent from that directory. The earlier 8d25 worktree is absent. Fourteen newer building/worker GLBs and forty card backups have been recovered separately. No complete approved card-derived leader set has been verified.

Continuation: restore a supported multi-view inference environment, inspect foreground masks and camera/view orientation, test one leader at adequate quality, compare against all four card views in Blender, then batch the remaining assets with source hashes and durable per-asset receipts. Determine current GPU support before selecting settings; do not silently downgrade to one-step reference output. Texture generation, topology cleanup, rigging and animation remain subsequent stages.

The manually assembled owl study from September 7 is paused and is not the selected production method.
