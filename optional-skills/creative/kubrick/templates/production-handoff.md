# Production Handoff Template

For approved scenes, produce:

## production_scene
- scene_id, script_pages, estimated_screen_duration
- location, time_of_day, cast, extras
- wardrobe, props, practical_effects, vfx, stunts
- sound_requirements, music_function
- environment, weather, lighting_intent
- visual_motifs, continuity_images_required
- safety flags

## visual_identity (for AI generators / previs)
- character_model_ids
- wardrobe_state_ids
- location_id, prop_ids
- lighting_state, lens_language, composition_rules
- movement_rules, color_logic
- prohibited_drift: [list]

### Single-frame and AI image prompt extensions
When the target is a generative image model or cinematic still:
- Provide `symbolic_intent` (dramatic_function required; keep it functional, not aesthetic).
- Motif as `observed_form` + current visible state.
- **State differentials** that make mutation or transformation legible in one frame (different ages of traces, material conditions, light interactions, object states).
- **Convergence point**: the exact location/interaction where the figure's behavior, the motif, geometry, and light meet and modify each other.
- Relational elements: objects, shadows, or environmental forces that actively participate in or alter the motif.
- Integration test: the motif must cross channels and would change how the image is read if removed.
- Translate `cinematic_encoding` into prompt language: relational/asymmetric composition, negative space function, geometry as structure, light behavior (including shadows as active), precise material response.
- Figure blocking: posture, weight shift, foot/hand position, and gaze must directly engage the active site of the motif.

Script remains the authority. Generation prompts must not invent canon. All elements must stay strictly observable.
### Corpus Usage (Expanded Profound Systems)
When writing visual_identity or image prompts, pull directly from cinematic-symbolism-corpus.md:
- Expanded Cinematic Systems: Light Systems, Geometry & Negative Space, Material & Trace, Inversion & Reflection, Perceptual Layering & Residue.
- Single-Frame & Generative Image Translation section for mappings and rules.
- Must use Corpus Integration Rules: at least one technique from each major system, interlocking at convergence point, state differentials, document in cinematic_encoding.

## Esoteric Integration Note (for visual work)
For image prompts and single-frame work, apply the Esoteric Structural Translation section of cinematic-symbolism-corpus.md. All ancient magical concepts (threshold, trace-binding, witness objects, erasure operation, inversion/crossing, dual preservation-dissolution) must be realized through interlocking Light + Geometry + Material/Trace + Inversion systems at a single convergence point with visible state differentials. Hidden correspondence stays private.

