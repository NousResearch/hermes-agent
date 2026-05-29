# Signal Room Long-Form Adult Animated Retention Quality Bar

Date: 2026-05-28
Status: working production spec
Scope: Signal Room long-form adult animated/vector explainer lane

## Purpose

Raise Signal Room long-form retention and production value by translating the
reference-video lesson into a concrete animated explainer system.

The target is not to copy The Infographics Show. The useful lesson is its
production discipline: a clear narrative spine, constant visual progression,
strong thumbnail/first-frame packaging, clean narration, and motion that turns
abstract claims into visible mechanisms. Signal Room should keep stronger
evidence standards and a more adult documentary tone.

## Current Lane Lock

Use the active Signal Room long-form lane:

- adult animated/vector documentary;
- reusable character rigs, props, and mechanisms;
- deterministic 2D animation via Moho/Cavalry/HyperFrames/HTML/SVG/GSAP;
- voice-realism audio with phone-audible music and SFX;
- no static card-heavy long-form;
- no generic AI-image slideshow;
- no childish cartoon tone;
- no avatar-first presenter format unless explicitly reopened.

## Retention Pattern

Every long-form segment should use this repeating pattern:

1. **Human problem:** show a person facing a simple, concrete situation.
2. **Visible contradiction:** the simple situation does not behave as expected.
3. **Mechanism reveal:** the hidden machinery opens up and explains why.
4. **Escalation:** add a second layer that makes the system feel bigger.
5. **Memory anchor:** end the beat with one strong visual that can survive as a
   contact-sheet frame.

For a 10-15 second proof, this becomes:

- 0.0-2.5s: character sees the ordinary bill;
- 2.5-5.0s: final number splits into fee labels;
- 5.0-8.5s: wall opens to reveal the fee machine;
- 8.5-12.0s: character points or reacts while the machine drives the fees;
- 12.0-15.0s: one final lever pull creates a readable hold frame.

## Visual Quality Bar

The frame should read as adult documentary animation, not a template explainer.

Required:

- a clear foreground character, middle-ground object, and background mechanism;
- a visible cause/effect relationship in every mechanism scene;
- one meaningful visual change every 2-3 seconds in short proofs;
- one larger structural reveal every 5-8 seconds in long-form;
- restrained but readable character acting;
- phone-readable silhouettes before captions are considered;
- local typography only when it clarifies the visual, not as the visual itself.

Reject:

- a slide with animated stickers;
- busy gears/arrows that do not explain cause and effect;
- labels appearing all at once;
- character poses too subtle to read muted;
- mascot-like bounce or childish exaggeration;
- generic stock-business art;
- text outside safe zones.

## Character Acting Gate

Before assembling the next fee-machine V2 proof, the chosen adult character must
pass a contact-sheet review with these poses:

- neutral read;
- bill shock;
- look to machine;
- skeptical point or open-palmed explain gesture;
- final lean or weight shift;
- mouth closed;
- mouth `a`;
- mouth `o`.

Pass criteria:

- face and hands read at phone size;
- pose changes are clear without sound;
- pointing/explain gesture aims at the machine area;
- no crop on hair, elbows, hands, or feet;
- adult tone survives the pose set;
- transparent edges do not show halos;
- source license is documented.

Fail criteria:

- sample-rig look dominates the scene;
- arms or hands look broken;
- expression is too subtle to matter;
- character feels cute, goofy, or corporate-stock;
- contact sheet cannot explain the acting sequence at a glance.

## Motion Rules

Use motion to explain, not decorate.

- Fee labels enter sequentially from the total, not from random screen edges.
- The wall reveal should expose the machine as a spatial answer to the bill.
- The machine needs one obvious driver: lever, gear, piston, or belt.
- Money/data flow should have one direction and one destination per beat.
- Camera movement should be motivated: push in for discovery, lateral reveal
  for mechanism, slight settle for the memory-anchor hold.
- Avoid continuous background motion that competes with the bill, character, or
  fee labels.

Suggested rhythm for the fee-machine proof:

`hold -> split -> reveal -> mechanical pulse -> acting beat -> settle`

## Audio Quality Bar

Audio should make the proof feel produced even before final narration.

Minimum:

- paper snap or envelope sound for the bill;
- distinct ticks for the fee split;
- low mechanical thump for the wall/machine reveal;
- small lever/click impact on the final pulse;
- light room tone or music bed audible on phone speakers;
- narration or caption optional for the 10-15 second proof, but the visual must
  remain understandable muted.

Reject:

- silent proof presented as production-quality;
- thin UI blips only;
- music masking the narrator;
- narrator line doing all the explanatory work.

## Packaging Gate

Every proof needs a first-frame and contact-sheet test.

The first frame must answer:

- who is affected?
- what object starts the story?
- why should the viewer keep watching?

The final contact sheet must show:

- ordinary bill;
- number split;
- machine reveal;
- character acting read;
- memory-anchor hold.

If the contact sheet looks like five similar cards, the proof fails.

## Evidence Discipline

The reference-video style is useful for pacing but risky for factual tone.
Signal Room should keep the production energy while using cleaner evidence.

Rules:

- do not use old theme-language crutches such as "the signal is," "STORY
  SIGNAL," or "hidden system";
- avoid broad claims unless they can be sourced in the eventual script packet;
- convert claims into visible systems, not fear language;
- keep the viewer's practical question centered: "what is happening to me, and
  what mechanism causes it?"

## Next Concrete Build Step

The next production step is still the rig-acting gate, not a full video:

0. Run the local production environment gate:

   ```bash
   python scripts/signal_room_video_env_gate.py \
     --out /path/to/review-package/video_env_scorecard.json
   ```

   If `render_mode` is `external_pose_export_required`, either install Blender
   locally or export the required pose PNGs from a Blender/Moho-capable machine.
   Generate the intake pack before handing the work to another workstation:

   ```bash
   python scripts/signal_room_pose_export_intake_pack.py \
     --out /path/to/review-package/pose_export_intake \
     --candidate-name Suit_Male
   ```

   The pack includes the exact filenames, manifest template, validation command,
   and contact-sheet command expected by the downstream gates.
   To keep choreography work moving while waiting on that pass, generate the
   review-only temporary vector rig:

   ```bash
   python scripts/signal_room_temp_rig_seed.py \
     --out /path/to/review-package/signal_room_adult_investigator_v0_20260528
   ```

   This temporary rig is for blocking, timing, and retention review only; it
   must be replaced by the approved Blender/Moho candidate before publication.
1. Render or export `Suit_Male` first, `Suit_Female` second, and
   `OldClassy_Male/Female` only if the first two fail.
2. Produce the required transparent pose frames and mouth switches.
3. For any generated scaffold, run the offline scaffold gate before preview:

   ```bash
   python scripts/signal_room_scaffold_gate.py /path/to/scaffold \
     --out /path/to/review-package/scaffold_scorecard.json
   ```

4. Run the deterministic rig gate:

   ```bash
   python scripts/signal_room_rig_acting_gate.py \
     /path/to/review-package/character_frames \
     --out /path/to/review-package/rig_acting_scorecard.json
   ```

5. Build a 1080x1920 contact sheet for each passing candidate:

   ```bash
   python scripts/signal_room_contact_sheet.py \
     /path/to/review-package/character_frames/Suit_Male \
     --out /path/to/review-package/Suit_Male_contact_sheet.svg
   ```

6. Install the passing pose export into the HyperFrames scaffold:

   ```bash
   python scripts/signal_room_pose_export_installer.py \
     --candidate /path/to/review-package/character_frames/Suit_Male \
     --scaffold /path/to/review-package/fee_machine_v2_scaffold \
     --out /path/to/review-package/pose_export_install_scorecard.json
   ```

   This maps `lean_weight_shift.png` into the scaffold's `slight_lean.png`,
   rewrites the scaffold pose references from temporary SVGs to approved PNGs,
   and records the source candidate in `source_rig_manifest.json`.

7. After installation, run the refresh pipeline to rerender the proof, resample
   retention frames, remux audio, refresh scorecards, and rebuild the handoff:

   ```bash
   python scripts/signal_room_post_pose_pipeline.py \
     --package-dir /path/to/review-package \
     --candidate /path/to/review-package/character_frames/Suit_Male
   ```

   Use `--dry-run` first if you want to inspect the full command plan without
   executing HyperFrames or FFmpeg.

8. Score adult tone, phone readability, gesture clarity, license status, and
   conversion effort.
9. Source or design the minimum SFX pass from the scaffold's
   `audio_cue_sheet.json`: room-tone bed, bill snap, sequential fee ticks, wall
   reveal thump, machine drive loop, pointing accent, final lever click, and
   memory-hold tail.
   If final SFX assets are blocked, seed the audio package for gate plumbing:

   ```bash
   python scripts/signal_room_audio_asset_seed.py \
     --cue-sheet /path/to/scaffold/audio_cue_sheet.json \
     --out /path/to/review-package/audio_assets
   ```

   Then validate the package:

   ```bash
   python scripts/signal_room_audio_asset_gate.py \
     --cue-sheet /path/to/scaffold/audio_cue_sheet.json \
     --assets /path/to/review-package/audio_assets \
     --out /path/to/review-package/audio_asset_scorecard.json
   ```

   Seeded WAV files are placeholders and must be replaced by final sound design
   before editorial audio review.
8. Build the proof contact sheet from the scaffold's
   `retention_frame_plan.json` sample times: ordinary bill, number split,
   machine reveal, acting read, and memory anchor. Reject the proof if those
   frames read as five similar cards.
   If real HyperFrames sampled exports are blocked, seed the frame package for
   gate plumbing only:

   ```bash
   python scripts/signal_room_proof_frame_seed.py \
     --plan /path/to/scaffold/retention_frame_plan.json \
     --out /path/to/review-package/proof_frames
   ```

   Seeded proof frames are placeholders and must be replaced by actual sampled
   frames before visual review.
9. Run the deterministic retention-frame gate on those sampled proof frames:

   ```bash
   python scripts/signal_room_retention_frame_gate.py \
     --plan /path/to/scaffold/retention_frame_plan.json \
     --frames /path/to/review-package/proof_frames \
     --out /path/to/review-package/retention_frame_scorecard.json
   ```

10. Build the proof retention contact sheet:

   ```bash
   python scripts/signal_room_proof_contact_sheet.py \
     --plan /path/to/scaffold/retention_frame_plan.json \
     --frames /path/to/review-package/proof_frames \
     --out /path/to/review-package/proof_retention_contact_sheet.svg
   ```

11. Only then assemble the 10-15 second fee-machine V2 proof.
12. Build the consolidated review package index:

   ```bash
   python scripts/signal_room_review_package_index.py \
     /path/to/review-package \
     --out /path/to/review-package/review_package_index.json \
     --markdown-out /path/to/review-package/REVIEW_STATUS.md
   ```

   This index is the handoff status: it should list no blockers before the
   package is treated as ready for editorial review.

Local blocker on this machine: Blender is not installed, so the actual rig pose
render must happen on a Blender/Moho-capable machine or after Blender is added.
