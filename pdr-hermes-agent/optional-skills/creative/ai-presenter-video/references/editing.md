# Editing

Read this file for the visual plan, deterministic timeline, openings, closes, screen demos, captions, keyword callouts, covers, previews, and exports.

## Visual routes

- **Presenter-led:** the presenter carries the explanation; add one concise visual idea per beat.
- **Screen-demo:** preserve real UI readability; introduce the presenter large at chapter starts, then shrink the same continuous video into a verified safe region.
- **Mixed explainer:** assign supporting media only where it proves or clarifies narration.

Do not fabricate product evidence. Label conceptual visuals clearly.

## Timeline contract

The locked narration defines duration and semantic boundaries. Every clip keeps three independent values:

- authored start in the final video;
- authored duration;
- source-media start.

When an opening changes length, shift authored starts while preserving presenter source offsets. Keep video sources muted and use approved external narration as the program clock. Prefer one continuous presenter source and source-time slices over regenerated chapter clips.

Use hard cuts at settled semantic pauses. Add transitions only when they clarify structure. Preserve a quiet tail so the last word, gesture, and mouth position finish naturally.

## Opening, body, and close

- Establish the topic and payoff within the first few seconds.
- Design the time-zero frame as a deliberate cover frame.
- Keep opening copy away from eyes, mouth, and the hand-motion path.
- Use the same visual system across the cover, chapters, captions, keywords, progress, and close.
- End with a conclusion or one requested next step. Do not add promotional copy without a brief requirement.

## Captions

Generate word timestamps from the final audio. Split into short semantic phrases, normally one readable line. A practical baseline is about `40ms` visual lead and `120ms` tail hold when neighboring phrases allow it.

Keep captions clear of the face, UI, watermark, progress line, and platform controls. Highlight one meaningful term per phrase. Make entrances and exits quick enough to preserve reading stability.

## Presenter-side keyword presets

Use keyword callouts to amplify selected spoken beats. Avoid repeating one identical card throughout the video.

Rotate a small family of presets such as:

- radial burst;
- tilted ribbon or sticker;
- hand-drawn circle or marker stroke;
- large/small type contrast;
- staggered word-chip cluster;
- double-layer outline lockup.

Bind each callout to a real spoken stress or ASR word anchor. Separate the entry timing of the kicker, main word, secondary word, marker, rays, and outline by a few frames. Let the callout settle briefly, then collapse before the next idea. Keep it inside a consistent presenter-side safe region and verify the complete motion path does not cover the face or hands.

## Composition and preview

Use a deterministic compositor available in the environment. For HyperFrames:

1. load the current HyperFrames workflow and domain instructions;
2. give all timed media explicit start, duration, track, and source offset;
3. run the required composition check with representative emphasis and transition samples;
4. inspect time zero, chapter cuts, layout changes, keyword beats, captions, close, and final frame;
5. open the final Studio preview and obtain approval before rendering;
6. render at delivery quality only after approval.

Measure the assembled stereo program after rendering. Mono narration duplicated into stereo can measure about 3 LU louder, so perform final program normalization from the rendered file.
