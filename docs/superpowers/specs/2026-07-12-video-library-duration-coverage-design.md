# Video Library Duration Coverage Design

## Goal

Prevent a long narration from immediately recycling the first matched local
shots. Video Studio must consume semantically relevant, non-repeating library
clips first and only repeat the minimum number of clips when the authorized
library cannot cover the narration duration.

## Current behavior

The Desktop matcher selects exactly one clip for each script sentence. The
named-library timeline therefore contains one shot per sentence regardless of
the eventual narration duration. MoneyPrinter correctly avoids black frames by
cycling that short list until the audio is covered, but the first shot can
reappear even though unused relevant clips remain in the library.

## Chosen approach

Use an ordered, round-robin candidate pool at the video-library edge.

1. Search multiple ranked candidates for every script segment.
2. Build round one from the best unused candidate for each segment in script
   order. This preserves the existing primary narrative sequence.
3. Build subsequent rounds from the next unused candidate for each segment,
   again in script order.
4. Prefer a different source asset globally. If every remaining candidate
   belongs to an already-used asset, allow a different clip from that asset
   before allowing an identical clip.
5. Materialize the complete ordered pool into the renderer-neutral timeline
   and cache it in MoneyPrinter's local-material whitelist.
6. MoneyPrinter consumes the pool sequentially until the narration plus its
   safety margin is covered. Only after the complete pool is exhausted may its
   existing cycle fallback repeat clips.

This keeps video-library retrieval out of MoneyPrinter and avoids coupling the
renderer to Hermes configuration or SQLite state.

## Alternatives rejected

- **Query the video library from MoneyPrinter after TTS:** this gives exact
  audio duration but couples the upstream renderer to Hermes-specific storage,
  credentials, and library contracts.
- **Slow or freeze the final shot:** this avoids repetition but produces a
  visibly static result and ignores unused relevant footage.
- **Stop rendering when footage is short:** this offers the strictest quality
  gate but conflicts with the selected requirement to always produce a video.

## Components

### Candidate-pool planner

Add a pure planner beside `named-library-matching.ts`. It accepts script
segments and their ranked candidates and returns ordered selections containing
both the segment and clip. The function is deterministic and globally avoids
duplicate clip IDs.

The primary round preserves the existing selection contract. Supplemental
rounds improve duration coverage without allowing one sentence with many
candidates to crowd later narrative beats out of the beginning of the video.

### Automatic timeline creation

`useNamedVideoLibrary.createAutomaticTimeline()` requests a larger bounded
candidate set for each segment, calls the pool planner, and sends every planned
clip plus its originating script row to the existing timeline API. Manual
matching remains one confirmed clip per sentence and is unchanged.

The bounded query protects Desktop latency and file-copy cost. The initial
limit is 12 candidates per segment; the planner uses every returned unique
candidate. This is enough to cover normal short-form narration while keeping a
hard upper bound for large libraries.

### MoneyPrinter fallback

No new Hermes core tool and no dynamic library query are added. MoneyPrinter's
existing sequential combiner already stops after enough video has been
processed and cycles only when the provided pool is too short. Tests will lock
this fallback contract so future refactors cannot shuffle or repeat clips
before all supplied unique paths are consumed.

## Data and provenance

Every supplemental timeline row retains:

- the original `segmentId` and script text;
- clip and asset IDs;
- source path, source hash, source time range, tags, quality, and confidence;
- its deterministic position in the video track.

The UI may continue showing one primary confirmation per segment. The complete
shot pool remains inspectable in the generated timeline and acceptance JSON.

## Failure and fallback behavior

- No candidate for a segment: use the current unfiltered-library fallback.
- No usable clips in the library: fail before creating a render task.
- Some segments have fewer candidates: continue round-robin with other
  segments while preserving their order.
- Candidate list contains duplicate clip IDs or assets: never repeat a clip;
  prefer new assets, then allow additional clips from used assets.
- Candidate pool shorter than narration: MoneyPrinter repeats the smallest
  prefix needed to cover the remaining duration, preserving continuous video
  and avoiding black frames.

## Testing

1. Pure planner tests prove primary narrative order, round-robin supplements,
   global clip deduplication, and new-asset preference.
2. Hook tests prove automatic timeline creation requests expanded candidates
   and sends supplemental script rows in the planned order.
3. MoneyPrinter tests prove supplied unique local paths are processed before
   the cycle fallback is entered.
4. Real acceptance uses the beef-noodle library with narration longer than the
   original three shots, verifies more than three distinct source clips were
   cached, confirms H.264/AAC portrait output, and visually samples the output
   for premature repetition.

## Non-goals

- Sentence-level forced alignment between TTS timestamps and shots.
- Vector retrieval or a new embedding database.
- Changing manual shot confirmation.
- Removing the final repeat fallback when the entire candidate pool is short.
- Adding a Hermes Agent Core model tool.
