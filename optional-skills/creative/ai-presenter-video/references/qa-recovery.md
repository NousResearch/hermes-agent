# QA and recovery

Read this file before accepting generated media or delivery, and whenever a production fault appears.

## Acceptance gates

### Inputs

- Topic or script exists.
- Presenter image decodes and contains one reviewed adult face.
- Image rights, adult confirmation, and remote-upload permission are recorded.
- Any voice sample contains one authorized speaker.
- Supporting media is probed and its provenance is recorded.

### Audio

- Final ASR matches the intended script once, without material omissions or additions.
- Sections have consistent loudness, commonly near `-17 LUFS`.
- Assembled program loudness is commonly `-16 ± 0.5 LUFS` unless the destination specifies another target.
- No clipped words, doubled tracks, echo, clicks, unexpected silence, or truncated tail.
- Measure the final stereo file; section measurements alone are insufficient.

### Presenter

- Expected resolution, frame rate, duration, and decodable streams.
- No black frames, long exact-frame freezes, progressive darkening, or duration extension.
- Identity, face geometry, hair, glasses, clothing, accessories, background, crop, exposure, and white balance remain coherent.
- Mouth follows names, numbers, English tokens, plosives, and phrase endings.
- Blinks are sparse and bilateral; gestures occur once and settle; hands remain plausible and away from the face.
- Tail ends with a settled face and resting mouth.

### Composition and delivery

- No flash, duplicate presenter, source-time reset, missing overlay, overflow, face obstruction, caption collision, or unreadable UI.
- Opening, callouts, captions, watermark, progress, and platform safe zones remain compatible.
- Master and share files have the expected duration, dimensions, frame rate, codec, pixel format, and audio rate.
- Both files fully decode.
- Contact sheet covers opening, chapters, emphasis graphics, close, and final frame.
- Watch the complete video at normal speed before delivery.

## Recovery rules

### Interrupted remote task

Poll the saved task ID and download a succeeded result. Resubmit only after a confirmed failure or cancellation.

### Voice identity or loudness changes

Confirm one synthesis identity and configuration. Regenerate the mismatching section, normalize again, and rerun ASR. If the final render is louder than the source, confirm video is muted and apply whole-program two-pass loudness correction.

### Presenter changes between chapters

Use one continuous audio-driven source with source-time slices. When splitting is unavoidable, lock identity, framing, parameters, light, and quiet edge holds.

### Body motion works but mouth is late

Keep the accepted motion plate. Apply lip-sync repair with the exact locked audio and no duration extension. Check speech anchors at numbers, plosives, English tokens, and final syllables.

### Face, glasses, hands, or light drift

Retry from the original image with reduced motion and explicit camera, exposure, and white-balance locks. Simplify the gesture and keep hands low. Structural face or finger faults require regeneration; grading cannot correct them.

### Frozen picture-in-picture

Confirm the layout uses moving video, source offsets advance continuously, and no poster frame or identical-frame padding replaced the presenter.

### Captions or keyword graphics feel wrong

Regenerate timings from the final audio. Split by meaning, shorten visible phrases, and adjust small lead/hold margins. Bind keyword motion to actual spoken anchors and verify its complete path against the face, hands, UI, and captions.

### Three rejected paid candidates

Stop. Preserve candidates, prompts, task IDs, cost, and rejection notes. Summarize the recurring failure, remaining options, expected additional cost, and the single variable proposed for the next attempt.
