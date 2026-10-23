# Generation

Read this file for intake, content, narration, capability selection, paid generation, presenter prompts, or provider changes.

## Inputs and authorization

Required:

- a topic or script;
- one local presenter image with one clear adult face;
- confirmed image rights before remote upload;
- confirmed adult status from the user.

Optional:

- authorized voice sample;
- screen recordings, B-roll, images, charts, brand assets, or source links;
- audience, duration, language, aspect, style, watermark, music, CTA, provider preference, or previous episode.

Inspect the real image. Record visible anchors only: framing, hair, glasses, clothing, accessories, hands, furniture, background geometry, camera height, crop, light, exposure, white balance, embedded text, logos, occlusion, multiple faces, and resolution. Do not infer sensitive traits.

When a voice sample exists, listen for one speaker, audible speech duration, language, noise, echo, clipping, music, silence, and codec damage. Record whether cloning is authorized. Keep the original unchanged.

## Content and narration

For a topic, use one clear spine:

```text
hook → promise → 2–4 useful beats → synthesis → close
```

For a supplied script, preserve factual meaning while improving breath, pronunciation, sentence length, and transitions. Flag material factual edits for approval.

Generate the complete approved narration before visual work. Keep one voice identity and one synthesis configuration. Split only for operational limits or clean semantic sections. Preserve raw and normalized copies.

Run ASR against final audio. Review names, numbers, English tokens, omitted words, repeated words, and tail speech. Use actual audio durations for every later cut.

## Capability routing

Resolve capabilities at execution time:

| Capability | Acceptance priority |
|---|---|
| Voice generation | authorization, identity, pronunciation, rate control, clean audio |
| Main presenter | exact audio support, semantic lip sync, identity stability, required duration |
| Short motion | stable face, one controllable gesture, fixed camera, economical retry |
| Lip-sync repair | preserves accepted motion while replacing mouth timing with exact audio |
| Word-timestamp ASR | accurate word start/end times and reviewable output |
| Timeline compositor | explicit source offsets, frame-accurate seek, deterministic render |
| Encoder and QA | local probe, loudness, decode, black-frame and contact-sheet support |

Inspect current installed tools and official documentation when availability, duration, pricing, or request fields may have changed. Prefer a user-selected provider when it passes the gates. Otherwise choose the simplest available capability that does.

Record the actual provider, model, version, region, parameters, price evidence date, task ID, and sanitized request body in `job.json`. Ask before a provider change that affects cost, privacy, voice, appearance, or quality.

## Billing and pilot gate

Before the first remote or paid call, state:

- files or data being uploaded;
- capability and selected tool;
- requested seconds or units;
- known price or that the cost is unknown;
- pilot duration and retry ceiling;
- expected output and main risks.

Generate the smallest useful pilot. Continue to a full run only after technical checks and visual review pass. Do not generate a long clip when the edit uses only a short range.

## Presenter prompt structure

Write prompts in this order:

1. Bind the user image as the sole presenter and scene reference.
2. List only visible identity, clothing, accessory, camera, framing, background, light, exposure, and white-balance anchors.
3. Request realistic skin, hair, eye moisture, fabric, breathing, bilateral blinking, and restrained head motion.
4. Supply exact dialogue through the tool's supported audio or dialogue field.
5. Tie one physically plausible gesture to one phrase and approximate time.
6. Reserve a settled, closed-mouth tail.
7. Exclude identity, face, glasses, hair, wardrobe, skin tone, background, camera, lighting, finger, hand, dialogue, text, logo, watermark, and extra-person drift.

For the main presenter, prioritize mouth timing and identity. Keep hands low or outside frame. For an opening or close, one small gesture may carry the hook or CTA. Keep hands below the collarbone, away from the face and lens, and end the movement before the tail hold.

If the body performance is accepted but mouth timing is visibly late, preserve the motion plate and apply lip-sync repair with the exact locked audio and no duration extension.
