# Mewtwo Tail Structure Correction Failure Pattern — 2026-05-31

## Context

Nick generated Entei and Mewtwo in the neon sticker-bomb Pokémon workflow, then asked to regenerate Mewtwo. Multiple fresh-generation attempts for Mewtwo returned `empty_response`. Nick then corrected the successful existing Mewtwo image: the tail structure was wrong.

## What happened

- A previous Mewtwo image succeeded and was delivered.
- Fresh reroll attempts using both direct `Mewtwo` naming and generic `pale-gray psychic laboratory creature` wording repeatedly returned `empty_response`.
- Nick asked to modify the existing image because the tail was structurally wrong.
- `image_edit` with a local file path failed with `404 page not found` from the backend.
- `image_generate` with the previous image as reference and a tail-only correction prompt also returned `empty_response`.

## Durable lesson

For Mewtwo-like creature generations, tail correctness must be treated as a first-class anatomy requirement from the initial prompt, not left for later editing. The edit path may be unavailable or unreliable, and rerolling after a successful but anatomically flawed output can repeatedly fail.

## Prompt pattern to use up front

Put this near the top, before style language:

```text
TAIL STRUCTURE PRIORITY: exactly one long thick smooth purple tail, clearly attached to the lower back/base of spine, continuous from attachment point to tip, with a clean natural S-curve behind/around the body. The tail must not split, fork, duplicate, detach, become an arm/ribbon/tentacle, or connect to the shoulder, neck, belly, front torso, or hands. The purple belly/torso patch must remain visually separate from the tail.
```

Add to the anatomy lock:

```text
Mewtwo has one head, one torso, two arms, two legs, exactly one long thick purple tail attached at the lower back, two matching hands with exactly three rounded fingers each, and two matching feet with exactly three rounded toes each. Keep the tail attachment point visible or logically unambiguous; use a simpler side or three-quarter pose if needed.
```

Add negatives:

```text
no second tail, no forked tail, no detached tail, no tail from shoulder, no tail from neck, no tail from belly/front, no tail replacing an arm, no ribbon tails, no tentacle bundle, no purple appendage fragments, no torso patch merging into tail
```

## Composition guidance

Avoid tail-orbit compositions for Mewtwo when anatomy fidelity matters. Tail rings and circular psychic-orbit layouts can encourage extra/forked/detached tail fragments. Prefer:

- side-profile split with one tail sweeping cleanly behind the body,
- upper-body/three-quarter diagonal hover where the lower-back attachment remains readable,
- controlled S-curve tail along the lower frame, not wrapping all around the torso.

If Nick flags the tail after generation, do not claim success from a plausible-looking style result. Report whether edit/reference reroll actually produced a new file. Never reuse the old path as a corrected image.
