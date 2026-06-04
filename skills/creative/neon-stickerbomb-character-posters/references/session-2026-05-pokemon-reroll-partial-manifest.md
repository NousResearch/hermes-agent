# Pokémon Reroll: Partial Manifest After Repeated Empty Responses

Session pattern: Nick asked to regenerate Mew / 梦幻 and Mewtwo / 超梦 in the neon sticker-bomb style after a prior rare-Pokémon batch.

## What happened

- Mew regenerated successfully with a fresh composition: chrome scanner / glitch sticker-bomb foreground lens.
- Mewtwo failed three consecutive generation attempts with `empty_response`:
  1. full low-angle fashion diagonal + psychic foreground depth prompt
  2. shorter side-profile split + psychic energy depth prompt
  3. compact anchor-first off-center low-angle prompt
- The correct handling was to save a fresh partial manifest containing:
  - successful new Mew path and composition
  - Mewtwo status as `failed_empty_response_after_3_attempts`
  - `local_path: null` for the failed slot
  - attempted composition notes

## Durable lesson

For small named reroll batches, never reuse an older sibling or previous-batch image path to fill a failed slot. Preserve successful fresh outputs, write a partial manifest immediately, and report the failure plainly. This prevents later commands like `发布12` from accidentally mapping to a stale image.

## Good manifest shape

```json
{
  "status": "partial",
  "theme": "Pokemon reroll: Mew and Mewtwo",
  "results": [
    {
      "index": 1,
      "title": "Mew / 梦幻",
      "status": "done",
      "local_path": "/abs/path/to/new_mew.png",
      "composition": "Chrome scanner + glitch sticker-bomb"
    },
    {
      "index": 2,
      "title": "Mewtwo / 超梦",
      "status": "failed_empty_response_after_3_attempts",
      "local_path": null,
      "composition_attempts": [
        "Low-angle fashion diagonal + psychic foreground depth",
        "Side profile split + psychic energy depth",
        "Compact off-center low-angle diagonal"
      ]
    }
  ]
}
```
