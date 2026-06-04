# The Boys live-action two-main-characters default

## Trigger

Use when Nick asks for a very small batch such as `生成2张剧版黑袍纠察队的主要角色图` without naming specific characters.

## Default roster

Prefer the two most immediately recognizable live-action core leads:

1. Homelander
2. Billy Butcher

Reasoning: for a two-image set, this pairing gives the strongest instant recognizability and ideological contrast.

## Prompting guidance

- Anchor to live-action-recognizable face, hair, costume, and hero/antihero prop cues.
- Do not drift into comic-only redesigns when `剧版` was explicitly requested.
- Keep the neon sticker-bomb treatment as the presentation layer, not a replacement for actor/costume recognizability.
- If one item fails or returns no image, retry only that slot with a shorter prompt first.

## Delivery/QC note

- Do not attach `MEDIA:` for any failed or unverified slot.
- Do not duplicate another image path to fill a missing slot.
- If substitution is needed because the user requested only a count, report it explicitly; if the user named exact characters, do not substitute silently.
