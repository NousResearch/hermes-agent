# Florida Email Photo-Led Map Cards

Session lesson from improving Ron's Florida daily email packet (`~/.hermes/scripts/florida-email-task.py`).

## Pattern

For recurring coastal/environmental email packets, make the message feel like a polished briefing instead of a stack of raw maps:

- Use a wide hero photo at the top with a short dated briefing label.
- Wrap each environmental product in a card: photo header, concise human summary, then the authoritative map/chart.
- Use distinct photo themes per product so repeated sections do not feel generic:
  - Sargassum: beach/shore condition photo.
  - Tropical outlook: storm clouds/ocean sky photo.
  - Radar: lightning/rain/storm photo.
  - Rainfall totals: rain/water photo.
  - Tide graph: shoreline/waves/tide photo.
  - Red tide/current status: Gulf water/coastal water photo.
- Keep the authoritative map/chart prominent; photos are visual lead-ins, not replacements for source products.
- Label FWC red tide as current status, not trajectory/forecast, unless the actual forecast product is used.

## Implementation notes

- Cache downloaded public photo assets under the email task cache directory so repeated cron runs do not depend on re-fetching unchanged images unnecessarily.
- Attach photos as inline CID images just like maps, and include all photo CIDs in the HTML alternative's `add_related` list.
- If using dynamic photo providers, choose specific query terms per section; avoid reusing one generic photo for unrelated products.
- In the footer, describe photos generically as cached public photos or visual section headers unless a specific provider/license is guaranteed.

## Verification recipe

After a dry run, parse the generated `.eml` and check:

- Subject updated to match the improved theme.
- HTML contains the expected number of map cards.
- All CID images are present: hero photo, each section photo, and each authoritative map/chart.
- Image payload sizes are non-trivial.
- Browser-preview the extracted HTML with CIDs converted to local file URIs and visually inspect for broken images, huge whitespace, unreadable text, or repeated/wrong photos.

Example expectations from this session: 8 map cards and 15 inline images (7 photos + 8 maps/charts).