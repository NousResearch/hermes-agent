# Manasota Key Fun Pack daily email notes

Use this reference when maintaining Ron's daily Manasota Key environmental/entertainment email generator.

## Branding and recipient expectations
- Brand the email as **Manasota Key Fun Pack** throughout: sender display name, subject prefix, header/title, and greeting.
- Avoid broad/old branding such as "Ara1bot Florida Fun Pack".
- Send samples only to `reisworth@gmail.com` unless Ron explicitly authorizes another recipient.

## Gmail display and clipping/collapsing
- Gmail clips HTML bodies around ~102 KB; verify generated HTML stays well below this threshold.
- Keep repeated inline CSS and wrapper markup lean.
- Add a generated `Message-ID` and `X-Entity-Ref-ID` so repeated test sends are less likely to thread/collapse together in Gmail.
- If same-day test samples still collapse in conversation view, add a small test-only timestamp/token to the subject; avoid doing this for the normal scheduled daily email.

## Forecast chart layout standards
- For 7-day weather/wind/swell charts, use high-resolution canvases suitable for email/PDF screenshots (the current target is ~2200×1360).
- Do not put legends in the plot area if they can collide with day/date labels or source/footer text.
- Reserve separate vertical bands for:
  1. title/subtitle/header,
  2. condition/callout chips,
  3. plotted data,
  4. secondary rows such as rain probability/amount,
  5. legend,
  6. source/footer.
- Specific weather pitfall: rain percentage/amount labels near the bottom must not overlap low-temperature labels. Put rain data in its own row below the temp plot.
- Specific weather pitfall: high-temperature labels must not crowd conditions callouts at the top. Put condition callouts in their own row above the plot or leave enough top padding.
- Use polished coastal styling: soft gradients, day panels/cards, readable typography, high-contrast series colors, and labeled axes/gridlines.

## Entertainment source and format
- Primary entertainment source is SWFLLive Englewood events, e.g. `https://swfllive.com/events?city=Englewood` with date parameters as needed.
- Parse the structured data from the page rather than searching daily.
- Filter to these venues unless Ron changes the list:
  - Beachcomber Trading Post
  - SandBar Tiki & Grille
  - White Elephant Pub
- Music horizon preference changed during the session: Ron requested 5 days after initially trying 3 days. Use 5 days for the Manasota Key Fun Pack unless he explicitly asks to compact it again.
- Organize entertainment by day, then venue. Keep it compact: time, performer/event name, optional genre.
- Brave Search is useful as a fallback/discovery tool if SWFLLive changes or venue URLs move, but do not use search as the daily source of truth when structured SWFLLive data is available.

## Secret handling
- Never hard-code Brave Search API keys or other credentials into the email script.
- If Brave Search is used, rely on Hermes/environment configuration and do not echo the key value in responses, logs, or references.

## Verification checklist
- Run Python compile/syntax check on the email script.
- Generate a dry-run `.eml`.
- Parse the `.eml` to verify sender/subject/header branding, HTML byte size, entertainment date range, and no secret leakage.
- Generate a contact sheet or preview for the 7-day weather/wind/swell images and visually check overlap/readability before sending a sample.
