# Coach brand packet branding assets

This reference captures the durable pattern for coach-specific packet branding when the packet engine already exists but the coach/club brand layer is separate.

## Session lesson
A premium lesson packet workflow may already be proven with a generic/shared brand template, while the coach-specific branded variant is still missing or unverified.

Do not collapse those into one claim.

Report them separately:
- packet workflow capability may be GREEN
- coach-specific branding variant may still be YELLOW until the brand assets and renderer wiring are verified

## Staging pattern
When a user provides coach/club logos directly in chat:
1. copy the provided assets into the target coach workspace lane under a clear local path such as:
   - `inputs/brand-assets/<brand-slug>/...`
2. rename them to stable descriptive filenames
3. record sha256 hashes
4. add a short local README with provenance, descriptions, and intended use
5. state clearly whether the assets are only staged locally or actually wired into the packet renderer/template

## Truthfulness rule
Staged brand assets mean:
- the target coach lane has local access to the files
- the files can be referenced by future branding/template work

Staged brand assets do NOT by themselves mean:
- the renderer automatically uses them
- the packet workflow is already branded correctly
- the end user has received a branded output

## Bryan/Hermitage example
Example local staging path used for Bryan/Hermitage CC packet branding work:
- `projects/golf/coach-agents/bryan/inputs/brand-assets/hermitage-cc/`

Example contents:
- `hermitage-cc-logo-wreath.webp`
- `hermitage-cc-logo-circle.png`
- `README.md`

Use this pattern whenever a coach-specific brand layer needs to be introduced without overstating template integration.
