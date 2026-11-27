# Visual Verification

Do not trust code alone for UI work. Verify the rendered result whenever possible.

## When Verification Is Required

- After implementing a new page/screen.
- After redesigning layout, type, color, or motion.
- Before final audit.
- When the user says it looks wrong, generic, cramped, cheap, or not native.
- When responsive/native behavior matters.

## Web Verification

Check at least:
- desktop viewport
- mobile viewport
- navigation/focus states if interactive
- loading/empty/error if implemented
- key breakpoints that could cause overflow

Before judging the visual result, confirm the app is serving fresh CSS/assets:
- If the page looks like raw browser-default HTML, inspect computed styles or CSS network responses before critiquing the design.
- In Next.js/dev-server workflows, running a production build while `next dev` is active can leave dev CSS paths stale; restart the dev server and reload before drawing design conclusions. See `references/nextjs-dev-cache-visual-verification.md` for the recovery recipe.
- Use browser console/network checks when visual output contradicts the intended classes.
- If the cleanup command would combine process killing with `rm -rf .next`, split it into approval-friendly steps or use a fresh port; do not retry an identical blocked destructive command.

Look for:
- visual hierarchy
- CTA visibility
- spacing rhythm
- text wrapping
- horizontal overflow
- cropped content
- contrast
- animation jank
- focus visibility

For responsive web pages with code blocks, tables, or long tokens inside CSS grid/flex layouts:
- Check `document.documentElement.scrollWidth` vs `clientWidth` at mobile widths; screenshots can miss off-screen overflow.
- Remember grid/flex children default to `min-width: auto`, so a `pre`, long token, table, or URL can make the whole page wider even when the element has `overflow-x-auto`.
- Add `min-w-0`/`min-width: 0` to the grid/flex child columns and keep long content inside an `overflow-x-auto` container.
- Verify the page itself has no horizontal overflow while the code/table container can scroll internally.

## Native Mobile Verification

If only specs are available, reason through the checks. If screenshots/simulator are available, inspect them.

Check:
- safe areas and status/navigation bars
- thumb reach for primary actions
- Dynamic Type/text scaling
- keyboard open state
- bottom sheets/modals
- back navigation
- permission flows
- empty/loading/error states
- dark/light mode

## Screenshot Review Heuristics

Ask:
- What is the first thing my eye sees?
- Is that the intended thing?
- Can I predict the next action?
- Does anything feel equally important when it should not?
- Does the layout still work with real content?
- Does it look like a known generic template?

## Verification Notes Format

```markdown
## Visual Verification
- Environment/viewport/screen:
- What was checked:
- Observations:
- Issues found:
- Follow-up fixes:
- Not checked:
```

If no rendered UI or screenshot is available, explicitly say verification was not performed and provide a checklist instead.
