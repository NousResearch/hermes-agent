# DESIGN — <project>

> The style contract. Written once after the build passes QA; read at the START of every subsequent edit. New work that contradicts this file is wrong even if it looks good in isolation — consistency IS the design. Update the file deliberately when the system itself evolves; never drift it silently.

## Identity
- **The one feeling:** <!-- from the storyboard -->
- **Signature / peak:** <!-- what a visitor describes to a friend; do not dilute it with competing spectacles -->
- **Register:** build | direct

## Tokens (verbatim from the shipped CSS)
```css
:root {
  /* colors (OKLCH) — bg, surface, ink, muted, accent, + roles */

  /* type scale (clamp-based) */

  /* spacing scale */

  /* radii, shadows, z-scale */

  /* easing custom properties */
}
```

## Typography
- **Display:** <family, weights, where used, letter-spacing rules>
- **Text:** <family, sizes, line-heights, measure>
- **Numerals/mono:** <if any>
- Pairing axis and the reason (from COMMIT-SHEET). New text styles must come from this system — no new fonts.

## Color rules
- Commitment tier: <restrained/committed/full/drenched> — accent carries ~N% of surface.
- What each role means semantically (e.g. "green = money-positive, never decorative").
- Forbidden in this project: <e.g. gradients entirely; any warm neutral; ...>

## Motion vocabulary
- The ≤3 scroll families used, with their exact easing/duration/stagger values.
- Hover/press/focus recipes (copy the shipped values).
- New sections must reuse an existing family or consciously replace one (budget stays ≤3) — never add a fourth.

## Layout patterns
- Grid system + the named grid-break(s) in play.
- Section-opening patterns used (list them; rotate among these, don't invent an eyebrow).
- Component patterns that exist (cards? tables? ledger rows?) — reuse before inventing.

## Copy voice
- 2–3 adjectives + one example headline that nails it.
- Button/label conventions. Banned words stay banned.

## Project ban additions
- auteur-allow suppressions in force (rule + reason).
- Anything this project additionally forbids beyond SKILL.md's list.

## Rejected — decisions that already have a history

Everything below was tried and taken out, for a reason the finished page no longer shows. Without this list the next session — or the next model — sees an apparent mistake, confidently "corrects" it to the obvious answer, and reintroduces a problem that was already paid for once. The obvious answer is what got rejected; that is the whole point of the row. Write the row the moment you reject something, not at the end when the reason has evaporated.

| What was tried | Why it lost | What is there instead |
|---|---|---|
| <e.g. subject lighter than its field> | <silhouette dissolved; no lighting fixed it> | <subject below the field in value, edges carry the light> |
| <e.g. volumetric cones for the light shafts> | <read as cheap geometry; three rescue attempts, all worse> | <cut; the glow comes from the interior source alone> |

The same discipline inside the code: a bare constant teaches nothing, so every tuned number says why it is that number and what breaks otherwise — `--scrub-smooth: 0.45; /* >0.6 and the scene visibly lags the cursor */`. Numbers that carry their reason survive being cleaned up by someone who does not know the history.

## Editing protocol
1. Read this file fully before touching anything.
2. New section → pick an existing section-opening pattern + an existing motion family + existing tokens.
3. After any edit: `node <skill>/scripts/slopscan.mjs <src>` and re-shoot the changed viewport(s); compare against neighboring sections for family consistency.
4. If the edit genuinely needs a new pattern — update THIS file first (that's a design decision, not a patch).
5. An edit that "fixes" something in the Rejected table is a regression, however reasonable it looks. Disagree in the file first — argue the row, change it, then edit the page.
