# SYSTEM-SHEET — <product>

> Filled BEFORE any markup. The route map says what exists; the inventory says what it is built from
> and — critically — **how many variants of each thing are allowed**. That number is the budget
> `systemscan` enforces after the build. Declaring four button variants and shipping nine is the
> disease this file exists to prevent.

## Product
- **What it does:**
- **Who uses it, how often:** <!-- daily tools want less personality than monthly ones -->
- **Stack / where the markup lands:** <!-- static, React, Vue, Rails views… and who plugs it in -->
- **The moment of care:** <!-- the ONE place this product is more than correct. Not a wow moment — an empty state that genuinely helps, a table that does something smart, a keyboard flow that feels designed. There is no peak in this register. -->

## Route map

<!-- Every screen. A route nobody can describe in one line is a route nobody designed. -->

| route | job (one line) | layout family | in the shell? | traffic |
|---|---|---|---|---|
| `/` |  |  | yes | high |
| `/…` |  |  |  |  |

**The shell** <!-- what persists across routes: nav, sidebar, header, page frame. Its mistakes are on every screen, so it is built first, right after tokens. -->
- structure:
- collapses to (mobile):
- current-route indicator:

**Build order** <!-- by traffic, not by interest. The boring high-traffic screen sets the patterns. -->
1.
2.

## Component inventory

<!-- Every control the product needs, with its variant count DECLARED. Add a row only when a screen
     genuinely needs it — a component nobody has a route for is speculative work. -->

<!-- `systemscan` counts a variant as a PAINTED SIGNATURE: background | colour | border-colour |
     border-width | radius | font-size/weight | padding | shadow. A disabled button and a small
     button are separate variants whether or not you name them. `<a class="btn">` counts as a
     button, not a link. Pass these numbers per kind:
       --max-variants button=3,link=4,input=2,select=1 -->

| component | variants (name them all) | budget | where used |
|---|---|---|---|
| button | primary / ghost / danger | 3 | everywhere |
| input |  |  |  |
| select |  |  |  |
| table |  |  |  |
| … |  |  |  |

**Non-control components** — pills, badges, banners, skeletons, avatars, tags. `systemscan` does not
count these (no focus, no interaction), which makes them exactly where drift breeds unseen. Declare
them here anyway and police them by eye against `components.png`:

| component | variants | where used |
|---|---|---|
| status pill |  |  |

**Rule:** a variant not on this list does not get built. If a screen needs one, that is a design
decision — edit this file first, then build it.

## State matrix

<!-- One row per component. Mark n/a where the state genuinely cannot occur, and say why.
     focus-visible is never n/a on anything interactive: systemscan presses Tab and looks. -->

| component | default | hover | focus-visible | active | disabled | loading | empty | error | selected |
|---|---|---|---|---|---|---|---|---|---|
| button | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | n/a | n/a | n/a |
|  |  |  |  |  |  |  |  |  |  |

**Contrast survives every state.** A degraded, stale or disabled view usually dims something, and
dimming is how a designed state quietly becomes unreadable — text at `opacity: .5` on a coloured
surface can fall from 6.9:1 to under 3:1 while every linter stays silent, because they all read the
undimmed computed colour. State the consequence for each state that changes opacity or colour:

| state | what it dims | measured contrast after dimming |
|---|---|---|
| error / stale |  |  |
| disabled |  |  |

**How each state is reachable.** Five beautiful empty states nobody can open are five states nobody
reviewed. Name the mechanism — a `:target` fragment, a query param, a fixture flag — preferring one
that survives with JS off, and list the URLs:

- mechanism:
- URLs:

**Empty / loading / error are not edge cases.** They are the first thing a new user sees. For each
screen that can be empty, write the actual words:

| screen | empty state says | the one action that fills it |
|---|---|---|
|  |  |  |

## Density

- **Tables:** rows per screen · sticky header? · sort affordance · `tabular-nums` on numerals
- **Charts:** route to the `dataviz` skill — do not improvise a series palette here
- **What gets truncated, and how the full value is reachable:**

## Gate checklist
- [ ] every route has a one-line job and a layout family
- [ ] the shell is described, including its mobile collapse and its current-route indicator
- [ ] every component has a named, justified variant count — not "a few"
- [ ] every interactive component's `focus-visible` is designed, not defaulted
- [ ] empty / loading / error are written as real copy for every screen that can hit them
- [ ] build order is by traffic
- [ ] no peak. If a screen has a wow moment, justify it or cut it.
