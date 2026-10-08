# Hermes Codex Graphite Design System

## Product context

Hermes is a local-first AI agent dashboard with a persistent embedded TUI chat, session history, files, models, logs, automation, integrations, configuration, and system controls. The redesign must preserve the native information architecture, route hierarchy, compact operator workflow, accessibility, and responsive navigation. It changes presentation only.

## Visual direction

Codex-inspired dark productivity UI: calm, matte graphite surfaces; precise spacing; crisp typography; restrained blue interaction color; subtle borders instead of decorative effects. Borrow only the useful layered-dark hierarchy from the selected Neural Noir reference. Do not use its gold palette, serif typography, marketing composition, glowing nodes, glass blur, gradients, or ornamental animation.

## Color tokens

- Canvas: `#0d0f12`
- Sidebar: `#090b0e`
- Elevated surface: `#13161b`
- Card: `#161a20`
- Hover surface: `#1b2028`
- Input surface: `#111419`
- Primary text: `#f2f4f7`
- Secondary text: `#a3acb9`
- Tertiary text: `#737e8e`
- Disabled text: `#535d6b`
- Border: `#282e38`
- Subtle border: `rgba(255,255,255,0.07)`
- Primary blue: `#4c8dff`
- Primary blue hover: `#6aa0ff`
- Primary blue wash: `rgba(76,141,255,0.12)`
- Focus ring: `rgba(76,141,255,0.55)`
- Success: `#32d583`
- Warning: `#fdb022`
- Destructive: `#f97066`
- Terminal background: `#080a0d`
- Terminal foreground: `#e6eaf0`
- Input-series accent: `#8aaeff`
- Output-series accent: `#32d583`

## Typography

- UI sans: `Inter`, followed by the native system UI stack.
- Technical and terminal: `JetBrains Mono`, followed by the native monospace stack.
- Base size: 14px; line height: 1.5; letter spacing: -0.005em.
- Headings use 600 weight, sentence case, and tight spacing.
- Navigation and controls use 500 weight. Avoid wide uppercase labels except where Hermes already requires compact status metadata.

## Geometry and spacing

- Base radius: 10px; controls 7-9px; large dialogs/cards 12px.
- Comfortable density with compact operator controls.
- Sidebar remains 256px expanded and 56px collapsed.
- Use 1px borders and small tonal changes to establish hierarchy.
- Shadows are rare and restrained: `0 10px 30px rgba(0,0,0,0.28)` only for floating menus/dialogs.
- No glassmorphism, bevels, neon, grain, or large glow effects.

## Components

- Sidebar: distinct darkest surface, thin right border, blue-tinted active row, 3px active indicator, muted icons, stronger text on hover.
- Cards: matte elevated surface, subtle border, no heavy shadow, 10-12px radius.
- Buttons: primary blue solid; secondary graphite; ghost controls use tonal hover. Preserve destructive/status semantics.
- Inputs/selects: dark recessed surface, visible border, blue focus ring, high-contrast placeholder.
- Tabs/segmented controls: compact graphite track with the selected segment one tonal step brighter.
- Dialogs/popovers: elevated graphite, crisp border, soft shadow, clear hierarchy.
- Tables/lists: low-contrast row separators, blue-tinted selection, visible keyboard focus.
- Status indicators: semantic colors only; no ambient pulsing except an existing live state.
- Embedded TUI: use the terminal colors above; retain JetBrains Mono and all terminal behavior.

## Motion

- Transitions: 140-180ms, `cubic-bezier(0.2, 0, 0, 1)`.
- Limit motion to hover, focus, menu/dialog entrance, and sidebar width changes.
- Respect `prefers-reduced-motion` and disable non-essential animation.

## Responsive behavior

- Preserve the existing 1024px sidebar-to-drawer transition and 768px document overflow behavior.
- Keep touch targets at least 40px on mobile.
- Avoid horizontal overflow on Sessions, Chat, Models, and Config.

## Hard constraints

- Preserve all Hermes routes, controls, icons, information, and plugin slots.
- Keep the native embedded TUI as the primary chat surface.
- Use only the colors, fonts, spacing, and component styles defined here.
- Do not introduce gold, purple, teal, decorative serif fonts, gradients, translucent glass panels, marketing sections, or invented product features.
