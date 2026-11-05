# Responsive Breakpoints and Patterns

Breakpoints are named by the layout change they cause, not by a device. Design the content first, then add breakpoints wherever the layout starts to fail.

## Default Breakpoint Set

| Name | Range | Typical layout change |
|------|-------|-----------------------|
| small | 0–639px | Single column, stacked navigation, full-bleed images, 16px margins |
| medium | 640–1023px | 2 columns, wider margins (24px), larger type, expanded nav |
| large | 1024–1439px | 12-column grid, sidebar, max-width containers |
| xlarge | 1440px+ | Generous whitespace, larger imagery, enhanced motion |

## Content-Driven Breakpoints

Add or remove a breakpoint when:

- Line length exceeds 75 characters.
- A component becomes too wide or too narrow to remain useful.
- A two-column layout becomes unreadable.
- The navigation no longer fits or becomes ambiguous.

## Responsive Patterns

- **Column drop:** 4 columns → 2 columns → 1 column as the screen narrows.
- **Reflow:** horizontal groups become vertical stacks.
- **Off-canvas:** secondary content moves behind a toggle.
- **Priority+:** show the most important, overflow the rest behind "more."
- **Container shift:** full-bleed on small, centered max-width on large.

## Avoid

- Device-specific breakpoints (e.g., `iphone-pro-max`).
- Breakpoints that only change a single property.
- More than 5 active breakpoints in one project.
