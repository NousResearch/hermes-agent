# Component Patterns

Components are the building blocks of the interface. Each one should have a clear job, consistent states, and a predictable relationship with the design tokens.

## Cards

- One idea per card. Do not overload with actions.
- Use `surface` color, `md` padding, and `sm` border-radius.
- On mobile, cards can be full-bleed. On desktop, keep them in a grid or list.
- Cards of icon + heading + text as the page structure are lazy. Break the pattern unless the brief earns it.

## Forms

- Label every input clearly. Place labels above the field.
- Helper text below the field, not inside the placeholder.
- Error messages explain how to fix the problem.
- Primary action is obvious; secondary actions are visually quieter.
- Preserve user input on error.

## Navigation

- Mobile: priority+ or hamburger with a visible primary CTA.
- Desktop: horizontal or sidebar, depending on the number of items.
- Active state is visible without relying on color alone.

## Tables and Lists

- Mobile: turn wide tables into cards or stacked rows.
- Desktop: keep tables with clear headers and zebra striping.
- Lists use consistent padding and clear separators.

## Buttons

- One primary action per screen or section.
- Disabled states must look disabled.
- Loading states preserve layout width to prevent shift.
- Touch target minimum 44x44px.

## States

Every interactive or data-bearing component must be designed for:

- **Default:** the resting state.
- **Hover:** pointer feedback, but never the only path on touch.
- **Focus:** visible, logical, and keyboard-friendly.
- **Active/pressed:** the moment of engagement.
- **Disabled:** clearly inactive, without removing the element.
- **Loading:** skeleton, spinner, or progress that matches the final shape.
- **Empty:** explain what is missing and what to do next.
- **Error:** say what happened and how to recover.
- **Success:** confirm the action without over-celebrating.

## Empty and Error States

- Empty: explain what is missing and what to do next.
- Error: say what happened and how to recover.
- Loading: show a skeleton or progress that matches the final shape.
- Permission denied: explain why and how to get access.
