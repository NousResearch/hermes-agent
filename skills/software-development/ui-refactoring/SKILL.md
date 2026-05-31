---
name: ui-refactoring
description: Deep UI code refactoring workflow for enforcing strict component ownership of visual styling. Use when invited to perform systematic UI refactoring or when pages/screens/routes need cleanup so visual styles live inside reusable components while external code controls only layout composition.
---

# UI Refactoring

## Core Principle

Keep every component's visual styling inside the component itself. External callers must not pass cosmetic overrides such as `style`, `className`, CSS objects, utility classes, or framework-specific equivalents to change how a component looks.

External code may control only:

- semantic component modes, such as `variant`, `size`, `tone`, `state`, `disabled`, `colorScheme`, `fullWidth`, and similar domain-level props;
- layout composition through wrapper/layout components that own direction, spacing, alignment, positioning, and grid/flex behavior.

Use the decision rule: if a style describes how a component looks, move it inside the component. If it only describes how the component is placed among other elements, keep it outside in a layout wrapper.

## Bad Practices

Treat these as refactoring targets:

- Passing `style`, `className`, inline CSS, utility classes, or equivalent props to change component appearance.
- Splitting a component's visual definition between the component and its callers.
- Letting wrappers override padding, color, border, background, dimensions, radius, shadow, typography, or other cosmetic properties.
- Using a component as an unfinished base whose final visual design is assembled at the call site.
- Hardcoding the same visual rule in one place while exposing it as a prop in another.
- Adding props that mirror CSS directly, such as `padding`, `background`, `borderRadius`, `shadow`, or `borderColor`, when those props affect visual identity rather than layout composition.

## Target Architecture

Build components with clear, semantic APIs. Express visual variants through explicit product-level props such as:

- `variant`
- `size`
- `tone`
- `state`
- `colorScheme`
- `fullWidth`
- `disabled`

Use layout wrappers for placement concerns only. A wrapper may own properties such as:

- direction: row or column;
- spacing: gap, padding, margin when it describes external layout rhythm;
- flex/grid behavior;
- alignment and justification;
- positioning and responsive placement.

Do not use wrapper components to change the visual skin of their children.

## Workflow

Work sequentially through files. For each file, finish the refactor and validation for that slice before moving to the next one.

1. Inspect the current UI system.
   - Read nearby components, existing design primitives, layout wrappers, style utilities, and route/page conventions.
   - Check whether a suitable component already exists in `components/` before creating a new one.
   - Prefer extending an existing component with a semantic prop over creating a duplicate.

2. Open the next target file.
   - Identify style props, class overrides, inline styles, utility-class assembly, local page components, and repeated cosmetic fragments.
   - Separate each style into external layout versus internal visual styling.

3. Move visual styling into components.
   - Extract local page/screen/route components into a flat `components/` folder, unless the repository already has a clear equivalent component convention.
   - Keep pages, screens, and routes as composition layers assembled from `components/`.
   - Implement each extracted component as a self-contained visual unit with its own surface, padding, radius, typography, color, borders, shadows, sizing, and state styling.

4. Keep layout outside.
   - Use or create wrapper/layout components for direction, gap, padding, margin, flex, grid, alignment, and positioning.
   - Keep wrapper APIs layout-oriented, not cosmetic.

5. Simplify component APIs.
   - Remove harmful `style`, `className`, override, and cosmetic pass-through props.
   - Remove unused props after the refactor.
   - Merge vague props into clearer semantic props when that reduces ambiguity.
   - Avoid CSS-mirroring props for visual design. Prefer named variants and semantic modes.

6. Preserve behavior.
   - Do not change product behavior, data flow, permissions, routing, persistence, or business logic unless required to preserve the UI during refactoring.
   - Keep interactions, accessibility behavior, loading states, empty states, error states, and responsive behavior intact.

7. Validate before continuing.
   - Run the smallest meaningful formatter, typecheck, lint, unit test, component test, build, or browser check for the touched surface.
   - If validation fails, fix it before moving to the next file.

## Decision Rules

- In disputed cases, choose the stricter component boundary.
- Do not leave temporary override mechanisms behind.
- Do not keep a local page component just because it is used once.
- Do not create a new component when an existing component can be reused or cleanly extended.
- Do not make the smallest textual diff if it leaves unclear ownership of visual styling.
- Do not weaken the existing design system. Align with established primitives and naming when they exist.

## Per-File Report

After each file or coherent file group, report:

- what was wrong before;
- what visual styling moved inside components;
- what stayed outside as layout;
- which props were added, simplified, or removed;
- which local components were moved into `components/`;
- why the result is cleaner;
- what validation was run.

## Expected Result

Pages, screens, and routes should become clean composition layers. Components should become predictable, reusable, visually self-contained units. The codebase should no longer rely on cosmetic style pass-throughs, scattered hardcoded styling, or call-site overrides to assemble final UI.
