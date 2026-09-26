---
version: alpha
name: MyBrand
description: One-sentence description of the visual identity.
colors:
  primary: "#0F172A"
  secondary: "#64748B"
  tertiary: "#2563EB"
  neutral: "#F8FAFC"
  on-primary: "#FFFFFF"
  on-tertiary: "#FFFFFF"
typography:
  h1:
    fontFamily: Inter
    fontSize: 3rem
    fontWeight: 700
    lineHeight: 1.1
    letterSpacing: "-0.02em"
  h2:
    fontFamily: Inter
    fontSize: 2rem
    fontWeight: 600
    lineHeight: 1.2
  body-md:
    fontFamily: Inter
    fontSize: 1rem
    lineHeight: 1.5
  label-caps:
    fontFamily: Inter
    fontSize: 0.75rem
    fontWeight: 600
    letterSpacing: "0.08em"
rounded:
  sm: 4px
  md: 8px
  lg: 16px
  full: 9999px
spacing:
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 48px
components:
  button-primary:
    backgroundColor: "{colors.tertiary}"
    textColor: "{colors.on-tertiary}"
    rounded: "{rounded.sm}"
    padding: 12px
  button-primary-hover:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.on-primary}"
  card:
    backgroundColor: "{colors.neutral}"
    textColor: "{colors.primary}"
    rounded: "{rounded.md}"
    padding: 24px
---

## Overview

Describe the voice and feel of the brand in one or two paragraphs. What mood
does it evoke? What emotional response should a user have on first impression?

## Colors

- **Primary ({colors.primary}):** Core text, headlines, high-emphasis surfaces.
- **Secondary ({colors.secondary}):** Supporting text, borders, metadata.
- **Tertiary ({colors.tertiary}):** Interaction driver — buttons, links,
  selected states. Use sparingly to preserve its signal.
- **Neutral ({colors.neutral}):** Page background and surface fills.

## Typography

Inter for everything. Weight and size carry hierarchy, not font family. Tight
letter-spacing on display sizes; default tracking on body.

## Layout

Spacing scale is a 4px baseline. Use `md` (16px) for intra-component gaps,
`lg` (24px) for inter-component gaps, `xl` (48px) for section breaks.

## Shapes

Rounded corners are modest — `sm` on interactive elements, `md` on cards.
`full` is reserved for avatars and pill badges.

## Components

- `button-primary` is the only high-emphasis action per screen.
- `card` is the default surface for grouped content. No shadow by default.

## Do's and Don'ts

- **Do** use token references (`{colors.primary}`) instead of literal hex in
  component definitions.
- **Don't** introduce colors outside the palette — extend the palette first.
- **Don't** nest component variants. `button-primary-hover` is a sibling,
  not a child.
