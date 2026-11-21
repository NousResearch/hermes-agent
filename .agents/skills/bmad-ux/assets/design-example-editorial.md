---
name: Linen & Logic
colors:
  surface: '#fbf9f4'
  surface-dim: '#dbdad5'
  surface-bright: '#fbf9f4'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f5f3ee'
  surface-container: '#f0eee9'
  surface-container-high: '#eae8e3'
  surface-container-highest: '#e4e2dd'
  on-surface: '#1b1c19'
  on-surface-variant: '#4e453d'
  inverse-surface: '#30312e'
  inverse-on-surface: '#f2f1ec'
  outline: '#80756b'
  outline-variant: '#d1c4b9'
  surface-tint: '#715a3f'
  primary: '#59452b'
  on-primary: '#ffffff'
  primary-container: '#735c41'
  on-primary-container: '#f5d6b4'
  inverse-primary: '#e0c1a1'
  secondary: '#a43b2c'
  on-secondary: '#ffffff'
  secondary-container: '#fd7d69'
  on-secondary-container: '#71160b'
  tertiary: '#374a5f'
  on-tertiary: '#ffffff'
  tertiary-container: '#4f6278'
  on-tertiary-container: '#caddf8'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#fdddbb'
  primary-fixed-dim: '#e0c1a1'
  on-primary-fixed: '#281804'
  on-primary-fixed-variant: '#58432a'
  secondary-fixed: '#ffdad4'
  secondary-fixed-dim: '#ffb4a7'
  on-secondary-fixed: '#400200'
  on-secondary-fixed-variant: '#842417'
  tertiary-fixed: '#d0e4ff'
  tertiary-fixed-dim: '#b4c8e2'
  on-tertiary-fixed: '#071d30'
  on-tertiary-fixed-variant: '#35485d'
  background: '#fbf9f4'
  on-background: '#1b1c19'
  surface-variant: '#e4e2dd'
typography:
  display-lg:
    fontFamily: Libre Caslon Text
    fontSize: 48px
    fontWeight: '400'
    lineHeight: '1.1'
    letterSpacing: -0.02em
  display-lg-mobile:
    fontFamily: Libre Caslon Text
    fontSize: 36px
    fontWeight: '400'
    lineHeight: '1.1'
  headline-md:
    fontFamily: Libre Caslon Text
    fontSize: 32px
    fontWeight: '400'
    lineHeight: '1.2'
  headline-sm:
    fontFamily: Libre Caslon Text
    fontSize: 24px
    fontWeight: '400'
    lineHeight: '1.3'
  body-lg:
    fontFamily: DM Sans
    fontSize: 18px
    fontWeight: '400'
    lineHeight: '1.6'
    letterSpacing: 0.01em
  body-md:
    fontFamily: DM Sans
    fontSize: 16px
    fontWeight: '400'
    lineHeight: '1.6'
  label-caps:
    fontFamily: DM Sans
    fontSize: 12px
    fontWeight: '500'
    lineHeight: '1.4'
    letterSpacing: 0.1em
  caption:
    fontFamily: DM Sans
    fontSize: 13px
    fontWeight: '400'
    lineHeight: '1.4'
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  unit: 8px
  gutter: 24px
  margin-mobile: 20px
  margin-desktop: 64px
  editorial-gap: 80px
---

## Brand & Style

The design system is rooted in the philosophy of "Slow Design"—an intentional departure from the frantic pace of fast fashion. It evokes a tactile, "linen-weight" sensation through high-end editorial layouts and a restrained aesthetic. The target audience values provenance over presence, seeking a reflective and sophisticated discovery experience that feels as much like a boutique magazine as a digital marketplace.

The style is **Editorial Minimalism** with **Tactile** accents. It prioritizes breathable white space, asymmetrical layouts that mimic printed lookbooks, and a soft, sun-faded palette. Every interaction is designed to be deliberate and "anti-hype," eschewing aggressive animations for subtle transitions and quiet confidence.

## Colors

The palette is inspired by natural fibers and weathered landscapes. 
- **Warm White (#F9F7F2)** serves as the primary canvas, providing a soft, non-clinical background that reduces eye strain.
- **Bone (#E3DED1)** and **Dust (#C2B9A7)** are used for structural depth, subtle dividers, and secondary surfaces.
- **Tobacco (#735C41)** is the primary ink color, used for high-contrast typography and essential UI elements.
- **Sun-faded Red (#B84A39)** and **Wool Blanket Blue (#4A5D73)** are used sparingly as "organic accents"—highlighting editorial picks or signifying subtle state changes without disrupting the tranquil atmosphere.

## Typography

Typography is the primary vehicle for the brand’s sophisticated voice. 
- **Libre Caslon Text** is the voice of the curator. Its classic proportions and elegant serifs provide the editorial weight required for discovery and storytelling.
- **DM Sans** provides a quiet, functional counterpoint. It is used for body copy and navigational elements, ensuring clarity without competing with the headlines. 

Large display titles should often use "optical sizing" logic—tighter leading and slightly negative letter spacing to create a cohesive visual block. Labels are always tracked out (0.1em) to maintain a sense of airy premiumness.

## Layout & Spacing

This design system employs a **Fluid Editorial Grid**. While it follows a 12-column structure on desktop, it encourages "asymmetrical breathing room"—intentionally leaving columns empty to direct focus toward high-quality imagery.

Spacing is generous. The `editorial-gap` (80px+) should be used between major content sections to allow the user to pause and reflect. Mobile layouts should maintain a minimum of 20px side margins to ensure the content feels framed like a page, rather than bleeding to the edges of the device. Elements should lean toward vertical stacks to mimic the scroll of a digital journal.

## Elevation & Depth

Depth is communicated through **Tonal Layering** and **Ambient Shadows** rather than sharp borders.
- **Surfaces:** Use the "Bone" color to define containers against the "Warm White" base. 
- **Shadows:** Shadows are highly diffused and tinted with the "Tobacco" hue (`rgba(115, 92, 65, 0.08)`). They should feel like a soft glow of light hitting fabric, with large blur radii (20px+) and very low opacity.
- **Borders:** When borders are necessary, they are 1px thick and rendered in "Dust," creating a "ghost" outline that barely separates elements from the background.

## Shapes

The shape language is **Soft (0.25rem)**. While a sharp edge feels too aggressive and a pill-shape feels too digital/tech-heavy, a subtle rounding of corners mimics the natural softening of woven textiles over time. 

Larger containers (Cards, Modals) may use `rounded-lg` (0.5rem) to emphasize their tactile, object-like quality. Imagery should always follow these corner radii to maintain a cohesive, "framed" appearance.

## Components

- **Buttons:** Primary buttons use a solid "Tobacco" fill with "Warm White" text. Secondary buttons are "Bone" with "Tobacco" text or simply "Tobacco" text with a 1px "Dust" border. Padding is generous horizontally to create an elegant, elongated silhouette.
- **Cards:** Editorial cards feature large imagery, a "headline-sm" title, and a "caption" subline. Shadows are only applied on hover to simulate a gentle lift.
- **Inputs:** Minimalist underlines in "Dust" that transition to "Tobacco" on focus. Label text remains in "label-caps" above the field.
- **Chips/Tags:** Used for material types (e.g., "100% Linen"). These are rendered in "Bone" backgrounds with "Tobacco" text, using the "Soft" corner radius.
- **Icons:** Must be "Hand-drawn" or "Fine-line" style. Lines should have slight imperfections and vary in weight to reinforce the tactile, artisanal nature of the fashion being discovered.
- **Navigation:** A simple, centered bottom bar or a top-weighted "Ghost" header that disappears on scroll to maximize the editorial viewport.