---
name: plenishd-design-tokens
description: Design tokens for Plenishd — UK voice-first smart kitchen assistant. Colors, typography, spacing, component guidelines.
version: 1.0.0
metadata:
  hermes:
    tags: [design-tokens, plenishd, brand]
adoption_status: provisional
---

# Plenishd Design Tokens

## Brand colors
- **Plenishd Yellow:** `#FBBF24` — primary accent, CTAs, highlights
- **Warm Dark:** `#2C2A28` — backgrounds, containers
- **Warm Light:** `#E0DCD6` — body text, icons
- **Card Surface:** `#3A3836` — cards, input fields
- **Error:** `#EF4444` — validation errors
- **Success:** `#22C55E` — confirmations, stock available
- **Warning:** `#F59E0B` — low stock alerts

## Typography
- **Family:** System font (SF Pro on iOS, Roboto on Android)
- **Headings:** Bold, 20-28px
- **Body:** Regular, 14-16px
- **Caption:** Regular, 11-13px
- **Voice hints:** Larger than standard body text (voice-first interface)
- **Line height:** 1.4 (body), 1.2 (headings)

## Spacing
- **XS:** 4px
- **SM:** 8px
- **MD:** 12px
- **LG:** 16px
- **XL:** 24px
- **2XL:** 32px
- **Section:** 40px

## Border radius
- **SM:** 6px (input fields, chips)
- **MD:** 10px (cards, modals)
- **LG:** 16px (full-screen sheets)

## Elevation
- **Card shadow:** 0px 2px 8px rgba(0,0,0,0.25)
- **Modal shadow:** 0px 8px 24px rgba(0,0,0,0.35)

## Iconography
- Line icons, 24px default
- Stroke width: 1.5px
- Color: Warm Light `#E0DCD6` or Plenishd Yellow `#FBBF24` for active states

## Voice-first considerations
- Touch targets: minimum 44x44px
- Voice hint UI: appears near relevant input fields
- Loading states: skeleton screens with warm dark background
- Empty states: illustration + Plenishd Yellow accent CTA
