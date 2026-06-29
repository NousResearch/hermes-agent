# Stitch Prompt Template

Use this template to prepare an effective Stitch prompt from a WDS specification.

---

## How to Use

1. **Copy this template** into your Stitch dialog
2. **Fill in each section** using your spec and design system
3. **Remove Object IDs, translations, technical details** - Stitch doesn't need them
4. **Keep one language only** - typically the primary language (English or Swedish)
5. **Paste the filled template** as your Stitch prompt

---

## Template Structure

```
=== PROJECT CONTEXT ===

App: {App name} - {One-line description}
Target: {Target audience}
Brand feel: {2-3 adjectives describing the feel}
Market: {Market focus if relevant}

=== DESIGN SYSTEM ===

Colors:
- Background: {color name} ({hex})
- Primary/CTA: {color name} ({hex})
- Text: {color name} ({hex})
- Secondary text: {color name} ({hex})
- Success: {hex}
- Error: {hex}

Typography:
- Font: {font family}
- Headlines: {weight}, {characteristics}
- Body: {weight}, {size}

Component styles:
- Buttons: {style description - rounded, gradient, shadow, etc.}
- Inputs: {style description - border, focus state, etc.}
- Cards: {style description if relevant}

=== SCREEN DETAILS ===

Screen: {Screen name}
Purpose: {What this screen does, one sentence}
User context: {Where user is coming from, what they need}

Layout structure:
1. {Section 1}: {elements}
2. {Section 2}: {elements}
3. {Section 3}: {elements}

Key elements:
- {Element 1}: "{Actual content/text}"
- {Element 2}: "{Actual content/text}"
- {Element 3}: "{Actual content/text}"

Key interactions:
- Primary action: {what happens}
- Secondary action: {what happens}

=== CURRENT STATE NOTES ===

{Note any elements currently using default/unstyled components}
- {Component}: Currently ShadCN default, should match brand style
- {Component}: Uses custom gradient button

=== GENERATION INSTRUCTIONS ===

Generate this screen matching:
- Visual style of the attached reference image
- Layout structure of the attached sketch
- All content and elements listed above

Viewport: {Mobile 390px / Desktop 1440px}
```

---

## Example: Dog Week Sign-In

```
=== PROJECT CONTEXT ===

App: Dog Week - Family dog walk coordination app
Target: Swedish families (all ages from teens to grandparents)
Brand feel: Warm, friendly, trustworthy
Market: Sweden

=== DESIGN SYSTEM ===

Colors:
- Background: Cream (#FEF3CF), gradient to #FFFBED
- Primary/CTA: Orange (#FD6408), gradient #FD8002 to #FF2714
- Text: Brown (#2F1A0C)
- Secondary text: Gray (#686868)
- Success: Green (#28C54A)
- Error: Red (#DB0000)

Typography:
- Font: Inter
- Headlines: Bold/Extra Bold, tight letter spacing
- Body: Regular weight, 16px base

Component styles:
- Buttons: Rounded (8px), orange gradient for primary, subtle shadow
- Inputs: Light background, rounded corners, brown text
- Cards: Cream background, subtle shadow

=== SCREEN DETAILS ===

Screen: Sign In
Purpose: Authenticate users with email magic link or Google SSO
User context: Coming from Start Page, ready to access the app

Layout structure:
1. Header: Logo (left), Back button (right)
2. Main form: Email input, magic link button, divider, Google SSO
3. Trust section: Privacy and security messages
4. Help links: Support links at bottom

Key elements:
- Email input label: "Email address"
- Email placeholder: "your@email.com"
- Helper text: "We'll send you a magic link to sign in"
- Primary button: "Send magic link"
- Divider text: "Or sign in with"
- Google button: "Continue with Google"
- Trust message 1: "Your information is secure and private"
- Trust message 2: "We'll never spam you or share your details"
- Trust message 3: "Safe for all family members to use"

Key interactions:
- Primary: Enter email → Send magic link → Check email
- Secondary: Click Google → OAuth flow → Signed in

=== CURRENT STATE NOTES ===

- Input fields: Currently ShadCN default styling, should use cream background
- Google button: Should match brand's rounded style with Google colors
- Trust icons: Need checkmark or shield icons in success green

=== GENERATION INSTRUCTIONS ===

Generate this sign-in screen matching:
- Visual style of the attached Start Page screenshot (warm, cream, orange CTAs)
- Layout structure of the attached sketch
- All content and elements listed above

Viewport: Mobile 390px
```

---

## Checklist Before Pasting to Stitch

- [ ] Project context filled (app name, target, brand feel)
- [ ] Design system colors accurate (from Color-Palette.md)
- [ ] Typography correct (from Typography-System.md)
- [ ] Component styles described (buttons, inputs)
- [ ] Screen content in ONE language only (no translations)
- [ ] No Object IDs included
- [ ] No technical implementation details
- [ ] Current state notes added (what's ShadCN default)
- [ ] Viewport specified

---

_Stitch Prompt Template — Freya WDS Designer_
