# Open Questions — Auto-Population Guide

**Purpose:** During page specification creation or audit, automatically add relevant questions based on page characteristics.

---

## How to Use

When creating or auditing a page specification:
1. Review the checklist below
2. For each applicable category, check if the page specification addresses it
3. If not addressed, add to the Open Questions section

---

## Responsive Behavior

**Trigger:** Page metadata indicates multiple viewports OR page is responsive

| Condition | Add Question |
|-----------|--------------|
| No responsive sketches | "What are the responsive breakpoint layouts? (Mobile/Tablet/Desktop)" |
| Mobile-first but no desktop spec | "How does the layout adapt for desktop users?" |
| Desktop-first but no mobile spec | "How does the layout adapt for mobile users?" |
| Touch + mouse interaction | "Are there hover states that need touch alternatives?" |

---

## Loading & Error States

**Trigger:** Page fetches data OR has async operations

| Condition | Add Question |
|-----------|--------------|
| API data but no loading state | "What does the user see while data is loading?" |
| No error state documented | "What happens if the data fails to load?" |
| No empty state documented | "What does the user see when there's no data?" |
| Async actions (save, submit) | "What feedback does the user get during async operations?" |
| Network-dependent features | "What happens if the user is offline?" |

---

## SEO & Meta Content

**Trigger:** Page is public (visibility = Public)

| Condition | Add Question |
|-----------|--------------|
| No page title specified | "What is the page title for SEO?" |
| No meta description | "What is the meta description for search results?" |
| No Open Graph tags | "What should the social sharing preview show?" |
| Dynamic content | "How do we handle SEO for dynamic/personalized content?" |

---

## Accessibility

**Trigger:** All pages (accessibility is always relevant)

| Condition | Add Question |
|-----------|--------------|
| Live updating content (timers, feeds) | "Should live updates announce to screen readers (aria-live)?" |
| Modal/drawer interactions | "Where does focus go when modal opens/closes?" |
| Color-coded states | "Is information conveyed by color alone?" |
| Custom components | "Do custom components have proper ARIA roles?" |
| Animations | "Are animations respecting prefers-reduced-motion?" |
| Complex interactions | "What is the keyboard navigation pattern?" |

---

## User Permissions & Roles

**Trigger:** Page has authenticated users OR multiple user types

| Condition | Add Question |
|-----------|--------------|
| Multi-user feature | "What does User B see when User A is performing an action?" |
| Role-based access | "Which elements are visible/hidden per role?" |
| Shared resources | "What happens if two users act simultaneously?" |
| Destructive actions | "Should destructive actions require confirmation?" |

---

## Time-Sensitive Features

**Trigger:** Page has countdowns, timers, or time-based state changes

| Condition | Add Question |
|-----------|--------------|
| Countdown timer | "What happens when the countdown reaches zero?" |
| Time windows | "Can users act before the time window opens?" |
| Time windows | "What happens after the time window closes?" |
| Background behavior | "Does the timer continue when app is backgrounded?" |
| Session timeout | "What happens when the session expires?" |

---

## Form Interactions

**Trigger:** Page has form inputs

| Condition | Add Question |
|-----------|--------------|
| No validation rules | "What are the validation rules for each field?" |
| No error messages | "What error messages are shown for each validation failure?" |
| No success state | "What happens after successful form submission?" |
| Partial completion | "Can users save partial progress?" |
| Sensitive data | "Are there security considerations for this form data?" |

---

## Navigation & Flow

**Trigger:** Page is part of a multi-step flow

| Condition | Add Question |
|-----------|--------------|
| No back navigation | "Can users go back to the previous step?" |
| Browser back button | "What happens when user presses browser back?" |
| Unsaved changes | "Should we warn users about unsaved changes?" |
| Deep linking | "Can this page be accessed via direct URL?" |

---

## Integration Checklist

When creating a page specification, check these categories:

```
[ ] Responsive — Do we have all breakpoint layouts?
[ ] Loading — Is the loading state documented?
[ ] Error — Is the error state documented?
[ ] Empty — Is the empty state documented?
[ ] SEO — Is meta content defined (if public)?
[ ] Accessibility — Are a11y requirements specified?
[ ] Permissions — Are role-based views documented?
[ ] Time — Are time-sensitive behaviors defined?
[ ] Forms — Are validation rules specified?
[ ] Navigation — Is back/forward behavior defined?
```

---

## Example Open Questions Section

For a responsive page with API data and timer:

```markdown
## Open Questions

| # | Question | Context | Status |
|---|----------|---------|--------|
| 1 | What are the tablet/desktop layouts? | Only mobile sketch provided | 🔴 Open |
| 2 | What does user see while loading? | API fetch on page load | 🔴 Open |
| 3 | What happens if API call fails? | Error handling | 🔴 Open |
| 4 | Does timer continue when app backgrounded? | Mobile behavior | 🔴 Open |
| 5 | Should timer announce to screen readers? | Accessibility | 🔴 Open |

**Status Legend:** 🔴 Open | 🟡 In Discussion | 🟢 Resolved
```

---

_Use this guide during specification creation and audits to ensure comprehensive coverage._
