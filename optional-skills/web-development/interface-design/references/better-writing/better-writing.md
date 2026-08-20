# Writing that disappears into the interface

Clear and brief beats clever, consistency beats variety, and the best error message is the interaction redesigned so the error can't happen. Apply these principles when writing or reviewing any user-facing text.

How copy renders (capitalization via `text-transform`, truncation, smart punctuation) is covered by the `better-typography` skill; error markup and announcements (`aria-invalid`, live regions) by the `better-accessibility` skill; room for translated strings by the `better-layout` skill.

## Quick Reference

| Category | When to Use |
| --- | --- |
| [Review Output Format](review-output.md) | Severity scale, findings table, verification, verdict |

## Core Principles

### 1. Recon the Existing Voice

Before writing or reviewing, inspect nearby interface copy, the product's terminology, localization conventions, and any voice or content style guide. Preserve intentional brand character when it remains clear and appropriate to the stakes. Treat a difference from generic plain language as a finding only when it creates inconsistency, ambiguity, translation risk, or an inappropriate tone.

### 2. One Voice, Flexible Tone

The product has one voice, established by its existing system rather than invented during a local edit. Keep terms consistent: if it's "Archive" in the menu, it isn't "Move to storage" in the toast. Tone flexes with the stakes:

| Context | Tone |
| --- | --- |
| Success, onboarding, empty states | Warm, can be light |
| Routine actions, settings | Neutral, minimal |
| Errors, destructive confirmations | Calm, plain, zero playfulness |
| Data loss, security | Serious, explicit |

### 3. Address the Reader Directly

In instructional interface copy, address the reader directly as "you" rather than "the user." Avoid “we” in errors when it creates ambiguity or reads as deflection: prefer “Unable to load content” over “We're having trouble loading this content.” Preserve an established first-person brand voice in low-stakes contexts when it remains clear. Use possessives sparingly (“Favorites” over “Your Favorites”) and never switch perspective accidentally.

### 4. Plain Words Over Clever Ones

Choose easily understood words and delete every word that isn't needed. No idioms, colloquialisms, or humor that won't translate. Skip unnecessary gender: "Subscribers can post recipes", not "each subscriber can post his or her recipes". Match the input device: "tap" on touch, "click" with a pointer, "select" when both are possible. Never build sentences by concatenating fragments around variables (`"You have " + n + " new messages"`); word order changes per language, so use full templated strings with proper pluralization.

### 5. Verb-First Buttons

Button labels start with a verb naming the specific action: "Send", "Save draft", "Delete project". Never "OK!", "Let's go!", or bare "Yes"/"No" on consequential actions. Confirmation buttons repeat the consequence so the dialog is answerable without reading the body: "Delete this project?" offers `Delete project` and `Cancel`, not `Yes` and `No`.

### 6. Consistent Flow Vocabulary

Multi-step flows use one vocabulary: "Get Started" to enter, "Continue" or "Next" (pick one) to advance, "Done" to finish. Alternating synonyms across steps makes users wonder if the buttons do different things.

### 7. Links Describe Their Destination

Link text makes sense out of context; screen-reader users navigate by a list of the page's links. "Read the billing docs", never "Click here" (which also fails the device-verb rule on touch), and never a bare "Learn more" when several appear on one page. Suffix each: "Learn more about exports".

### 8. One Capitalization Policy

Pick title case or sentence case per element type (all buttons, all headings) and apply it consistently; sentence case is the safer default: calmer, no per-word case rules, localizes cleanly. "Save Changes" beside "Discard changes" reads as sloppiness.

### 9. Settings Describe the ON State

Label a toggle for what happens when it's on: "Send read receipts", and users infer the off state. Never label the negative ("Don't send read receipts"), which turns the toggle into a double negative. Link directly to a referenced setting instead of describing the path to it: a "Notification settings" link, not "Go to Settings > Notifications > Email".

### 10. Errors Say How to Fix, Next to Where It Broke

An error is an instruction, adjacent to the failing field:

| Bad | Good |
| --- | --- |
| That password is too short | Choose a password with at least 8 characters |
| Invalid name | Use only letters for your name |
| Oops! Something went wrong. | Unable to save. Check your connection and try again. |

No blame, no "oops", no exclamation marks. Phrase hints positively ("Use only letters", not "Don't use numbers or symbols") and show them before the mistake, not after. If the same error keeps firing for many users, redesign the interaction instead of rewording it.

### 11. Empty States Point Forward

An empty state says what this place is and how to fill it, with one clear next action:

```html
<!-- Bad: a shrug -->
<p>No results.</p>

<!-- Good: orientation plus a next step -->
<p class="font-medium">No projects yet</p>
<p class="text-sm text-zinc-500">Projects keep your tasks and files together.</p>
<button class="mt-4">Create a project</button>
```

Search and filter empty states name the query and offer an exit: "No results for 'quarterly'. Clear filters". Never park crucial persistent information in an empty state; it disappears the moment content exists.

### 12. Placeholders Are Examples, Not Labels

Placeholders show the expected format (`name@example.com`, `DD/MM/YYYY`). A placeholder is never the field's only label: it vanishes on input, and every field keeps a visible label.

## Common Mistakes

| Mistake | Fix |
| --- | --- |
| Local rewrite ignores the product's established terminology or voice | Inspect nearby copy and the style guide before proposing a change |
| "The user" in instructional interface copy | Address the reader directly as "you" |
| "We're having trouble…" obscures responsibility or recovery | Use a direct status and next step: "Unable to load content" |
| `OK` / `Yes` confirming a destructive dialog | Repeat the consequence: "Delete project" |
| "Continue" on step 2, "Next" on step 3 | One flow vocabulary throughout |
| "Click here" or bare "Learn more" link | Describe the destination: "Read the billing docs" |
| "Save Changes" beside "Discard changes" | One capitalization policy per element type |
| "Don't send read receipts" toggle | Label the ON state: "Send read receipts" |
| "Oops! Something went wrong." | Say what to do, next to the failing field |
| "No results." as the whole empty state | Orient and point forward with a next action |
| Placeholder doing the label's job | Visible label; placeholder shows the format |
| `"You have " + n + " messages"` | Full templated strings with pluralization |

## Reporting

A standalone writing review is finished when every confirmed finding is reported in the format in [review-output.md](review-output.md), with verification and a verdict. Under `better-interface`, its format governs instead.
