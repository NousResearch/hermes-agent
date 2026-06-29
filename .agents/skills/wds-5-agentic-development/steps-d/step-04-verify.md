---
name: 'step-04-verify'
description: 'Systematically confirm that the implementation satisfies every requirement in the spec'

# File References
nextStepFile: './step-05-finalize.md'
---

# Step 4: Verify

## STEP GOAL:

Systematically confirm that the implementation satisfies every requirement in the spec.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are an Implementation Partner guiding structured development activities
- ✅ If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring software development methodology expertise, user brings domain knowledge and codebase familiarity
- ✅ Maintain clear and structured tone throughout

### Step-Specific Rules:

- 🎯 Focus only on verifying acceptance criteria, responsive behavior, interactive states, accessibility, and visual fidelity
- 🚫 FORBIDDEN to add new features or refactor — only verify and fix issues found
- 💬 Approach: Walk through each acceptance criterion with user, testing concretely
- 📋 Fix failures immediately as they are found — do not batch them

## EXECUTION PROTOCOLS:

- 🎯 Every acceptance criterion tested and passing
- 💾 Document verification results and any fixes applied
- 📖 Reference acceptance criteria from Step 1 and the original spec
- 🚫 Do not add scope — only verify what was planned

## CONTEXT BOUNDARIES:

- Available context: Acceptance criteria from Step 1; implementation from Step 3; spec
- Focus: Systematic verification against spec requirements
- Limits: No new features, no refactoring beyond fixing issues
- Dependencies: Step 3 must be complete (implementation done)

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Walk Through Every Acceptance Criterion

Open the acceptance criteria checklist from Step 01. Go through each criterion one by one:

- Does the implementation satisfy it? Test it concretely, do not assume.
- If it passes, check it off.
- If it fails, note what is wrong and fix it before continuing.

Do not batch failures for later. Fix as you find them.

### 2. Test All Responsive Breakpoints

For each page and component, test at every breakpoint defined in the spec:

- Mobile (typically 375px)
- Tablet (typically 768px)
- Desktop (typically 1024px+)
- Any custom breakpoints specified

At each breakpoint, verify:

- Layout adapts correctly (stacking, reordering, hiding/showing elements)
- Text remains readable — no overflow, no truncation unless intended
- Touch targets meet minimum size (44x44px) on touch devices
- Images and media scale appropriately
- No horizontal scroll unless intended

### 3. Test All Interactive States

For every interactive element, verify each state:

| State | Verify |
|-------|--------|
| **Default** | Renders correctly on load |
| **Hover** | Visual feedback appears |
| **Focus** | Focus ring or indicator visible (keyboard users) |
| **Active / Pressed** | Visual response on click/tap |
| **Disabled** | Visually distinct, not interactive |
| **Loading** | Spinner or skeleton shown, interactions blocked |
| **Error** | Error message displayed, field highlighted |
| **Empty** | Empty state message or placeholder shown |
| **Success** | Confirmation feedback displayed |

### 4. Test Accessibility

Minimum accessibility checks for every feature:

- **Keyboard navigation:** Can you reach and operate every interactive element using only Tab, Enter, Space, Escape, and arrow keys?
- **Screen reader:** Do headings, labels, buttons, and form fields have meaningful text? Are ARIA labels present where needed?
- **Color contrast:** Does text meet WCAG AA contrast ratios (4.5:1 for normal text, 3:1 for large text)?
- **Focus management:** After modal open/close, form submit, or route change — is focus placed logically?
- **Alt text:** Do images have descriptive alt text (or empty alt for decorative images)?

### 5. Cross-Browser Check (If Specified)

If the spec requires specific browser support:

- Test in each listed browser
- Check for layout differences, font rendering, and JavaScript behavior
- Note any browser-specific issues and whether they are acceptable

### 6. Compare Implementation to Spec Side by Side

With the spec open next to the running implementation:

- Compare visual layout at each breakpoint
- Compare text content word for word
- Compare colors to spec hex values
- Compare spacing and proportions
- Note any discrepancies — fix or document as intentional deviations

For projects using Puppeteer, follow the verification process in INLINE-TESTING-GUIDE.md: measure what you can measure programmatically, and present only qualitative questions to the user.

### 7. Verify Checklist

- [ ] Every acceptance criterion tested and passing
- [ ] All responsive breakpoints verified
- [ ] All interactive states working (hover, focus, disabled, loading, error, empty, success)
- [ ] Keyboard navigation works for all interactive elements
- [ ] Screen reader labels and ARIA attributes present
- [ ] Color contrast meets WCAG AA
- [ ] Focus management correct after state changes
- [ ] Cross-browser tested (if required by spec)
- [ ] Visual comparison to spec completed — no unintended differences
- [ ] All found issues fixed or documented as intentional deviations

### 8. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Step 5: Finalize"

#### Menu Handling Logic:
- IF C: Update design log, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN all acceptance criteria are verified passing and all issues fixed will you then load and read fully `{nextStepFile}` to execute.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Every acceptance criterion tested and passing
- All responsive breakpoints verified
- All interactive states working
- Accessibility checks completed
- Visual comparison to spec completed
- All found issues fixed or documented

### ❌ SYSTEM FAILURE:
- Assuming criteria pass without testing concretely
- Skipping responsive or accessibility verification
- Batching failures instead of fixing immediately
- Not comparing implementation to spec visually

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
