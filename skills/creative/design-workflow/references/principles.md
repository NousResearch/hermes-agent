# Design Workflow Principles

These principles override style presets. Use them for web, responsive web, and native iOS/Android product UI.

## 1. Clarity Before Beauty

A screen is successful only if the target user can understand what it is, why it matters, and what to do next. Visual polish is secondary to comprehension.

Ask:
- Can a new user explain the screen's purpose in 5 seconds?
- Is the primary action obvious?
- Does every section support the page/screen goal?

## 2. One Job Per View

Each page, section, modal, or mobile screen should have one dominant job.

Good signs:
- One clear hierarchy.
- One dominant CTA or user action.
- Secondary actions are visibly secondary.
- The screen does not ask the user to make multiple unrelated decisions at once.

## 3. Design Is Persuasion Order

For landing/product pages, the default order is:

```text
Understand → Relevance → Trust → Action
```

For product apps, the default order is:

```text
Orient → Decide → Act → Recover
```

For dashboards, the default order is:

```text
Status → Priority → Diagnosis → Action
```

## 4. Visuals Serve Hierarchy

Use contrast, spacing, typography, color, elevation, and motion to explain importance and state. Do not add decoration that competes with the primary job.

## 5. Real States Matter

Always consider:
- loading
- empty
- error
- long text
- extreme data
- disabled
- focus
- hover or pressed
- success
- warning
- permission denied
- offline or retry states

## 6. Platform Conventions Are Product Quality

For web, respect semantic HTML, keyboard navigation, focus, responsive layout, browser constraints, and performance.

For iOS, respect safe areas, Dynamic Type, VoiceOver, standard navigation/sheets/tabs, SF Symbols conventions, and expected gestures.

For Android, respect system back, Material conventions where appropriate, TalkBack, status/navigation bars, density buckets, and edge-to-edge layout.

## 7. Taste Is Adjustable

Do not argue with vague adjectives. Convert style requests into dials:

- variance
- motion
- density

Then make trade-offs explicit.

## 8. Existing Systems Win

Existing brand guidelines, design tokens, component libraries, accessibility requirements, and product constraints outrank generic rules from this skill.
