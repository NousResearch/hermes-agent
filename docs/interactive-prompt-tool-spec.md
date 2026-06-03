# `interactive_prompt` Tool — Design Spec

## Motivation

PR #19413 originally bolted a generic `components` schema onto `send_message`, letting the model build raw Discord component JSON. While functional, this approach has problems:

1. **Model must understand Discord component structure** — action rows, style enums, custom_id conventions. That's implementation leakage into the LLM layer.
2. **Discord-only** — no path to Telegram inline keyboards, Slack block kit, or any other platform.
3. **Fire-and-forget** — components render on the message, but the tool call completes before the user interacts. The model doesn't block-and-wait for a response.
4. **Fights upstream direction** — teknium1's clarify implementation (`1dca6a696`) uses gateway-level adapter overrides, not model-generated component specs. Our PR's approach doesn't align with that pattern.

**This spec proposes pivoting to a purpose-built tool** that abstracts the component rendering behind a clean schema. The model describes *what* it wants (a question, options, display type); the tool + adapter handle *how* to render it per platform.

## Design Goals

| Goal | Why |
|------|-----|
| **Model-agnostic schema** | Model doesn't need to know about Action Rows, ButtonStyle enums, or custom_id prefixes |
| **Blocking tool call** | Tool blocks until user responds, returns the choice — natural LLM tool pattern |
| **Platform-adaptable** | Discord renders buttons/selects/modal; future platforms render native equivalents |
| **Composable with freeform input** | Structured choices + "type your own" fallback via modal popup |
| **File collection** | Modal file upload field for requesting documents from the user |
| **Matches upstream pattern** | Same gateway-level rendering approach as teknium1's clarify buttons |

## Tool Schema

```
interactive_prompt(
    question: str,                    # Required — the prompt/message text
    options: list[Option],            # Required — structured choices
    display_type: "buttons" | "select",# How to render options on the message (only "buttons" implemented in v1)
    allow_custom: bool,                # Whether to show "Other" option
    custom_fields: list[CustomField], # Modal fields when allow_custom=True
)
```

### Returns

```
{
    "choice": "strict",               # The selected option value, or null if custom
    "custom_response": {               # Populated if user went through modal
        "fields": {
            "reason": "We have compliance requirements...",
            "attachment": "<file_name>"  # If file upload field was used
        }
    }
}
```

## Option Schema

```python
Option:
    label: str              # Display text (max 80 chars)
    value: str              # Machine-readable value returned on selection
    description: str?        # Tooltip/subtitle (optional, max 100 chars)
    emoji: str?              # Emoji prefix (optional)
    style: "primary" | "secondary" | "success" | "danger"?  # Buttons only, default "secondary"
```

## CustomField Schema (Modal Fields)

```python
CustomField:
    key: str                # Identifier — used to populate custom_response.fields
    label: str              # Display label for the field
    description: str?       # Help text below the label
    type: "text" | "select" | "file_upload" | "radio" | "checkbox"
    required: bool?         # Default True
    placeholder: str?       # For text/select fields
    options: list[Option]?  # For select/radio/checkbox fields
    min_length: int?        # For text fields
    max_length: int?        # For text fields (max 4000)
    multiline: bool?        # For text fields — paragraph vs single-line
```

## Rendering Behavior

### Message Phase

The tool sends a Discord message with:

1. The `question` text as the message content (supports markdown)
2. Options rendered as either:
   - **Buttons** (display_type="buttons"): one `discord.ui.Button` per option, max 25 buttons across 5 action rows. Discord's 25-component cap leaves room for an "Other" button when `allow_custom=True`.
   - **Select Menu** (display_type="select"): one `discord.ui.Select` with all options. Better for 6+ choices where buttons would be cluttered.
3. If `allow_custom=True`: an additional "Other (type your answer)" button with `style="primary"`

### Interaction Phase

- **Option selected**: All buttons/select disables. Tool returns `{"choice": "<value>", "custom_response": null}`.
- **"Other" clicked**: Opens a modal popup containing the `custom_fields`. User fills in and submits.

### Modal Phase (triggered by "Other")

The modal renders `custom_fields` as native Discord modal components:

| CustomField.type | Discord Component | Notes |
|-----------------|-------------------|-------|
| `text` | Text Input (type 4) | `short=True/False` based on `multiline`. Wrapped in Label (type 18) with `label` + `description`. |
| `select` | String Select (type 3) | Wrapped in Label. Options from the field's `options` list. |
| `file_upload` | File Upload (type 19) | Wrapped in Label. Discord handles the file picker natively. |
| `radio` | Radio Group (type 21) | Single-choice set. Wrapped in Label. |
| `checkbox` | Checkbox Group (type 22) | Multi-select. Wrapped in Label. |

On modal submit, the tool returns:
```json
{
    "choice": null,
    "custom_response": {
        "fields": {
            "reason": "Compliance requires strict mode",
            "attachment": "ScubaReport_TenantA_2026-05-29.json"
        }
    }
}
```

## Example Usage

### Simple: Remediation approach selection
```python
interactive_prompt(
    question="**Tenant A** — Which remediation approach for this M365 deployment?",
    options=[
        {"label": "🔒 Strict", "value": "strict", "description": "Full MFA + CA + modern auth enforcement", "style": "success"},
        {"label": "🛡️ Standard", "value": "standard", "description": "MFA + modern auth, conditional access optional", "style": "primary"},
        {"label": "📋 Basic", "value": "basic", "description": "Password policy + audit logging only", "style": "secondary"},
    ],
    display_type="buttons",
    allow_custom=True,
    custom_fields=[
        {"key": "reason", "label": "Your reasoning", "type": "text", "multiline": True, "placeholder": "Why this approach?"},
    ]
)
```

**Renders:**
```
[Message]
**Tenant A** — Which remediation approach for this M365 deployment?

[🔒 Strict]  [🛡️ Standard]  [📋 Basic]  [✏️ Other (type your answer)]
```

**If "Other" clicked → Modal:**
```
┌─────────────────────────────────────────┐
│ Custom Response                          │
│                                          │
│ Your reasoning                           │
│ ┌────────────────────────────────────┐   │
│ │ Why this approach?                  │   │
│ │                                    │   │
│ └────────────────────────────────────┘   │
│                                          │
│              [Cancel]  [Submit]          │
└─────────────────────────────────────────┘
```

### File collection: Request Scuba report
```python
interactive_prompt(
    question="I need the **ScubaGear JSON report** for this tenant to continue the remediation analysis.",
    options=[
        {"label": "📎 Attach Report", "value": "attach", "style": "primary"},
        {"label": "⏭️ Skip for now", "value": "skip", "style": "secondary"},
    ],
    display_type="buttons",
    allow_custom=False,
)
# If attach → immediately opens modal:
# But actually we want the modal to have the file field...
```

Better pattern — always use modal for file collection:
```python
interactive_prompt(
    question="I need the **ScubaGear JSON report** for this tenant to continue the remediation analysis.",
    options=[
        {"label": "📎 Attach Report", "value": "attach", "style": "primary", "description": "Opens file picker"},
        {"label": "⏭️ Skip for now", "value": "skip", "style": "secondary"},
    ],
    display_type="buttons",
    allow_custom=True,
    custom_fields=[
        {"key": "notes", "label": "Additional context", "type": "text", "multiline": True, "placeholder": "Any notes about the report...", "required": False},
        {"key": "report", "label": "ScubaGear Report", "type": "file_upload", "required": True},
    ]
)
```

### Multi-field modal: Slice planning discussion
```python
interactive_prompt(
    question="**Slice Planning** — Let's scope the first iteration of the document migration.",
    options=[
        {"label": "✅ Use suggested slices", "value": "accept", "style": "success", "description": "3 slices, ~2 days each"},
        {"label": "✏️ Modify the plan", "value": "modify", "style": "primary"},
        {"label": "❌ Start over", "value": "restart", "style": "danger"},
    ],
    display_type="select",
    allow_custom=True,
    custom_fields=[
        {"key": "which_slices", "label": "Which slices to include?", "type": "checkbox",
         "options": [
             {"label": "Slice 1: Assessment", "value": "s1"},
             {"label": "Slice 2: Migration", "value": "s2"},
             {"label": "Slice 3: Validation", "value": "s3"},
         ], "required": True},
        {"key": "notes", "label": "Modification notes", "type": "text", "multiline": True, "placeholder": "What should change?"},
    ]
)
```

## Platform Adaptability

The tool schema is platform-agnostic. Each platform adapter implements its own rendering:

| Concept | Discord | Telegram (future) | Slack (future) |
|---------|---------|-------------------|-----------------|
| `buttons` | `discord.ui.Button` in Action Rows | `InlineKeyboardButton` | `Block Kit Button` |
| `select` | `discord.ui.Select` | Not native — fallback to buttons | `Static Select` |
| Modal with text | Modal + `TextInput` | Not native — fallback to freeform reply | `Modal` |
| Modal with file | Modal + `File Upload` | Not native — fallback to `send_file` reply | Not supported |
| Blocking wait | `ComponentStore` + interaction callback | Callback query handler | `block_actions` |

The tool implementation lives in `tools/` (not in the Discord adapter). Each platform adapter registers a handler for rendering interactive prompts via a common interface. This mirrors how `BasePlatformAdapter.send_clarify` works today.

## Relationship to Existing Infrastructure

### Reuses from PR #19413
- `ComponentStore` — in-memory view tracking by message_id
- `ComponentView` — but simplified (no raw spec parsing, tool generates views directly)
- `TrackedComponentView` — message_id → view → session_key mapping
- `_format_interaction_text` — interaction → MessageEvent formatting
- `_handle_interaction` — the interaction handler pattern

### New components
- `tools/interactive_prompt_tool.py` — tool registration + schema
- `InteractivePromptView` — replaces `ComponentView`'s generic spec parsing with purpose-built rendering from the tool's structured args
- `ModalBuilder` — builds Discord Modal from `CustomField[]`
- Platform adapter hooks — `send_interactive_prompt(prompt, callback)` on `BasePlatformAdapter`

### Alignment with upstream
- Same gateway-level rendering as teknium1's `ClarifyChoiceView` (`1dca6a696`)
- Same auth pattern (`_component_check_auth`)
- Same blocking/timeout behavior
- The tool is to structured Q&A what `clarify` is to clarification — a higher-level abstraction over raw components

### What we build on from teknium1's existing work

Directly reused from `plugins/platforms/discord/adapter.py`:
- **`_component_check_auth()`** (line 5053) — shared auth gate for all interactive views. Handles user/role allowlists, no-allowlist deployments, fail-closed. Proven by `ExecApprovalView`, `SlashConfirmView`, `UpdatePromptView`, `ModelPickerView`, and `ClarifyChoiceView`.

Pattern/conventions we follow:
- **`ClarifyChoiceView` UX shape** — buttons from choices, single-use (first click disables all), embed footer updates with who answered, `discord.ui.View` subclass with timeout.
- **"Other" → freeform fallback** — teknium1's `ClarifyChoiceView` uses `mark_awaiting_text()` to switch to text-capture mode. We extend this pattern but with a modal popup instead.
- **Embed styling** — orange for pending, green for resolved, blue for awaiting text input.

Entirely new (no upstream equivalent):
- **Modal support** — none of the 5 existing views open modals. First modal-based interaction in the adapter.
- **Select menus** — `ClarifyChoiceView` only renders buttons. New `discord.ui.Select` rendering for `display_type="select"`.
- **File Upload fields** — doesn't exist anywhere in the codebase.
- **Radio/Checkbox groups** — modal-only, not implemented upstream.
- **Blocking tool pattern** — all existing views are fire-and-forget (resolve via gateway callbacks). Our tool blocks the LLM turn and returns the result directly.

## Security Considerations

- **Auth**: All interactions validated through `_component_check_auth` (same as clarify buttons) — ensures the interacting user owns the session
- **File uploads**: File size limits enforced at the platform level (Discord: 25MB free, up to 500MB with boost). Files land as attachments in the agent's context, not written to disk automatically.
- **Timeout**: Interactive prompt views should have a configurable timeout (default 15 min) after which the tool returns `{"choice": null, "custom_response": null, "timed_out": True}`
- **Component cap**: Respect Discord's 25-component-per-message limit when rendering buttons. If options > 24 (leaving room for "Other"), auto-fallback to select menu display_type.

## Open Questions

> **Note:** The following features are aspirational for v2 — only `display_type="buttons"` is implemented in v1.
> - `display_type="select"` (select menu rendering, auto-fallback from buttons at 25+ options)
> - `file_upload` modal field type (requires Discord modal file-upload permissions)
> - Auto-fallback from buttons to select menu when options > 24

1. **Should the tool auto-acknowledge selection?** After user clicks a button, should the message update with "✅ Selected: Strict" or leave the buttons disabled as-is?
2. **Multi-select for buttons?** Discord buttons are inherently single-click. For "pick multiple slices", should we use checkbox group in a modal instead?
3. **Re-prompt after timeout?** Should the tool re-send the interactive prompt or let the model handle it in its next turn?
4. **Modal title character limit**: Discord modals max 45 chars. Long `question` text doesn't fit as the modal title — should we truncate to a summary?

## Next Steps

1. Comment on PR #19413 noting the pivot, tag teknium1, link to new PR when ready
2. Create new branch from main (not from #19413's branch)
3. Implement `tools/interactive_prompt_tool.py` — tool schema + handler
4. Implement `InteractivePromptView` — Discord message rendering
5. Implement `ModalBuilder` — Discord modal from CustomField[]
6. Wire into `BasePlatformAdapter` as `send_interactive_prompt()`
7. Add Discord adapter implementation
8. Tests: message rendering, modal rendering, interaction routing, timeout, auth, file upload
9. Live test against Pepper's Discord channel
