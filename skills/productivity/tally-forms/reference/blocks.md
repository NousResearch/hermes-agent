# Block Types Reference

Blocks are the building units of Tally forms. Every form is an ordered array of blocks. Each block has a `type`, a `uuid`, a `groupUuid` (linking it to a parent group), a `groupType`, and a `payload` with type-specific data.

## Block Structure

```json
{
  "uuid": "550e8400-e29b-41d4-a716-446655440000",
  "type": "INPUT_TEXT",
  "groupUuid": "660e8400-e29b-41d4-a716-446655440000",
  "groupType": "QUESTION",
  "payload": {
    "isRequired": false,
    "isHidden": false,
    "placeholder": "",
    "name": "question_abc12345"
  }
}
```

## Question Pattern

Most input blocks follow the same 3-block pattern within a shared `groupUuid`:

1. **QUESTION** block — container with `isRequired` and `isHidden`
2. **TITLE** block — the question label (HTML in `payload.html`)
3. **Input block** — the actual field (INPUT_TEXT, INPUT_EMAIL, DROPDOWN, etc.)

```json
[
  { "type": "QUESTION",   "groupUuid": "G1", "groupType": "QUESTION", "payload": { "isRequired": true } },
  { "type": "TITLE",      "groupUuid": "G1", "groupType": "QUESTION", "payload": { "html": "<p>Your name?</p>" } },
  { "type": "INPUT_TEXT",  "groupUuid": "G1", "groupType": "QUESTION", "payload": { "placeholder": "Jane Doe" } }
]
```

---

## All Block Types

### Form Structure

| Type | Description | Key Payload Fields |
|------|-------------|-------------------|
| FORM_TITLE | Form title/header block | `html`, `logo`, `cover` |
| TITLE | Question label / title text | `html` |
| QUESTION | Question container (groups an input + title) | `isRequired`, `isHidden` |
| TEXT | Free text / description paragraph | `html` |
| LABEL | Static label text | `html` |
| PAGE_BREAK | Page separator for multi-page forms | `name` |
| DIVIDER | Visual horizontal rule | `isHidden` |

### Headings

| Type | Description | Key Payload Fields |
|------|-------------|-------------------|
| HEADING_1 | Large heading | `html` |
| HEADING_2 | Medium heading | `html` |
| HEADING_3 | Small heading | `html` |

### Text Inputs

| Type | Description | Key Payload Fields |
|------|-------------|-------------------|
| INPUT_TEXT | Single-line text | `isRequired`, `placeholder`, `name` |
| TEXTAREA | Multi-line text | `isRequired`, `placeholder`, `name` |

### Specialized Inputs

| Type | Description | Key Payload Fields |
|------|-------------|-------------------|
| INPUT_EMAIL | Email with validation | `isRequired`, `placeholder`, `name` |
| INPUT_NUMBER | Numeric input | `isRequired`, `name`, `hasMinNumber`, `minNumber`, `hasMaxNumber`, `maxNumber` |
| INPUT_LINK | URL input | `isRequired`, `placeholder`, `name` |
| INPUT_PHONE_NUMBER | Phone number | `isRequired`, `placeholder`, `name` |
| INPUT_DATE | Date picker | `isRequired`, `name` |
| INPUT_TIME | Time picker | `isRequired`, `name` |

### Choice Inputs

| Type | Description | Key Payload Fields |
|------|-------------|-------------------|
| MULTIPLE_CHOICE | Radio button group (container) | — |
| MULTIPLE_CHOICE_OPTION | Single radio option | `label` |
| CHECKBOXES | Checkbox group (container) | — |
| CHECKBOX | Single checkbox option | `label` |
| DROPDOWN | Dropdown select (container) | — |
| DROPDOWN_OPTION | Single dropdown option | `label` |
| MULTI_SELECT | Multi-select tags (container) | — |
| MULTI_SELECT_OPTION | Single multi-select option | `label` |
| RANKING | Ranking container | — |
| RANKING_OPTION | Single ranking option | `label` |

### Scale & Rating

| Type | Description | Key Payload Fields |
|------|-------------|-------------------|
| RATING | Star rating | `isRequired`, `maxRating` |
| LINEAR_SCALE | Numeric scale | `isRequired`, `min`, `max`, `minLabel`, `maxLabel` |

### Matrix

| Type | Description | Key Payload Fields |
|------|-------------|-------------------|
| MATRIX | Matrix/grid container | — |
| MATRIX_ROW | Matrix row label | `label` |
| MATRIX_COLUMN | Matrix column label | `label` |

### Media

| Type | Description | Key Payload Fields |
|------|-------------|-------------------|
| IMAGE | Embedded image | `url`, `alt` |
| EMBED | Generic embed (iframe) | `url` |
| EMBED_VIDEO | Video embed | `url` |
| EMBED_AUDIO | Audio embed | `url` |

### File & Signature

| Type | Description | Key Payload Fields |
|------|-------------|-------------------|
| FILE_UPLOAD | File upload field | `isRequired`, `allowedFileTypes`, `maxFileSize` |
| SIGNATURE | Signature pad | `isRequired` |

### Payment

| Type | Description | Key Payload Fields |
|------|-------------|-------------------|
| PAYMENT | Payment collection (Stripe) | `amount`, `currency`, `description` |
| WALLET_CONNECT | Crypto wallet connect | — |

### Advanced / Logic

| Type | Description | Key Payload Fields |
|------|-------------|-------------------|
| HIDDEN_FIELDS | Pre-filled hidden fields | `fields` (array of key-value) |
| CONDITIONAL_LOGIC | Show/hide blocks based on answers | `conditions`, `action` |
| CALCULATED_FIELDS | Computed values from other fields | `formula` |
| CAPTCHA | Bot protection | — |
| RESPONDENT_COUNTRY | Auto-detected country | — |

---

## Nesting Rules

- **QUESTION** groups: `groupType` = `"QUESTION"`, children share the same `groupUuid`
- **Choice containers** (MULTIPLE_CHOICE, CHECKBOXES, DROPDOWN, MULTI_SELECT, RANKING): options use the container's `uuid` as their `groupUuid` and the container type as `groupType`
- **MATRIX**: rows and columns use the matrix block's `uuid` as `groupUuid`
- **Standalone blocks** (DIVIDER, PAGE_BREAK, HEADING_*, IMAGE, etc.): `groupUuid` = own `uuid`, `groupType` = own `type`

## Example: Complete Dropdown Question

```json
[
  {
    "uuid": "q1",
    "type": "QUESTION",
    "groupUuid": "g1",
    "groupType": "QUESTION",
    "payload": { "isRequired": true, "isHidden": false }
  },
  {
    "uuid": "t1",
    "type": "TITLE",
    "groupUuid": "g1",
    "groupType": "QUESTION",
    "payload": { "html": "<p>Select your country</p>" }
  },
  {
    "uuid": "d1",
    "type": "DROPDOWN",
    "groupUuid": "g1",
    "groupType": "QUESTION",
    "payload": {}
  },
  {
    "uuid": "o1",
    "type": "DROPDOWN_OPTION",
    "groupUuid": "d1",
    "groupType": "DROPDOWN",
    "payload": { "label": "United States" }
  },
  {
    "uuid": "o2",
    "type": "DROPDOWN_OPTION",
    "groupUuid": "d1",
    "groupType": "DROPDOWN",
    "payload": { "label": "United Kingdom" }
  },
  {
    "uuid": "o3",
    "type": "DROPDOWN_OPTION",
    "groupUuid": "d1",
    "groupType": "DROPDOWN",
    "payload": { "label": "Canada" }
  }
]
```
