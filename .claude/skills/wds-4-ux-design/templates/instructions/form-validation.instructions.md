# Form Validation

**Include when:** Page has forms or input fields

---

## Validation Rules

| Field | Rule | Error Code | Error Message |
|-------|------|------------|---------------|
| `{field-name}` | {validation-rule} | `{ERR_CODE}` | {message} |

---

## Error Messages

| Error Code | Trigger | EN | SE | Recovery |
|------------|---------|-----|-----|----------|
| `ERR_001` | {When occurs} | "{English}" | "{Swedish}" | {How to fix} |

---

## Form States

### Valid State
- All fields pass validation
- Submit button enabled
- No error indicators

### Invalid State
- Error fields highlighted
- Error messages visible
- Submit button disabled until fixed

### Submitting State
- Submit button shows loading
- Fields disabled
- Cancel option available

---

## Field Specifications

### {Field Name}

| Property | Value |
|----------|-------|
| **OBJECT ID** | `{form-field-id}` |
| Type | {text / email / password / select / etc.} |
| Required | {yes / no} |
| Placeholder EN | "{Placeholder text}" |
| Placeholder SE | "{Swedish placeholder}" |
| Validation | {rules} |
| Error Message | {message} |
