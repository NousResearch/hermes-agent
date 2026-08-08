# Tally API Documentation Guides

Practical how-to guides from the official Tally documentation. Each section shows the exact API calls needed for common tasks.

---

## Creating a Form

Create an empty form with just a title:

```bash
curl -X POST 'https://api.tally.so/forms' \
-H 'Authorization: Bearer <token>' \
-H 'Content-Type: application/json' \
-d '{
  "status": "PUBLISHED",
  "blocks": [
    {
      "uuid": "6ef8675d-33cb-419b-a81e-93982e726f2e",
      "type": "FORM_TITLE",
      "groupUuid": "073c835f-7ad4-459c-866d-4108b6b7e2e1",
      "groupType": "TEXT",
      "payload": {
        "title": "Test",
        "html": "Test"
      }
    }
  ]
}'
```

Each block requires a unique UUID. Response returns `201` with form metadata including the `id`.

---

## Creating a Contact Form

A form with name (INPUT_TEXT), email (INPUT_EMAIL), and message (TEXTAREA) fields:

```bash
curl -X POST 'https://api.tally.so/forms' \
-H 'Authorization: Bearer <token>' \
-H 'Content-Type: application/json' \
-d '{
  "status": "PUBLISHED",
  "blocks": [
    {
      "uuid": "6ef8675d-33cb-419b-a81e-93982e726f2e",
      "type": "FORM_TITLE",
      "groupUuid": "073c835f-7ad4-459c-866d-4108b6b7e2e1",
      "groupType": "TEXT",
      "payload": { "html": "Contact form" }
    },
    {
      "uuid": "48b4cdf3-2c9d-47d3-b8fb-b0ccabc5cd84",
      "type": "TITLE",
      "groupUuid": "93034250-5f05-4710-b8e0-5c9145c5b9ea",
      "groupType": "QUESTION",
      "payload": { "html": "Name" }
    },
    {
      "uuid": "884ff838-97f9-4ac9-8db1-31aa052df988",
      "type": "INPUT_TEXT",
      "groupUuid": "93034250-5f05-4710-b8e0-5c9145c5b9ea",
      "groupType": "QUESTION",
      "payload": { "isRequired": true, "placeholder": "Enter your name" }
    },
    {
      "uuid": "7d9c2e31-b5aa-4c8b-9c2d-123456789abc",
      "type": "TITLE",
      "groupUuid": "3287d15c-c2b2-4f84-a915-bc57380a4b51",
      "groupType": "QUESTION",
      "payload": { "html": "Email" }
    },
    {
      "uuid": "9b3f5d2a-1c8e-4f7d-b6a9-def012345678",
      "type": "INPUT_EMAIL",
      "groupUuid": "3287d15c-c2b2-4f84-a915-bc57380a4b51",
      "groupType": "QUESTION",
      "payload": { "isRequired": true, "placeholder": "Enter your email" }
    },
    {
      "uuid": "abc12345-6789-def0-1234-56789abcdef0",
      "type": "TITLE",
      "groupUuid": "456789ab-cdef-4321-b8e0-987654321def",
      "groupType": "QUESTION",
      "payload": { "html": "Message" }
    },
    {
      "uuid": "456789ab-cdef-0123-4567-89abcdef0123",
      "type": "TEXTAREA",
      "groupUuid": "456789ab-cdef-4321-b8e0-987654321def",
      "groupType": "QUESTION",
      "payload": { "isRequired": true, "placeholder": "Enter your message" }
    }
  ]
}'
```

Pattern: each question uses a shared `groupUuid` for its TITLE + input block pair.

---

## Creating a Dropdown

Dropdown options are `DROPDOWN_OPTION` blocks sharing the same `groupUuid`. Use `index` for ordering and `text` for the label.

```bash
curl -X POST 'https://api.tally.so/forms' \
-H 'Authorization: Bearer <token>' \
-H 'Content-Type: application/json' \
-d '{
  "status": "PUBLISHED",
  "blocks": [
    {
      "uuid": "6ef8675d-33cb-419b-a81e-93982e726f2e",
      "type": "FORM_TITLE",
      "groupUuid": "073c835f-7ad4-459c-866d-4108b6b7e2e1",
      "groupType": "TEXT",
      "payload": { "title": "Dropdown example", "html": "Dropdown example" }
    },
    {
      "uuid": "2515b4dd-54e3-4502-afe6-074ad5019b44",
      "type": "TITLE",
      "groupUuid": "22a0af81-0117-4931-806f-2b83e374275b",
      "groupType": "QUESTION",
      "payload": { "html": "What'\''s your favorite color?" }
    },
    {
      "uuid": "338631d5-64b0-4a55-8219-17658e66196b",
      "type": "DROPDOWN_OPTION",
      "groupUuid": "aa64831b-8695-4887-afba-31c07034cd77",
      "groupType": "DROPDOWN",
      "payload": { "index": 0, "text": "Red" }
    },
    {
      "uuid": "4c8fe10a-b07e-407e-9347-f64d73a9ba9a",
      "type": "DROPDOWN_OPTION",
      "groupUuid": "aa64831b-8695-4887-afba-31c07034cd77",
      "groupType": "DROPDOWN",
      "payload": { "index": 1, "text": "Green" }
    },
    {
      "uuid": "3bddc101-571b-4f00-aa7a-30ee629441bc",
      "type": "DROPDOWN_OPTION",
      "groupUuid": "aa64831b-8695-4887-afba-31c07034cd77",
      "groupType": "DROPDOWN",
      "payload": { "index": 2, "text": "Blue" }
    }
  ]
}'
```

All dropdown options must share the same `groupUuid`.

---

## Creating a Form with Settings

Settings cover language, closing rules, submission limits, email notifications, redirects, mentions in settings, and custom themes.

Key settings fields:
- `language` — form language code (e.g. `"fr"`)
- `closeDate`, `closeTime`, `closeTimezone` — auto-close schedule
- `submissionsLimit` — max submissions allowed
- `uniqueSubmissionKey` — enforce one submission per unique field value (uses mentions)
- `redirectOnCompletion` — URL redirect after submit (can use mentions for dynamic URLs)
- `hasSelfEmailNotifications`, `selfEmailTo`, `selfEmailReplyTo`, `selfEmailBody` — email notifications with mention support
- `styles` — custom theme with `theme: "CUSTOM"`, `color` object, and `css` for custom CSS

Example settings object:
```json
{
  "language": "fr",
  "closeDate": "2026-01-09",
  "closeTime": "09:41",
  "closeTimezone": "Europe/Paris",
  "submissionsLimit": 42,
  "styles": {
    "theme": "CUSTOM",
    "color": {
      "background": "#ffffff",
      "text": "#37352f",
      "accent": "#007aff",
      "buttonBackground": "#007aff",
      "buttonText": "#ffffff"
    },
    "css": ".tally-submit-button svg { display: none; }",
    "direction": "ltr"
  }
}
```

---

## Creating a Mention

Mentions let you reference field values dynamically in titles, descriptions, and settings. They use a `<span class="mention" data-uuid="...">@fieldName</span>` HTML pattern paired with a `mentions` array.

```json
{
  "html": "Hello <span class=\"mention\" data-uuid=\"0f7b3637-61b6-4faa-a93a-a6a31ff5ac63\">@name</span>",
  "mentions": [
    {
      "uuid": "0f7b3637-61b6-4faa-a93a-a6a31ff5ac63",
      "field": {
        "uuid": "16826368-6cce-4066-b1da-be466f851c2d",
        "type": "HIDDEN_FIELD",
        "questionType": "HIDDEN_FIELDS",
        "blockGroupUuid": "203e6532-22a0-4421-b720-5d88d603e618",
        "title": "name"
      },
      "defaultValue": "there"
    }
  ]
}
```

Hidden fields are populated via query parameters: `https://tally.so/r/{formId}?name=John`

---

## Fetching Form Submissions

```bash
curl -X GET 'https://api.tally.so/forms/:id/submissions' \
-H 'Authorization: Bearer <token>'
```

Response structure:
```json
{
  "page": 1,
  "limit": 50,
  "hasMore": false,
  "totalNumberOfSubmissionsPerFilter": {
    "all": 4,
    "completed": 4,
    "partial": 0
  },
  "questions": [
    {
      "id": "EKOE2N",
      "type": "INPUT_TEXT",
      "title": "First name",
      "fields": [
        {
          "uuid": "21dd98ef-4c54-4e77-bcc6-7ec79409b3ea",
          "type": "INPUT_FIELD",
          "questionType": "INPUT_TEXT",
          "title": "First name"
        }
      ]
    }
  ],
  "submissions": [
    {
      "id": "GG6z5L",
      "formId": "mexJoq",
      "respondentId": "jzQdR9",
      "isCompleted": true,
      "submittedAt": "2024-12-30T09:02:01.000Z",
      "responses": [
        {
          "id": "4r5rAWb",
          "questionId": "EKOE2N",
          "answer": "Filip"
        }
      ]
    }
  ]
}
```

Note: The response includes a `questions` array mapping question IDs to field metadata, and a `submissions` array with `responses` keyed by `questionId`.

---

## Styling Title Blocks

Use HTML markup in the `html` payload field to style text:

```json
{
  "html": "What's your <span style=\"color: #eb4d4b\"><i><b>name</b></i></span>?"
}
```

Supported HTML tags: `<b>`, `<i>`, `<u>`, `<span style="color: ...">`, `<a href="...">`.

---

## Adding Blocks to a Form

Use `PATCH /forms/:id` with the **complete** blocks array. You must include all existing blocks plus your new ones — the API replaces the entire blocks array.

```bash
curl -X PATCH 'https://api.tally.so/forms/:id' \
-H 'Authorization: Bearer <token>' \
-H 'Content-Type: application/json' \
-d '{
  "name": "Test",
  "status": "PUBLISHED",
  "blocks": [
    ... all existing blocks ...,
    ... new blocks ...
  ]
}'
```

**Important:** Always GET the form first to retrieve current blocks, then append your new blocks to that array before PATCHing.
