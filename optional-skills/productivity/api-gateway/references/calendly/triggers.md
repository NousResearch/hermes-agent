# Calendly Trigger Reference

## Event Types

- [`invitee.created`](#inviteecreated)
- [`invitee.canceled`](#inviteecanceled)
- [`invitee_no_show.created`](#invitee_no_showcreated)

---

## `invitee.created`

### Sample Payload

```json
{
  "created_at": "2026-06-24T22:30:45.105381Z",
  "created_by": "https://api.calendly.com/users/00000000-0000-0000-0000-0000000000a1",
  "event": "invitee.created",
  "payload": {
    "cancel_url": "https://calendly.com/cancellations/00000000-0000-0000-0000-000000000001",
    "created_at": "2026-06-09T21:37:17.731525Z",
    "email": "john.doe@example.com",
    "event": "https://api.calendly.com/scheduled_events/00000000-0000-0000-0000-000000000002",
    "first_name": null,
    "invitee_scheduled_by": null,
    "last_name": null,
    "name": "John Doe",
    "new_invitee": null,
    "no_show": null,
    "old_invitee": null,
    "payment": null,
    "questions_and_answers": [],
    "reconfirmation": null,
    "reschedule_url": "https://calendly.com/reschedulings/00000000-0000-0000-0000-000000000001",
    "rescheduled": false,
    "routing_form_submission": null,
    "scheduled_event": {
      "created_at": "2026-06-09T21:37:17.707255Z",
      "end_time": "2026-06-10T07:00:00.000000Z",
      "event_guests": [],
      "event_memberships": [
        {
          "user": "https://api.calendly.com/users/00000000-0000-0000-0000-0000000000a1",
          "user_email": "support@example.com",
          "user_name": "Acme"
        }
      ],
      "event_type": "https://api.calendly.com/event_types/00000000-0000-0000-0000-0000000000e1",
      "invitees_counter": {
        "total": 1,
        "active": 1,
        "limit": 1
      },
      "location": {
        "data": {
          "id": 10000000001,
          "settings": {},
          "password": "••••••",
          "extra": null
        },
        "join_url": "https://us05web.zoom.us/j/10000000001?pwd=REDACTED",
        "status": "pushed",
        "type": "zoom"
      },
      "meeting_notes_html": null,
      "meeting_notes_plain": null,
      "name": "Acme - Intro",
      "start_time": "2026-06-10T06:30:00.000000Z",
      "status": "active",
      "updated_at": "2026-06-09T21:37:17.707255Z",
      "uri": "https://api.calendly.com/scheduled_events/00000000-0000-0000-0000-000000000002"
    },
    "scheduling_method": "api",
    "status": "active",
    "text_reminder_number": null,
    "timezone": "America/Los_Angeles",
    "tracking": {
      "utm_campaign": null,
      "utm_source": null,
      "utm_medium": null,
      "utm_content": null,
      "utm_term": null,
      "salesforce_uuid": null
    },
    "updated_at": "2026-06-09T21:37:17.731525Z",
    "uri": "https://api.calendly.com/scheduled_events/00000000-0000-0000-0000-000000000002/invitees/00000000-0000-0000-0000-000000000001"
  }
}
```

---

## `invitee.canceled`

### Sample Payload

```json
{
  "created_at": "2026-06-24T22:30:45.574849Z",
  "created_by": "https://api.calendly.com/users/00000000-0000-0000-0000-0000000000a1",
  "event": "invitee.canceled",
  "payload": {
    "cancel_url": "https://calendly.com/cancellations/00000000-0000-0000-0000-000000000003",
    "cancellation": {
      "canceled_by": "Jane Doe",
      "canceler_type": "invitee",
      "created_at": "2026-06-17T01:40:57.223808Z",
      "reason": "test creation"
    },
    "created_at": "2026-06-17T01:40:15.793613Z",
    "email": "jane.doe@example.com",
    "event": "https://api.calendly.com/scheduled_events/00000000-0000-0000-0000-000000000004",
    "first_name": null,
    "invitee_scheduled_by": null,
    "last_name": null,
    "name": "Jane Doe",
    "new_invitee": null,
    "no_show": null,
    "old_invitee": null,
    "payment": null,
    "questions_and_answers": [
      {
        "answer": "Test event create",
        "position": 0,
        "question": "Please share anything that will help prepare for our meeting."
      }
    ],
    "reconfirmation": null,
    "reschedule_url": "https://calendly.com/reschedulings/00000000-0000-0000-0000-000000000003",
    "rescheduled": false,
    "routing_form_submission": null,
    "scheduled_event": {
      "cancellation": {
        "canceled_by": "Jane Doe",
        "canceler_type": "invitee",
        "created_at": "2026-06-17T01:40:57.223808Z",
        "reason": "test creation"
      },
      "created_at": "2026-06-17T01:40:15.775543Z",
      "end_time": "2026-07-01T07:00:00.000000Z",
      "event_guests": [],
      "event_memberships": [
        {
          "user": "https://api.calendly.com/users/00000000-0000-0000-0000-0000000000a1",
          "user_email": "support@example.com",
          "user_name": "Acme"
        }
      ],
      "event_type": "https://api.calendly.com/event_types/00000000-0000-0000-0000-0000000000e1",
      "invitees_counter": {
        "total": 1,
        "active": 0,
        "limit": 1
      },
      "location": {
        "data": {
          "id": 10000000002,
          "settings": {},
          "password": "••••••",
          "extra": null
        },
        "join_url": "https://us05web.zoom.us/j/10000000002?pwd=REDACTED",
        "status": "pushed",
        "type": "zoom"
      },
      "meeting_notes_html": null,
      "meeting_notes_plain": null,
      "name": "Acme - Intro",
      "start_time": "2026-07-01T06:30:00.000000Z",
      "status": "canceled",
      "updated_at": "2026-06-17T01:40:57.264331Z",
      "uri": "https://api.calendly.com/scheduled_events/00000000-0000-0000-0000-000000000004"
    },
    "scheduling_method": "api",
    "status": "canceled",
    "text_reminder_number": null,
    "timezone": "America/Los_Angeles",
    "tracking": {
      "utm_campaign": null,
      "utm_source": null,
      "utm_medium": null,
      "utm_content": null,
      "utm_term": null,
      "salesforce_uuid": null
    },
    "updated_at": "2026-06-17T01:40:57.238765Z",
    "uri": "https://api.calendly.com/scheduled_events/00000000-0000-0000-0000-000000000004/invitees/00000000-0000-0000-0000-000000000003"
  }
}
```

---

## `invitee_no_show.created`

### Sample Payload

```json
{
  "created_at": "2026-06-24T22:34:33.909091Z",
  "created_by": "https://api.calendly.com/users/00000000-0000-0000-0000-0000000000a1",
  "event": "invitee_no_show.created",
  "payload": {
    "cancel_url": "https://calendly.com/cancellations/00000000-0000-0000-0000-000000000001",
    "created_at": "2026-06-09T21:37:17.731525Z",
    "email": "john.doe@example.com",
    "event": "https://api.calendly.com/scheduled_events/00000000-0000-0000-0000-000000000002",
    "first_name": null,
    "invitee_scheduled_by": null,
    "last_name": null,
    "name": "John Doe",
    "new_invitee": null,
    "no_show": {
      "uri": "https://api.calendly.com/invitee_no_shows/00000000-0000-0000-0000-0000000000f1",
      "created_at": "2026-06-24T22:34:08.683444Z"
    },
    "old_invitee": null,
    "payment": null,
    "questions_and_answers": [],
    "reconfirmation": null,
    "reschedule_url": "https://calendly.com/reschedulings/00000000-0000-0000-0000-000000000001",
    "rescheduled": false,
    "routing_form_submission": null,
    "scheduled_event": {
      "created_at": "2026-06-09T21:37:17.707255Z",
      "end_time": "2026-06-10T07:00:00.000000Z",
      "event_guests": [],
      "event_memberships": [
        {
          "user": "https://api.calendly.com/users/00000000-0000-0000-0000-0000000000a1",
          "user_email": "support@example.com",
          "user_name": "Acme"
        }
      ],
      "event_type": "https://api.calendly.com/event_types/00000000-0000-0000-0000-0000000000e1",
      "invitees_counter": {
        "total": 1,
        "active": 1,
        "limit": 1
      },
      "location": {
        "data": {
          "id": 10000000001,
          "settings": {},
          "password": "••••••",
          "extra": null
        },
        "join_url": "https://us05web.zoom.us/j/10000000001?pwd=REDACTED",
        "status": "pushed",
        "type": "zoom"
      },
      "meeting_notes_html": null,
      "meeting_notes_plain": null,
      "name": "Acme - Intro",
      "start_time": "2026-06-10T06:30:00.000000Z",
      "status": "active",
      "updated_at": "2026-06-09T21:37:17.707255Z",
      "uri": "https://api.calendly.com/scheduled_events/00000000-0000-0000-0000-000000000002"
    },
    "scheduling_method": "api",
    "status": "active",
    "text_reminder_number": null,
    "timezone": "America/Los_Angeles",
    "tracking": {
      "utm_campaign": null,
      "utm_source": null,
      "utm_medium": null,
      "utm_content": null,
      "utm_term": null,
      "salesforce_uuid": null
    },
    "updated_at": "2026-06-09T21:37:17.731525Z",
    "uri": "https://api.calendly.com/scheduled_events/00000000-0000-0000-0000-000000000002/invitees/00000000-0000-0000-0000-000000000001"
  }
}
```