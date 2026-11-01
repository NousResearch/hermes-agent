# Linear Trigger Reference

## Event Types

- [`issue.created`](#issuecreated)
- [`issue.status_changed`](#issuestatus_changed)
- [`comment.created`](#commentcreated)

---

## `issue.created`

### Parameters

- `team_id` (string, optional): Team ID to scope to a specific team. Leave blank to match all teams.

### Sample Payload

```json
{
  "actor": {
    "name": "Jane Doe",
    "id": "00000000-0000-0000-0000-0000000000b2",
    "type": "user",
    "email": "jane@example.com",
    "url": "https://linear.app/acme/profiles/jane"
  },
  "organizationId": "00000000-0000-0000-0000-0000000000b1",
  "createdAt": "2026-06-20T00:24:05.106Z",
  "webhookTimestamp": 1781915045396,
  "webhookId": "25ec010c-3128-45e6-8cb8-c6fabc494c78",
  "data": {
    "creatorId": "00000000-0000-0000-0000-0000000000b2",
    "description": "",
    "title": "test issue",
    "createdAt": "2026-06-20T00:24:05.106Z",
    "number": 580,
    "id": "b451181e-5bd9-4be1-b803-72c1ac662295",
    "state": {
      "name": "Todo",
      "id": "1ab9475f-eb91-4207-a5a3-1176e38b85be",
      "color": "#e2e2e2",
      "type": "unstarted"
    },
    "updatedAt": "2026-06-20T00:24:05.106Z",
    "identifier": "ENG-580",
    "priorityLabel": "No priority",
    "stateId": "1ab9475f-eb91-4207-a5a3-1176e38b85be",
    "team": {
      "name": "Engineering",
      "key": "ENG",
      "id": "70c49a0d-6973-4563-a743-8504f1a5171b"
    },
    "priority": 0,
    "url": "https://linear.app/acme/issue/ENG-580/test-issue",
    "teamId": "70c49a0d-6973-4563-a743-8504f1a5171b"
  },
  "action": "create",
  "type": "Issue",
  "url": "https://linear.app/acme/issue/ENG-580/test-issue"
}
```

---

## `issue.status_changed`

### Parameters

- `new_state` (string, optional): Filter to a specific target status name (e.g. `Done`, `In Review`). Leave blank to match any status change.

### Sample Payload

```json
{
  "actor": {
    "name": "John Doe",
    "id": "00000000-0000-0000-0000-0000000000b2",
    "type": "user",
    "email": "john.doe@example.com",
    "url": "https://linear.app/acme/profiles/john.doe"
  },
  "organizationId": "00000000-0000-0000-0000-0000000000b1",
  "createdAt": "2026-06-24T22:01:29.377Z",
  "webhookTimestamp": 1782338489808,
  "webhookId": "b229420a-a5bf-4234-8867-62b70c82ac46",
  "data": {
    "reactionData": [],
    "creatorId": "00000000-0000-0000-0000-0000000000b3",
    "description": "Example issue description",
    "title": "Example issue",
    "createdAt": "2026-03-04T01:11:59.954Z",
    "number": 564,
    "labelIds": [],
    "prioritySortOrder": -139227.99,
    "id": "00000000-0000-0000-0000-0000000000b4",
    "state": {
      "name": "Backlog",
      "id": "f21dfa65-7951-4742-a202-00ceb0ff6e9f",
      "color": "#bec2c8",
      "type": "backlog"
    },
    "updatedAt": "2026-06-24T22:01:29.376Z",
    "identifier": "ENG-564",
    "previousIdentifiers": [],
    "priorityLabel": "No priority",
    "stateId": "f21dfa65-7951-4742-a202-00ceb0ff6e9f",
    "team": {
      "name": "Engineering",
      "key": "ENG",
      "id": "70c49a0d-6973-4563-a743-8504f1a5171b"
    },
    "priority": 0,
    "inheritsSharedAccess": false,
    "url": "https://linear.app/acme/issue/ENG-564/example-issue",
    "labels": [],
    "subscriberIds": [
      "00000000-0000-0000-0000-0000000000b3"
    ],
    "slaType": "all",
    "addedToTeamAt": "2026-03-04T01:11:59.981Z",
    "sortOrder": -72957.76,
    "teamId": "70c49a0d-6973-4563-a743-8504f1a5171b",
    "descriptionData": "{\"type\":\"doc\",\"content\":[{\"type\":\"paragraph\",\"content\":[{\"type\":\"text\",\"text\":\"Example issue description\"}]}]}"
  },
  "updatedFrom": {
    "stateId": "1ab9475f-eb91-4207-a5a3-1176e38b85be",
    "sortOrder": -139130,
    "updatedAt": "2026-03-04T01:11:59.954Z"
  },
  "action": "update",
  "type": "Issue",
  "url": "https://linear.app/acme/issue/ENG-564/example-issue"
}
```

---

## `comment.created`

### Sample Payload

```json
{
  "actor": {
    "name": "John Doe",
    "id": "00000000-0000-0000-0000-0000000000b2",
    "type": "user",
    "email": "john.doe@example.com",
    "url": "https://linear.app/acme/profiles/john.doe"
  },
  "organizationId": "00000000-0000-0000-0000-0000000000b1",
  "createdAt": "2026-06-24T22:05:23.926Z",
  "webhookTimestamp": 1782338725257,
  "webhookId": "b229420a-a5bf-4234-8867-62b70c82ac46",
  "data": {
    "createdAt": "2026-06-24T22:05:23.926Z",
    "issueId": "00000000-0000-0000-0000-0000000000b4",
    "reactionData": [],
    "issue": {
      "identifier": "ENG-564",
      "id": "00000000-0000-0000-0000-0000000000b4",
      "team": {
        "name": "Engineering",
        "key": "ENG",
        "id": "70c49a0d-6973-4563-a743-8504f1a5171b"
      },
      "title": "Example issue",
      "url": "https://linear.app/acme/issue/ENG-564/example-issue",
      "teamId": "70c49a0d-6973-4563-a743-8504f1a5171b"
    },
    "isArtificialAgentSessionRoot": false,
    "id": "0603e91b-2568-4766-bdba-53248a50b9b3",
    "body": "test",
    "userId": "00000000-0000-0000-0000-0000000000b2",
    "user": {
      "name": "John Doe",
      "id": "00000000-0000-0000-0000-0000000000b2",
      "email": "john.doe@example.com",
      "url": "https://linear.app/acme/profiles/john.doe"
    },
    "updatedAt": "2026-06-24T22:05:23.908Z"
  },
  "action": "create",
  "type": "Comment",
  "url": "https://linear.app/acme/issue/ENG-564/example-issue#comment-0603e91b"
}
```