---
name: google-people-api-sync
description: Synchronizing contacts to Google People API using a robust pattern to avoid library TypeErrors and authentication scope pitfalls.
---

# Google People API Sync

## Trigger conditions
- When you need to create, update, or search for contacts in Google Contacts via the People API.
- When the standard `googleapiclient` library fails with `TypeError: Got an unexpected keyword argument` during `searchContacts` calls.

## Procedure

### 1. Authentication
Ensure the OAuth token has the correct scope: `https://www.googleapis.com/auth/contacts`. If a `403 ACCESS_TOKEN_SCOPE_INSUFFICIENT` error occurs during `createContact` or `updateContact`, the user must re-authorize with this specific scope.

### 2. Search for Existing Contact
To avoid the common `TypeError` in the Python client library, use a direct REST call via `curl` or a raw HTTP request.
- **Endpoint:** `POST https://people.googleapis.com/v1/people:searchContacts`
- **Payload:** `{"queries": ["query=Name Of Person"]}`
- **Header:** `Authorization: Bearer <TOKEN>`

### 3. Upsert Logic
- **If contact exists:**
    1. Fetch the person's `etag` via `GET https://people.googleapis.com/v1/people/{personId}?personFields=names,emailAddresses,phoneNumbers`.
    2. Use `PATCH https://people.googleapis.com/v1/people/{personId}`.
    3. Include `updatePersonFields` in the query parameters (e.g., `updatePersonFields=names,emailAddresses,phoneNumbers`).
    4. Include the `etag` in the request body to prevent version conflicts.
- **If contact does not exist:**
    1. Use `POST https://people.googleapis.com/v1/people:createContact`.
    2. Provide the person object containing `names`, `emailAddresses`, and `phoneNumbers`.

## Pitfalls & Lessons
- **Library Instability:** The `google-api-python-client` often has mismatched keyword arguments for the People API's `searchContacts` method. Use direct REST calls to bypass this.
- **Scope Confusion:** `contacts.readonly` is insufficient for creating/updating. Ensure the full `contacts` scope is active.
- **ETag Requirement:** Updates to existing contacts will fail if the current `etag` is not provided in the request body.