# Zoho CRM (API v8)

Base: `<api_domain>/crm/v8`, where `api_domain` comes with the access token (`https://www.zohoapis.eu` for eu). The helper adds it: `zoho_api.py request <METHOD> crm <path>`.

## Read

```
request GET crm Leads --query fields=Last_Name,Company,Email,Lead_Status --query per_page=50
request GET crm Leads/<id>
request GET crm Contacts/search --query email=jane@example.com
request GET crm Contacts/search --query "criteria=(Last_Name:equals:Doe)"
request GET crm Accounts/search --query word=Example
```

- List calls without `fields` fail with `REQUIRED_PARAM_MISSING`.
- No hits: HTTP 204 and an empty body. Treat as "not found".
- Paging: `info.more_records` and `info.next_page_token` in the response. Pass `page_token=<token>` for the next page.
- `criteria` operators: `equals`, `starts_with`, `in`, `between`, `greater_than`, combined with `and` / `or`, for example `((Last_Name:equals:Doe)and(Company:starts_with:Ex))`.
- For reports and joins, COQL: `request POST crm coql --data '{"select_query":"select Last_Name, Email from Leads where Lead_Status = \u0027New\u0027 limit 10"}'`.
- Field API names differ from labels. List them once: `request GET crm settings/fields --query module=Leads` and use `api_name`.

## Write (ask the user first)

Bodies are `{"data": [ ... ]}` with up to 100 records. Check `data[i].status` and `data[i].code` per record.

Create:

```
request POST crm Leads --data '{"data":[{"Last_Name":"Doe","First_Name":"Jane","Company":"Example Ltd","Email":"jane@example.com"}]}'
```

Upsert (create or update by a unique field):

```
request POST crm Leads/upsert --data '{"data":[{"Last_Name":"Doe","Company":"Example Ltd","Email":"jane@example.com","Lead_Status":"Contacted"}],"duplicate_check_fields":["Email"]}'
```

Each record comes back with `"action": "insert"` or `"action": "update"`. Use upsert whenever a record might already exist; it prevents duplicates from repeated runs.

Update one record:

```
request PUT crm Leads/<id> --data '{"data":[{"Lead_Status":"Qualified"}]}'
```

Note on a record:

```
request POST crm Leads/<id>/Notes --data '{"data":[{"Note_Title":"Call","Note_Content":"Asked for an offer."}]}'
```

Tags (names with spaces must be URL encoded by the query, the helper does that):

```
request POST crm Contacts/actions/add_tags --query ids=<id1>,<id2> --query "tag_names=Event 2026"
```

## Order and links

- Create the Account first, then the Contact with `"Account_Name": {"id": "<account id>"}` (or `{"name": "..."}`). Without an existing Account, Zoho creates an empty stub and the address data is lost.
- Deals need `Deal_Name`, `Stage` and usually `Account_Name`; `Contact_Name` links a person.
- Mandatory fields depend on the org's layout. On `MANDATORY_NOT_FOUND`, read `details.api_name` and ask the user for the value.

## Limits

- API credits per day depend on the edition. Bulk work: batch up to 100 records per call and pause briefly between calls.
- `DUPLICATE_DATA` on create means a unique field already exists; switch to upsert or update.
