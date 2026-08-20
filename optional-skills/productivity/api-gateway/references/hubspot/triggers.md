# HubSpot Trigger Reference

## Event Types

- [`contact.created`](#contactcreated)
- [`contact.updated`](#contactupdated)
- [`contact.deleted`](#contactdeleted)
- [`company.created`](#companycreated)
- [`company.updated`](#companyupdated)
- [`company.deleted`](#companydeleted)
- [`deal.created`](#dealcreated)
- [`deal.updated`](#dealupdated)
- [`deal.deleted`](#dealdeleted)
- [`deal.stage_changed`](#dealstage_changed)
- [`ticket.created`](#ticketcreated)
- [`ticket.updated`](#ticketupdated)
- [`ticket.deleted`](#ticketdeleted)

---

## `contact.created`

### Sample Payload

```json
{
  "attemptNumber": 0,
  "sourceId": "39522895",
  "eventId": 60775291,
  "changeSource": "INTEGRATION",
  "occurredAt": 1782371969536,
  "subscriptionType": "contact.creation",
  "portalId": 48975575,
  "appId": 39522895,
  "changeFlag": "CREATED",
  "subscriptionId": 6527595,
  "objectId": 508369434347
}
```

---

## `contact.updated`

### Parameters

- `property_name` (string, optional): Internal property name to match, such as `email`. Leave blank to match any property change.

### Sample Payload

```json
{
  "attemptNumber": 0,
  "sourceId": "39522895",
  "eventId": 2572854942,
  "changeSource": "INTEGRATION",
  "occurredAt": 1782371969536,
  "subscriptionType": "contact.propertyChange",
  "propertyName": "firstname",
  "portalId": 48975575,
  "appId": 39522895,
  "propertyValue": "Maton",
  "subscriptionId": 6527612,
  "objectId": 508369434347
}
```

---

## `contact.deleted`

### Sample Payload

```json
{
  "attemptNumber": 0,
  "sourceId": "39522895",
  "eventId": 1641846346,
  "changeSource": "INTEGRATION",
  "occurredAt": 1782371970538,
  "subscriptionType": "contact.deletion",
  "portalId": 48975575,
  "appId": 39522895,
  "changeFlag": "DELETED",
  "subscriptionId": 6547065,
  "objectId": 508369434347
}
```

---

## `company.created`

### Sample Payload

```json
{
  "attemptNumber": 0,
  "sourceId": "39522895",
  "eventId": 3425709231,
  "changeSource": "INTEGRATION",
  "occurredAt": 1782371970942,
  "subscriptionType": "company.creation",
  "portalId": 48975575,
  "appId": 39522895,
  "changeFlag": "CREATED",
  "subscriptionId": 6547117,
  "objectId": 330222354146
}
```

---

## `company.updated`

### Parameters

- `property_name` (string, optional): Internal property name to match, such as `name`. Leave blank to match any property change.

### Sample Payload

```json
{
  "attemptNumber": 0,
  "sourceId": "39522895",
  "eventId": 1764670272,
  "changeSource": "INTEGRATION",
  "occurredAt": 1782371971396,
  "subscriptionType": "company.propertyChange",
  "propertyName": "city",
  "portalId": 48975575,
  "appId": 39522895,
  "propertyValue": "San Francisco",
  "subscriptionId": 6547073,
  "objectId": 330222354146
}
```

---

## `company.deleted`

### Sample Payload

```json
{
  "attemptNumber": 0,
  "sourceId": "39522895",
  "eventId": 1434030517,
  "changeSource": "INTEGRATION",
  "occurredAt": 1782371971709,
  "subscriptionType": "company.deletion",
  "portalId": 48975575,
  "appId": 39522895,
  "changeFlag": "DELETED",
  "subscriptionId": 6547066,
  "objectId": 330222354146
}
```

---

## `deal.created`

### Sample Payload

```json
{
  "attemptNumber": 0,
  "sourceId": "39522895",
  "eventId": 731756599,
  "changeSource": "INTEGRATION",
  "occurredAt": 1782371971992,
  "subscriptionType": "deal.creation",
  "portalId": 48975575,
  "appId": 39522895,
  "changeFlag": "CREATED",
  "subscriptionId": 6547118,
  "objectId": 332507367156
}
```

---

## `deal.updated`

### Parameters

- `property_name` (string, optional): Internal property name to match, such as `amount`. Leave blank to match any property change.

### Sample Payload

```json
{
  "attemptNumber": 0,
  "sourceId": "39522895",
  "eventId": 2995724042,
  "changeSource": "INTEGRATION",
  "occurredAt": 1782371972438,
  "subscriptionType": "deal.propertyChange",
  "propertyName": "amount",
  "portalId": 48975575,
  "appId": 39522895,
  "propertyValue": "2000",
  "subscriptionId": 6547123,
  "objectId": 332507367156
}
```

---

## `deal.deleted`

### Sample Payload

```json
{
  "attemptNumber": 0,
  "sourceId": "39522895",
  "eventId": 3838838948,
  "changeSource": "INTEGRATION",
  "occurredAt": 1782371973155,
  "subscriptionType": "deal.deletion",
  "portalId": 48975575,
  "appId": 39522895,
  "changeFlag": "DELETED",
  "subscriptionId": 6547119,
  "objectId": 332507367156
}
```

---

## `deal.stage_changed`

### Sample Payload

```json
{
  "attemptNumber": 0,
  "sourceId": "39522895",
  "eventId": 1839643432,
  "changeSource": "INTEGRATION",
  "occurredAt": 1782371972793,
  "subscriptionType": "deal.propertyChange",
  "propertyName": "dealstage",
  "portalId": 48975575,
  "appId": 39522895,
  "propertyValue": "qualifiedtobuy",
  "subscriptionId": 6527619,
  "objectId": 332507367156
}
```

---

## `ticket.created`

### Sample Payload

```json
{
  "attemptNumber": 0,
  "sourceId": "39522895",
  "eventId": 627582561,
  "changeSource": "INTEGRATION",
  "occurredAt": 1782371973446,
  "subscriptionType": "ticket.creation",
  "portalId": 48975575,
  "appId": 39522895,
  "changeFlag": "CREATED",
  "subscriptionId": 6527637,
  "objectId": 318514647758
}
```

---

## `ticket.updated`

### Parameters

- `property_name` (string, optional): Internal property name to match, such as `subject`. Leave blank to match any property change.

### Sample Payload

```json
{
  "attemptNumber": 0,
  "sourceId": "39522895",
  "eventId": 1048958506,
  "changeSource": "INTEGRATION",
  "occurredAt": 1782371973873,
  "subscriptionType": "ticket.propertyChange",
  "propertyName": "content",
  "portalId": 48975575,
  "appId": 39522895,
  "propertyValue": "Updated by Maton test",
  "subscriptionId": 6547148,
  "objectId": 318514647758
}
```

---

## `ticket.deleted`

### Sample Payload

```json
{
  "attemptNumber": 0,
  "sourceId": "39522895",
  "eventId": 3923359118,
  "changeSource": "INTEGRATION",
  "occurredAt": 1782371974198,
  "subscriptionType": "ticket.deletion",
  "portalId": 48975575,
  "appId": 39522895,
  "changeFlag": "DELETED",
  "subscriptionId": 6547143,
  "objectId": 318514647758
}
```
