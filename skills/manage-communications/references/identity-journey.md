# Identity and journey

`Person` is canonical. `PlatformIdentity` is a person's provider profile,
`ConnectedAccount` is the user's authenticated account, and `ContactEndpoint`
is their exact pair. Equal names, avatars, or message text are not merge proof.

```text
hermes communication people search QUERY
hermes communication people show PERSON_ID
hermes communication people merge WINNER DUPLICATE --evidence TEXT
hermes communication people unmerge MERGE_AUDIT_ID
hermes communication timeline show PERSON_ID
```

Messages keep their provider/account provenance. A `CommunicationJourney`
links episodes without copying raw messages across namespaces. A person request
or a verified inbound message can resume a prior endpoint. Delivery failure
creates an issue; it never authorizes fallback to an old channel.
