# Groups, segments, and greetings

Explicit groups have stable membership. Smart segments are computed at preview
time and explain every match; never treat preview membership as approval.

```text
hermes communication groups list
hermes communication groups show GROUP_ID
hermes communication groups create NAME [--exclude]
hermes communication groups preview GROUP_ID
hermes communication greetings plan [--date YYYY-MM-DD]
hermes communication greetings list [--date YYYY-MM-DD]
```

Greetings are keyed by canonical person, event, and local date so one person is
not duplicated across platforms. Use the person's timezone. Exclusion groups
produce an `excluded` plan record and no draft. Planning is read/draft work; it
does not send.
