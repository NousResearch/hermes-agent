# Learn Mode

Learn is an opt-in workflow discovery surface for the Desktop app and CLI.
It identifies repeated local workflow patterns and turns them into pending
automation suggestions. It does not schedule, enable, or run learned workflows
without explicit approval.

## Controls

Desktop users can open **Skills & Tools > Learn** and use:

- **Start Learn** - starts profile-local metadata collection.
- **Pause Learn** - pauses collection without deleting configuration.
- **Resume Learn** - resumes collection.
- **Stop Learn** - stops runtime collection.
- **Review suggestions** - analyzes collected metadata and creates pending
  `cron.suggestions` records with `source="usage"`.
- **Delete data** - deletes Learn-collected artifacts for the active profile.

The same controls are available from the CLI and slash command surfaces:

```text
hermes learn status
hermes learn start [learn]
hermes learn pause
hermes learn resume
hermes learn stop
hermes learn review
hermes learn delete-data
hermes learn config --allowlist code.exe chrome.exe --denylist slack.exe --retention-days 14
```

```text
/learn status
/learn start
/learn review
/suggestions
```

## Data Boundary

Learn data is stored under the active Hermes profile returned by
`get_hermes_home()`, in the profile-local `learn/` directory. Default collection
is metadata-only:

- foreground process name
- redacted window title
- redacted domain only, not full URLs
- coarse category
- timestamp
- idle state and idle seconds
- duration seconds

Learn does not collect keystrokes, clipboard contents, screenshots, document or
message bodies, passwords, MFA codes, payment data, cookies, browser profiles,
or full URL query strings by default.

## Suggestions

The analyzer aggregates repeated metadata categories and creates conservative
read/prepare jobs such as follow-up summaries, development summaries, research
packets, and checklist reviews. These are added as pending cron suggestions.
The user must accept a suggestion through `/suggestions` or the suggestions
command before Hermes creates a cron job.

Ask-first memory drafting, auto-draft skill creation, and explicit Teach Mode
workflow capture are product directions for later iterations. This MVP ships
only opt-in metadata collection plus approval-gated automation suggestions.
