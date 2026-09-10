# Herdr lessons for orchestration control loops

Herdr demonstrates why persistent, event-driven execution feels faster than
chat-driven coordination. Apply these lessons without confusing runtime state
with delivery evidence.

## What makes the runtime fast

- **Persistent server ownership:** terminals survive client disconnects, so a
  coordinator reconnects instead of recreating workers and context.
- **Stable pane and agent identity:** CLI/socket operations target returned IDs,
  not guessed labels.
- **Event-driven status:** integrations report `working`, `idle`, and `blocked`;
  plugins can react to transitions instead of repeatedly reading transcripts.
- **Direct agent transport:** prompts and reads go to the existing terminal
  session; they do not resume a large historical chat for every status request.
- **One small native runtime:** terminal rendering and multiplexing stay in one
  server process, reducing repeated process and UI startup.
- **Detach/reattach:** the human interface is disposable while work continues.

## What not to copy blindly

- `idle` or `done` is not a tested commit.
- A pane title is not process, repository, or writer identity.
- A successful prompt submit is not durable completion.
- Terminal persistence does not establish Git authority, test independence, or
  permission to push, merge, deploy, migrate, or delete.

## Recommended hybrid

Use a persistent runtime for fast execution and event notifications, then bind
it to an evidence contract:

1. dispatch one immutable task packet to one writer;
2. require a `working` acknowledgement;
3. receive state-change events without busy-polling;
4. require an atomic completion report tied to task ID and exact SHA;
5. verify Git and tests independently;
6. re-read upstream and PR state before any consequential action.

This preserves Herdr's speed while preventing the common failure where a fast
terminal reports completion but the wrong branch, stale feature, missing test,
or overlapping writer reaches the PR gate.
