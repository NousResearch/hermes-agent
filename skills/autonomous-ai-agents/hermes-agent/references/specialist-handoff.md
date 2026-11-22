# Give a specialist its first task from chat

Start with the outcome, not a profile wizard. A one-off lookup needs no new Bot. If the user wants an ongoing specialist, check the roster (`hermes profile list` through `terminal`) and reuse a suitable Bot rather than silently creating a second owner. Propose a new Bot if that would help; don't create one from a passing remark.

**Clarify only what changes the work.** Use the current conversation first. For “Make a release watcher; tell me only if a release slips,” if two projects are in scope, ask “Which project should it watch?” Don't ask for a name or repeat the alert condition. If the destination is a shared room and the source is private, clarify the audience before passing anything. Don't include source material in that question.

**Create the requested role.** For a fresh local profile, the native path is `hermes profile create release-watch --description "Checks release dates and reports slips" --no-credentials --no-alias` through `terminal`. Read back its role with `hermes profile describe release-watch`. Don't clone another Bot's accounts, memory, or channels to carry context. `--no-credentials` skips copying credentials and the model; it is not a security sandbox. Default provider logins can still be readable. Verify the intended model and source access before calling it ready; use a separate trust boundary if the task requires one.

**Make one real handoff.** In a Bot Chat with `message_agent`, give the specialist only the goal, the authorized source or facts, and the decision you need back. For example:

```text
target: release-watch
message: Check the public release page for the May milestone. The announced date was May 12.
constraints: Read only that page. No posts or changes.
expecting: The current date and its source, or the reason you couldn't verify it.
```

A queued delivery is not an answer. The recipient returns its answer as its final response; it doesn't message the sender back to acknowledge. Use the completion for that delivery to answer the user. If the route isn't available in this conversation, say so; don't relay through an unrelated account or claim the Bots spoke.

**Add the routine separately.** A successful first exchange isn't a running watch. Have one owner propose the job through `cronjob_manage` where a schedule card is available; don't bypass the card through the CLI. Specify the source, who sees the result, when to speak (a verified date change), and when to stop (release shipped). Use a change monitor when possible. Unchanged means silence; a failed read is not evidence that nothing changed. Where the user wants a conversation, use a supported single-chat delivery with `attach_to_session` rather than a broadcast. Verify the saved job after approval; don't describe a proposed, declined or unverified job as active.
