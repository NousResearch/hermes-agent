# Subagent coordination prompt

Children inherit no reliable shell environment. Put board lineage and resource
scope directly in the delegated prompt, but let each child register its own id.

```text
You are a subagent of board session <PARENT_BOARD_ID>.

Before mutating anything:
1. Run `python3 <session_coord_path> status`.
2. Register your own child id:
   `CID=$(python3 <session_coord_path> register --task "<child task>" --surface subagent --parent <PARENT_BOARD_ID> --slot <a|b|c> | head -1)`
3. Atomically claim every resource you will mutate:
   `python3 <session_coord_path> claim --id "$CID" --res "<key>" --res "<key2>" --task "<child task>"`
4. Continue only after CLAIMED. If exit 75 / HELD / QUEUED, stop before mutation and report the holder to the parent.
5. At task end run `python3 <session_coord_path> done --id "$CID"` and report verification evidence.

Never rank yourself, preempt, or steal. Do not use the parent's board id as your
own id. Do not assume files you create in shared space are private.
```

Assign distinct slots in intended order (`a`, `b`, ...), and assign disjoint
resource keys. The parent must not hold keys children need to claim. It may claim
final merge/publication keys after children call `done`, and should stop/steer
children before the parent itself pauses.

## Optional native child continuation

Do not add `--yield`, native session ids, or wake targets to the prompt unless
`hermes_setup.py check` succeeds for the exact parent/profile path, the receiver
was started after configuration, and the runtime supplies trustworthy
identities. In native mode a yielded child process
is terminal; the exact parent receiver redispatches a fresh child from the saved
checkpoint. Never try to revive the old child process, never guess the parent's
session id, and never treat configuration files as proof that a resident process
loaded the plugin. See `references/automatic-resume.md`.
