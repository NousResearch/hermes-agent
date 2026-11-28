# Native session library

`hermes-library` owns the desktop/gateway conversation catalog. It uses native
Rust filesystem and SQLite libraries and does not invoke the Python Hermes CLI.

```rust,no_run
use hermes_library::{HermesHistory, Library};
use std::path::Path;

let library = Library::open_default()?;
let project = library.add_project(Path::new("/chosen/workspace"))?;
library.select_project(Some(&project.id))?;
let chat = library.create_session(Some(&project.id), None)?;
let workspace = library.workspace_for_session(&chat.id)?.unwrap();
let mut session = hermes_engine::Session::open_in(&chat.directory, &workspace)?;
// Drive session.turn(...), then refresh the sidebar's title and activity:
library.touch_session(&chat.id)?;

// Only after the user chooses an existing database and a listed conversation:
let history = HermesHistory::open(Path::new("/chosen/hermes/state.db"))?;
let available = history.sessions()?;
if let Some(chosen) = available.first() {
    let result = library.import_hermes(&history, &chosen.id, Some(&project.id))?;
    // Display result.warnings and switch to result.session.
}
# Ok::<(), anyhow::Error>(())
```

## Storage and concurrency

The default root is `hermes_engine::Session::default_dir()` and can be overridden
with `HERMES_RUST_DATA_DIR`. Tests pass private temporary directories explicitly.

```
library.json                   # versioned catalog, projects and selection
library.lock                   # brief cross-process catalog lock
sessions/<native-id>/
  conversation.json            # engine transcript, atomically saved
  session.lock                 # engine's exclusive conversation lock
  hermes-source.json           # imported upstream rows, when applicable
```

The catalog uses same-directory atomic replacement plus file and directory
syncing. Projects are existing canonical folders; imported upstream `cwd` is
provenance and never automatically grants execution access. A selected native
conversation supplies its own workspace, so desktop and gateway share history
without depending on process cwd. The catalog lock is never kept for a model
turn. The engine separately rejects two owners opening the same conversation.

`set_archived(id, bool)` soft-hides or restores a conversation without opening
its engine or changing its transcript. Manual archive survives subsequent opens;
restoration is explicit. `set_pinned(id, bool)` persists append-order pins without
changing conversation activity time. Archived pins stay out of the active list
and retain their pin when restored.

`sessions(project_id)` lists active conversations. `list_sessions(&SessionQuery)`
combines a workspace, `ArchiveFilter::{Active, Archived, All}`, an optional pin
filter, and case-insensitive title/native-ID/imported-Hermes-ID search. Pinned
sessions appear first in durable pin order, followed by recent conversations.
`session(id)` remains available for direct lookup of an archived conversation.
Version-one catalogs written before these fields existed upgrade additively.

An original one-conversation Rust `conversation.json` at the library root is
copied into the catalog once. Its source file is retained unchanged.

## Manual Hermes import

The compatibility reader follows upstream `persistence/sqlite/_schema_ddl.py`
and `persistence/sqlite/conversation/_store/messages.py`. It validates real
`sessions` and `messages` tables, reads optional newer columns when present,
decodes `\0json:` content and JSON tool calls, and combines compression ancestry
and its preferred continuation without merging explicitly marked branches,
delegates or resets. It deduplicates copied display/message identities.

SQLite never opens the original database. Read-only file handles copy the main
database and its optional WAL into a private temporary directory. Source size,
mtime and inode are checked across the copy; changing sources are retried three
times. Active rollback journals are rejected. SQLite opens only the disposable
snapshot in read-only/query-only mode with trusted schema disabled. The source
database, WAL, and shared-memory file are not written or migrated.

The native copy supports text conversation and recorded tool transcripts.
Every selected lineage row, including inactive messages, multimodal JSON,
reasoning, and unknown columns, is retained in `hermes-source.json`. BLOBs use a
typed hexadecimal wrapper. Import does not install upstream system prompts or
provider settings, resolve attachments, or execute historical tools. Historical
tool entries use `approved=false` because native approval cannot be inferred
from an upstream transcript. The report explains text-only conversion and other
limitations; callers should display it.

Limits are 4 GiB of source database/WAL, 100,000 listed sessions / 64 MiB of
picker metadata, and 20,000
messages / 64 MiB of raw transcript JSON per imported conversation. Missing or
unsupported sources fail without creating or modifying them. Reimporting an
already copied conversation returns the existing native copy; later upstream
changes do not overwrite native conversation progress.

## Intentional follow-up scope

- Media attachments are preserved as upstream row references, not copied or
  rendered as native image/audio messages.
- Imported sessions are snapshots; there is no bidirectional SQLite sync or
  automatic import of a profile.
- Legacy unmarked compression forks and advanced upstream display/checkpoint
  reconstruction can differ from upstream's full presentation layer. The raw
  archive preserves the selected lineage for inspection.
- Catalog mutations do not include chat deletion, project
  removal, or project moves. These can be added without rewriting the engine.

Run `./work/cargo-local test -p hermes-library --offline` from the enclosing
task workspace. Tests use synthetic databases only, including a live WAL fixture
and comparisons proving source bytes and mtimes remain unchanged.
