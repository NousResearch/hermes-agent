# Capability map

Baseline: Himalaya v2.1.0, official default-feature Linux release. This map covers all callable leaf commands in that release; aliases and per-command options are discovered through `--help`. It is a routing aid, not a promise that a configured backend/server supports every operation.

Contents: choose shared or native; command-family map; integration utilities; later development features.

## Choose the appropriate surface

| Surface | Reach for it when |
| --- | --- |
| Shared mailbox/envelope/flag/message/attachment | Ordinary cross-backend mail workflows. Use the mail-operations and composition references for verified syntax. |
| IMAP | Mailbox lifecycle/subscriptions/status; native search/sort/thread; flag storage; UID fetch/append/copy/move; ID negotiation; expunge and raw protocol access. Match UID versus sequence-number semantics and server capabilities to command help. |
| JMAP | Structured email/mailbox queries, patches and destruction; import/export/parse; thread/identity/submission resources; vacation responses; raw JMAP method requests through `query`. Submission creation can send mail. |
| Gmail | Gmail search/labels, native drafts, batch label changes, trash/untrash versus permanent delete, threads, history, attachments, forwarding/filter/delegate/send-as/vacation settings. Native draft resources differ from merely appending to a mailbox. |
| Microsoft Graph | Profile, folder hierarchy/lifecycle, message CRUD/send/copy/move, attachment resources. Inspect native query/pagination flags and API permissions. |
| Maildir | Local mailbox create/rename/delete/list, message save/copy/move and flag operations. |
| m2dir | Local mailbox create/delete/list, message save and flag operations. Do not assume the richer Maildir native surface exists here. |
| pimdir | Shared offline browsing and staged replica mutations, selected with `--backend pimdir`. V2.1 has no top-level `pimdir` command. Synchronization is performed by Neverest, not Himalaya. |
| SMTP | Native submission or raw SMTP access. Raw protocol access can send mail or change state; it is not an authorization bypass. |

In v2.1.0, Gmail REST and Graph do not implement shared `envelope search` or shared `message add`. Use native search and draft creation. Gmail-specific operational recipes are in [gmail-workflows.md](gmail-workflows.md); Hotmail/Outlook recipes and Graph pagination limits are in [msgraph-workflows.md](msgraph-workflows.md).

Native queries are not interchangeable: the shared DSL, IMAP search keys, Gmail search syntax, JMAP filters and Graph query options differ. Inspect help for the selected native operation and preserve its pagination tokens/state where applicable. History/delta-like resources are not a generic local sync command.

## Complete v2.1 command-family map

Prefix each family with `himalaya`; run `himalaya FAMILY ACTION --help` for its exact options and arguments. The table groups **leaf commands**, so nested resource names are expanded into their own rows. Global switches include `--config`, `--account`, `--backend`, `--json`, `--log-level`, `--log-file`, `--help` and `--version`.

| Command family | Leaf actions |
| --- | --- |
| `mailbox` | `list` |
| `envelope` | `list`, `search` |
| `flag` | `add`, `set`, `remove` |
| `message` | `add`, `compose`, `copy`, `delete`, `forward`, `move`, `read`, `reply`, `send` |
| `attachment` | `list`, `download` |
| `imap` | `id`, `select`, `create`, `delete`, `rename`, `subscribe`, `unsubscribe`, `list`, `status`, `close`, `unselect`, `expunge`, `search`, `sort`, `thread`, `store`, `flags`, `fetch`, `append`, `copy`, `move`, `raw` |
| `jmap` | `query` |
| `jmap mailbox` | `get`, `query`, `create`, `update`, `destroy` |
| `jmap email` | `get`, `query`, `read`, `update`, `delete`, `copy`, `export`, `import`, `parse` |
| `jmap thread` | `get` |
| `jmap identity` | `get`, `create`, `update`, `delete` |
| `jmap submission` | `get`, `query`, `create`, `cancel` |
| `jmap vacation-response` | `get`, `set` |
| `gmail profile` | `get` |
| `gmail labels` | `list`, `get`, `create`, `update`, `delete` |
| `gmail messages` | `list`, `get`, `send`, `import`, `insert`, `modify`, `trash`, `untrash`, `delete`, `batch-modify`, `batch-delete` |
| `gmail attachments` | `get` |
| `gmail drafts` | `list`, `get`, `create`, `update`, `send`, `delete` |
| `gmail threads` | `list`, `get`, `modify`, `trash`, `untrash`, `delete` |
| `gmail history` | `list` |
| `gmail settings vacation` | `get`, `set` |
| `gmail settings imap` | `get`, `set` |
| `gmail settings pop` | `get`, `set` |
| `gmail settings language` | `get`, `set` |
| `gmail settings auto-forwarding` | `get`, `set` |
| `gmail settings filters` | `list`, `get`, `create`, `delete` |
| `gmail settings forwarding-addresses` | `list`, `get`, `create`, `delete` |
| `gmail settings delegates` | `list`, `get`, `create`, `delete` |
| `gmail settings send-as` | `list`, `get`, `create`, `update`, `delete`, `verify` |
| `msgraph profile` | `get` |
| `msgraph mail-folder` | `list`, `child-folders`, `get`, `create`, `rename`, `copy`, `move`, `delete` |
| `msgraph message` | `list`, `get`, `create`, `update`, `send`, `copy`, `move`, `delete` |
| `msgraph attachment` | `list`, `get`, `create`, `delete` |
| `maildir` | `create`, `rename`, `delete`, `list` |
| `maildir messages` | `save`, `copy`, `move` |
| `maildir flags` | `list`, `add`, `set`, `remove` |
| `m2dir` | `create`, `delete`, `list` |
| `m2dir messages` | `save` |
| `m2dir flags` | `list`, `add`, `set`, `remove` |
| `smtp` | `send`, `raw` |
| `account` | `list`, `check` |

Standalone top-level commands: `configure`, `completion`, `manual`, `json-schema`, and generated `help`. `configure` changes local account configuration; the others generate help/integration artifacts. The installed help tree is authoritative for feature-reduced builds and later versions.

Gmail forwarding, delegates and send-as changes affect access or future delivery. Vacation responses and Sieve rules can generate future messages. Require user authorization for that actual effect, even though these are called “settings.” Native mailbox/message destruction and batch-delete can be permanent; establish the requested scope before execution.

## Integration utilities and external companions

- `himalaya json-schema ./schemas`: generate command-specific JSON output schemas. Use a fresh directory because existing generated files are overwritten.
- `himalaya manual ./man`: generate manual pages; existing pages may be overwritten.
- `himalaya completion bash > himalaya.bash`: generate shell completion; bash, elvish, fish, powershell and zsh are supported in this release.
- MML: external rich MIME composition/interpretation, editor integration and signing/encryption; check its own capabilities separately.
- Ortie or another token broker: external OAuth acquisition/refresh.
- Sirup: external pre-authenticated IMAP/SMTP session reuse over a Unix socket.
- Neverest: external synchronization that can populate pimdir. A local body may be missing even though its envelope is available.

No native v2 `template`, `folder`, Notmuch, Sendmail, arbitrary mailto handler, calendar or contact-management capability is implied by this skill.

## Development branch: discover, do not assume

At commit `eaada0f940b1ccd85888c33e3e38a51b67eb1874` (checked 2026-09-08), upstream master additionally exposes:

| Feature | Discovery path | Boundary |
| --- | --- | --- |
| ManageSieve | `himalaya sieve --help`: capability, list, get, put, check, rename, delete, activate, deactivate, raw | Uses a separate Sieve account block/build feature. Validate a candidate script with `check` before an authorized upload/activation. Activation affects future server mail processing. |
| pimdir outgoing queue | `himalaya pimdir queue --help`: list, cancel | Inspect queue semantics and exact message IDs before cancellation; this is not present as a native namespace in v2.1.0. |

Master also advertises SOCKS5/HTTP proxy support through environment variables. Confirm the installed build's documentation before configuring transport behavior. Do not present master-only features as released v2.1 capabilities or install a development build without a task-specific reason.

Sources: [v2.1 command root](https://github.com/pimalaya/himalaya/blob/v2.1.0/src/cli.rs), [v2.1 build features](https://github.com/pimalaya/himalaya/blob/v2.1.0/Cargo.toml), [v2.1 native source tree](https://github.com/pimalaya/himalaya/tree/v2.1.0/src), [pinned development Sieve](https://github.com/pimalaya/himalaya/blob/eaada0f940b1ccd85888c33e3e38a51b67eb1874/src/sieve/cli.rs), [pinned development queue](https://github.com/pimalaya/himalaya/blob/eaada0f940b1ccd85888c33e3e38a51b67eb1874/src/pimdir/queue/cli.rs).
