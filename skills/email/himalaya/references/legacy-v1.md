# Legacy v1 and migration

Use this reference only when `himalaya --version` reports v1.x. The uploaded skill primarily targeted this interface but mixed in incompatible syntax.

## Keep the interfaces separate

| Operation | v1.2-era interface | v2.1.0 interface |
| --- | --- | --- |
| JSON | `--output json` | `--json` |
| Mailboxes | `folder list` | `mailbox list` |
| Source selection | `--folder` | `--mailbox`; move/copy use `--from` |
| Search | `envelope list QUERY` | `envelope search QUERY` |
| Add seen | `flag add 42 seen` | `flag add --flag seen 42` |
| Move | `message move Archive 42` | `message move --to Archive 42` |
| MIME export | `message export 42 --full` | `message read --raw 42` |
| Save raw mail | `message save` | `message add` (`save` alias retained) |
| Compose | `message write`, interactive | `message compose`, noninteractive (`write` alias retained) |
| MML/templates | `template write/reply/forward/save/send` | Removed; external composer produces MIME |
| Attachment output directory | `--downloads-dir` | `--dir` |
| Setup/check | `account configure`, `account doctor` | `configure`, `account check` |
| Native Notmuch/Sendmail | Available in feature-enabled v1 builds | Removed |
| OAuth/keyring | In-process v1 configuration | External secret/token commands |

Verify the exact v1 command help before execution. In particular, v1 account selection is not uniformly a top-level flag; place it on the command that advertises it, for example `himalaya envelope list --account work`.

V1.2 flags use positional IDs and flag names; the uploaded `flag add 42 --flag seen` is not v1.2 syntax. Search examples should include explicit Boolean operators: `envelope list from alice@example.org and subject meeting`.

For v1 draft/reply handling, generate the template to a file, edit deliberately, then invoke the requested template save/send operation separately. Avoid pipes that transform arbitrary blank lines and immediately send. Reading can affect seen state on older versions; check `message read --help` for `--preview` when preserving unread status matters.

Older installations may additionally offer envelope threading, envelope watch, folder add/delete/purge/expunge, flag set, message edit, template generation/saving and feature-dependent MML/PGP support. Discover these through their local subcommand help rather than applying v2 native equivalents.

## Migration

Do not automatically upgrade an existing v1 workflow. When migration is requested:

1. Preserve the original config and create a separate v2 file from the [v2.1 sample](https://github.com/pimalaya/himalaya/blob/v2.1.0/config.sample.toml).
2. Translate the entire backend/authentication structure: `backend.*` → `imap.*` (or the selected backend), `message.send.backend.*` → `smtp.*`, and secrets `.cmd` → `.command`. Configure an external OAuth broker if needed.
3. V1 folder aliases become v2 `mailbox.alias`. The original skill's “always use plural folder.aliases” advice must stay version-scoped; it is wrong for v2. Match the installed v1 schema before modifying an older config.
4. Update scripts for command/argument changes, output schemas, compose behavior, read state and send/save ordering.
5. Validate the new account with an explicit config path and inspect mailbox listings before making it the default.

The [migration guide](https://github.com/pimalaya/himalaya/blob/v2.1.0/MIGRATION.md) is useful for the v2.0 refactor, but some statements are superseded by the [v2.1 release notes](https://github.com/pimalaya/himalaya/releases/tag/v2.1.0) and source. V2.1 restored shared delete, email/display-name, signatures and readable message output.

Reference for v1 positional flags: [v1.2 parser](https://github.com/pimalaya/himalaya/blob/v1.2.0/src/email/envelope/flag/arg/ids_and_flags.rs).

