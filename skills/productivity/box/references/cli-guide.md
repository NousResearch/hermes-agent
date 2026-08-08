# Box CLI guide

Run Box commands through Hermes' `terminal` tool. Prefer the documented command in this skill over exploratory help calls. Use help only when a required option is absent here or the installed CLI rejects the syntax.

## Use one command runner

If `box` is on `PATH`, run the examples in this guide as written. If it is missing, install the CLI once under the current Hermes home at `tools/box-cli`. `HERMES_HOME` is optional: Hermes uses its platform default when it is unset (`~/.hermes` on macOS/Linux and `%LOCALAPPDATA%\hermes` on Windows). Choose the command for the current shell.

Require Node.js and npm in the runtime where Hermes executes commands. If they are unavailable or the filesystem is not writable, ask for the runtime-appropriate installation or writable Hermes home; do not assume a system package manager, a desktop, or elevated privileges.

On macOS/Linux:

```bash
BOX_CLI_HOME="${HERMES_HOME:-$HOME/.hermes}/tools/box-cli"
npm install --prefix "$BOX_CLI_HOME" @box/cli
```

On Windows PowerShell:

```powershell
$boxCliHome = Join-Path $(if ($env:HERMES_HOME) { $env:HERMES_HOME } else { Join-Path $env:LOCALAPPDATA "hermes" }) "tools\box-cli"
npm install --prefix $boxCliHome @box/cli
```

Keep the resolved directory for the whole task. Then run every example by replacing its leading `box` with the runner for the current shell.

On macOS/Linux:

```bash
npm exec --prefix "$BOX_CLI_HOME" -- box
```

On Windows PowerShell:

```powershell
npm exec --prefix $boxCliHome -- box
```

For example on macOS/Linux:

```bash
npm exec --prefix "$BOX_CLI_HOME" -- \
  box users:get me --json --fields id,name,login
```

Do not attempt a global npm install, use `sudo`, change npm's global prefix, or change `PATH`.

## Check identity and control output

```bash
command -v box
box --version
box users:get me --json --fields id,name,login
box folders:items 0 --json --max-items 20 --fields id,name,type
```

Use `--json` for machine-readable output and `--fields` to return only needed fields. Folder `0` is the current actor's root, not a complete access inventory: do not use its listing to reject a shared file or folder, and never use it to discover Box Hubs.

## Environments and actors

```bash
box configure:environments:list
box configure:environments:set-current <ENVIRONMENT_NAME>
box users:get me --json --fields id,name,login
```

The CLI has one current environment. Confirm before switching it, then verify the actor. Perform ordinary Hermes work as the OAuth identity selected for that environment; do not impersonate another user.

## Pagination and search

```bash
box folders:items <FOLDER_ID> --json --max-items 100 --fields id,name,type
box search "quarterly review" --json --limit 20 --fields id,name,type,parent
box metadata-query enterprise_12345.contractTemplate <ANCESTOR_FOLDER_ID> \
  --query "status = :status" --query-param status=active --json
```

Paginate inventories fully before bulk work. Metadata queries require the template scope/key and an ancestor folder ID.

## REST escape hatch

When the CLI has no dedicated command, preserve its configured auth with `box request` and perform the ordinary requested operation. Do not stop to ask simply because this uses REST; read [REST API fallback](rest-api.md) for endpoint-specific bodies and headers.

```bash
box request /files/<FILE_ID> --json
box request /files/<FILE_ID> -X PUT --body '{"name":"renamed.pdf"}' --json
box request /folders -X POST --body '{"name":"New folder","parent":{"id":"0"}}' --json
```

Use `box request` as the CLI-based REST fallback. Use an SDK or raw HTTP only when the CLI is unavailable or application code genuinely needs direct REST.

## Batch inputs and mutations

Many Box CLI commands accept `--bulk-file-path` for CSV or JSON input. Use it only after inventorying the target set and confirming material writes. For ordered moves, version updates, and other recoverable mutations, keep an operation log and process serially. Use bounded concurrency in application SDK code only when its retry and rate-limit behavior is explicit.

## Confirmation rules

- Confirm before deletes, access changes, identity changes, broad moves, or an ambiguous target.
- Confirm the scope before an AI-unit-consuming bulk request.
- Do not pass `--yes` unless the user has already approved the exact operation.
