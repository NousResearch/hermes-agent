# Box CLI guide

Run Box commands through Hermes' `terminal` tool. Prefer the documented command in this skill over exploratory help calls. Use help only when a required option is absent here or the installed CLI rejects the syntax.

## Use one command runner

Resolve one command runner before any Box operation:

1. Check whether `box` already resolves in the runtime shell (`command -v box` on macOS/Linux or `Get-Command box` in PowerShell). If it does, use that command as-is, regardless of where Hermes or Box CLI was installed.
2. If it does not resolve, install and verify an isolated CLI under a writable, persistent Hermes runtime directory. Prefer the current Hermes home at `tools/box-cli`; `HERMES_HOME` is optional, and Hermes uses its platform default when it is unset (`~/.hermes` on macOS/Linux and `%LOCALAPPDATA%\hermes` on Windows).
3. If that directory is not writable, ask for a writable persistent directory in the runtime. Do not assume Hermes's source checkout, a global npm prefix, or a user home is writable. If a nonstandard existing CLI is not on `PATH`, ask for its executable path instead of scanning the machine.

Only use `npm exec --prefix` after Hermes installed and verified that exact local copy. Run each installation block below as one terminal call, record the verified absolute prefix it prints, and use that literal path in later calls. Never depend on a shell variable surviving a separate Hermes terminal call, and never give the user an unverified `npm exec --prefix` command to run.

Box CLI 4 requires Node.js 18 or newer. Before installing, run `node --version` and `npm --version` in the same runtime and shell Hermes will use. If Node is missing or its major version is below 18, ask for approval to install or activate a supported Node runtime using the environment's normal mechanism, then rerun both checks. If npm is unavailable or the filesystem is not writable, ask for the runtime-appropriate installation or writable Hermes home; do not assume a system package manager, a desktop, or elevated privileges.

On macOS/Linux:

```bash
node --version
npm --version
BOX_CLI_HOME="${HERMES_HOME:-$HOME/.hermes}/tools/box-cli"
mkdir -p "$BOX_CLI_HOME"
npm install --prefix "$BOX_CLI_HOME" @box/cli
npm exec --prefix "$BOX_CLI_HOME" -- box --version
cd "$BOX_CLI_HOME" && pwd -P
```

On Windows PowerShell:

```powershell
node --version
npm --version
$boxCliHome = Join-Path $(if ($env:HERMES_HOME) { $env:HERMES_HOME } else { Join-Path $env:LOCALAPPDATA "hermes" }) "tools\box-cli"
New-Item -ItemType Directory -Force -Path $boxCliHome | Out-Null
npm install --prefix $boxCliHome @box/cli
npm exec --prefix $boxCliHome -- box --version
Resolve-Path $boxCliHome
```

Keep the resolved runner for the whole task. When Hermes installed the local copy, replace the leading `box` in every example with the applicable `npm exec --prefix` runner below. Otherwise run the examples with the already-resolved `box` command.

The examples in other references use `bash` fences and POSIX `\` continuations. In PowerShell, keep the same Box arguments but run the command on one line or use PowerShell's backtick continuation. Use PowerShell variables only in PowerShell examples.

On macOS/Linux:

```bash
npm exec --prefix "<VERIFIED_ABSOLUTE_PREFIX>" -- box
```

On Windows PowerShell:

```powershell
npm exec --prefix "<VERIFIED_ABSOLUTE_PREFIX>" -- box
```

For example on macOS/Linux:

```bash
npm exec --prefix "<VERIFIED_ABSOLUTE_PREFIX>" -- box users:get me --json --fields id,name,login
```

Do not attempt a global npm install, use `sudo`, change npm's global prefix, or change `PATH`.

## Check identity and control output

On macOS/Linux:

```bash
command -v box
box --version
box users:get me --json --fields id,name,login
box folders:items 0 --json --max-items 20 --fields id,name,type
```

On Windows PowerShell:

```powershell
Get-Command box -ErrorAction SilentlyContinue
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

An isolated npm installation isolates the CLI executable, not its authenticated environments. Box CLI stores environments and tokens for the runtime's OS user, using the platform credential store when available and `~/.box` as a fallback. Hermes profiles and concurrent sessions running as the same OS user can therefore share the current Box environment. Warn about this shared state during setup, verify the actor before every task, and explain that changing the current environment can affect other Hermes sessions and ordinary Box CLI use under that OS account.

On Linux, Box CLI secure storage depends on Secret Service/libsecret support. If the CLI reports a plaintext fallback, warn that credentials may be stored in `~/.box/box_environments.json` and token-cache files. Do not read or print those files. Recommend configuring the runtime's supported Secret Service/libsecret package or using a properly isolated runtime user before production use; do not assume a package manager or require another confirmation merely to deliver the warning.

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
