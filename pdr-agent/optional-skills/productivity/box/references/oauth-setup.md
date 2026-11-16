# OAuth setup

Use OAuth for every Hermes-to-Box connection. OAuth follows the signed-in Box user's permissions and the app's scopes; it does not grant enterprise-wide access.

## Choose the OAuth account

Authorize the Box account that Hermes should act as. OAuth follows that account's permissions. If the user wants a narrower permission boundary, authorize an account that is invited only to the files, folders, or Hubs Hermes should access. Do not make that account an administrator merely to unlock an exceptional operation.

Everyone who uses a shared or background Hermes deployment receives the access of the one Box account it authorizes, so do not connect it to a broader personal or administrator account. Before starting the browser flow, make sure the authorization browser is signed in as the intended Box account.

Choose a descriptive environment name, such as `hermes-box-oauth`. Do not overwrite or reauthorize an existing environment until its identity is confirmed.

## Same-host interactive path

First resolve the Box command runner using [CLI guide](cli-guide.md). Then ask whether Hermes runs on the same computer as the browser the user will use to authorize Box. Use this path only when they confirm that it does. This is normally a local computer setup. Do not infer this from the operating system alone. Use the resolved runner; do not reconstruct a local npm prefix unless Hermes installed and verified that exact local copy.

Start one official local login operation without `--code`, leave its terminal process running until it exits, then verify the actor. The examples below use `box`; replace that executable with the previously verified local runner only when Hermes installed and verified a private CLI copy:

```bash
box login --default-box-app --name <ENVIRONMENT_NAME>
box users:get me --json --fields id,name,login
```

The browser flow creates and selects the named environment. Run the action through Hermes's terminal rather than asking the user to copy a runner command. Announce the pending authorization, wait for the CLI process to finish, then continue with the actor check. Let the CLI open the authorization page and receive the local callback. Do not use browser tools, inspect browser tabs, request the resulting URL, navigate to Box, or ask the user to paste a code.

If the callback server cannot bind port 3000, the browser opens an unusable authorization result, or the callback never reaches the waiting CLI, stop that login process before retrying. Retry the official app on the supported ports `3001`, `4000`, `5000`, and `8080`, one at a time, and verify the actor after each successful completion:

```bash
box login --default-box-app --port 3001 --name <ENVIRONMENT_NAME>
```

Do not switch a same-host setup to `--code` merely because port 3000 failed. Use `--code` only after the supported local ports fail or the user confirms that the authorization browser is on another host.

## Separate-host or headless path

Use this path only after the user explicitly confirms that Hermes runs on a remote host—such as a VPS, container, or cloud VM—or that it is headless and the authorization browser is on a different computer. Do not use this path when the same-host callback is available.

With `--code`, the CLI waits at interactive prompts for the `code` and `state` values from the browser's final localhost address. Those values must go from the user's browser straight into those prompts, so Hermes does not start this login itself: anything Hermes relays to a process it started lands in its context and transcript. The CLI saves the login under `$HOME/.box`, and Hermes's terminal can run with a different `HOME` from the user's shell (`terminal.home_mode: profile`, or `{HERMES_HOME}/home` in containers such as the official Docker image), so first run `echo "$HOME"` in Hermes's terminal. Give the user the exact command, built from the runner Hermes already resolved and verified and prefixed on macOS/Linux with `HOME=<that path>`, to run in a terminal they control on the Hermes host as the same OS user that runs Hermes (for example an SSH session, or `docker exec -it -u hermes <container> sh` for the official image):

```bash
box login --default-box-app --code --name <ENVIRONMENT_NAME>
```

Tell the user that the CLI prints an authorization URL, that the browser will end on an unreachable `localhost` page (expected), and that they should copy `code` and then `state` from that page's address bar into the two CLI prompts within about five minutes. Never ask for these values or the full callback URL in chat, never send them to a Hermes-started process with process `write`/`submit`, and never put them on a command line (for example `curl` against the loopback callback). After the user reports that the CLI finished, verify the actor with `box users:get me --json --fields id,name,login`. If the user cannot open a terminal on the Hermes host as that OS user, pause the login rather than improvising another channel for these values. If the CLI exits with `Port 3000 is already in use` before it prompts (it still binds the loopback callback in `--code` mode), the user reruns it with `--port 3001`, `4000`, `5000` or `8080`; for a custom Platform App, only on a port whose `http://localhost:<PORT>/callback` URI is registered.

## Existing environments

The Box CLI stores multiple named environments but uses one current default:

```bash
box configure:environments:list
box configure:environments:set-current <ENVIRONMENT_NAME>
box users:get me --json --fields id,name,login
```

Request approval before switching the current environment, especially on a shared or background installation. Switch it only after approval and verify the resulting actor. If the returned identity is API-only or has no normal Box login, do not use it for Hermes; connect a normal Box account through OAuth instead.

## Custom OAuth Platform App

Use this path only when the requested operation needs a scope unavailable through the official CLI app, such as **Manage webhooks**. Open the [Box Developer Console](https://app.box.com/developers/console), create or select a Platform App with **User Authentication (OAuth 2.0)**, and enable only the required scopes. Never broaden scopes merely to avoid an authorization error.

Use the same topology decision as the official app. Every command in this section, retries included, is run by the user in their own terminal on the Hermes host (see the last paragraph); Hermes gives the verified command and checks the actor afterwards. For a same-host browser, add `http://localhost:3000/callback` as an OAuth redirect URI in the app's **Configuration** tab, save it, then run:

```bash
box login --platform-app --port 3000 --name <ENVIRONMENT_NAME>
```

If port 3000 cannot bind, choose another free local port, add the exact `http://localhost:<PORT>/callback` URI to the Platform App, save the configuration, stop the failed login process, and retry with the matching `--port`. Unlike the official app, a custom Platform App may use any port whose exact callback URI is registered.

For a remote or headless Hermes runtime whose authorization browser is on a different computer, register the same loopback callback URI and add `--code`:

```bash
box login --platform-app --code --port 3000 --name <ENVIRONMENT_NAME>
```

This remote flow uses the same user-run terminal handoff as the official app's separate-host path: the user runs the command in their own terminal on the Hermes host and types `code` and `state` from the unreachable localhost page's address bar directly into the CLI prompts. Hermes never requests those values or the callback URL in chat and never passes them to a process or command line; it verifies the actor after the user reports completion. Do not inspect unrelated browser tabs or switch to the local callback workflow on a remote host.

Let the CLI prompt for the Client ID and Client Secret. The CLI echoes the Client Secret as typed, so every `--platform-app` login, local or remote, is run by the user in a terminal they control on the Hermes host, with the same `HOME` prefix as the separate-host path, never started through Hermes's terminal with the secret relayed by process `write`/`submit`. Do not ask the user to paste the Client Secret into chat, write it to Hermes configuration, or give the user an unverified local-runner command to copy. Authenticate the intended user in the browser, then verify the resulting actor. Keep administrator-only operations outside the normal Hermes OAuth identity.

## Official links

- [Box CLI quick start](https://developer.box.com/guides/cli/quick-start/)
- [Box CLI headless login](https://developer.box.com/guides/cli/headless-login/)
- [OAuth 2.0 guide](https://developer.box.com/guides/authentication/oauth2/)
- [Box OAuth scopes](https://developer.box.com/guides/api-calls/permissions-and-errors/scopes/)
