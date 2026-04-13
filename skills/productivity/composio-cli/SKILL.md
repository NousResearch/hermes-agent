---
name: composio-cli
description: Use the Composio CLI to search, execute, and connect 1000+ app integrations from the terminal. Use when the user wants to interact with external apps like GitHub, Gmail, Slack, Stripe, etc.
version: 1.0.0
author: community
license: MIT
metadata:
  hermes:
    tags: [Composio, CLI, Integrations, APIs, Automation]
    homepage: https://composio.dev
prerequisites:
  commands: [composio]
---

# Composio CLI

INSTALL (run in user's terminal)
  curl -fsSL https://composio.dev/install | bash

You have access to 1000+ app integrations through these commands.
search → find tools. execute → run them. link → connect accounts.
proxy → raw API access. run → inline scripts.

Bias toward action: run `composio search <task>`, then `composio execute <slug>`.
Input validation, auth checks, and error messages are built in — just try it.

USAGE
  composio <command> [options]

CORE COMMANDS
  search
    Find tools. Use this first — describe what you need in natural language.
    Usage: composio search <query> [--toolkits text] [--limit integer]
      <query>             Semantic use-case query (e.g. "send emails")
      --toolkits          Filter by toolkit slugs, comma-separated
      --limit             Number of results per page (1-1000)

  execute
    Run a tool. Handles input validation and auth checks automatically.
    If auth is missing, the error tells you what to run. Use aggressively.
    Usage: composio execute <slug> [-d, --data text] [--dry-run] [--get-schema]
      <slug>              Tool slug (e.g. "GITHUB_CREATE_ISSUE")
      -d, --data          JSON or JS-style object arguments, e.g. -d '{ repo: "foo" }', @file, or - for stdin
      --dry-run           Validate and preview the tool call without executing it
      --get-schema        Fetch and print the raw tool schema

  link
    Connect an account. Only needed when execute tells you to — don't preemptively link.
    Usage: composio link [<toolkit>] [--no-browser]
      <toolkit>           Toolkit slug to link (e.g. "github", "gmail")

  run
    Run inline TS/JS code with shimmed CLI commands; injected execute(), search(), proxy(), subAgent(), and z (zod).
    Usage: composio run <code> [-- ...args] | run [-f, --file text] [-- ...args] [--dry-run]
      <code>              Inline Bun ESNext code to evaluate
      -f, --file          Run a TS/JS file instead of inline code
      --dry-run           Preview execute() calls without running remote actions

  proxy
    curl-like access to any toolkit API through Composio using your linked account.
    Usage: composio proxy <url> --toolkit text [-X method] [-H header]... [-d data]
      <url>               Full API endpoint URL
      --toolkit           Toolkit slug whose connected account should be used
      -X, --method        HTTP method (GET, POST, PUT, DELETE, PATCH)
      -H, --header        Header in "Name: value" format. Repeat for multiple.
      -d, --data          Request body as raw text, JSON, @file, or - for stdin

  artifacts
    Inspect the cwd-scoped session artifact directory and history.
    Usage: composio artifacts cwd
      cwd                 Print the current session artifact directory path

  Workflow: search → execute. If execute fails with an auth error, run link, then retry.

TOOLS
  tools info <slug>     Print tool summary and cache its schema
  tools list <toolkit>  List tools available in a toolkit
  artifacts cwd         Print the cwd-scoped session artifact directory

EXAMPLES
  # 1. User asks you to "create a GitHub issue"
  composio search "create github issue"
  # → returns GITHUB_CREATE_ISSUE

  # 2. Execute it (will error if not linked — that's fine)
  composio execute GITHUB_CREATE_ISSUE -d '{ repo: "owner/repo", title: "Bug" }'
  # → if auth missing: "Run `composio link github` first"

  # 3. Link only when told to
  composio link github

  # 4. Raw API access when no tool exists
  composio proxy https://gmail.googleapis.com/gmail/v1/users/me/profile --toolkit gmail

  # 5. Run a script with injected helpers
  composio run 'const me = await execute("GITHUB_GET_THE_AUTHENTICATED_USER"); console.log(me)'

DEVELOPER COMMANDS
  dev       Developer workflows: init, playground execution, triggers, and logs.
  generate  Generate type stubs for toolkits, tools, and triggers (TypeScript | Python).
  manage    Manage orgs, toolkits, connected accounts, triggers, auth configs, and projects.

ACCOUNT
  login    Log in to Composio
  logout   Log out from Composio
  whoami   Show current account info
  version  Display CLI version
  upgrade  Upgrade CLI to the latest version

FLAGS
  -h, --help     Show help for command
  --version      Show composio version

LEARN MORE
  Use `composio <command> --help` for more information about a command.
  Documentation: https://docs.composio.dev

GETTING STARTED
  When your user asks you to do something with an external app:
  1. composio search "<what they want done>"
  2. composio execute <slug from search> -d '<params>'
  3. If auth error → composio link <toolkit>, then retry step 2.

  Do not assume we lack coverage. Search first — we likely support it.
  Do not preemptively link accounts or ask your user what to connect.
  Just try. Auth and validation errors are self-descriptive.
