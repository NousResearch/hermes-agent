# Forgejo issue to pull request

This procedure composes existing `forgejo_*` MCP tools and Hermes terminal/file
tools. It is a workflow guide, not a new Forgejo client, task broker,
credential issuer, approval engine, evidence store, or merge queue.

## 1. Read the task

Use the configured Forgejo tools to read the issue and comments, including the
full comment thread. Read repository metadata, `AGENTS.md`, contribution
documents, and the relevant source with `read_file` and `search_files`. Treat the newest comment
as current task state. Record the requested behavior, non-goals, and unanswered
questions.

Done when the issue, comments, repository instructions, and relevant source are
read.

## 2. Check for duplicate work

Use the configured Forgejo search/list tools to find existing pull requests,
branches, and recent changes for the issue and its keywords. Do not create a
second branch or pull request when an active task already owns the issue.

Done when duplicate work is ruled out or the existing task is used.

## 3. Create the disposable workspace

Create one disposable Agent Sandbox through the configured `terminal` backend. The
checkout must live under `/workspace`, not on the Hermes runtime, host
filesystem, or persistent Hermes PVC. Use only the admitted repository,
revision, image, and task-bound credential. The task Pod must not receive a
long-lived Matrix, OpenBao, Kubernetes, or unrelated Forgejo credential.

If the sandbox or checkout fails, stop. Keep every secret out of commands,
files, output, and Matrix reports. Report the failed command and the
steps completed before failure. Do not fall back to host execution.

Done when the repository is checked out inside the disposable Sandbox.

## 4. Inspect and implement

Read the repository instructions and relevant source again from the checkout.
Make the smallest change that addresses the issue. Use `read_file`,
`search_files`, and `patch`. Add a regression test before a non-trivial
behavior fix. Keep unrelated cleanup out of the branch.

Done when the intended files and tests are changed and the working tree shows
only the task change.

## 5. Run applicable tests

Run the repository's documented test and lint commands through `terminal` in
the Sandbox. Record each command, exit status, and concise result. A failing
test is a workflow failure, not a reason to hide output or continue to
publication.

Done when the applicable tests pass, or the workflow stops with their real
failure and completed-step report.

## 6. Create one branch and pull request

Create exactly one task branch from the checked-out base. Commit the change
with the repository's author and message rules. Push through the task-bound
Forgejo credential. Use the configured `forgejo_create_pull_request` tool to
create one pull request, link it to the source issue using the repository's
native syntax, and read the created pull request back. Do not merge it.

If the credential, branch, push, or pull-request operation fails, stop and
report the service error. Never print the credential or claim that publication
succeeded.

Done when the exact pull-request URL and source-issue link are read from
Forgejo.

## 7. Report and clean up

Send the Matrix result with:

- the exact pull-request URL;
- the changed files;
- the test result;
- the branch name;
- the changed file list;
- every test command and its real result;
- the issue and pull-request references;
- the completed steps and any failure;
- the Sandbox cleanup result.

Delete the Sandbox and verify that its task Pod and generated task resources
are absent. A cleanup failure remains visible and is not converted to success.
Do not claim live deployment, production health, merge, or CI success.

Done when the report is sent and cleanup is verified, or the remaining failure
is explicit.

## Boundaries

The Forgejo credential is the authorization boundary. The skill must not add a
second approval system or infer permission from tool names. A Forgejo write
permission does not authorize a live infrastructure change. Infrastructure
PRs remain subject to the repository's owner approval, protected merge queue,
and separate action-specific live approval. The first release stops after pull
request creation.
