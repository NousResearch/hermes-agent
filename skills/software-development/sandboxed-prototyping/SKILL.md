---
name: sandboxed-prototyping
description: "Run pasted snippets, new packages or cloned repos sandboxed."
version: 1.0.0
author: John Paul Soliva (jonpol01), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [sandbox, untrusted-code, prototype, experiment, pip, npm, clone, snippet, docker, isolation]
    related_skills: [spike, github, requesting-code-review]
---

# Sandboxed Prototyping Skill

Run code you did not write (a snippet from the web, a package you are trying for the first time, a repo, someone's PR) with `hermes sandbox run` instead of on the host. It copies a directory into a throwaway container with no credentials, no host environment, a read-only root and no access to the user's files, runs the command with no network, and deletes it afterwards. On the host, that code runs as the user and can read their SSH keys, tokens and `~/.hermes/.env`, and installing a package runs its install scripts the same way.

## When to Use

Load this before you run, install or import anything whose source you have not read:

- a snippet, gist, Stack Overflow answer or example the user pasted or you found online;
- trying an unfamiliar library or CLI ("try out X", "see if package Y does Z", "demo W");
- a quick prototype that needs `pip install`, `npm install` or similar;
- cloning a repo to run its tests, examples or build;
- anything with install or build scripts you have not read (`setup.py`, npm lifecycle scripts, a `Makefile`).

Not for: the user's own project and normal edits in their working tree, code you wrote yourself in this session that uses only the standard library, or reading files (reading is safe; running is not). Someone else's pull request in a review goes through the `github` skill, which uses the same command.

## Prerequisites

Docker or Podman, running. The first run pulls the sandbox image (`terminal.docker_image`, by default `nousresearch/hermes-sandbox:desktop`, which has Python, pip, uv, Node, npm and git).

## How to Run

Every run goes through the `terminal` tool:

```
hermes sandbox run --path <dir> [--setup '<install command>' --setup-network open] -- <command>
```

- `--path <dir>`: the directory copied in; the command runs with that copy as its working directory. Never your home directory itself or anything inside the Hermes home (it refuses those).
- `--setup '<cmd>'`: runs first, in its own container, for installs. It has **no network** unless you add `--setup-network open`, so an install that downloads packages needs both flags.
- `-- <command>`: the run step. It never has a network.
- `--pr N` or `--ref REF` (with `--repo DIR`) instead of `--path`: a pull request or git ref of a local clone, exported without a checkout.
- `--timeout SECONDS`: per step, default 900.

Setup and run share the copy, so what setup installs (into the copy or `$HOME`, e.g. `pip install --user`, `node_modules`) is there for the run. Everything is deleted afterwards: only stdout, stderr and the exit status come back, so print what you need.

## Quick Reference

| Task | Command |
|---|---|
| Snippet, standard library only | `hermes sandbox run --path proto -- python3 main.py` |
| Python with packages | `hermes sandbox run --path proto --setup 'pip install --user rich' --setup-network open -- python3 main.py` |
| Node with packages | `hermes sandbox run --path proto --setup 'npm install' --setup-network open -- node index.js` |
| Remote repo's tests | `hermes sandbox run --path proto --setup 'git clone --depth 1 <url> repo && cd repo && pip install --user -e . pytest' --setup-network open -- sh -c 'cd repo && python3 -m pytest -q'` |
| Local clone's tests | `hermes sandbox run --path repo --setup 'npm ci' --setup-network open -- npm test` |
| A pull request | `hermes sandbox run --repo repo --pr 123 --setup 'pip install --user -e . pytest' --setup-network open -- python3 -m pytest -q` |

The default image does not ship `pytest`: install it (and any other test tool the project uses) in `--setup`, or use `python3 -m unittest`.

Exit status: the command's own, `124` timed out, `69` no container runtime (see Pitfalls), `2` refused or bad arguments; a failed `--setup` returns its own status.

## Procedure

1. **Make a scratch directory per experiment** in the working directory, e.g. `proto-rich/`, and write the code into it with `write_file`. For packages, put the install in `--setup` (or write `requirements.txt` / `package.json` and install from it).
2. **For a remote repo you only need to run, clone it inside the setup step** (Quick Reference, "Remote repo's tests"): nothing from it lands on the host. If you also need to read it, cloning on the host with `terminal` is fine (`git clone` runs none of its code); read it with `read_file` and `search_files`, then run it with `--path`. Never run its install, build, test or example commands on the host.
3. **Run it in the sandbox** with the matching command. Installs go in `--setup` with `--setup-network open`; the run step has no network.
4. **Read the output and iterate**: edit with `write_file` or `patch` and rerun. Each run starts from a fresh copy, so setup reruns too.
5. **Report** the result and that it ran in the sandbox. If the code needs network at run time (an API call, a download), say it cannot work in the sandbox; do not move it to the host.

Example, Python:

```
write_file("proto-rich/main.py", "from rich.console import Console\nfrom rich.table import Table\nt = Table('fruit', 'price')\nt.add_row('apple', '1.20')\nConsole().print(t)\n")
terminal("hermes sandbox run --path proto-rich --setup 'pip install --user rich' --setup-network open -- python3 main.py")
```

Example, Node:

```
write_file("proto-dayjs/index.js", "const dayjs = require('dayjs');\nconsole.log(dayjs('2026-01-15').add(45, 'day').format('YYYY-MM-DD'));\n")
terminal("hermes sandbox run --path proto-dayjs --setup 'npm install dayjs' --setup-network open -- node index.js")
```

## Pitfalls

- **No container runtime: fail closed.** If `hermes sandbox run` exits `69` or reports that no container runtime is available, do NOT run the code on the host instead: not with `terminal`, not with `execute_code`, not "just this once", and do not offer to. Tell the user it was not run because no isolated runtime is available (it needs Docker or Podman), then read the code with `read_file` and report what it would do: imports, network calls, file writes, install scripts, anything suspicious. The same applies when `hermes` or its `sandbox` subcommand is not found where the terminal runs (an older Hermes, or a remote or container terminal backend).
- **`--setup` without `--setup-network open` has no network**: `pip` and `npm` fail with name-resolution errors. Add the flag; do not move the install to the host.
- **`--setup-network open` gives install scripts the container runtime's ordinary network**: the internet, the local network and services on this machine (still no credentials or user files). Use it only for installs.
- **`execute_code` and plain `terminal` commands run on the host.** Importing an untrusted package there (even `python3 -c "import x"`) runs its code as the user.
- **This is guidance, not a security boundary.** Nothing forces code through the sandbox except following this skill. It covers only what runs through `hermes sandbox run`; code loaded into the Hermes process itself (a plugin, an MCP server, a skill's scripts) is outside it.

## Verification

- The `terminal` output starts with `hermes sandbox: ... (no network, no credentials, read-only root)`.
- A script printing `os.getuid()` and `platform.system()` from the sandbox prints `65534` and `Linux`, whatever the host is.
- Nothing was installed on the host: no new packages in the host Python, no `node_modules` outside the scratch directory.
