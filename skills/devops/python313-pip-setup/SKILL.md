---
name: python313-pip-setup
description: Fix broken pip, distutils, and setuptools on Ubuntu with Python 3.13. Use when pip3 is missing, `python3 -m pip` fails, or packages fail to build due to missing distutils or setuptools.
tags: [python, pip, ubuntu, distutils, setuptools, python3.13, devops]
---

# Python 3.13 pip/distutils Setup on Ubuntu

## When to use

- `pip3: command not found` on Ubuntu with Python 3.13
- `python3 -m pip` fails with "No module named pip"
- Package build fails: `ModuleNotFoundError: No module named 'distutils'`
- `ImportError: cannot import name 'setup' from 'setuptools'`
- Installing packages like twikit, or anything with legacy `setup.py`

## Root cause

Python 3.13 on Ubuntu ships WITHOUT pip and WITHOUT distutils by default.
- `pip3` binary may not exist at all
- `distutils` was removed from stdlib in Python 3.12+
- Many legacy packages use `setup.py` with `from distutils.core import setup` — this breaks

## Fix (in order)

```bash
# Step 1: Install pip + dev headers
sudo apt install -y python3-pip python3-dev

# Step 2: Install setuptools + distutils compatibility layer
sudo apt install -y python3-setuptools python3-distutils-extra

# Step 3: Verify
python3 -m pip --version
pip3 --version
```

## Install packages after fix

```bash
# Use --break-system-packages on Ubuntu 24+ (PEP 668 enforced)
python3 -m pip install <package> --break-system-packages

# Or install to user space (no flag needed)
pip3 install --user <package>
```

## sudoers for agent (Cody/subagents)

If a subagent needs to install packages without password:

```bash
# Write to /tmp/agent-sudoers first, then:
sudo tee /etc/sudoers.d/agent-name < /tmp/agent-sudoers
sudo chmod 440 /etc/sudoers.d/agent-name
```

Content of the sudoers file (use scp + tee — avoid heredoc over SSH due to shell escaping):
```
# Agent - limited NOPASSWD
sander ALL=(ALL) NOPASSWD: /usr/bin/python3
sander ALL=(ALL) NOPASSWD: /usr/bin/pip3
sander ALL=(ALL) NOPASSWD: /usr/local/bin/pip3
```

**Pattern**: write file locally → `scp` to remote → `sudo tee /etc/sudoers.d/` on remote.
Do NOT try to write sudoers content via heredoc or inline SSH strings — shell escaping breaks.

## Pitfalls

- `sudo tee` requires tee to already be in NOPASSWD — check existing sudoers first with `sudo -l`
- `sudo visudo -c` may not be in NOPASSWD; skip it or use `cat` to verify file content
- `pip3` binary lands in `/home/sander/.local/bin` when installed with `--break-system-packages` — PATH may need updating for scripts
- Do NOT try inline Python heredoc over SSH with complex quoting — write locally, scp, then execute
- `python3-distutils-extra` is the correct apt package name (not `python3-distutils` alone)
