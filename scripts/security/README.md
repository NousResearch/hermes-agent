# Hermes Verify Sandbox

The privileged helper is used by agent/verify/runner.py to run
project-controlled verification commands in transient systemd units.

## Prerequisites

Linux with systemd, the hermes and hermes-verify accounts, and a workspace
base at /home/hermes/purchase-inventory-e2e. Workspaces must be beneath
that base and satisfy the ownership and permission checks in the helper.

## Installation

Install only from a trusted, reviewed source revision. Do not install a
root-owned helper from an unreviewed working tree writable by hermes.

Do not install privileged files directly from the checkout writable by
hermes. First copy both files to root-owned review files with mode 0600.
Inspect those copies and their checksums, then install only the reviewed
copies as root:root mode 0700 for the helper and 0440 for the sudoers file.
Run sudo visudo -c after installing the sudoers entry.

Verify that the reviewed source and installed helper checksums match.
The sudoers rule grants passwordless root execution only for this helper.

## Testing

Tests under tests/verify use a local unit-test adapter and do not alone
validate production isolation. Test the installed helper separately for
execution, rejected inputs, startup, readiness, cleanup, and timeouts.
These controls are defense in depth, not a guarantee against every attack.
