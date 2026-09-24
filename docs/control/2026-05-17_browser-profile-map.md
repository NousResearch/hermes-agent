# Browser Profile Map

> Map each account family to one browser profile. Keep state isolated.

**Support doc**
**Decision owner:** Haz

## Purpose

Keep roles separate.

## Core rule

One profile per account family. No shared cookies. No cross-login. No reuse.

## Recommended profiles

### `profile-qa-testing`

- **Purpose:** testing and demos
- **Linked account:** `qa-testing`
- **Trust level:** low
- **Use for:** disposable work
- **Do not use for:** business, recovery, or secure research

### `profile-research-secure`

- **Purpose:** secure research
- **Linked account:** `research-m3-global`
- **Trust level:** high
- **Use for:** locked-down sessions
- **Do not use for:** testing clutter

### `profile-recovery-admin`

- **Purpose:** recovery only
- **Linked account:** `recovery-admin`
- **Trust level:** highest
- **Use for:** setup and emergencies
- **Do not use for:** daily work

### `profile-business-admin-<brand>`

- **Purpose:** live admin for one brand
- **Linked account:** `business-admin-<brand>`
- **Trust level:** high
- **Use for:** billing and admin
- **Do not use for:** testing or unrelated work

### `profile-business-support-<brand>`

- **Purpose:** support and ops
- **Linked account:** `business-support-<brand>`
- **Trust level:** medium-high
- **Use for:** inbox and support work
- **Do not use for:** admin or recovery

## Setup rules

- Create the profile before first login.
- Name it by role.
- Keep secure profiles minimal.
- Do not sync across profiles unless needed.
- Split mixed-use profiles.

## Expansion rule

Only add a profile when a real role exists.

## Safe handling

This document should never contain passwords, backup codes, or live credential material.
