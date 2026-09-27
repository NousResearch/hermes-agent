# Local employee layout

Status: layout agreed; runtime implementation pending.

## Ownership

One Hermes profile owns one employee's knowledge. Resolve its home through
`get_hermes_home()`; never hardcode the default `~/.hermes` path. Different
profiles keep separate responsibilities, service manuals and memory.
The working directory holds actual work. Product guides ship with the code.

```text
<hermes-profile-home>/
  responsibilities/
    <name>/
      RESPONSIBILITY.md
      STATE.md
      references/
      state/
      archive/
      scripts/
      schedules/
      webhooks/
    .archive/                 # retired responsibility packages
  connections/
    <service>/
      manual.md
      references/
      scripts/
  ...existing Hermes state...

<working-directory>/
  documents/
  repos/
  tmp/

<installed-repository>/
  guides/
    <guide>/
      guide.md
      references/
```

The tree describes allowed locations, not a requirement to eagerly create every
optional directory. Preserve responsibility creation and lifecycle semantics,
including the distinction between package `archive/` records and retired
packages under `responsibilities/.archive/`.

## Profile-owned knowledge

Responsibilities and service manuals are employee-owned durable state. They
survive product updates. Ordinary file tools access them directly, with the
agreed responsibility validation, limits and feedback. There is no generated
`credentials.md` or connection-store dependency.

Authored shared memory and per-person profiles belong to the same employee
boundary. Their exact local storage representation is a separate implementation
detail; this layout does not collapse personal profiles into one shared USER.md.
Hindsight integration must retain the agreed employee scope as well.

## Working files

Use `documents/` for retained documents and cloud-document stubs, `repos/` for
checkouts, and `tmp/` for disposable work. The actual working directory must be
resolved from native configuration/session execution context and communicated
to the model; it is not a fixed `/workspace` path.

These conventions do not authorize moving existing user files, replacing
occupied paths, automatic deletion, or a new retention policy. A shared working
directory is not itself a security boundary between profiles.

## Product guides

Guides live in the installed repository's `guides/` directory and update with
the product. Resolve their real paths and supply them through prompt assembly
and guide links. Do not copy mutable guide instances into each profile or
reintroduce a skill installation/discovery mechanism.

Guides are product-owned; the employee is instructed not to edit them. This is
an ownership rule, not a claim that the local OS prevents writes.

## Native integration

Keep native Telegram transport, channel setup and display behavior. Integrate
the layout directly into shared prompt assembly, file operations, responsibility
discovery, cron reconciliation and learning. Bind the owning profile explicitly
for background work and person-memory lookup.

Do not create `/alfred` or `/workspace` aliases or a hosted filesystem emulation
layer. Preserve applicable guide wording, substituting resolved local paths.

## Verification when implemented

Verify two profiles cannot accidentally discover or mutate each other's
knowledge through these integrations; scheduled work resolves its owning
profile; installed guide paths work outside the checkout's current directory;
manuals survive product updates; and changing discovery listings does not
rewrite an existing conversation's cached prompt. Existing user files must
survive setup unchanged.

## Runtime lifecycle

Decision: retain native Hermes runtime lifecycle, gateway/service operation,
scheduling, missed-run handling and restart behavior. Do not add a new
supervisor or emulate hosted uptime, automatic cloud wakeup or machine
replacement. Unattended work uses the existing native service mechanisms.

Adapt guides to describe the actual native lifecycle and its sleep/offline
limits. Responsibility execution preserves its agreed employee contract within
that lifecycle; do not import hosted infrastructure to change native availability.
