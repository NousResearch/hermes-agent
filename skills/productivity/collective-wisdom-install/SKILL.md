---
name: collective-wisdom-install
description: Install a shared team skill with explicit consent.
version: 0.1.0
author: Shannon (Shannon), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [skills, collective-wisdom, install, team]
    related_skills: []
---

# Collective Wisdom Install Skill

Resolve a copied Collective Wisdom install request into a local compatibility
plan, then apply it only after explicit user confirmation. This skill never
adds a core tool, changes the active toolset, or installs dependencies.

## When to Use

- The user pastes `Install this Collective Wisdom skill: <portal-link>`.
- The user asks to install a Collective Wisdom skill ID or version.
- Don't use for Hub skills or the legacy `_org` Skill Sync mirror.

## Prerequisites

- The profile must already be signed into Nous Portal.
- `hermes wisdom setup` must have verified a team organization and installation
  identity for this profile.

## Procedure

1. Extract only the authenticated Portal URL, skill ID, or `skill-id@vN` from
   the user's request. Do not infer another organization or version.
2. Use `terminal(command="hermes wisdom install '<reference>' --plan --json")`.
   Treat a not-found response as opaque; never probe nearby identifiers.
3. Present the returned version, author copy, local compatibility outcome,
   setup actions, permissions, and known limitations. State that
   SkillEvaluator is advisory and separate from compatibility.
4. Use `clarify` to ask whether to apply this exact receipt. Do not treat the
   original natural-language request as the apply confirmation.
5. Only after an affirmative answer, use
   `terminal(command="hermes wisdom install --apply-receipt '<receipt>' --accept-partial --json")`.
   Omit `--accept-partial` unless the user explicitly accepted a partial or
   setup-required outcome.
6. Report the managed path, installed version, content hash, and effective
   update mode from the apply response.

## Pitfalls

- Never pass a browser-supplied organization, generation, content hash, or
  Gateway token. The CLI re-fetches authoritative state.
- Never convert a blocked compatibility result into a force install.
- Never say an unavailable advisory scan passed.
- A Portal raw copy is an unmanaged fork, not a managed install.

## Verification

The apply response must say `installed: true`, include the pinned version and
content hash, and point below the active profile's `_wisdom/<org-id>/` root.
