# hermes-ngit

Hermes skill for decentralized Git over Nostr via [ngit](https://gitworkshop.dev/ngit).

## What is this?

A skill file for Hermes that provides instructions and workflows for working with
Nostr-based Git repositories. ngit is a sovereign, decentralized Git hosting protocol
built on Nostr — no central forge, no daemon required.

## Features

- Announce repos to Nostr (`ngit init`) and keep gitworkshop current via `git push`
- Native `git clone` / `git push` against `nostr://` remotes (no daemon)
- Session repo detection for Nostr-enabled projects
- Full PR and issue lifecycle: open, view, comment, checkout, merge, close, reopen, label
- Account management (bunker/NIP-46 login, inline `--nsec` for CI) and key-flags reference
- Dual/triple-push configuration (GitHub + Radicle + Nostr) with fail-closed fallbacks

## Key Lesson

`ngit sync` only pushes git refs — it does **not** update the NIP-34 announcement that
gitworkshop.dev reads. To keep gitworkshop current, add the `nostr://` URL as an `origin`
pushurl so `git push` drives `git-remote-nostr` and re-announces HEAD automatically.

## Usage

Install via `hermes skills install official/devops/hermes-ngit`, or copy `SKILL.md`
(plus `LICENSE` and `NOTICE`) into your Hermes skills directory.

## Credit / License

This skill builds on **Dan Conway's** ngit command reference
(`DanConwayDev/ngit-cli`, `skills/ngit/SKILL.md`), incorporated under the
**Creative Commons Attribution-ShareAlike 4.0 International (CC-BY-SA-4.0)** license in
accordance with the source material's share-alike terms. Original framing and multi-mirror
packaging by Joey Stanford.

The full license text is in the `LICENSE` file; attribution is also recorded in `NOTICE`.
