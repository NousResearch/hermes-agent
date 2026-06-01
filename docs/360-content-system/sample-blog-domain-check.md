---
title: "Sample blog (EN) — written with Blog Skill v9"
source_card: kc-2026-0601-001
brand: Hi Logic Labs
status: sample-for-review
created: 2026-06-01
note: บล็อกตัวอย่างพิสูจน์ skill v9 · ยังไม่เผยแพร่ · รอพี่นัทตรวจ
---

# Your domain checker is lying to you

From 340 name checks, a "fast" tool reported dozens of taken domains as free.
If you trust it, you'll fall in love with a name you can't have.

## The problem

Picking a brand name is easy. Confirming the `.com` is free is where it breaks.

The usual move is one quick lookup — a single DNS check that returns in
milliseconds. It feels authoritative. It isn't. A registered domain with no
nameservers set will look exactly like an open one. So the fast check says
"free," and a slow human says "taken" three days later, after the name is
already on the pitch deck.

## What we tested

We needed to screen hundreds of candidate names quickly, so we ran two layers
instead of one.

Layer one was the fast scan — good for throwing out obvious dead ends in bulk.
Layer two was a direct query straight to the registry that actually owns `.com`,
reading for the exact "no match" signal that means a name is truly open.

Then we added a control. Before trusting a batch, we checked two known answers:
a name we knew was taken, and a random string we knew was free. If the tool got
either control wrong, the whole batch was suspect — usually a sign the lookup
service had started rate-limiting and returning empty noise.

## The result

The fast scan alone was wrong often. Across the run, it flagged names as open
that the registry confirmed were already registered — dozens of them. Easy,
readable `.com` names turned out to be almost entirely gone: of 340 candidates,
the clean, real-word options were taken at nearly 100%.

The two-layer check with a control caught every false positive the fast scan let
through.

## The takeaway

Speed and accuracy aren't the same tool. A fast checker is fine for a first
pass, but anything you'll act on needs a second layer that reads the source of
truth — plus a control to prove the tool is still telling you the truth at all.

This holds well beyond domains. Any automated check you build into a decision
deserves a known-answer test running next to it.

## Try this

- Run your bulk check first, but treat every "available" as a maybe, not a yes.
- Confirm the survivors against the authoritative registry, not a cache.
- Add two controls — one known-taken, one known-free — and re-run them every
  batch. If a control flips, stop and check for rate limits before trusting
  anything.
