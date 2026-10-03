# Example 1 - shipping a backlog to production

Source: a real session report, translated and anonymised. Kept because it shows
every failure mode at once. The BEFORE is abridged wherever you see an ellipsis.
The AFTER is what the skill produced from the session itself, not a rewrite of
this excerpt.

## BEFORE - the engineer-to-engineer version (~35 lines)

> Cleanup complete - prod == main across all three surfaces. Final report:
> ✅ Prod promote: everything shipped
> | Surface | SHA | Delivered | Proof |
> | Backend | `a1b2c3d` | redis-py 8.0, structlog 26, patch group | deploy success, health green |
> | Bot | `e4f5a6b` | fastapi + typing-extensions, floor bump | deploy success |
> | Verify | tip (image rebuilt) | #687 canvas tab-dismiss + #697 fontconfig | containers healthy, edge 200, bundle fingerprint matches local dist |
>
> Two episodes worth mentioning: 1) The digest gate worked in anger. The bot
> deploy on a1b2c3d correctly failed closed: backend-only commit → no bot image
> was built for it. Correct move: promote from the last SHA with a successful
> bot build (e4f5a6b), having first proven with an empty diff that app/ is
> identical to tip… 2) The server disproved my notes: a signing fix I believed
> was live had never shipped, so live users went a day without the fix…
>
> Tails (non-blocking): dependabot config v2, transitive pins for pip, ABC-169
> may unblock aiohttp≥3.14 → possibly drop two --ignore-vuln flags…
>
> Net for the session: root cause of 14 hanging PRs found → grouping + auto-merge
> introduced and battle-tested → queue drained to zero → lock file repaired →
> whole backlog delivered to prod. Clean.

**What is wrong with it**

| Defect | Where |
|---|---|
| Best news is last | "14 hanging PRs → queue drained to zero" is the final line |
| Machine identifiers as facts | `a1b2c3d`, `e4f5a6b`, `#687`, bundle fingerprints |
| Untranslated jargon | surface, fail-closed, digest gate, tip, empty diff |
| A table instead of meaning | 3 rows × 4 columns to answer "is it all shipped?" |
| Bad news hidden mid-paragraph | "live users went a day without the fix" sits inside item 2 |

## AFTER - done-or-not mode output (13 lines)

> Cleared the whole backlog. The live product is now current across all three
> parts: the app, the bot, and document signing. Also found why our library
> updates kept getting stuck - 14 of them had been hanging, now zero.
>
> | | |
> |---|---|
> | ✅ | Shipped to the live product - all three parts |
> | ✅ | Checked: tests green, the site responds, signing verified by hand |
> | ⚠️ | Surprise: a signing fix I believed was live had not shipped. Real users went a day without it. It is live now. |
> | ⏳ | Deferred to Monday: two questions about automatic library updates |
>
> **Verdict**
>
> | | |
> |---|---|
> | Fully done? | Yes |
> | Anything needed from you? | No |
> | New debt? | Two small ones, written down |
> | Safe to close? | Yes |

(Outcome: #1 - shipped and verified. Note that "it reached users" is not a
verdict row: here it was verified, so it sits in the second ✅ line. Had the work
merged without reaching users, it would be a ⚠️ row instead.)

## What this example changed in the format

1. A ⚠️ row is mandatory. The two most important facts fit neither "done" nor
   "not done" - they were surprises.
2. The verdict block exists because of this example: four explicit answers, so
   the reader never has to infer completion from prose.
3. Some things cannot be translated after the fact. "canvas tab-dismiss" never
   says what the user saw before and after. Plain meaning has to be captured
   while the work happens, not reconstructed later.
