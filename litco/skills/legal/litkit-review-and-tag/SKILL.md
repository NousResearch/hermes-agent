---
name: litkit-review-and-tag
description: Build, propose, change, and follow a LitKit Review & Tag run from chat.
version: 0.1.0
author: LitCo, Hermes Agent
platforms: [linux, macos]
metadata:
  hermes:
    tags: [legal, litkit, ediscovery, review, tagging, criteria]
    related_skills: [litkit-corpus-pull, production-data-analysis]
---

# LitKit Review & Tag Skill

Ana owns a review from the lawyer's request to the finished tags. She registers each part of the review in LitKit herself: the tags, the criteria set, the scope, and the proposed run. The person makes one decision, which is to press Launch on the card in the thread. Ana never writes the criteria to a file and hands back steps for the Review screen.

## When to Use

- A lawyer asks to start a review, run criteria over a set, or tag documents against requests ("run Part 11 over the Crowder emails").
- A lawyer changes criteria for a review that exists or is pending ("add REV-NR as Part 11 item 6").
- A question needs consistent judgments over more than a few dozen documents. Proposing a run is free; the person sees the count and the price before anything is spent.
- A run Ana proposed has launched and needs following to completion.

## Prerequisites

The `litkit` toolset on a matter host. Every call acts for the lawyer on the turn. A refusal (`permission_denied`) means that lawyer may not create tags, write criteria, or propose runs on this matter. Tell them so and name who can do it; do not route around it.

## Quick Reference

| Step | Tool | Result |
|---|---|---|
| See what exists | `litkit_tags` action=list; `litkit_review` action=criteria (criteriaAction=list) | tag names, criteria sets |
| Register tags | `litkit_tags` action=create, name, kind | a tag per judgment the run returns |
| Register criteria | `litkit_review` action=criteria, criteriaAction=create, name, criteria | a criteria set, version 1 |
| Scope | `litkit_review` action=work_sets; `litkit_work_sets` action=create | a work set id, or document ids, a filter, a Bates range |
| Propose | `litkit_review` action=create | proposalId, estimatedCount, estimate; or requiresApproval with a quote |
| Change criteria | criteriaAction=update (new version), then publish | a new version; propose again if the run is pending |
| Follow | `litkit_review` action=list, status, records | progress, then per-document results |
| Finish | `litkit_review` action=accept_tags, jobId | a finished run's proposed tags accepted, when the person asks |

## Procedure

1. **Read the request into parts.** Name the judgments the lawyer wants back (one tag each), the standard for each judgment, and the documents in scope. Ask only if the scope or a judgment is genuinely ambiguous; a numbered request list is criteria, not a question.
2. **Tags.** List the matter's tags. Reuse an existing tag where the name fits, spelled as LitKit spells it. Create the rest with `litkit_tags` action=create (kind `issue` for request-by-request tags, `privilege` or `responsive` where those fit). A run can write only the tags named in its `tags` list, so every judgment needs its own tag.
3. **Criteria set.** `litkit_review` action=criteria, criteriaAction=create, with a name and one criterion per judgment: `title`, `description` (what counts and what does not, in the lawyer's terms), `tagName` (the tag that criterion writes), and optionally `seedQuery`. Keep the lawyer's numbering in the titles ("Part 11, item 6: REV-NR"). Then publish it (criteriaAction=publish).
4. **Scope.** Use a work set when the documents are a named batch (`litkit_review` action=work_sets lists them; `litkit_work_sets` action=create makes one from up to 500 ids). Otherwise send `documentIds`, a `filter` (the review grid's filters: custodian, dateFrom, dateTo, query, tagIds, productionIds), or a `batesRange` {start, end}. The scope takes exactly one of these.
5. **Propose.** `litkit_review` action=create with `criteriaSetId`, `scope`, and `tags` (every tag the run may write). Leave `applyTags` off so tags land as reviewable proposals unless the lawyer asked for direct tagging. Set `createMissingTags` only for tags the lawyer named that the matter lacks.
6. **Price.** If the result has `requiresApproval`, nothing is proposed yet. Give the person the price and the document count and ask whether to proceed. After an explicit yes, and only then, call create again with the same arguments plus `quoteId` and `userConfirmed: true`. If the result says `needsSecondApprover`, a different matter admin must approve the quote in LitKit billing first; say so and wait.
7. **Report the proposal.** Tell the person in two or three sentences what you proposed: the scope and its document count, the criteria set with its version, the tags, and the estimated cost. Say that Launch is on the card in this thread.
8. **Changes.** When the lawyer changes a criterion, fetch the set (criteriaAction=get), edit that criterion or add the new one, and send the full list with criteriaAction=update and a `changeNote` in the lawyer's words. That saves a new version. Publish it. If the earlier proposal has not launched, propose again so the card carries the new version, and say that the earlier card is superseded. If a run already launched on the old version, say so and ask whether to propose a new run over the same scope.
9. **Follow the run.** After the person launches, find the job with action=list and note its jobId. Check it with action=status. If it will run past this turn, schedule a check with `cronjob_manage` action=create: a self-contained prompt naming the jobId, the lawyer's user id, and the tags, telling the job to call `litkit_review` action=status, and, when the run is done, to count the results by tag from action=records and send them with `litkit_notify` kind=`review_complete` and that userId. Remove the scheduled job once it has reported.
10. **Completion.** Report the run in the thread when next asked, or through the notification: documents reviewed, the count under each tag, documents that failed or were skipped, and whether the tags are applied or waiting as proposals. Accept all proposed tags (action=accept_tags) only when the person asks.

## Pitfalls

- Criteria prose does not route tags. A tag missing from `tags` is never applied, however clearly the criteria name it.
- An estimate is an estimate. The count is taken again at launch, and the corpus may have grown.
- Never invent a `quoteId`, and never send `userConfirmed` before the person has said yes to the quoted price.
- A cron check has no lawyer on its turn, so a notification from it must name the userId.
- Do not send the person to the Review screen to finish the setup. If a step fails, report the refusal or error and what would clear it.

## Verification

- Each judgment the lawyer asked for has a tag, a criterion, and a place in the run's `tags` list.
- The proposal's count matches the scope's size within reason; a large gap means the scope is wrong.
- After a criteria change, the pending proposal carries the new version.
