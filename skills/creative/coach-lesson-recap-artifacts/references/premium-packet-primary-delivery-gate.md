# Premium packet primary delivery gate

Use this when a coach-agent can generate lesson recap drafts and can also render local premium/Connor packets, but the live handoff still exposes Google Docs or chat summaries as the apparent finished product.

## Session lesson
A draft Google Doc can be useful, but it is not the finished premium lesson recap product when robust lesson media should route to the Connor/premium packet workflow. The failure mode is subtle: the agent may create the premium PDF locally, then still send/share a basic Doc as the primary visible artifact. From the coach's perspective, the premium product did not happen.

## Gate before saying done
For robust lesson media where the premium packet path is expected, do not report GREEN/DONE unless all required delivery truth is explicit:
- premium packet PDF/HTML/JSON generated through the canonical packet renderer
- PDF exists and is non-empty
- intended brand/logo variant applied or explicit blocker stated
- PDF/packet is surfaced first as the primary deliverable
- any Google Doc is labeled as a secondary editable companion
- if Drive upload/share of the PDF is unavailable, say the PDF is local/prepped and name the exact blocker; do not replace it with a Doc and call the task done
- when the locked coach Workspace seam exposes `drive-upload-pdf`, the correct recovery path is: upload the existing verified premium PDF, share that Drive file to the explicit @thesystem.golf recipient, and return the PDF link before any Doc companion link

## NO-GO cases
- Only a chat recap exists.
- Only a rough/basic Google Doc exists.
- A polished Doc exists but the expected premium packet PDF is absent.
- The local PDF exists but is not surfaced or is treated as secondary while the Doc is presented as final.
- The agent says it has the Connor format when the exact gold reference has only been recorded as a URL and not extracted/applied to the renderer.

## Correct handoff shape
Lead with packet truth:
1. `Premium packet PDF: <path or Drive link>`
2. `HTML/JSON source: <paths>`
3. `Google Doc companion: <link, if created>`
4. `Sharing truth: shared yes/no, to whom, via which seam`
5. `Blocker: <only if PDF upload/share or brand application is not wired>`
