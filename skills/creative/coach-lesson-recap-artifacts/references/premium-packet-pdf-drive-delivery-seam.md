# Premium packet PDF Drive delivery seam

Use this when a coach agent successfully renders local premium packet PDFs but the coach/user needs clickable shared PDF links.

## Lesson
A local premium PDF plus a shared Google Doc is still a NO-GO when the requested deliverable is the premium packet PDF link. The coach-facing product is the shared, openable artifact — not the local filesystem path.

Do not let an easier Docs path mask a missing PDF upload/share seam.

## Correct delivery sequence
For premium packet requests where the coach asks for PDFs, links, or shared access:

1. Verify the local packet PDF exists and is non-empty.
2. Upload the PDF to Drive through the approved coach Workspace seam.
3. Share the uploaded PDF only with the explicit approved recipient(s), normally the coach's `@thesystem.golf` address for that lane.
4. Return the Drive PDF link(s) first.
5. Label any Google Doc as a secondary editable companion, not the primary deliverable.
6. Audit file size, SHA256, MIME type, Drive file ID/link, recipient, role, and permission ID.

## Narrow approved operation shape
If the Workspace seam lacks PDF upload, add a bounded operation rather than falling back to Docs:

```text
operation: drive-upload-file / share-file
allowed input: explicit local file path
initial allowed MIME: application/pdf
recipient: explicit @thesystem.golf address only
role: reader or commenter
forbidden: public links, broad ACLs, deletes, folder sweeps, Gmail, arbitrary file mutation
outputs: file_id, webViewLink, permission_id, audit record
```

## GO / NO-GO
GREEN only when:
- local PDF exists and is non-empty
- PDF was uploaded through the approved seam
- PDF was shared to the intended explicit recipient
- returned link is a real Drive file link, not invented

NO-GO when:
- only local paths are available
- only Google Docs links are shared while PDF links were requested
- the agent says the PDF exists but cannot surface it to the coach
- the agent provides chat summaries or Docs as substitutes for the packet product

## Handoff wording
Lead with:

```text
Premium packet PDF links:
- Student A: <Drive PDF link>
- Student B: <Drive PDF link>

Editable companions, if created:
- Student A Doc: <Google Doc link>

Sharing truth:
- shared to <recipient> as <role>
```

If upload/share is blocked, say exactly:

```text
Local packet PDFs exist, but PDF Drive upload/share is not wired in the approved Workspace seam yet. Google Docs are only secondary companions and do not satisfy the PDF-link request.
```
