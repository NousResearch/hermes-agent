# Anbu's item-wise replies to NDR (voice-dictated, one item per draft)

Verified 2026-10-23. NDR fires a batch of numbered action-item emails at
Anbu late at night (several threads, minutes apart). Anbu then works through
them **one item at a time**, each by its own voice note, and says things like
*"want to reply for the first item …"*, *"respond to the item number 2"*.

## 1. Find the right parent message — the number alone will betray you

- "Item 1", "item 2" are **not** re-usable across email batches, and different
  items live in **different threads** sent the same night. Item 1 and item 2
  arrived 3 minutes apart on 20-Oct-2026 in two unrelated threads (O.S. 05/2026
  legal, and Ranka Iris / Shankar Electricals).
- Identify the thread **from the subject the user describes**, not the number:
  search his mailbox on the case/project vocabulary, read the candidate mails,
  and match the numbered asks inside each body.
- Confirm before drafting by naming the subject back to him in the reply.

## 2. Threading — the To-address trap that silently drafts to himself

Replying to a mail NDR sent **to Anbu** means the parent's `To:` header is
`pm2.blr@draas.com` — i.e. Anbu's own address. A naive `To = parent.To` builds
a draft addressed to the user himself (caught in-session 2026-10-23; the draft
had to be deleted and rebuilt).

```python
msg['To']        = <parent From:>          # ndr@draas.com — the SENDER
msg['Cc']        = <parent Cc verbatim>     # reply-all keeps his loops (Sarthak, Roshini)
msg['In-Reply-To'] = <parent Message-ID>
msg['References']  = <parent References> + ' ' + <parent Message-ID>
drafts().create(body={'message':{'raw': raw, 'threadId': parent['threadId']}})
```
Delete a wrong draft before creating the corrected one; verify after create by
reading back To/Cc/In-Reply-To and `message.threadId == parent.threadId`.

## 3. Dictation arrives in pieces — never finalise on the first note

Anbu sends a stream of short notes that **correct the previous one**. Same
session: date "12 December" → later "9th of December"; phone "944327 245" →
"944-327-0245"; "KVA" → **"caveat"**. Rules:

- Treat the first note as a draft of the content, not the final content. Build
  the draft, say what is still unverified, and wait.
- **Verify every date/number against a document** before putting it in a mail
  NDR will hold him to: the court notice says `appear 09.12.2026`, so a dictated
  "12th December" is wrong on its face.
- Leave an explicit `[___]` / `[one line on …]` placeholder rather than guessing
  a figure. He fills it in one keystroke; a wrong number in front of NDR is far
  more expensive.
- Garbled fragments that cannot be resolved (e.g. *"I, Panjait, Muttalika, all
  parties"*) get **dropped from the draft and raised as a question**, not
  paraphrased into something plausible.

## 4. Attaching a legacy Office file to a draft

Native draft tools take no attachment parameter — build the MIME message
(`EmailMessage` + `add_attachment`) as in `threaded-reply-pdf-attachments.md`.
When the source is a legacy `.doc` and no converter is installed
(`antiword`, `catdoc`, `wvText`, `soffice` all absent on this box), convert
**through Drive**:

```python
doc = g.files().create(
    body={'name':'TMP_CONVERT','mimeType':'application/vnd.google-apps.document'},
    media_body=MediaFileUpload(src, mimetype='application/msword'),
    fields='id').execute()                  # this CONVERTS the .doc
pdf = g.files().export(fileId=doc['id'], mimeType='application/pdf').execute()
g.files().delete(fileId=doc['id']).execute()          # tidy up
```
Then `pdftotext -layout` the PDF to confirm content landed, attach under a
dated descriptive filename (`20260119_Legal_Opinion_..._Sevaganapalli.pdf`).

## 5. "Pull it from the previous chat" = the workbook on disk, not Drive

When he says *"we prepared the Excel sheet previously in this chat / pull it from
the previous chart, that has the exact value"*, he means a workbook a past
session **generated into `/opt/data`** under a descriptive name
(`/opt/data/Shankar_Electrical_RankaIris_WorkOrder_Payment_Summary.xlsx`).

- Search `/opt/data` (and the session workspace) for descriptively named
  `.xlsx` before telling him nothing exists — a Drive/Gmail sweep will come back
  empty because the file was never uploaded.
- Read it with `openpyxl` (`data_only=True`) and quote the sheet's own totals in
  the draft; attach that same workbook so NDR gets the "working sheet" he asked
  for.
- Flag any staleness in the sheet (e.g. payments list ends 27-01-2026) rather
  than silently presenting it as current.

## 6. Draft shape that works for these replies

1. One line naming the matter + the WO/reference and its value.
2. The **scope position stated once**, plainly, early — for Anbu that is
   *coordination and execution follow-up only; he did not issue the WO; a
   full-time site team was in position*. Then stop arguing it; the rest of the
   mail is facts.
3. **Item-by-item, mirroring NDR's numbering**, each ending in an explicit
   recommendation (debit / do not debit / quantify at WO rate).
4. The numbers block, then what is still pending on Anbu's side, with a date.
5. One forward-looking flag (e.g. deductions must be raised during execution,
   not reconstructed at settlement). Keep it to a single paragraph — it is a
   process point, not a grievance.
6. Sign as `Anbu / Anbarasan M / VP, DRA Realty Private Limited / +91 81500 29900`.

## 7. Cross-checks worth doing unprompted

- **A document beats the dictation.** The advocate's own letterhead inside the
  attached opinion gave his name and `Cell: 94432 70245` — that is what
  verified the dictated phone number.
- **Read the exhibit for its actual scope before framing it.** That opinion
  certifies the *vendor's* title (Venkatamma & children, Patta 1343, clean EC);
  it is not an opinion that the suit is false. Say what the document says, and
  tell Anbu plainly that the "false case" line rests on his assertion, so NDR
  does not over-read the attachment.
- Photographs circulated later show **condition, not timing** — say exactly that
  when a damage claim is raised at settlement.
