# Triaging an automated "declined / rejected / failed" status email

Trigger (voice): *"There is an email saying my DSC has been declined and it's got some image in it.
I can't understand. Can you analyze the image and see if there's any reason why it has been declined?"*

These are machine-generated mails from CAs, banks, registrars, exchanges and KYC vendors
(Pantasign, eMudhra, NSDL/CSDL, MCA V3, K-RERA, banks). The reason is almost always **plain text**;
the "image" the user saw is a hosted logo that his mail client rendered inline. Do not go hunting for
a vision target.

## Order of operations

1. **Enumerate the MIME structure FIRST**, before any vision call:
   fetch `format='full'` and walk `payload.parts`, printing `mimeType / filename / body.size /
   attachmentId` for every part. If there are no parts with `filename` + `attachmentId`, **there is no
   attachment.** Say that explicitly — it is the single most useful thing you can tell him.
2. **Check for *embedded* images**, which are not attachments: fetch `format='raw'`, base64-decode, and
   grep the MIME source for `cid:` and `<img src=`. A remote `<img src="https://vendor.com/logo.png">`
   is chrome, not evidence. `cid:` count 0 + `Content-Type` parts of only `text/html` = text-only mail.
3. **Extract the visible text** by stripping HTML — that is where the reason lives:
   ```python
   import re, html
   t = re.sub(r'(?is)<(script|style)[^>]*>.*?</\1>', ' ', raw)
   t = re.sub(r'(?i)<br\s*/?>', '\n', t)
   t = re.sub(r'(?i)</(p|div|tr|td|h[1-6]|li)>', '\n', t)
   t = re.sub(r'<[^>]+>', ' ', t)
   t = html.unescape(t); t = re.sub(r'[ \t\xa0]+', ' ', t)
   print('\n'.join(l.strip() for l in t.split('\n') if l.strip()))
   ```
   Automated templates bury the reason in one unlabelled paragraph ("…has been declined due to the
   following reason: …"). Quote it **verbatim** and then translate it into what actually failed.
4. **Harvest the identifiers** the mail gives — application/certificate number, PAN, registered mobile,
   registered email, validity, PIN, decline timestamp. He will need every one of them to chase it, and
   the status portal link is usually in an `href` (list `href`s that are not `unsubscribe`/fonts).
5. **Pull the neighbouring messages from the same sender/domain around the same timestamp.** Automated
   pipelines emit contradictory sequences and the user is usually looking at the wrong one.

## Worked example — Pantasign DSC, 23-Oct-2026

`kyc.enrollment@pantasign.com` → NDR@DRAHOMES.IN, subject **"DSC HAS BEEN DECLINED"**,
Application No **2922790**, PAN AHVPR5168E, CLASS3 / 3-year, declined 23-10-2026 10:48:22.

Reason quoted: *"Video Script not read properly in video applicant is not available in video"* — i.e.
**two** separate eKYC failures (declaration script not read aloud verbatim; applicant not visible in
frame), either enough to fail on its own.

The MIME walk settled the "image" question in one call: one part, `text/html`, **no attachment, no
`cid:`, no `<img src>`** — the picture was Pantasign's hosted logo.

Same sender, same Application ID, same morning:

- 10:46:19 — "Video Recording Completed"
- **10:48:22 — DSC DECLINED**
- 10:50:10 — "Video Recording Completed" *again*
- 10:59 — NDR forwarded the 10:50 message onward, believing it was done

So the decline landed between two "completed" notices and a second recording followed it — but no
approval or re-issue mail ever arrived. **The application is still formally Declined as of the last
written notice.** Flag that anyone he forwarded the "completed" mail to may be working off stale
information, and check the vendor status portal by application number before assuming either way.

Remedy shape for a video-KYC decline: face fully in frame and well lit at eye level, read the script
aloud verbatim and slowly with no paraphrasing or looking away, one continuous take, ID visible if the
script calls for it. The decline mail itself is auto-generated and says *"please do not reply"* — route
the query to the vendor's support address (here `info@pantasign.com`), quoting the application number.

## Pitfalls

- **`vision_analyze` needs a real image file.** A remote `https://…/logo.png` URL comes back
  *"Only real image files are supported for vision analysis"* — download the asset to disk first and
  pass the local path. Do not read that error as a vision outage.
- **Do not treat "video recording completed" as success.** In eKYC pipelines the recording-completed
  notice is emitted by the capture step, *before* human/automated review; the decline is issued after.
  They can arrive in either order, and a later "completed" does not revoke an earlier decline.
- **Quote the vendor's own wording before interpreting it.** These templates are cryptic and the user
  wants to know exactly what was said as well as what it means.
- **Note the registered email address.** DSC/KYC applications are often filed under an alias
  (e.g. NDR@DRAHOMES.IN) rather than the primary work address, so searching only the main mailbox's
  `from:` will still work but the certificate will be issued against the alias. Worth telling him when
  the two differ.
