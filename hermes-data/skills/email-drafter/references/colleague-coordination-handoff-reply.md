# Relaying a vendor's submissions + quotation to a COLLEAGUE and handing off coordination

Class of task NDR dictates as: *"take the latest email where his quotation is there as well as the
images / reference images, which I am not sure whom I had sent to — that email I want to send to
<colleague> as well, and tell <colleague> to please coordinate with him."*

Worked 2026-09-25 (AR Portable Cabin / Sayed → Gowri Singh, Ranka Udaya site marketing office).
Sibling of `oversized-forward-attachment-split.md` (that one is a FORWARD to a new recipient; this
one is a REPLY on the colleague's own existing thread, usually well under the size cap).

## Shape

1. **Recipient** — resolve from Gmail headers, not the contacts sheet. Gowri = `gsingh@draas.com`
   (marketing). The sheet's `gowrisingh1341@gmail.com` / `gowrisingh72@yahoo.com` are NOT the work
   identity. She may have already asked NDR a direct question (here: "please share his email ID and
   contact number") — **answer that question in the first block of the reply**, don't bury it.
2. **Thread identity** — reply on the COLLEAGUE's thread (subject like "Ranka Udaya Site Marketing
   Office"), anchored on the colleague's LATEST message, even though the material lives on a
   completely different vendor thread. Do NOT reply on the vendor thread.
3. **Threading** — fetch the anchor with `messages().get(format='raw')`, read its own `Message-ID`,
   set `In-Reply-To` = anchor's OWN Message-ID and `References` = anchor's chain + anchor Message-ID,
   plus explicit `threadId` in the create body.
4. **Vendor contact block** — always give name, both the personal and the `info@` address, both
   mobiles, and the vendor's own offices. The colleague is about to cold-start a relationship.

## The reuse check that matters most

Before re-uploading or re-sharing anything, **search Drive for the mirror folder a PRIOR relay
already created** and read its permissions:

```
d.files().list(q="name contains 'AR Portable' and trashed = false", fields="files(id,name,mimeType,parent)")
d.permissions().list(fileId=FOLDER, fields="permissions(id,type,role,emailAddress,expirationTime)")
```

Ranka Udaya refs folder: `Container Site Office (Prefab) — AR Portable Cabin Refs`,
id `1mq2pdtX_sXNeZhgjZoTEAyqjsd2v-BVc` — holds the 3 heavy JPGs (~7.7 MB each) and the 3 videos
(Prestige Somerville 143 MB, Prestige Serenity Shores 199 MB, Assetz 59 MB). Gowri was **already
`reader`** on it from the 22-Sep run. The correct move was to LINK it, not re-upload or re-grant.
Colleague already on it → link. Not on it → add `reader` (visibility) / `writer` (if they must act
on the files), then verify the returned role.

## ⚠️ Audit the permission list before you link a folder into an outbound email

Enumerating that folder surfaced **three unrelated external grants with `reader`**:
`sudheer@vgrofinserv.com`, `praveen@vgrofinserv.com`, `vamshiraj@vgrofinserv.com` (an NBFC's team,
almost certainly a mis-grant from an unrelated session). Linking a folder into a mail without
reading its ACL means shipping a folder whose audience you have not verified — and silently
publishing the counterparty's material to whoever else happens to be on it. **Always
`permissions().list()` a shared folder before linking it. Surface any anomalous external grant to
NDR and ask before revoking — do not act unilaterally.**

## Attach / link split (well under the cap at ~3.7 MB)

Attach: the vendor's quotation PDF, OUR render PNG, OUR interior execution spec PDF, and 4 of the
small watermarked reference JPGs (≈600 KB). Link: the mirror folder (heavy JPGs + videos) and the
Google Earth / KML view of the reference sites. Keep the small-photo filenames role-scoped
("AR Portable - Executed Reference 1.jpg"), not the raw `IMG-20191023-WA0027_Watermarked.jpg`.

## Body shape NDR accepted (HTML, 4 colour-coded blocks)

1. blue — **vendor's contact details** (the answer to what they asked)
2. amber — **"everything he has sent us is here"**: attachment inventory with figures quoted
   (e.g. the three quote options), plus the Drive folder link and the Earth link
3. green — **what the vendor has already confirmed**: timeline commitment, container-vs-structural
   recommendation, factory/site split, and the list of executed sites they can show
4. red — **what I need from here**: (a) share the updated plan/render with him, (b) get a final
   ITEMISED quotation against the new look + new spec, spelled out element by element, "the version
   that will let us place the work order"

Then two closers: NDR is happy to work with the colleague on spec compliance — *"you take the call
and finalise with him"* on open technical questions — and *push for the work order and the payment
so his clock starts*. End "Keep me posted at each step."

## Verify (all four, every time)

- `drafts().list()` contains the draft id (authoritative — a fetched draft resource can read back
  `['DRAFT']` and later be SENT)
- MIME walk prints each `parts[].filename` + size → expected attachment count and exact bytes
- decode the `text/html` part and assert it `startswith('<div')` (proves it renders as HTML, not
  literal markup) and that the key links/phrases are present
- From == `Nishant Ranka <ndr@draas.com>` (run with `HERMES_SESSION_USER_ID=7449813913`)

## Pitfall — `%` in the body text kills `%`-formatting

Real bodies contain `18% GST`, `95% of the work`, `60% advance`. Building the string with
`html = """...""" % (FOLDER, EARTH)` then raises `TypeError: must be real number, not str`. Use
token placeholders and `.replace()`:

```python
html = "...<a href=\"__FOLDER__\">...  18% GST ..."
html = html.replace("__FOLDER__", FOLDER).replace("__EARTH__", EARTH)
text = text.replace("__FOLDER__", FOLDER).replace("__EARTH__", EARTH)
```

And because the script builds BOTH a `text` and an `html` body, a bulk `str.replace()`-style patch
of the script can silently point the `text` closing line at `html` — leaving `text` un-rewritten and
sending literal `__FOLDER__` tokens in the plain-text alternative. After any programmatic edit of a
two-body build script, grep the assignment lines and confirm each variable name matches.
