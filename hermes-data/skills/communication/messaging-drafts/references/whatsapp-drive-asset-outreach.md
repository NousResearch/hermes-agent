# WhatsApp Outreach Messages That Carry Drive/Web Asset Links

Pattern: user wants a WhatsApp intro message to a specific person (e.g. a golf-course
manager near the project, an SME owner) that also hands over project assets —
brochure, videos, completed photos, Google Maps pin. ALWAYS delivered as a
whatsapp_link deep link (tool-mandated), NEVER constructed by hand, NEVER claimed sent.

## Before drafting: inventory the assets and make them shareable
Drive files are PRIVATE by default (team-only perms). A WhatsApp recipient with no
Google login gets a permission error on a non-public link. For EVERY Drive asset in the
message:
1. Verify the file exists and find it by name/query (folder listing, not guesswork).
2. Add the public permission: `gws_drive_permissions_add(file_id, perm_type='anyone',
   role='reader')` — this creates the anyoneWithLink condition.
3. Local-only assets (e.g. site photos the user attached earlier) must be UPLOADED to
   Drive first with a public permission before they can be linked:
   build_service('drive','v3', service_name=<resolved>) + MediaIoBaseUpload +
   permissions().create(type=anyone, role=reader). Use /opt/hermes/.venv for the GWS
   client. Then format https://drive.google.com/file/d/<id>/view.

## Compress big attachments BEFORE upload (WhatsApp-friendly)
User asks to "add all documents compressed version" or sends a brochure PDF / project
video / ad PDF as chat attachments that are 13–29 MB — WhatsApp recipients with slow
networks never open those. Compress locally first, then upload the SMALL version public
and link THAT (verified 2026-10, 29 MB brochure + 23 MB ad PDF + 13 MB video → 1.9 MB +
0.46 MB + 3.6 MB):
- PDFs via ghostscript (host has /usr/bin/gs): `gs -sDEVICE=pdfwrite
  -dCompatibilityLevel=1.5 -dPDFSETTINGS=/ebook -dNOPAUSE -dQUIET -dBATCH
  -sOutputFile=out.pdf in.pdf` — /ebook is a safe default (~94% reduction on image-heavy
  brochures). /screen shrinks more but can wash out renders; try /ebook first.
- Video via ffmpeg (host has /usr/bin/ffmpeg): check specs with `ffprobe -v error
  -select_streams v:0 -show_entries stream=width,height,duration -of csv=p=0`; then
  `ffmpeg -y -v error -i in.mp4 -vf "scale=640:360" -c:v libx264 -crf 28 -preset fast
  -c:a aac -b:a 96k -movflags +faststart out.mp4` — 720p/13MB → 640x360/~3.6MB,
  WhatsApp-faststart friendly.
- Extract a single ad/offer image from a multi-page ad PDF for the message lead-in:
  `/opt/hermes/.venv/bin/python` with pymupdf — `pix = page.get_pixmap(dpi=100);
  pix.save('ad.jpg', jpg_quality=80)` per page, then vision-check which page is the
  right creative before linking it. It is not an email body embed (a WhatsApp deep link
  carries TEXT ONLY) — the image rides as its own Drive link placed right after the
  offer intro line, and tell the user that is how the ad travels.
- Report the size reduction honestly in the reply (29MB → 1.9MB etc.) so the user knows
  the link is the small version; the local originals stay in /opt/data/ranka_udaya_whatsapp/
  (uploads from /mnt/uploads are read-only — copy first, then compress).
4. Google Maps pin: use the VERIFIED project coordinates, not a guess:
   https://maps.google.com/?q=<lat>,<lng>

## Honesty rule for assets that don't exist
If the user asks for a "drone view link" or "location video" and no such file exists
in Drive or public portals, SAY SO explicitly and link what does exist (project film,
RG video), offering to swap when they provide the hosted link. Never fabricate a URL
for an asset you can't verify.

## Structure of the message
- Open with neighbourhood/common-ground framing (e.g. "we are your neighbours at
  Sevaganapalli — Ranka Udaya is right next to your golf course") — hooks read better
  than a sales intro.
- Numbered bullets for the assets: BROCHURE, PROJECT FILM, RENTAL GUARANTEE VIDEO,
  COMPLETED PHOTOS, GOOGLE MAPS, WEBSITE — each on its own line with the URL.
- One-line offer hook (guaranteed rent Rs 60,000/month, 5 years, 5% escalation).
- Signature block: name, role, RERA No., phone, website.
- Optional add-on asks (e.g. promote to members / standee space) framed as a
  low-risk partnership, not a hard sell.

## Delivery
- Use whatsapp_link with phone= and platform='telegram'; deliver the display_link
  verbatim (never the bare URL).
- If the recipient NAME or number isn't known/resolvable, generate the link with a
  [Name] placeholder and tell the user exactly what to fill; offer to regenerate with
  the number pre-filled when they provide it.
- Report as a link-only delivery: nothing sent; the recipient taps it on a device
  with WhatsApp.