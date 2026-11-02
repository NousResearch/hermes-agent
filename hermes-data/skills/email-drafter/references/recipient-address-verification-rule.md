# Recipient Address Verification — NDR's Standing Rule (stated 2026-10-07)

Every email addressee must be verified BEFORE the draft is created:

1. The address must appear in NDR's contacts (online contact sheet
   'NDR DRAAS Google contacts' OR Google Contacts People API across
   google-draas / google-ahfl / google-gmail).
2. There must be prior non-bounced mail to that address (search Gmail sent
   items). If the address was used before AND BOUNCED, do not reuse it —
   look up the correct address from the contact sheet / prior sent mail.
3. If you cannot verify an address this way, do NOT put it on the draft;
   flag it to NDR instead of guessing.

Verified example (Sep-2026): `arvindsarchitects@gmail.com` hard-bounced
(550 5.1.1 address not found) on an area-statement email. The correct AJ
Architects address on the thread was Shrinidhi GV —
`shrinidhi.a.j.architects@gmail.com` (contacts row 4249). Arvind Jain =
`arch_arvind2000@yahoo.co.in`.

Voice-note names map to real people on the thread — "Sriniti / Srin / his
Sridevi" = Shrinidhi GV; "Sinchana" = sgowda@draas.com (in-house architect);
"Anbu" = pm2.blr@draas.com. Resolve every name via contact_resolver or the
thread headers; never hand-type a guessed domain.