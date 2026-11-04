# Ranka Oasis — Area-Discrepancy Confirmation-Request Reply (2026-09-15)

Working example of the multi-recipient CONFIRMATION-REQUEST email pattern (see
`templates/confirmation-request-email.html` for the reusable HTML skeleton).

## Source thread
- Thread = message id: `1a0a30b2fb179125` (single-message thread)
- Subject: "Discrepancy in Oasis Master Plan and Approval Plan Areas"
- From: Sinchana Gowda <sgowda@draas.com>, 2026-09-15 08:39 +0530
- To: ndr@draas.com, Anbarasan M <anbarasan@draas.com>, Kantesh B G <kanteshbg@draas.com>,
  Bhuvanesh Krishnan <bk@findingform.design>, Bhuvanesh S Krishnan <mail.bskrishnan@gmail.com>
- Body includes comparison sheet:
  https://docs.google.com/spreadsheets/d/1-gowmf9Tp4GfnLrU_it6-z87JGB590i4OQbUy0GeC74/edit

## Discrepancy reported (Sinchana)
- Sum of P-line plot areas = 17,564 sq m vs total area in text = 17,569 sq m → ~5 sq m gap.
- Cross-check on Bhurnesh's sorted master plan comparison = ~11 sq m gap.
- Assumed cause: minor drafting errors (line movements invisible in PDF but affecting
  AutoCAD area calculation).

## NDR's decision (dictated reply)
1. Governing basis = **area statement as sanctioned IN WORDS**. P-line / drawings are
   ignored — the approval is a PDF, not a DWG, so minor drafting variation is acceptable.
2. **ANBU (execution)** confirms: each on-ground plot area = exactly the sanctioned total
   area statement AND the sanctioned plot-wise inventory; total boundary = title area (as
   submitted while obtaining sanction) or MORE, never less (slightly more is OK); excludes
   the additional road-entry plots whose title has not yet come to DRA.
3. **SINCHANA (architect of record, after Anbu)** confirms: on-ground reality + area
   statement + what is sold to customers all reconcile; each plot = exactly sanctioned;
   sum of plot areas = sanctioned total in words; total boundary ≥ title area (excluding
   road-entry plots).
4. Closure **today**; if any blocker prevents the above → both get on a call with NDR and
   explain the blocker so he can take a call.

## Recipients ("respond to X, bring in Y" rule)
- To: Sinchana Gowda <sgowda@draas.com>
- Cc: Anbarasan M <anbarasan@draas.com> (thread address — NOT contacts-sheet
  pm2.blr@draas.com; headers win), Kantesh B G <kanteshbg@draas.com> (internal, on thread)
- Excluded deliberately: Bhuvanesh ×2 (design consultants) — their scope (drawings) was
  being set aside. Tell NDR who was left off and offer to re-add.

## Draft created
- Bridge call: `draft_reply_create(service_name="google-draas",
  message_id="1a0a30b2fb179125", to=..., cc=..., body=<HTML>, html=True)`
- Draft id `r9053375907132227802`, message `1a0a33645ebc775c`, thread `1a0a30b2fb179125`
- Verified via drafts().get(): payload.mimeType = text/html; labelIds ['DRAFT'];
  In-Reply-To / References = Sinchana's Message-ID
  `<CAM_+Haxtvz2HDtKhocKUAyKU01rgtAE8vLpK1X5bUKEbumFw4Q@mail.gmail.com>`
- Script pattern: write the draft script to /tmp via write_file (avoids the '&' heredoc
  backgrounding guard), run with /opt/hermes/.venv/bin/python3.