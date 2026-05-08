# KOBUS HTTP/API Probe Notes

Session-proven on 2026-05-08. Goal: avoid browser automation where possible.

## Base

```text
https://www.kobus.co.kr
```

Use a desktop User-Agent, HTTP/1.1 if needed, a cookie jar, and referers.

## Tested Flow

### Route / Terminal Candidates

```text
POST /mrs/readRotLinInf.ajax
```

Observed JSON keys:

```text
tfrLen
tfrInfList
len
codeYn
rotInfList
```

One probe returned about 1,208 route records in `rotInfList`.

### Timetable

```text
POST /mrs/alcnSrch.do
```

Example tested route/date:

```text
서울경부(010) -> 부산(700), 2026-05-09
```

Observed result:

```text
42 schedule links/cards
25 selectable seat snippets
first selectable: 00:30 / 천일고속 / 심야우등 / 10 seats
```

Typical POST fields:

```text
deprCd=010
arvlCd=700
pathDvs=sngl
pathStep=1
deprDtm=YYYYMMDD
busClsCd=0
rtrpChc=1
timeLinkMin=00
timeLinkMax=23
```

Parse `fnSatsChc(...)` onclick values for the next step. Example:

```text
fnSatsChc('20260509','003000','003000','010','700','3','07','0','Y','N','010','700','N','N','N','N')
```

### Seat / Fare Stage

```text
POST /mrs/satschc.do
```

Send the original `alcnSrchFrm` hidden fields plus selected values from `fnSatsChc(...)`, including values such as:

```text
deprTime=003000
alcnDeprTime=003000
alcnDeprTrmlNo=010
alcnArvlTrmlNo=700
indVBusClsCd=3
cacmCd=07
dcDvsCd=0
prvtBbizEmpAcmtRt=N
chldSftySatsYn=N
dsprSatsYn=N
```

Observed response contained `form#satsChcFrm` and hidden values:

```json
{
  "deprTime": "003000",
  "alcnDeprTrmlNo": "010",
  "alcnArvlTrmlNo": "700",
  "adltFee": "47600",
  "rmnSatsNum": "10",
  "totSatsNum": "28"
}
```

### Temporary Hold

```text
POST /mrs/setPcpy.ajax
```

Observed successful response markers:

```text
MSG_CD=S0000
pcpyNoAll
satsNoAll
ESTM_AMT
DC_AMT
TISSU_AMT
```

### Hold Cancellation

```text
POST /mrs/cancPcpy.ajax
```

Observed success marker:

```text
MSG_CD=S0000
```

### Checkout Entry

```text
POST /mrs/stplcfmpym.do
```

The POST body must include the selected schedule/seat form values plus temporary hold identifiers and fare amounts. A helper page can auto-submit this form to the official KOBUS endpoint.

For mobile browsers, use:

```text
POST /mrs/stplcfmpym.do?keep=/mrs/pay
```

This preserves the POST body while placing `/mrs/pay` in `location.href`, which avoids a KOBUS client-side mobile redirect condition observed in the common JavaScript.

## Interpretation

- Login was not required for route lookup, timetable lookup, seat-selection-page entry, temporary hold, or checkout-entry page display in the tested flow.
- Page HTML can include login or `grecaptchaToken` forms, but these did not block the tested lookup/seat-stage path.
- Final payment should remain a manual, explicitly confirmed stage.
- KOBUS mobile behavior is less stable than desktop because common JavaScript can redirect narrow screens to the mobile main page.
