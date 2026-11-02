# Gmail draft-creation gotchas (verified 2026-09-15)

## 1. Terminal `&`-guard breaks HTML-entity drafts in heredocs

Creating a Gmail draft via `terminal()` with an inline heredoc fails when the
HTML body contains entity references (`&mdash;`, `&amp;`, `&nbsp;`, raw `&` in
text like "BOQ & Area Statement"):

- Error: `Foreground command uses '&' backgrounding...` exit -1/130 — the
  foreground guard treats `&` as shell backgrounding and refuses the command.
- Some of these getchas also abort mid-run with `[Command interrupted]`
  (exit 130) after printing just `WHOAMI:` — the draft may or may not have been
  created before the interrupt, so ALWAYS list drafts after an interrupted run
  and delete strays before continuing.

**Fix:** write the script to a file with `write_file()` and run
`python3 /tmp/script.py` — never inline-heredoc an HTML body containing raw `&`
entities. This also makes the script re-runnable/verifiable.

**Pattern (verified working):**
1. `write_file('/tmp/create_draft.py', ...)` — full script with `build_service()` + `drafts().create()`
2. `cd /opt/hermes && HERMES_SESSION_USER_ID=<ndr> /opt/hermes/.venv/bin/python3 /tmp/create_draft.py`
3. Verify with `drafts().get()` — print To/Cc/Subject/ThreadId and the compose link.

## 2. Interrupted draft runs leave stray artifacts

After a failed/interrupted draft creation, the Gmail thread list may show a
message with the intended subject but NO thread reply headers (thread =
`1a0a40abf206d7d8` style, its own thread). Check `drafts().list()` and delete
strays BEFORE creating the corrected version, so NDR's Drafts folder never holds
two versions of the same email.

## 3. Colleague work emails — People API surname/org search (verified)

The NDR DRAAS contacts sheet often carries only personal/Gmail addresses for
employees. When NDR says "use her work email / she has a @draas.com address":

- People API `searchContacts(query=<surname>)` across google-draas returned:
  - Gowri Singh → `gsingh@draas.com` (content lead; mobile +91 98454 30607)
  - Sai Neha Vaddadi ("Neha") → `nvaddadi@draas.com` (mobile +91 78993 98273;
    personal esotericarts.ani@gmail.com)
- The contacts sheet rows for the same people held only yahoo/gmail addresses —
  the @draas.com work identity lived in People API (org: DRA Realty Private Limited).
- Workspace Admin directory API (`admin.users().list`) FAILS with
  `insufficientAuthenticationScopes` on the google-draas vault token — do not
  rely on it; People API searchContacts is the sanctioned lookup.

## 4. Verify identity before drafting on behalf of NDR

Always print `WHOAMI:` (`gmail.users().getProfile()['emailAddress']`) first and
confirm it equals `ndr@draas.com` when drafting for NDR — a colleague's session
token can silently resolve to their own mailbox.