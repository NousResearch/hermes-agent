# Email paragraph & rich-text formatting rule (NDR — standing rule, always apply)

NDR stated this as a permanent rule covering his *whole* email drafting workflow:

> "This last rule has to be applied to my entire email drafting skill … so this rule is always and
> always followed."

Treat it as a hard default on every draft, not a per-request style choice.

## The rule

1. **No mid-sentence / mid-line line breaks.** In the `text/plain` part, a paragraph is ONE
   continuous line. A newline goes only where a new paragraph or a genuinely new line starts —
   never to wrap a long sentence. Do not hand-wrap at 70/80 columns.

2. **Never deliver a "text mask".** One undifferentiated block of prose is wrong. Break the content
   into proper paragraphs, and use **bullets** wherever the content is list-shaped.

3. **Rich text / HTML when asked for it** — triggers: "format it so it's visually easy to consume",
   "use rich text", "HTML", "bullet it correctly", "make it readable". Then emit a real `text/html`
   part: `<p>` per paragraph, `<ul>/<li>` for bullets, `<strong>` for emphasis. Always pair it with a
   matching `text/plain` part (`multipart/alternative`) — never HTML-only.

4. **Split multiple matters into numbered, bolded sections** instead of running them together as
   narrative. Real example: a five-matter mail became "1. Krishna's scope … 2. Document segregation
   … 3. Survey No. 40 … 4. CLU conversion … 5. Engineer Himanth …".

## Why rule 1 matters specifically

The plain-text part is what NDR reads in the mobile Gmail preview. Hard-wrapped or break-riddled
text renders with ragged, arbitrary gaps and reads as machine output — he has rejected drafts for
exactly this ("it reads as proper lines and paragraphs, no mid-sentence or mid-line written
characters at all").

## Checklist before saving any email draft

- [ ] `text/plain` part: every paragraph is a single unbroken line
- [ ] bullets used wherever the content is list-shaped
- [ ] `text/html` part present whenever "visual" / "HTML" / "rich text" was requested
- [ ] `text/plain` part present ALWAYS (HTML-only drafts break in clients that strip HTML)
- [ ] distinct matters split into numbered, bolded sections — not one continuous narrative
- [ ] sign-off on its own lines: `Regards,` / `Nishant Ranka` / `ndr@draas.com`

## Applying it to an existing draft

When NDR says "the rest of the draft is fine, only change X", do **not** recreate the draft — revise
it in place. Recipe: `gws-automation` skill, `references/email-draft-save-pattern.md` §4b.
Read the existing `text/plain` and `text/html` parts first so the untouched paragraphs are carried
over verbatim.
