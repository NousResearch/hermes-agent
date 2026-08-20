# Wrapping and punctuation

Where lines start, where they end, where they break and which characters they use.

## Measure (line length)

Long lines make it harder for the eye to find the start of the next line. For long-form text, aim for 60–75 characters per line.

Any unit works. `65ch` measures characters directly (one `ch` is the width of the `0` in the current font), but a pixel or rem cap is just as good: at a `16px` body size the 60–75 character range lands roughly between `560px` and `680px` depending on the font, so Tailwind's `max-w-xl` (`576px`) or `max-w-2xl` (`672px`) fit. What matters is that a cap exists and the resulting line length sits in range; recheck it if the body font size changes.

## Alignment

`text-align` controls where each line starts and ends. `text-align: justify` stretches spaces until both edges line up; it can work in specific editorial layouts but avoid it in most interfaces.

## Wrapping

| Property | Use |
| --- | --- |
| `text-wrap: balance` | Distributes text evenly across multiple lines |
| `text-wrap: pretty` | Avoids leaving a single short word on the final line |
| `overflow-wrap: break-word` | Lets long words, links and IDs break before escaping the container |
| `white-space: nowrap` | Keeps labels and badges on one line where a break looks broken |

Use `balance` on headings and `pretty` on descriptions; combined they give the best outcome. Skip both in long-form text: browsers ignore `balance` past a few lines anyway, and evening out a whole paragraph wastes space and makes it harder to read.

## Truncation

- Single line: `text-overflow: ellipsis`, which needs `overflow: hidden` and `white-space: nowrap`.
- Multiple lines: `line-clamp` allows any number of lines before the ellipsis.

Truncation hides content. If the missing text matters, make the full value available elsewhere (tooltip or expanded view).

## Case

`text-transform` changes how case appears without changing the underlying text. Write copy naturally and control presentation with CSS, so changing presentation never requires rewriting copy.

## Smart punctuation

Keyboard characters are not always the best characters:

| Instead of | Use |
| --- | --- |
| Straight quotes `"..."` | Curly quotes that curve around the text (keep straight quotes in code) |
| Hyphen in ranges | En dash: `2010–2020` |
| Two hyphens for an aside | Em dash character |
| Three periods `...` | The single ellipsis character `…` |
| Regular space in `16 px` | `&nbsp;` so the value never breaks apart |
| Uncontrolled word breaks | `&shy;` to mark where a word may break |

## Internationalization

- Set the `lang` attribute so browsers choose the right quotes, hyphenation and pronunciation.
- Set `dir` at the document or content boundary where direction changes. Spatial mirroring and the logical-property table live in `better-layout`.

Two refinements for mixed-direction text:

- **Long paragraphs align by their own language.** A one- or two-line snippet follows the surrounding UI's direction, but a paragraph of three or more lines aligns to its own script's direction: an English paragraph stays start-aligned LTR even inside an RTL interface. `text-align: start` with the correct `lang`/`dir` on the paragraph element handles this.
- **Never reverse digits.** Numbers keep their digit order in every direction: a phone number or "541" reads identically in RTL. Browsers handle this via the Unicode bidi algorithm; don't fight it with manual reordering, and wrap mixed number/text values in `<bdi>` if adjacent RTL text disturbs them.
