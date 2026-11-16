# Building Fillable Forms: spec format and workflow

The same JSON spec drives both `pdf_form_layout.py` (design lint) and
`pdf_make_form.py` (AcroForm build). Coordinates are PDF points, origin
at the bottom-left of the page (1 pt = 1/72 inch; A4 is 595.27 x 841.89,
letter is 612 x 792).

## Spec shape

```json
{
  "title": "Example Intake Form",
  "author": "example-author",
  "page_size": "A4",
  "page_count": 1,
  "fields": [
    {"name": "surname", "type": "text", "page": 1,
     "label": "Surname", "label_box": [72, 700, 150, 714],
     "entry_box": [160, 696, 400, 716],
     "value": "", "tooltip": "Family name"},

    {"name": "agree", "type": "checkbox", "page": 1,
     "label": "I agree", "label_box": [72, 660, 150, 674],
     "entry_box": [160, 658, 176, 674], "checked": false},

    {"name": "color", "type": "radio", "page": 1,
     "label": "Color", "label_box": [72, 620, 150, 634],
     "entry_box": [160, 616, 400, 636],
     "options": ["red", "blue"], "value": "blue"},

    {"name": "size", "type": "dropdown", "page": 1,
     "label": "Size", "label_box": [72, 580, 150, 594],
     "entry_box": [160, 576, 300, 596],
     "options": ["small", "large"], "value": "small"}
  ]
}
```

- `page_size`: `"A4"`, `"letter"`, or `[width, height]` in points.
- `page_count`: optional; extended automatically to the highest field page.
- Boxes are `[x0, y0, x1, y1]` with `x0 < x1`, `y0 < y1`.
- `label` is drawn as static text near `label_box`; omit it (and
  `label_box`) for unlabeled fields.
- `radio`: the buttons are laid out left-to-right inside `entry_box`,
  one slot per option, each with a small static caption. `value`
  pre-selects an option by its export name.
- `dropdown` maps to an AcroForm choice (combo) field.

## Field types → what pdf_read.py --fields reports

| Spec type | /FT | value format after fill |
|---|---|---|
| text | /Tx (`text`) | the string |
| checkbox | /Btn (`button`) | `/Yes` or `/Off` |
| radio | /Btn (`button`) | `/<export>`, e.g. `/red` |
| dropdown | /Ch (`choice`) | the option string |

When filling with `pdf_fill_form.py`, checkboxes accept `true`/`false`;
radio values need the leading slash (`"/red"`); dropdown values are the
plain option string.

## Layout lint rules (pdf_form_layout.py)

Per field, on its declared page:

- boxes must be well-formed and inside the page bounds;
- entry boxes must be at least 8x8 pt (12 pt tall for text/dropdown);
- no two entry boxes on the same page may overlap (the second and later
  fields of an overlapping cluster are flagged);
- a label must sit within 150 pt of its entry box and must not overlap it.

Exit code 0 = clean, 1 = at least one problem; the JSON report lists
per-field `problems`. Lint the spec BEFORE building — fixing numbers in
JSON is cheaper than debugging a rendered PDF.

## Visual review loop

```bash
python3 scripts/pdf_form_layout.py spec.json --render-overlay overlay.png [--pdf built.pdf]
```

Red rectangles = entry boxes (with field names), blue = label boxes.
Without `--pdf` the overlay is drawn on a blank page (PIL-only, always
works); with `--pdf` the real page is rasterized underneath
(needs pypdfium2 or pdftoppm — otherwise the report says
`"rendered": false` with install hints). Feed the PNG to `vision_analyze`
and ask specifically about collisions, alignment, and stray labels.

## Radio-group quirks (reportlab + pypdf)

- reportlab requires at least two `radio()` calls per group; a
  single-option radio group produces a broken field.
- Pre-selecting is done at build time via `"value"`; changing selection
  later via `pdf_fill_form.py` needs the slashed export name (`"/red"`).
- Some viewers render reportlab radio appearances inconsistently after a
  pypdf fill; verify with `--fields` (data truth) plus a rendered page
  image (visual truth) rather than either alone.
- Flattening radio groups is the least reliable flatten case — check the
  output image before shipping.
