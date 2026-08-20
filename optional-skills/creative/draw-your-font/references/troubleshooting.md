# Troubleshooting capture & segmentation

The binarizer is adaptive (local background estimate) with two knobs on
`segment`/`make`:

- `--delta N` (default 40): how much darker than the local paper a pixel must
  be to count as ink. Lower = more sensitive (catches faint pens, also more
  noise). Raise to 55–70 for photos with heavy shadows misread as ink.
- `--cap N` (default 165): absolute grey ceiling for ink after contrast
  normalization. Anything lighter is never ink - this is what erases the
  template's grey guides. Lower to 140 if printed guides survive into blobs;
  raise to 190 for a very faint pencil (expect more noise).

Re-running segment rewrites blobs.json and renumbers every blob - discard
any labels.json written earlier and relabel from the fresh contact sheet
(prefer a fresh `-d` workdir).

## Symptoms → fixes

**Hundreds of tiny blobs** - noisy paper texture or aggressive delta. Re-run
segment with `--delta 55`. If the photo is low light, ask for a brighter shot.

**A huge blob spanning the page** - a shadow edge or the page border got
thresholded. Crop the photo to just the paper (or re-shoot from directly
above), or raise `--delta`.

**Letters missing entirely** - pen too faint (pencil, gel on glossy paper).
Try `--delta 25 --cap 190`. If still missing, the honest fix is rewriting
with a darker pen; say so.

**Strokes broken into fragments** - thin ballpoint. Try `--delta 30`, then
`build --weight 1` to fatten what traced. Recommend a 0.5mm+ pen for the
re-shoot.

**Two letters in one box** - they touch on paper. No code fix; ask the user
to re-write just those letters with space between them, segment the new
photo, and merge via labels.

**Template guides appear as blobs** - printer printed the grey too dark.
Re-run with `--cap 140`. If their printer only does solid black, they can
still use the template - the guides will show as long thin blobs; label them
all `""`.

**i/j dots detached as separate blobs** - normally auto-merged; if the dot is
very far from the stem it may not be. Label the stem blob with the letter and
the dot `""` - or better, relabel both after a re-shoot. (A dotless i still
reads fine in most handwriting.)

**Rotated/skewed photo** - mild angles are fine and become part of the font's
character. A strongly rotated photo (>5°) will slant every glyph; ask for a
straighter shot rather than trying to compensate.
