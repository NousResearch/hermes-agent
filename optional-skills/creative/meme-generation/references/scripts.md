# Meme Generation Scripts

## `scripts/imgflip_download.py`

Use this script to discover and download blank Imgflip templates.

Responsibilities:
- search Imgflip templates by name or ID
- rank close matches by normalized name, prefix, substring, and token overlap
- download the raw blank template image to a local file

Use this when the task starts from a meme source/template request and a local blank image is needed before captioning.

## `scripts/meme_caption.py`

Use this script to caption an already-downloaded local image.

Responsibilities:
- add meme-style captions to a local image
- optionally trim near-white padding before rendering
- optionally render the text in bars above and below the image

Use this when the source image already exists locally, including an Imgflip download or any other blank template.

## `scripts/generate_meme.py`

Use this script as the main renderer and compatibility wrapper.

Responsibilities:
- resolve curated templates by ID or name
- resolve Imgflip templates as blank sources
- render captions directly onto a template image
- caption custom local images through `--image`
- list curated templates and search Imgflip templates

Prefer `imgflip_download.py` when the workflow should explicitly fetch a blank source first.
Prefer `meme_caption.py` when the workflow should explicitly caption a local image.
Use `generate_meme.py` when a single entry point is enough or when compatibility with the older combined flow is desired.
